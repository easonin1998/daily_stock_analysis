# -*- coding: utf-8 -*-
"""Account-aware portfolio strategy integration tests."""

from __future__ import annotations

from datetime import date, timedelta
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.repositories.portfolio_repo import PortfolioRepository
from src.services.portfolio_service import PortfolioService
from src.services.portfolio_strategy_service import (
    PortfolioStrategyService,
    update_martin_holding_plan as real_update_martin_holding_plan,
)
from src.services.screening.config import Config as ScreeningConfig
from src.storage import DatabaseManager


class PortfolioStrategyServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        DatabaseManager.reset_instance()
        self.db = DatabaseManager(db_url="sqlite:///:memory:")
        self.repo = PortfolioRepository(self.db)
        self.portfolio = PortfolioService(repo=self.repo)
        self.strategy = PortfolioStrategyService(
            repo=self.repo,
            screening_config=ScreeningConfig(
                strategies_dir=(
                    Path(__file__).resolve().parents[1]
                    / "src"
                    / "services"
                    / "screening"
                    / "strategies"
                ),
                daily_feature_library_path=Path(self.temp_dir.name) / "missing.json",
                data_dir=Path(self.temp_dir.name) / "screening",
            ),
        )
        self.start = date(2025, 1, 1)
        rows = []
        for index in range(130):
            close = 10.0 + index * 0.1
            rows.append({
                "date": self.start + timedelta(days=index),
                "open": close - 0.03,
                "high": close + 0.12,
                "low": close - 0.12,
                "close": close,
                "volume": 1_000_000 + index * 100,
                "amount": close * 1_000_000,
                "pct_chg": 0.5,
            })
        self.db.save_daily_data(pd.DataFrame(rows), code="600519", data_source="unit-test")
        self.as_of = self.start + timedelta(days=129)

        account = self.portfolio.create_account(
            name="策略账户",
            broker="Demo",
            market="cn",
            base_currency="CNY",
        )
        self.account_id = account["id"]
        self.portfolio.record_cash_ledger(
            account_id=self.account_id,
            event_date=self.start,
            direction="in",
            amount=100_000,
            currency="CNY",
        )
        self.portfolio.record_trade(
            account_id=self.account_id,
            symbol="600519",
            trade_date=self.start + timedelta(days=120),
            side="buy",
            quantity=100,
            price=22.0,
            market="cn",
            currency="CNY",
            strategy="martin_momentum_pullback",
            strategy_version="3.3",
            strategy_action="initial_entry",
            signal_run_id="run-initial",
            initial_stop_price=21.5,
        )

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        self.temp_dir.cleanup()

    def test_account_plan_updates_managed_holding_and_persists_state(self) -> None:
        with patch.object(
            self.strategy,
            "_load_feature_library",
            side_effect=AssertionError("local StockDaily should avoid loading the full feature library"),
        ):
            plan = self.strategy.get_account_plan(
                account_id=self.account_id,
                as_of=self.as_of,
                market_regime={"state": "risk_on", "profile": {"max_positions": 5}},
            )

        self.assertEqual(plan["account"]["managed_position_count"], 1)
        holding = plan["holdings"][0]
        self.assertTrue(holding["managed"])
        self.assertEqual(holding["symbol"], "600519")
        self.assertEqual(holding["quantity"], 100.0)
        self.assertEqual(holding["sellable_quantity"], 100.0)
        self.assertGreater(holding["current_r_multiple"], 0)
        self.assertEqual(holding["add_on_plan"]["status"], "ready")
        state = self.repo.get_strategy_position(
            account_id=self.account_id,
            symbol="600519",
            market="cn",
            strategy="martin_momentum_pullback",
        )
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.last_evaluated_date, self.as_of)
        self.assertIn("current_r_multiple", state.last_plan_json or "")

    def test_same_complete_bar_refresh_reuses_persisted_holding_plan(self) -> None:
        with patch(
            "src.services.portfolio_strategy_service.update_martin_holding_plan",
            wraps=real_update_martin_holding_plan,
        ) as update_mock:
            first = self.strategy.get_account_plan(
                account_id=self.account_id,
                as_of=self.as_of,
            )
            second = self.strategy.get_account_plan(
                account_id=self.account_id,
                as_of=self.as_of + timedelta(days=1),
            )

        self.assertEqual(update_mock.call_count, 1)
        self.assertEqual(
            first["holdings"][0]["ema9_below_streak"],
            second["holdings"][0]["ema9_below_streak"],
        )
        state = self.repo.get_strategy_position(
            account_id=self.account_id,
            symbol="600519",
            market="cn",
            strategy="martin_momentum_pullback",
        )
        self.assertEqual(state.last_evaluated_date, self.as_of)

    def test_screening_candidates_receive_sequential_account_sizing(self) -> None:
        response = {
            "strategy": "martin_momentum_pullback",
            "analysis_date": self.as_of.isoformat(),
            "market_regime": {
                "state": "strong_risk_on",
            },
            "candidates": [
                self._candidate("000001", 20.0, 19.0),
                self._candidate("000002", 20.0, 19.0),
            ],
        }

        enriched = self.strategy.enrich_screening_result(
            response,
            account_id=self.account_id,
        )

        first = enriched["candidates"][0]["trade_plan"]["position_sizing"]
        second = enriched["candidates"][1]["trade_plan"]["position_sizing"]
        self.assertEqual(first["recommended_quantity"], 500)
        self.assertEqual(second["recommended_quantity"], 500)
        self.assertLess(second["available_cash_before"], first["available_cash_before"])
        self.assertLess(
            second["available_heat_amount_before"],
            first["available_heat_amount_before"],
        )

    def test_risk_off_blocks_account_sizing(self) -> None:
        response = {
            "strategy": "martin_momentum_pullback",
            "analysis_date": self.as_of.isoformat(),
            "market_regime": {"state": "risk_off"},
            "candidates": [self._candidate("000001", 20.0, 19.0)],
        }

        enriched = self.strategy.enrich_screening_result(
            response,
            account_id=self.account_id,
        )

        sizing = enriched["candidates"][0]["trade_plan"]["position_sizing"]
        self.assertEqual(sizing["execution_status"], "market_blocked")
        self.assertEqual(sizing["recommended_quantity"], 0)
        self.assertFalse(enriched["portfolio_plan"]["rules"]["new_entries_allowed"])

    def test_account_plan_loads_persisted_market_regime_when_not_supplied(self) -> None:
        with patch(
            "src.services.portfolio_strategy_service.load_market_regime_history",
            return_value={"state": "risk_off", "date": self.as_of.isoformat()},
        ):
            plan = self.strategy.get_account_plan(
                account_id=self.account_id,
                as_of=self.as_of,
            )

        self.assertEqual(plan["account"]["market_regime_state"], "risk_off")
        self.assertEqual(plan["account"]["market_regime_date"], self.as_of.isoformat())
        self.assertFalse(plan["rules"]["new_entries_allowed"])

    def test_short_local_history_falls_back_without_using_future_library_row(self) -> None:
        short_history = pd.DataFrame([{
            "date": self.as_of,
            "open": 22.8,
            "high": 23.0,
            "low": 22.7,
            "close": 22.9,
            "volume": 1_000_000,
            "amount": 22_900_000,
        }])
        library = pd.DataFrame([
            {
                "code": "600519",
                "daily_latest_date": self.as_of.isoformat(),
                "daily_data_points": 130,
                "close": 22.9,
                "ema21": 22.0,
            },
            {
                "code": "600519",
                "daily_latest_date": (self.as_of + timedelta(days=1)).isoformat(),
                "daily_data_points": 131,
                "close": 30.0,
                "ema21": 25.0,
            },
        ])

        result = self.strategy._latest_features(
            symbol="600519",
            history=short_history,
            library=library,
            as_of=self.as_of,
        )

        self.assertEqual(result["daily_data_points"], 130)
        self.assertEqual(result["close"], 22.9)

    def test_cn_strategy_rejects_non_cn_execution_account(self) -> None:
        account = self.portfolio.create_account(
            name="美股账户",
            broker="Demo",
            market="us",
            base_currency="USD",
        )

        with self.assertRaisesRegex(ValueError, "supports"):
            self.strategy.get_account_plan(
                account_id=account["id"],
                as_of=self.as_of,
            )

    def test_account_sizing_rejects_ineffective_one_lot_position(self) -> None:
        response = {
            "strategy": "martin_momentum_pullback",
            "analysis_date": self.as_of.isoformat(),
            "market_regime": {"state": "strong_risk_on"},
            "candidates": [self._candidate("000001", 50.0, 46.0)],
        }

        enriched = self.strategy.enrich_screening_result(
            response,
            account_id=self.account_id,
        )

        sizing = enriched["candidates"][0]["trade_plan"]["position_sizing"]
        self.assertEqual(sizing["execution_status"], "blocked")
        self.assertEqual(sizing["recommended_quantity"], 0)
        self.assertIn("最小有效仓位", sizing["constraint_reason"])

    def test_add_on_rebuilds_peak_r_from_close_based_position_structure(self) -> None:
        self.strategy.get_account_plan(account_id=self.account_id, as_of=self.as_of)
        self.portfolio.record_trade(
            account_id=self.account_id,
            symbol="600519",
            trade_date=self.as_of,
            side="buy",
            quantity=100,
            price=22.8,
            market="cn",
            currency="CNY",
            strategy="martin_momentum_pullback",
            strategy_version="3.3",
            strategy_action="add_on",
            signal_run_id="run-add",
            initial_stop_price=22.3,
        )

        plan = self.strategy.get_account_plan(account_id=self.account_id, as_of=self.as_of)
        holding = plan["holdings"][0]
        self.assertTrue(holding["position_structure_reset"])
        self.assertEqual(holding["tranche_count"], 2)
        self.assertEqual(holding["add_on_count"], 1)
        self.assertLess(holding["peak_r_multiple"], 2.0)

    def test_partial_exit_does_not_turn_small_position_into_full_exit(self) -> None:
        rounded = self.strategy._round_partial_sell(
            quantity=100,
            sellable=100,
            fraction=0.2,
            lot_size=100,
        )

        self.assertEqual(rounded, 0.0)

    def test_partial_exit_rounds_down_to_a_share_lot(self) -> None:
        rounded = self.strategy._round_partial_sell(
            quantity=600,
            sellable=600,
            fraction=0.2,
            lot_size=100,
        )

        self.assertEqual(rounded, 100.0)

    def test_untagged_position_change_requires_reconciliation(self) -> None:
        self.portfolio.record_trade(
            account_id=self.account_id,
            symbol="600519",
            trade_date=self.as_of,
            side="buy",
            quantity=100,
            price=22.8,
            market="cn",
            currency="CNY",
        )

        plan = self.strategy.get_account_plan(account_id=self.account_id, as_of=self.as_of)

        holding = plan["holdings"][0]
        self.assertFalse(holding["managed"])
        self.assertEqual(holding["status"], "tracking_mismatch")

    def test_adopting_legacy_position_starts_tracking_on_adoption_date(self) -> None:
        account = self.portfolio.create_account(
            name="旧持仓",
            broker="Demo",
            market="cn",
            base_currency="CNY",
        )
        account_id = account["id"]
        self.portfolio.record_cash_ledger(
            account_id=account_id,
            event_date=self.start,
            direction="in",
            amount=100_000,
            currency="CNY",
        )
        self.portfolio.record_trade(
            account_id=account_id,
            symbol="600519",
            trade_date=self.start + timedelta(days=30),
            side="buy",
            quantity=100,
            price=13.0,
            market="cn",
            currency="CNY",
        )

        plan = self.strategy.adopt_position(
            account_id=account_id,
            symbol="600519",
            initial_stop_price=12.5,
            as_of=self.as_of,
        )

        holding = plan["holdings"][0]
        self.assertEqual(holding["opened_date"], self.as_of.isoformat())
        state = self.repo.get_strategy_position(
            account_id=account_id,
            symbol="600519",
            market="cn",
            strategy="martin_momentum_pullback",
        )
        self.assertEqual(state.opened_date, self.as_of)

    def test_adopted_position_does_not_replay_same_add_on_across_days(self) -> None:
        account = self.portfolio.create_account(
            name="纳管加仓",
            broker="Demo",
            market="cn",
            base_currency="CNY",
        )
        account_id = account["id"]
        self.portfolio.record_cash_ledger(
            account_id=account_id,
            event_date=self.start,
            direction="in",
            amount=100_000,
            currency="CNY",
        )
        self.portfolio.record_trade(
            account_id=account_id,
            symbol="600519",
            trade_date=self.start + timedelta(days=30),
            side="buy",
            quantity=100,
            price=13.0,
            market="cn",
            currency="CNY",
        )
        self.strategy.adopt_position(
            account_id=account_id,
            symbol="600519",
            initial_stop_price=12.5,
            as_of=self.as_of,
        )
        add_date = self.as_of + timedelta(days=1)
        self.portfolio.record_trade(
            account_id=account_id,
            symbol="600519",
            trade_date=add_date,
            side="buy",
            quantity=100,
            price=22.8,
            market="cn",
            currency="CNY",
            strategy="martin_momentum_pullback",
            strategy_version="3.3",
            strategy_action="add_on",
            initial_stop_price=22.3,
        )

        first = self.strategy.get_account_plan(account_id=account_id, as_of=add_date)
        second = self.strategy.get_account_plan(
            account_id=account_id,
            as_of=add_date + timedelta(days=1),
        )

        self.assertEqual(first["holdings"][0]["tranche_count"], 2)
        self.assertEqual(second["holdings"][0]["tranche_count"], 2)
        self.assertEqual(second["holdings"][0]["quantity"], 200.0)
        self.assertEqual(first["holdings"][0]["status"], "awaiting_close")
        self.assertEqual(second["holdings"][0]["status"], "awaiting_close")
        self.assertGreater(first["holdings"][0]["risk_amount"], 0)
        self.assertNotEqual(second["holdings"][0]["status"], "tracking_mismatch")

    def test_new_buy_waits_for_matching_completed_daily_bar(self) -> None:
        buy_date = self.as_of + timedelta(days=1)
        account = self.portfolio.create_account(
            name="待收盘账户",
            broker="Demo",
            market="cn",
            base_currency="CNY",
        )
        account_id = account["id"]
        self.portfolio.record_cash_ledger(
            account_id=account_id,
            event_date=self.start,
            direction="in",
            amount=100_000,
            currency="CNY",
        )
        self.portfolio.record_trade(
            account_id=account_id,
            symbol="600519",
            trade_date=buy_date,
            side="buy",
            quantity=100,
            price=22.8,
            market="cn",
            currency="CNY",
            strategy="martin_momentum_pullback",
            strategy_version="3.3",
            strategy_action="initial_entry",
            initial_stop_price=22.3,
        )

        plan = self.strategy.get_account_plan(account_id=account_id, as_of=buy_date)

        holding = plan["holdings"][0]
        self.assertEqual(holding["status"], "awaiting_close")
        self.assertEqual(holding["action"], "wait_for_complete_close")
        self.assertEqual(holding["data_date"], self.as_of.isoformat())
        self.assertEqual(holding["risk_amount"], 50.0)
        self.assertGreater(plan["account"]["strategy_risk_heat_pct"], 0)
        self.assertIsNone(self.repo.get_strategy_position(
            account_id=account_id,
            symbol="600519",
            market="cn",
            strategy="martin_momentum_pullback",
        ))

    def test_next_open_exit_uses_projected_t1_sellable_quantity(self) -> None:
        account = self.portfolio.create_account(
            name="次日退出账户",
            broker="Demo",
            market="cn",
            base_currency="CNY",
        )
        account_id = account["id"]
        self.portfolio.record_cash_ledger(
            account_id=account_id,
            event_date=self.start,
            direction="in",
            amount=100_000,
            currency="CNY",
        )
        self.portfolio.record_trade(
            account_id=account_id,
            symbol="600519",
            trade_date=self.as_of,
            side="buy",
            quantity=100,
            price=25.0,
            market="cn",
            currency="CNY",
            strategy="martin_momentum_pullback",
            strategy_version="3.3",
            strategy_action="initial_entry",
            initial_stop_price=24.5,
        )

        plan = self.strategy.get_account_plan(account_id=account_id, as_of=self.as_of)

        holding = plan["holdings"][0]
        self.assertEqual(holding["status"], "exit_signal")
        self.assertEqual(holding["sellable_quantity"], 0.0)
        self.assertEqual(holding["projected_next_open_sellable_quantity"], 100.0)
        self.assertEqual(holding["recommended_sell_quantity"], 100.0)
        self.assertEqual(holding["execution_status"], "ready")

    @staticmethod
    def _candidate(code: str, max_price: float, stop: float) -> dict:
        return {
            "code": code,
            "name": code,
            "trade_plan": {
                "signal_status": "buy_signal",
                "entry": {"max_price": max_price},
                "exit": {"initial_stop_price": stop},
                "position_sizing": {
                    "max_position_weight_pct": 20.0,
                    "initial_position_weight_pct": 10.0,
                    "max_portfolio_risk_per_trade_pct": 0.5,
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
