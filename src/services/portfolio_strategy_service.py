# -*- coding: utf-8 -*-
"""Account-aware Martin strategy planning for manually maintained portfolios."""

from __future__ import annotations

from datetime import date
import json
import math
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from src.repositories.portfolio_repo import PortfolioRepository
from src.services.portfolio_service import EPS, PortfolioService
from src.services.screening.adaptive_exit import is_profit_exhaustion_bar
from src.services.screening.config import Config as ScreeningConfig
from src.services.screening.daily import compute_daily_features
from src.services.screening.daily_library import load_daily_feature_library
from src.services.screening.market_regime import (
    load_market_regime_history,
    resolve_market_regime_limits,
)
from src.services.screening.strategy import load_all_strategies
from src.services.screening.trade_plan import (
    MartinTradePlanConfig,
    resolve_martin_entry_gap_bounds,
    update_martin_holding_plan,
)

DEFAULT_STRATEGY = "martin_momentum_pullback"


class PortfolioStrategyService:
    """Join portfolio events, daily features, and strategy execution rules."""

    def __init__(
        self,
        *,
        repo: Optional[PortfolioRepository] = None,
        screening_config: Optional[ScreeningConfig] = None,
    ) -> None:
        self.repo = repo or PortfolioRepository()
        self.portfolio = PortfolioService(repo=self.repo)
        self.screening_config = screening_config or ScreeningConfig.from_env()
        self._daily_feature_library: Optional[pd.DataFrame] = None

    def get_account_plan(
        self,
        *,
        account_id: int,
        as_of: Optional[date] = None,
        strategy: str = DEFAULT_STRATEGY,
        market_regime: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        as_of_date = as_of or date.today()
        strategy_model, rules = self._load_strategy(strategy)
        snapshot = self.portfolio.get_portfolio_snapshot(
            account_id=account_id,
            as_of=as_of_date,
            cost_method="fifo",
            include_realtime=False,
        )
        if not snapshot.get("accounts"):
            raise ValueError(f"Account not found: {account_id}")
        account = snapshot["accounts"][0]
        account_market = str(account.get("market") or "").lower()
        supported_markets = {
            str(item).lower() for item in strategy_model.screening.market_scope
        }
        if supported_markets and account_market not in supported_markets:
            raise ValueError(
                f"Strategy {strategy} supports {sorted(supported_markets)}, "
                f"but account {account_id} uses market {account_market or 'unknown'}"
            )
        positions = account.get("positions") or []
        trades = self.repo.list_trades(account_id, as_of=as_of_date)
        actions = self.repo.list_corporate_actions(account_id, as_of=as_of_date)
        states = {
            (row.symbol, row.market): row
            for row in self.repo.list_strategy_positions(
                account_id=account_id,
                strategy=strategy,
            )
        }
        regime = (
            dict(market_regime)
            if isinstance(market_regime, dict) and market_regime
            else self._load_market_regime(as_of_date)
        )
        regime_state = str(regime.get("state") or "unavailable")
        regime_limits = resolve_market_regime_limits(
            regime,
            base_max_positions=5,
            base_risk_per_trade_pct=rules.max_portfolio_risk_per_trade_pct,
            base_max_portfolio_heat_pct=rules.max_portfolio_heat_pct,
            base_max_position_weight_pct=rules.max_position_weight_pct,
            base_min_position_weight_pct=rules.min_position_weight_pct,
        )

        holding_plans: list[Dict[str, Any]] = []
        active_state_keys: set[tuple[str, str]] = set()
        for position in positions:
            symbol = str(position.get("symbol") or "")
            market = str(position.get("market") or account.get("market") or "cn")
            state = states.get((symbol, market))
            tranches, campaign = self._resolve_live_tranches(
                position=position,
                trades=trades,
                actions=actions,
                strategy=strategy,
                state=state,
                as_of=as_of_date,
            )
            if not tranches:
                holding_plans.append(self._unmanaged_holding(position))
                continue
            active_state_keys.add((symbol, market))
            holding_plans.append(
                self._build_holding_plan(
                    account_id=account_id,
                    position=position,
                    tranches=tranches,
                    campaign=campaign,
                    state=state,
                    strategy=strategy,
                    strategy_version=strategy_model.version,
                    rules=rules,
                    market_regime_state=regime_state,
                    as_of=as_of_date,
                    library=None,
                )
            )

        for key, state in states.items():
            if state.status == "active" and key not in active_state_keys:
                self.repo.upsert_strategy_position(
                    account_id=account_id,
                    symbol=state.symbol,
                    market=state.market,
                    currency=state.currency,
                    strategy=strategy,
                    values={
                        "status": "closed",
                        "closed_date": as_of_date,
                        "tracking_mode": "trade_events",
                        "tracking_trade_cutoff_id": 0,
                        "entry_tranches_json": "[]",
                        "last_evaluated_date": as_of_date,
                    },
                )

        equity = float(account.get("total_equity") or 0.0)
        managed_risk = sum(float(item.get("risk_amount") or 0.0) for item in holding_plans)
        risk_heat_pct = managed_risk / equity * 100.0 if equity > EPS else 0.0
        unmanaged_count = sum(1 for item in holding_plans if not item.get("managed"))
        summary = {
            "account_id": account_id,
            "account_name": account.get("account_name") or "",
            "as_of": as_of_date.isoformat(),
            "currency": account.get("base_currency") or "CNY",
            "market": account_market,
            "total_cash": float(account.get("total_cash") or 0.0),
            "total_market_value": float(account.get("total_market_value") or 0.0),
            "total_equity": equity,
            "position_count": len(positions),
            "managed_position_count": len(positions) - unmanaged_count,
            "unmanaged_position_count": unmanaged_count,
            "strategy_risk_amount": round(managed_risk, 2),
            "strategy_risk_heat_pct": round(risk_heat_pct, 4),
            "market_regime_state": regime_state,
            "market_regime_date": regime.get("date"),
        }
        result = {
            "status": "available",
            "strategy": strategy,
            "strategy_version": strategy_model.version,
            "account": summary,
            "holdings": holding_plans,
            "rules": {
                "max_positions": int(regime_limits["max_positions"]),
                "target_exposure_pct": float(regime_limits["target_exposure_pct"]),
                "new_entries_allowed": bool(regime_limits["new_entries_allowed"]),
                "max_portfolio_heat_pct": float(regime_limits["max_portfolio_heat_pct"]),
                "max_portfolio_risk_per_trade_pct": float(regime_limits["risk_per_trade_pct"]),
                "max_position_weight_pct": float(regime_limits["max_position_weight_pct"]),
                "min_position_weight_pct": float(regime_limits["min_position_weight_pct"]),
                "initial_position_weight_pct": rules.initial_position_weight_pct,
                "add_on_position_weight_pct": rules.add_on_position_weight_pct,
                "max_add_ons": rules.max_add_ons,
                "first_add_on_r_multiple": rules.first_add_on_r_multiple,
                "second_add_on_r_multiple": rules.second_add_on_r_multiple,
                "lot_size": rules.lot_size,
            },
            "limitations": (
                ["存在未纳入策略管理的持仓，其风险未计入策略热度。"]
                if unmanaged_count
                else []
            ),
        }
        self._apply_add_on_sizing(result, trade_rules=rules)
        return result

    def _load_market_regime(self, as_of: date) -> Dict[str, Any]:
        """Load the latest non-forward-looking persisted market state."""

        try:
            regime = load_market_regime_history(
                self.screening_config.data_dir / "market_regime_history.json"
            )
        except Exception:
            return {}
        regime_date = self._parse_date(regime.get("date"))
        if regime_date is not None and (
            regime_date > as_of or (as_of - regime_date).days > 14
        ):
            return {}
        return regime

    def enrich_screening_result(
        self,
        response: Dict[str, Any],
        *,
        account_id: int,
    ) -> Dict[str, Any]:
        analysis_date = self._parse_date(response.get("analysis_date")) or date.today()
        strategy = str(response.get("strategy") or DEFAULT_STRATEGY)
        account_plan = self.get_account_plan(
            account_id=account_id,
            as_of=analysis_date,
            strategy=strategy,
            market_regime=response.get("market_regime"),
        )
        screening_market = str(response.get("market") or "").lower()
        account_market = str(account_plan["account"].get("market") or "").lower()
        if screening_market and screening_market != account_market:
            raise ValueError(
                f"Screening market {screening_market} does not match account market {account_market}"
            )
        self._apply_candidate_sizing(response.get("candidates") or [], account_plan)
        response["portfolio_plan"] = account_plan
        response["account_id"] = account_id
        return response

    def adopt_position(
        self,
        *,
        account_id: int,
        symbol: str,
        initial_stop_price: float,
        as_of: Optional[date] = None,
        strategy: str = DEFAULT_STRATEGY,
        signal_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        as_of_date = as_of or date.today()
        snapshot = self.portfolio.get_portfolio_snapshot(
            account_id=account_id,
            as_of=as_of_date,
            cost_method="fifo",
            include_realtime=False,
        )
        account = (snapshot.get("accounts") or [{}])[0]
        normalized = self.portfolio._normalize_symbol_for_position(symbol)
        position = next(
            (
                item
                for item in account.get("positions") or []
                if self.portfolio._normalize_symbol_for_position(item.get("symbol") or "")
                == normalized
            ),
            None,
        )
        if position is None:
            raise ValueError(f"Position not found in account {account_id}: {symbol}")
        avg_cost = float(position.get("avg_cost") or 0.0)
        if initial_stop_price <= 0 or initial_stop_price >= avg_cost:
            raise ValueError("initial_stop_price must be above 0 and below current average cost")
        strategy_model, _ = self._load_strategy(strategy)
        position_market = str(position.get("market") or account.get("market") or "").lower()
        supported_markets = {
            str(item).lower() for item in strategy_model.screening.market_scope
        }
        if supported_markets and position_market not in supported_markets:
            raise ValueError(
                f"Strategy {strategy} supports {sorted(supported_markets)}, "
                f"but position {symbol} uses market {position_market or 'unknown'}"
            )
        trades = self.repo.list_trades(account_id, as_of=as_of_date)
        matching = [
            row
            for row in trades
            if self.portfolio._normalize_symbol_for_position(row.symbol) == normalized
        ]
        cutoff = max((int(row.id) for row in matching), default=0)
        # Adoption starts a new auditable strategy baseline. Reusing the
        # earliest historical trade would fabricate holding age and could
        # trigger the follow-through exit immediately.
        opened = as_of_date
        quantity = float(position.get("quantity") or 0.0)
        current_close = float(position.get("last_price") or avg_cost)
        tranches = [{
            "tranche_index": 0,
            "type": "adopted",
            "shares": quantity,
            "remaining_shares": quantity,
            "price": avg_cost,
            "cost_per_share": avg_cost,
            "protective_stop_price": float(initial_stop_price),
            "risk_per_share": avg_cost - float(initial_stop_price),
            "highest_close": current_close,
            "entry_date": opened.isoformat(),
            "source_trade_id": None,
        }]
        self.repo.upsert_strategy_position(
            account_id=account_id,
            symbol=str(position.get("symbol") or normalized),
            market=str(position.get("market") or account.get("market") or "cn"),
            currency=str(position.get("currency") or account.get("base_currency") or "CNY"),
            strategy=strategy,
            values={
                "strategy_version": strategy_model.version,
                "status": "active",
                "tracking_mode": "adopted_snapshot",
                "opened_date": opened,
                "closed_date": None,
                "initial_signal_run_id": (signal_run_id or "").strip() or None,
                "initial_stop_price": float(initial_stop_price),
                "highest_close": current_close,
                "highest_position_r_multiple": 0.0,
                "partial_exit_count": 0,
                "ema9_below_streak": 0,
                "tracking_trade_cutoff_id": cutoff,
                "entry_tranches_json": json.dumps(tranches, ensure_ascii=False),
                "last_plan_json": None,
                "last_evaluated_date": as_of_date,
            },
        )
        return self.get_account_plan(
            account_id=account_id,
            as_of=as_of_date,
            strategy=strategy,
        )

    def _load_strategy(self, strategy: str) -> tuple[Any, MartinTradePlanConfig]:
        strategies = load_all_strategies(self.screening_config.strategies_dir)
        model = strategies.get(strategy)
        if model is None:
            raise ValueError(f"Strategy not found: {strategy}")
        profile = model.screening.execution_profile or {}
        allowed = MartinTradePlanConfig.__dataclass_fields__
        values = {key: value for key, value in profile.items() if key in allowed}
        if bool(profile.get("pyramiding_enabled", False)):
            values.setdefault("min_position_weight_pct", 8.0)
        return model, MartinTradePlanConfig(**values)

    def _load_feature_library(self) -> pd.DataFrame:
        if self._daily_feature_library is not None:
            return self._daily_feature_library
        path = self.screening_config.daily_feature_library_path
        if path is None:
            self._daily_feature_library = pd.DataFrame()
            return self._daily_feature_library
        try:
            self._daily_feature_library = load_daily_feature_library(
                path,
                max_age_days=36_500,
            )
        except Exception:
            self._daily_feature_library = pd.DataFrame()
        return self._daily_feature_library

    def _build_holding_plan(
        self,
        *,
        account_id: int,
        position: Dict[str, Any],
        tranches: list[Dict[str, Any]],
        campaign: Dict[str, Any],
        state: Any,
        strategy: str,
        strategy_version: str,
        rules: MartinTradePlanConfig,
        market_regime_state: str,
        as_of: date,
        library: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        quantity = float(position.get("quantity") or 0.0)
        tracked_quantity = sum(float(item.get("remaining_shares") or 0.0) for item in tranches)
        symbol = str(position.get("symbol") or "")
        if abs(quantity - tracked_quantity) > max(EPS, quantity * 1e-6):
            return {
                **self._unmanaged_holding(position),
                "status": "tracking_mismatch",
                "reason": (
                    f"策略批次合计 {tracked_quantity:g} 股，与账户持仓 {quantity:g} 股不一致；"
                    "请修正成交记录或重新纳入策略。"
                ),
            }

        average_cost = sum(
            float(item["remaining_shares"]) * float(item["cost_per_share"])
            for item in tranches
        ) / tracked_quantity
        total_risk = sum(
            float(item["remaining_shares"]) * float(item["risk_per_share"])
            for item in tranches
        )
        aggregate_stop = average_cost - total_risk / tracked_quantity
        opened_date = campaign.get("opened_date") or as_of
        entry_dates = [
            parsed
            for item in tranches
            if (parsed := self._parse_date(item.get("entry_date"))) is not None
        ]
        latest_entry_date = max(entry_dates, default=opened_date)

        history = self._load_local_history(symbol=symbol, as_of=as_of)
        features = self._latest_features(
            symbol=symbol,
            history=history,
            library=library,
            as_of=as_of,
        )
        current_close = self._number(features.get("close")) or self._number(position.get("last_price"))
        ema9 = self._number(features.get("ema9"))
        ema21 = self._number(features.get("ema21"))
        adr20 = self._number(features.get("adr_20_pct"))
        daily_data_points = self._number(features.get("daily_data_points"))
        data_date = (
            self._parse_date(features.get("daily_latest_date"))
            or self._parse_date(position.get("price_date"))
            or as_of
        )
        data_age_days = max((as_of - data_date).days, 0)
        common_plan = {
            "managed": True,
            "symbol": symbol,
            "market": position.get("market"),
            "quantity": quantity,
            "sellable_quantity": float(position.get("sellable_quantity", quantity) or 0.0),
            "projected_next_open_sellable_quantity": quantity,
            "avg_cost": float(position.get("avg_cost") or average_cost),
            "current_close": current_close,
            "market_value": float(position.get("market_value_base") or 0.0),
            "strategy": strategy,
            "strategy_version": strategy_version,
            "opened_date": opened_date.isoformat(),
            "data_date": data_date.isoformat(),
            "data_age_days": data_age_days,
            "daily_source": features.get("daily_source") or position.get("price_source"),
            "daily_data_points": (
                int(daily_data_points) if daily_data_points is not None else None
            ),
            "adr_20_pct": adr20,
            "initial_stop_price": float(campaign.get("initial_stop_price") or aggregate_stop),
            "risk_amount": round(total_risk, 2),
            "tranche_count": len(tranches),
            "add_on_count": int(campaign.get("add_on_count") or 0),
            "market_regime_state": market_regime_state,
        }
        if (
            current_close is None
            or ema21 is None
            or (daily_data_points is not None and daily_data_points < 21)
            or data_age_days > 14
        ):
            return {
                **common_plan,
                "status": "data_insufficient",
                "action": "hold_without_rule_change",
                "reason": "缺少最新收盘价、至少21根完整日线，或行情已超过14天未更新；今日不自动改变持仓条件单。",
            }
        if data_date < latest_entry_date:
            return {
                **common_plan,
                "status": "awaiting_close",
                "action": "wait_for_complete_close",
                "reason": "最新完整日线早于最近一笔买入，等待该交易日收盘后再更新退出和加仓条件。",
            }

        structure_increased = self._position_structure_increased(state, tranches)
        structure_changed = self._position_structure_changed(state, tranches)
        cached_plan = self._json_dict(getattr(state, "last_plan_json", None)) if state else {}
        cached_close = self._number(cached_plan.get("current_close"))
        cached_regime = str(cached_plan.get("market_regime_state") or "unavailable")
        if (
            state is not None
            and state.last_evaluated_date == data_date
            and not structure_changed
            and cached_plan
            and cached_close is not None
            and abs(cached_close - current_close) <= EPS
            and cached_regime == market_regime_state
        ):
            cached_plan.update({
                "quantity": quantity,
                "sellable_quantity": float(position.get("sellable_quantity", quantity) or 0.0),
                "market_value": float(position.get("market_value_base") or 0.0),
            })
            return cached_plan
        persisted_high = self._number(getattr(state, "highest_close", None)) if state else None
        highest_close = max(current_close, persisted_high or current_close)
        if not history.empty and campaign.get("opened_date"):
            opened = campaign["opened_date"]
            closes = pd.to_numeric(
                history.loc[history["date"].dt.date >= opened, "close"],
                errors="coerce",
            ).dropna()
            if not closes.empty and state is None:
                highest_close = max(highest_close, float(closes.max()))
        for tranche in tranches:
            tranche["highest_close"] = max(
                self._number(tranche.get("highest_close")) or current_close,
                current_close,
            )

        previous_peak_r = self._number(
            getattr(state, "highest_position_r_multiple", None)
        ) if state and not structure_increased else None
        current_r = (
            (current_close * tracked_quantity - average_cost * tracked_quantity) / total_risk
            if total_risk > EPS
            else 0.0
        )
        historical_peak_r = (
            (highest_close * tracked_quantity - average_cost * tracked_quantity) / total_risk
            if total_risk > EPS
            else current_r
        )
        peak_r = max(
            current_r,
            (
                historical_peak_r
                if state is None or str(getattr(state, "status", "")) != "active"
                else current_r
            ),
            previous_peak_r if previous_peak_r is not None else current_r,
        )
        holding_days = self._holding_days(history, opened_date, data_date)
        partial_count = max(
            int(getattr(state, "partial_exit_count", 0) or 0) if state else 0,
            int(campaign.get("partial_exit_count") or 0),
        )
        previous_ema9_below_streak = (
            int(getattr(state, "ema9_below_streak", 0) or 0) if state else 0
        )
        if structure_increased:
            previous_ema9_below_streak = 0
        elif state is not None and state.last_evaluated_date == data_date:
            previous_ema9_below_streak = max(previous_ema9_below_streak - 1, 0)
        exhaustion = self._exhaustion_signal(features, rules)
        plan = update_martin_holding_plan(
            entry_price=average_cost,
            initial_stop_price=aggregate_stop,
            current_close=current_close,
            highest_close=highest_close,
            ema21=ema21,
            ema9=ema9,
            market_regime_state=market_regime_state,
            partial_exit_count=partial_count,
            previous_ema9_below_streak=previous_ema9_below_streak,
            adr_20_pct=adr20,
            exhaustion_signal=exhaustion,
            holding_days=holding_days,
            entry_tranches=tranches,
            highest_position_r_multiple=peak_r,
            estimated_exit_cost_bps=15.0,
            config=rules,
        )
        sellable = float(position.get("sellable_quantity", quantity) or 0.0)
        projected_next_open_sellable = quantity
        if plan.get("status") == "exit_signal":
            plan["recommended_sell_quantity"] = projected_next_open_sellable
            plan["execution_status"] = "ready" if projected_next_open_sellable > EPS else "blocked"
        else:
            partial_orders = plan.get("partial_exit_orders") or []
            if partial_orders:
                fraction = float(partial_orders[0].get("sell_fraction") or 0.0)
                partial_quantity = self._round_partial_sell(
                    quantity=quantity,
                    sellable=projected_next_open_sellable,
                    fraction=fraction,
                    lot_size=rules.lot_size,
                )
                plan["recommended_partial_sell_quantity"] = partial_quantity
                if partial_quantity > EPS:
                    plan["action"] = "partial_sell_next_open_call_auction"
                    plan["execution_status"] = "ready"
                else:
                    plan["execution_status"] = "blocked_min_lot"
                    plan["partial_exit_note"] = "目标减仓数量不足一手，保留持仓并于下一交易日重新评估。"

        plan.update({
            **common_plan,
            "sellable_quantity": sellable,
            "projected_next_open_sellable_quantity": projected_next_open_sellable,
            "position_structure_reset": structure_increased,
        })
        should_persist = (
            state is None
            or getattr(state, "last_evaluated_date", None) is None
            or data_date >= state.last_evaluated_date
        )
        if should_persist:
            self.repo.upsert_strategy_position(
                account_id=account_id,
                symbol=symbol,
                market=str(position.get("market") or "cn"),
                currency=str(position.get("currency") or "CNY"),
                strategy=strategy,
                values={
                    "strategy_version": strategy_version,
                    "status": "active",
                    "tracking_mode": campaign.get("tracking_mode") or "trade_events",
                    "opened_date": opened_date,
                    "closed_date": None,
                    "initial_signal_run_id": campaign.get("signal_run_id"),
                    "initial_stop_price": float(campaign.get("initial_stop_price") or aggregate_stop),
                    "highest_close": float(plan.get("position_high_water_price") or highest_close),
                    "highest_position_r_multiple": float(plan.get("peak_r_multiple") or peak_r),
                    "partial_exit_count": partial_count,
                    "ema9_below_streak": int(plan.get("ema9_below_streak") or 0),
                    "tracking_trade_cutoff_id": int(campaign.get("tracking_trade_cutoff_id") or 0),
                    "entry_tranches_json": json.dumps(tranches, ensure_ascii=False),
                    "last_plan_json": json.dumps(plan, ensure_ascii=False),
                    "last_evaluated_date": data_date,
                },
            )
        return plan

    def _resolve_live_tranches(
        self,
        *,
        position: Dict[str, Any],
        trades: Iterable[Any],
        actions: Iterable[Any],
        strategy: str,
        state: Any,
        as_of: date,
    ) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
        symbol = self.portfolio._normalize_symbol_for_position(position.get("symbol") or "")
        market = str(position.get("market") or "cn")
        currency = str(position.get("currency") or "CNY")
        campaign: Dict[str, Any] = {
            "tracking_mode": getattr(state, "tracking_mode", "trade_events") if state else "trade_events",
            "tracking_trade_cutoff_id": int(getattr(state, "tracking_trade_cutoff_id", 0) or 0) if state else 0,
            "opened_date": getattr(state, "opened_date", None) if state else None,
            "signal_run_id": getattr(state, "initial_signal_run_id", None) if state else None,
            "initial_stop_price": getattr(state, "initial_stop_price", None) if state else None,
            "partial_exit_count": 0,
            "add_on_count": 0,
        }
        tranches: list[Dict[str, Any]] = []
        if state and state.tracking_mode == "adopted_snapshot":
            tranches = self._json_list(state.entry_tranches_json)

        cutoff = campaign["tracking_trade_cutoff_id"] if tranches else 0
        events = []
        for row in actions:
            if (
                state
                and state.tracking_mode == "adopted_snapshot"
                and state.last_evaluated_date is not None
                and row.effective_date <= state.last_evaluated_date
            ):
                continue
            if (
                self.portfolio._normalize_symbol_for_position(row.symbol) == symbol
                and str(row.market) == market
                and str(row.currency) == currency
            ):
                events.append((row.effective_date, 0, int(row.id), "corp", row))
        for row in trades:
            if int(row.id) <= cutoff:
                continue
            if (
                self.portfolio._normalize_symbol_for_position(row.symbol) == symbol
                and str(row.market) == market
                and str(row.currency) == currency
            ):
                events.append((row.trade_date, 1, int(row.id), "trade", row))
        events.sort(key=lambda item: (item[0], item[1], item[2]))

        for event_date, _, _, event_type, event in events:
            if event_type == "corp":
                if (event.action_type or "").strip().lower() == "split_adjustment":
                    ratio = float(event.split_ratio or 0.0)
                    if ratio > 0:
                        for item in tranches:
                            item["remaining_shares"] = float(item["remaining_shares"]) * ratio
                            item["shares"] = float(item.get("shares") or item["remaining_shares"]) * ratio
                            item["price"] = float(item["price"]) / ratio
                            item["cost_per_share"] = float(item["cost_per_share"]) / ratio
                            item["protective_stop_price"] = float(item["protective_stop_price"]) / ratio
                            item["risk_per_share"] = float(item["risk_per_share"]) / ratio
                continue

            side = str(event.side or "").lower()
            quantity = float(event.quantity or 0.0)
            event_strategy = str(event.strategy or "")
            if campaign["tracking_mode"] == "adopted_snapshot":
                campaign["tracking_trade_cutoff_id"] = max(
                    int(campaign.get("tracking_trade_cutoff_id") or 0),
                    int(event.id),
                )
            if event_strategy != strategy:
                # Do not invent a stop or exit stage for manual transactions.
                # Leaving the event untracked makes the quantity reconciliation
                # fail visibly so the user can correct or re-adopt the position.
                continue
            if side == "buy":
                stop = self._number(event.initial_stop_price)
                if stop is None:
                    stop = self._number(campaign.get("initial_stop_price"))
                if stop is None or stop >= float(event.price):
                    continue
                cost = (quantity * float(event.price) + float(event.fee or 0.0) + float(event.tax or 0.0)) / quantity
                tranche = {
                    "tranche_index": len(tranches),
                    "type": str(event.strategy_action or ("initial" if not tranches else "manual_add")),
                    "shares": quantity,
                    "remaining_shares": quantity,
                    "price": float(event.price),
                    "cost_per_share": cost,
                    "protective_stop_price": stop,
                    "risk_per_share": max(cost - stop, 0.01),
                    "highest_close": float(event.price),
                    "entry_date": event_date.isoformat(),
                    "source_trade_id": int(event.id),
                }
                tranches.append(tranche)
                if campaign.get("opened_date") is None:
                    campaign["opened_date"] = event_date
                    campaign["signal_run_id"] = event.signal_run_id
                    campaign["initial_stop_price"] = stop
                if str(event.strategy_action or "") == "add_on":
                    campaign["add_on_count"] += 1
                continue
            if side == "sell" and tranches:
                remaining = quantity
                for item in tranches:
                    available = float(item.get("remaining_shares") or 0.0)
                    consumed = min(available, remaining)
                    item["remaining_shares"] = available - consumed
                    remaining -= consumed
                    if remaining <= EPS:
                        break
                tranches = [item for item in tranches if float(item.get("remaining_shares") or 0.0) > EPS]
                if str(event.strategy_action or "") == "partial_exit":
                    campaign["partial_exit_count"] += 1
                if not tranches:
                    campaign.update({
                        "opened_date": None,
                        "signal_run_id": None,
                        "initial_stop_price": None,
                        "add_on_count": 0,
                        "partial_exit_count": 0,
                    })
        return tranches, campaign

    def _load_local_history(self, *, symbol: str, as_of: date) -> pd.DataFrame:
        aliases = self.portfolio._build_symbol_filter_values(symbol)
        rows = self.repo.get_daily_history(symbols=aliases, as_of=as_of, limit=320)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame([
            {
                "date": row.date,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
                "amount": row.amount,
            }
            for row in rows
        ]).assign(date=lambda frame: pd.to_datetime(frame["date"], errors="coerce"))
        return (
            frame.dropna(subset=["date"])
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .tail(320)
            .reset_index(drop=True)
        )

    def _latest_features(
        self,
        *,
        symbol: str,
        history: pd.DataFrame,
        library: Optional[pd.DataFrame],
        as_of: date,
    ) -> Dict[str, Any]:
        local_result: Dict[str, Any] = {}
        if not history.empty:
            try:
                local_result = compute_daily_features(history)
                latest = history.iloc[-1]
                local_result["close"] = float(
                    pd.to_numeric(history["close"], errors="coerce").dropna().iloc[-1]
                )
                local_result["high"] = self._number(latest.get("high"))
                local_result["low"] = self._number(latest.get("low"))
                local_result["daily_source"] = "stock_daily"
                if int(local_result.get("daily_data_points") or 0) >= 21:
                    return local_result
            except Exception:
                local_result = {}
        if library is None:
            library = self._load_feature_library()
        if not library.empty and "code" in library.columns:
            normalized = self.portfolio._normalize_symbol_for_position(symbol)
            digits = normalized[-6:] if normalized[-6:].isdigit() else normalized
            rows = library[
                library["code"].astype(str).isin({normalized, digits, symbol})
            ].copy()
            if "daily_latest_date" in rows.columns:
                rows["_parsed_daily_date"] = pd.to_datetime(
                    rows["daily_latest_date"],
                    errors="coerce",
                )
                rows = rows[
                    rows["_parsed_daily_date"].dt.date.le(as_of)
                ].sort_values("_parsed_daily_date")
            if not rows.empty:
                library_result = rows.iloc[-1].to_dict()
                if int(self._number(library_result.get("daily_data_points")) or 0) > int(
                    self._number(local_result.get("daily_data_points")) or 0
                ):
                    return library_result
        return local_result

    def _apply_candidate_sizing(
        self,
        candidates: Iterable[Any],
        account_plan: Dict[str, Any],
    ) -> None:
        account = account_plan["account"]
        rules = account_plan["rules"]
        equity = float(account.get("total_equity") or 0.0)
        available_cash = max(float(account.get("total_cash") or 0.0), 0.0)
        exposure_room = max(
            equity * float(rules["target_exposure_pct"]) / 100.0
            - float(account.get("total_market_value") or 0.0),
            0.0,
        )
        heat_room = max(
            equity * (
                float(rules["max_portfolio_heat_pct"])
                - float(account.get("strategy_risk_heat_pct") or 0.0)
            ) / 100.0,
            0.0,
        )
        slots = max(int(rules["max_positions"]) - int(account["position_count"]), 0)
        held = {
            self.portfolio._normalize_symbol_for_position(str(item.get("symbol") or ""))
            for item in account_plan.get("holdings") or []
        }
        lot_size = max(int(rules["lot_size"]), 1)

        for item in candidates:
            if not isinstance(item, dict):
                continue
            plan = item.get("trade_plan") if isinstance(item.get("trade_plan"), dict) else {}
            sizing = plan.get("position_sizing") if isinstance(plan.get("position_sizing"), dict) else {}
            plan["position_sizing"] = sizing
            sizing.update({
                "account_id": account.get("account_id"),
                "account_name": account.get("account_name"),
                "account_equity": round(equity, 2),
                "available_cash_before": round(available_cash, 2),
                "available_heat_amount_before": round(heat_room, 2),
                "remaining_position_slots_before": slots,
            })
            code = self.portfolio._normalize_symbol_for_position(
                str(item.get("code") or "")
            )
            if plan.get("signal_status") != "buy_signal":
                sizing.update({"execution_status": "not_armed", "recommended_quantity": 0})
                continue
            if not bool(rules.get("new_entries_allowed", True)):
                sizing.update({
                    "execution_status": "market_blocked",
                    "recommended_quantity": 0,
                    "constraint_reason": "市场状态处于风险关闭，不允许新开仓。",
                })
                continue
            if code in held:
                sizing.update({
                    "execution_status": "already_held",
                    "recommended_quantity": 0,
                    "constraint_reason": "已有持仓；新增仓位仅按持仓计划中的盈利加仓信号执行。",
                })
                continue
            entry = plan.get("entry") if isinstance(plan.get("entry"), dict) else {}
            exit_plan = plan.get("exit") if isinstance(plan.get("exit"), dict) else {}
            price = self._number(entry.get("max_price"))
            stop = self._number(exit_plan.get("initial_stop_price"))
            if price is None or stop is None or price <= stop or equity <= EPS:
                sizing.update({
                    "execution_status": "data_insufficient",
                    "recommended_quantity": 0,
                    "constraint_reason": "缺少有效的最高竞价价或初始防守价。",
                })
                continue
            risk_per_share = price - stop
            max_weight = min(
                float(sizing.get("max_position_weight_pct") or rules["max_position_weight_pct"]),
                float(sizing.get("initial_position_weight_pct") or rules["initial_position_weight_pct"]),
            )
            notional_cap = min(
                available_cash,
                exposure_room,
                equity * max_weight / 100.0,
            )
            risk_cap = min(
                equity * float(sizing.get("max_portfolio_risk_per_trade_pct") or rules["max_portfolio_risk_per_trade_pct"]) / 100.0,
                heat_room,
            )
            raw_quantity = min(notional_cap / price, risk_cap / risk_per_share)
            quantity = math.floor(raw_quantity / lot_size) * lot_size if slots > 0 else 0
            notional = quantity * price
            risk_amount = quantity * risk_per_share
            min_weight = float(
                sizing.get("min_position_weight_pct")
                or rules["min_position_weight_pct"]
            )
            estimated_weight = notional / equity * 100.0 if equity > EPS else 0.0
            if quantity <= 0:
                reason = "组合持仓数量已达上限" if slots <= 0 else "现金、目标仓位或风险预算不足一手"
                sizing.update({
                    "execution_status": "blocked",
                    "recommended_quantity": 0,
                    "constraint_reason": reason,
                })
                continue
            if estimated_weight + EPS < min_weight:
                sizing.update({
                    "execution_status": "blocked",
                    "recommended_quantity": 0,
                    "constraint_reason": (
                        f"整数手计仓仅占账户 {estimated_weight:.2f}%，"
                        f"低于当前市场状态的最小有效仓位 {min_weight:.2f}%。"
                    ),
                })
                continue
            sizing.update({
                "execution_status": "ready",
                "recommended_quantity": quantity,
                "sizing_price": round(price, 4),
                "estimated_notional": round(notional, 2),
                "estimated_risk_amount": round(risk_amount, 2),
                "estimated_weight_pct": round(estimated_weight, 4),
                "estimated_heat_pct": round(risk_amount / equity * 100.0, 4),
                "constraint_reason": "按最高允许竞价价、技术止损、现金、仓位和组合热度共同取最小值。",
            })
            available_cash = max(available_cash - notional, 0.0)
            exposure_room = max(exposure_room - notional, 0.0)
            heat_room = max(heat_room - risk_amount, 0.0)
            slots -= 1

    def _apply_add_on_sizing(
        self,
        account_plan: Dict[str, Any],
        *,
        trade_rules: MartinTradePlanConfig,
    ) -> None:
        account = account_plan["account"]
        rules = account_plan["rules"]
        equity = float(account.get("total_equity") or 0.0)
        cash = max(float(account.get("total_cash") or 0.0), 0.0)
        exposure_room = max(
            equity * float(rules["target_exposure_pct"]) / 100.0
            - float(account.get("total_market_value") or 0.0),
            0.0,
        )
        heat_room = max(
            equity * (
                float(rules["max_portfolio_heat_pct"])
                - float(account.get("strategy_risk_heat_pct") or 0.0)
            ) / 100.0,
            0.0,
        )
        lot = max(int(rules["lot_size"]), 1)
        for plan in account_plan.get("holdings") or []:
            if not plan.get("managed") or plan.get("status") != "hold":
                continue
            add_count = int(plan.get("add_on_count") or 0)
            if (
                add_count >= int(rules.get("max_add_ons") or 0)
                or not bool(rules.get("new_entries_allowed", True))
            ):
                continue
            threshold = float(
                rules["first_add_on_r_multiple"]
                if add_count == 0
                else rules["second_add_on_r_multiple"]
            )
            current_r = float(plan.get("current_r_multiple") or 0.0)
            if current_r < threshold:
                plan["add_on_plan"] = {
                    "status": "not_armed",
                    "next_trigger_r": threshold,
                    "recommended_quantity": 0,
                }
                continue
            price = float(plan.get("current_close") or 0.0)
            active_exit = float(plan.get("active_exit_price") or 0.0)
            gap = resolve_martin_entry_gap_bounds(
                adr_20_pct=plan.get("adr_20_pct"),
                config=trade_rules,
            )
            min_price = price * (1.0 + float(gap["min_gap_pct"]) / 100.0)
            max_price = price * (1.0 + float(gap["max_gap_pct"]) / 100.0)
            risk_per_share = max(max_price - active_exit, 0.01)
            current_value = float(plan.get("market_value") or 0.0)
            name_room = max(equity * float(rules["max_position_weight_pct"]) / 100.0 - current_value, 0.0)
            notional_cap = min(
                cash,
                exposure_room,
                name_room,
                equity * float(rules["add_on_position_weight_pct"]) / 100.0,
            )
            tranche_risk_cap = min(
                heat_room,
                equity * float(rules["max_portfolio_risk_per_trade_pct"]) / 100.0,
            )
            quantity = math.floor(
                min(notional_cap / max_price, tranche_risk_cap / risk_per_share) / lot
            ) * lot if max_price > 0 else 0
            plan["add_on_plan"] = {
                "status": "ready" if quantity > 0 else "blocked",
                "next_trigger_r": threshold,
                "recommended_quantity": quantity,
                "reference_price": round(price, 4),
                "execution_session": "next_open_call_auction",
                "min_price": round(min_price, 4),
                "max_price": round(max_price, 4),
                "gap_model": gap["model"],
                "estimated_notional": round(quantity * max_price, 2),
                "protective_stop_price": round(active_exit, 4),
                "reason": (
                    "持仓盈利达到下一加仓R门槛，且现金、个股上限与组合热度允许。"
                    if quantity > 0
                    else "已达到触发R，但现金、个股上限或组合热度不足一手。"
                ),
            }
            if quantity > 0:
                cash -= quantity * max_price
                exposure_room -= quantity * max_price
                heat_room -= quantity * risk_per_share

    @staticmethod
    def _unmanaged_holding(position: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "managed": False,
            "status": "unmanaged",
            "action": "manual_review",
            "symbol": position.get("symbol"),
            "market": position.get("market"),
            "quantity": float(position.get("quantity") or 0.0),
            "sellable_quantity": float(
                position.get("sellable_quantity", position.get("quantity") or 0.0)
            ),
            "avg_cost": float(position.get("avg_cost") or 0.0),
            "current_close": float(position.get("last_price") or 0.0),
            "market_value": float(position.get("market_value_base") or 0.0),
            "reason": "该持仓尚未绑定策略来源和初始止损，不自动生成卖出条件。",
        }

    @staticmethod
    def _position_structure_increased(state: Any, tranches: list[Dict[str, Any]]) -> bool:
        if state is None:
            return False
        previous = PortfolioStrategyService._json_list(state.entry_tranches_json)
        previous_sources = {
            item.get("source_trade_id") for item in previous if item.get("source_trade_id") is not None
        }
        current_sources = {
            item.get("source_trade_id") for item in tranches if item.get("source_trade_id") is not None
        }
        return bool(current_sources - previous_sources)

    @staticmethod
    def _position_structure_changed(state: Any, tranches: list[Dict[str, Any]]) -> bool:
        if state is None:
            return True
        previous = PortfolioStrategyService._json_list(state.entry_tranches_json)

        def signature(items: list[Dict[str, Any]]) -> list[tuple[str, float, float, float]]:
            return sorted(
                (
                    str(item.get("source_trade_id") or "adopted"),
                    round(float(item.get("remaining_shares") or 0.0), 8),
                    round(float(item.get("cost_per_share") or 0.0), 8),
                    round(float(item.get("protective_stop_price") or 0.0), 8),
                )
                for item in items
            )

        return signature(previous) != signature(tranches)

    @staticmethod
    def _holding_days(history: pd.DataFrame, opened: date, as_of: date) -> int:
        if history.empty or "date" not in history.columns:
            return max((as_of - opened).days, 0)
        dates = pd.to_datetime(history["date"], errors="coerce").dropna().dt.date
        return int(((dates >= opened) & (dates <= as_of)).sum())

    @staticmethod
    def _exhaustion_signal(features: Dict[str, Any], rules: MartinTradePlanConfig) -> bool:
        return is_profit_exhaustion_bar(
            SimpleNamespace(
                high=features.get("high"),
                low=features.get("low"),
                close=features.get("close"),
                trigger_volume_vs_previous=features.get("trigger_volume_vs_previous"),
                sector_data_status="unavailable",
                sector_strength_score=None,
            ),
            adr_20_pct=features.get("adr_20_pct"),
            volume_ratio_min=1.5,
            close_location_max_pct=60.0,
            range_adr_multiple_min=1.0,
            weak_sector_score_max=40.0,
        )

    @staticmethod
    def _round_partial_sell(*, quantity: float, sellable: float, fraction: float, lot_size: int) -> float:
        target = min(quantity * fraction, sellable)
        if target <= 0:
            return 0.0
        rounded = math.floor(target / lot_size) * lot_size
        return min(float(rounded), sellable)

    @staticmethod
    def _json_list(value: Any) -> list[Dict[str, Any]]:
        if not value:
            return []
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        return [dict(item) for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []

    @staticmethod
    def _json_dict(value: Any) -> Dict[str, Any]:
        if not value:
            return {}
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value or "")[:10])
        except ValueError:
            return None
