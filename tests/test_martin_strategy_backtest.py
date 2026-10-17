# -*- coding: utf-8 -*-
"""Point-in-time and execution contracts for the Martin strategy backtest."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.run_martin_strategy_backtest import _compare_close_series
from src.services.screening.daily import compute_daily_features
from src.services.screening.models import HardFilterConfig, ScreeningConfig
from src.services.screening.strategy_backtest import (
    MartinBacktestConfig,
    build_historical_feature_panel,
    generate_martin_signals,
    render_backtest_markdown,
    run_martin_portfolio_backtest,
)


def _history(periods: int = 140) -> pd.DataFrame:
    close = np.linspace(10.0, 20.0, periods)
    spread = np.r_[np.full(periods - 3, 0.025), np.full(3, 0.012)]
    volume = np.r_[np.full(periods - 3, 10_000_000.0), np.full(3, 6_000_000.0)]
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-02", periods=periods),
        "open": close * 0.998,
        "high": close * (1.0 + spread),
        "low": close * (1.0 - spread),
        "close": close,
        "volume": volume,
        "amount": volume * close,
    })


def test_historical_panel_matches_live_feature_calculation_at_each_cutoff() -> None:
    history = _history()
    target = pd.Timestamp(history.iloc[-2]["date"])
    panel = build_historical_feature_panel(
        {"000001": history},
        start_date=target.date(),
        end_date=target.date(),
    )
    expected = compute_daily_features(history[history["date"].le(target)])
    row = panel.iloc[0]

    for key in (
        "change_20d",
        "change_60d",
        "adr_20_pct",
        "atr_20_pct",
        "ema9",
        "ema21",
        "ema50",
        "distance_to_120d_high_pct",
        "pullback_depth_10d_pct",
        "range_contraction_ratio",
        "volume_contraction_ratio",
        "anchored_vwap_60d_high",
        "trigger_distance_pct",
        "trigger_close_location_pct",
        "trigger_volume_vs_previous",
        "avg_amount_20d",
    ):
        assert float(row[key]) == pytest.approx(float(expected[key]), abs=1e-4)
    assert int(row["support_confluence_count"]) == expected["support_confluence_count"]
    assert 0.0 <= float(row["trigger_close_location_pct"]) <= 100.0
    assert float(row["trigger_volume_vs_previous"]) > 0.0
    assert 0.0 <= float(row["market_above_ema21_ratio"]) <= 100.0
    assert 0.0 <= float(row["market_above_ema50_ratio"]) <= 100.0


def test_source_validation_ignores_constant_price_scale_but_detects_path_drift() -> None:
    dates = pd.bdate_range("2026-08-03", periods=4)
    base = pd.DataFrame({"date": dates, "close": [10.0, 10.5, 10.2, 11.0]})
    scaled = pd.DataFrame({"date": dates, "close": [20.0, 21.0, 20.4, 22.0]})
    drifted = pd.DataFrame({"date": dates, "close": [20.0, 21.0, 19.0, 22.0]})

    compatible = _compare_close_series(base, scaled)
    incompatible = _compare_close_series(base, drifted)

    assert compatible["max_close_delta_pct"] == 50.0
    assert compatible["max_normalized_close_delta_pct"] == pytest.approx(0.0)
    assert compatible["max_daily_return_delta_pct"] == pytest.approx(0.0)
    assert incompatible["max_normalized_close_delta_pct"] > 0.5
    assert incompatible["max_daily_return_delta_pct"] > 0.5


def test_future_bar_changes_do_not_rewrite_an_earlier_signal_row() -> None:
    original = _history()
    target = pd.Timestamp(original.iloc[-2]["date"])
    mutated = original.copy()
    mutated.loc[mutated.index[-1], ["open", "high", "low", "close"]] = [80.0, 90.0, 75.0, 85.0]

    before = build_historical_feature_panel(
        {"000001": original},
        start_date=target.date(),
        end_date=target.date(),
    ).iloc[0]
    after = build_historical_feature_panel(
        {"000001": mutated},
        start_date=target.date(),
        end_date=target.date(),
    ).iloc[0]

    for key in (
        "change_60d",
        "ema21",
        "distance_to_120d_high_pct",
        "anchored_vwap_60d_high",
        "trigger_distance_pct",
    ):
        assert float(after[key]) == pytest.approx(float(before[key]))


def test_signal_book_confirms_previous_high_without_fixed_breakout_ceiling() -> None:
    signal_date = pd.Timestamp("2026-08-03")
    panel = pd.DataFrame([
        {
            "date": signal_date,
            "code": "000001",
            "name": "Watch Only",
            "trigger_distance_pct": -1.0,
            "relative_strength_60d_pctile": 90.0,
        },
        {
            "date": signal_date,
            "code": "000002",
            "name": "Confirmed",
            "trigger_distance_pct": 0.5,
            "relative_strength_60d_pctile": 95.0,
        },
        {
            "date": signal_date,
            "code": "000003",
            "name": "Strong Confirmed",
            "trigger_distance_pct": 3.0,
            "relative_strength_60d_pctile": 98.0,
        },
    ])
    screening = ScreeningConfig(
        hard_filters=HardFilterConfig(
            exclude_st=False,
            trigger_distance_pct_min=-4.0,
            trigger_distance_pct_max=None,
        )
    )

    book, diagnostics = generate_martin_signals(panel, screening)

    assert set(book[signal_date]["code"]) == {"000002", "000003"}
    assert diagnostics["setup_count"] == 3
    assert diagnostics["entry_signal_count"] == 2
    assert diagnostics["unconfirmed_setup_count"] == 1


def test_full_market_signal_mode_excludes_current_st_names() -> None:
    signal_date = pd.Timestamp("2026-08-03")
    panel = pd.DataFrame([{
        "date": signal_date,
        "code": "000004",
        "name": "*ST Example",
        "trigger_distance_pct": 0.5,
        "relative_strength_60d_pctile": 95.0,
    }])
    screening = ScreeningConfig(
        hard_filters=HardFilterConfig(
            exclude_st=True,
            trigger_distance_pct_min=-4.0,
            trigger_distance_pct_max=2.0,
        )
    )

    book, diagnostics = generate_martin_signals(
        panel,
        screening,
        current_static_prefiltered=False,
    )

    assert book == {}
    assert diagnostics["setup_count"] == 0
    assert "current ST names excluded" in diagnostics["historical_static_filters"]


def test_portfolio_executes_close_signal_at_next_open_only() -> None:
    dates = pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05"])
    panel = pd.DataFrame({
        "date": dates,
        "code": ["000001"] * 3,
        "name": ["Example"] * 3,
        "open": [9.8, 10.0, 10.8],
        "close": [10.0, 10.5, 11.0],
        "previous_close": [9.7, 10.0, 10.5],
        "ema21": [5.0, 5.0, 5.0],
        "change_pct": [1.0, 5.0, 4.7619],
    })
    signal = pd.DataFrame([{
        "date": dates[0],
        "code": "000001",
        "name": "Example",
        "screen_score": 80.0,
        "close": 10.0,
        "ema21": 5.0,
        "previous_high": 9.9,
        "trigger_distance_pct": 1.0,
        "adr_20_pct": 4.0,
        "anchored_vwap_60d_high": 9.5,
    }])
    result = run_martin_portfolio_backtest(
        panel,
        signal_book={dates[0]: signal},
        config=MartinBacktestConfig(
            start_date=date(2026, 8, 3),
            end_date=date(2026, 8, 5),
            initial_cash=100_000.0,
            max_positions=1,
            risk_per_trade_pct=100.0,
            max_portfolio_heat_pct=100.0,
            max_position_weight_pct=100.0,
            min_position_weight_pct=0.0,
            adr_stress_fraction=0.0,
            commission_bps=0.0,
            slippage_bps=0.0,
            sell_stamp_duty_bps=0.0,
        ),
    )

    assert result["metrics"]["trade_count"] == 1
    assert result["metrics"]["total_return_pct"] == pytest.approx(10.0)
    assert result["trades"][0]["signal_date"] == "2026-08-03"
    assert result["trades"][0]["entry_date"] == "2026-08-04"
    assert result["trades"][0]["entry_raw_price"] == 10.0
    assert result["trades"][0]["entry_session"] == "opening_call_auction"
    assert result["trades"][0]["exit_reason"] == "period_end_mark"
    assert result["trades"][0]["shares"] == 10_000
    assert result["trades"][0]["entry_notional"] == pytest.approx(100_000.0)
    assert result["trades"][0]["entry_total"] == pytest.approx(100_000.0)
    assert result["trades"][0]["exit_notional"] == pytest.approx(110_000.0)
    assert result["trades"][0]["exit_proceeds"] == pytest.approx(110_000.0)


def test_position_size_uses_technical_stop_and_reports_adr_stress() -> None:
    dates = pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05"])
    panel = pd.DataFrame({
        "date": dates,
        "code": ["000001"] * 3,
        "name": ["Example"] * 3,
        "open": [9.8, 10.0, 10.8],
        "close": [10.0, 10.5, 11.0],
        "previous_close": [9.7, 10.0, 10.5],
        "ema21": [5.0, 5.0, 5.0],
        "change_pct": [1.0, 5.0, 4.7619],
    })
    signal = pd.DataFrame([{
        "date": dates[0],
        "code": "000001",
        "name": "Example",
        "screen_score": 80.0,
        "close": 10.0,
        "ema21": 5.0,
        "previous_high": 9.9,
        "trigger_distance_pct": 1.0,
        "adr_20_pct": 4.0,
        "anchored_vwap_60d_high": 9.5,
    }])

    result = run_martin_portfolio_backtest(
        panel,
        signal_book={dates[0]: signal},
        config=MartinBacktestConfig(
            start_date=date(2026, 8, 3),
            end_date=date(2026, 8, 5),
            initial_cash=100_000.0,
            max_positions=5,
            risk_per_trade_pct=0.5,
            max_portfolio_heat_pct=2.5,
            max_position_weight_pct=20.0,
            adr_stress_fraction=1.0,
            commission_bps=0.0,
            slippage_bps=0.0,
            sell_stamp_duty_bps=0.0,
        ),
    )

    trade = result["trades"][0]
    assert trade["shares"] == 2000
    assert trade["position_weight_pct"] == pytest.approx(20.0)
    assert trade["technical_account_risk_pct"] == pytest.approx(0.32)
    assert trade["adr_stress_account_risk_pct"] == pytest.approx(0.8)
    assert result["metrics"]["max_risk_heat_pct"] <= 0.5
    assert result["metrics"]["max_adr_stress_heat_pct"] == pytest.approx(0.7921)
    assert result["metrics"]["one_lot_trade_count"] == 0


def test_position_is_skipped_when_board_lot_is_below_minimum_weight() -> None:
    dates = pd.to_datetime(["2026-08-03", "2026-08-04"])
    panel = pd.DataFrame({
        "date": dates,
        "code": ["000001"] * 2,
        "name": ["Example"] * 2,
        "open": [49.0, 50.0],
        "close": [50.0, 50.0],
        "previous_close": [49.0, 50.0],
        "ema21": [25.0, 25.0],
        "change_pct": [2.0, 0.0],
    })
    signal = pd.DataFrame([{
        "date": dates[0],
        "code": "000001",
        "name": "Example",
        "screen_score": 80.0,
        "close": 50.0,
        "ema21": 25.0,
        "previous_high": 49.0,
        "trigger_distance_pct": 2.0,
        "adr_20_pct": 4.0,
        "anchored_vwap_60d_high": 47.5,
    }])

    result = run_martin_portfolio_backtest(
        panel,
        signal_book={dates[0]: signal},
        config=MartinBacktestConfig(
            start_date=date(2026, 8, 3),
            end_date=date(2026, 8, 4),
            initial_cash=50_000.0,
            commission_bps=0.0,
            slippage_bps=0.0,
            sell_stamp_duty_bps=0.0,
        ),
    )

    assert result["metrics"]["trade_count"] == 0
    assert result["diagnostics"]["skipped_min_position_weight"] == 1


def test_opening_fill_is_skipped_when_technical_stop_exceeds_limit() -> None:
    dates = pd.to_datetime(["2026-08-03", "2026-08-04"])
    panel = pd.DataFrame({
        "date": dates,
        "code": ["000001"] * 2,
        "name": ["Example"] * 2,
        "open": [9.8, 10.2],
        "close": [10.0, 10.2],
        "previous_close": [9.7, 10.0],
        "ema21": [5.0, 5.0],
        "change_pct": [1.0, 2.0],
    })
    signal = pd.DataFrame([{
        "date": dates[0],
        "code": "000001",
        "name": "Example",
        "screen_score": 80.0,
        "close": 10.0,
        "ema21": 5.0,
        "previous_high": 9.9,
        "trigger_distance_pct": 1.0,
        "adr_20_pct": 4.0,
        "anchored_vwap_60d_high": 9.5,
    }])

    result = run_martin_portfolio_backtest(
        panel,
        signal_book={dates[0]: signal},
        config=MartinBacktestConfig(
            start_date=date(2026, 8, 3),
            end_date=date(2026, 8, 4),
            initial_cash=50_000.0,
            commission_bps=0.0,
            slippage_bps=0.0,
            sell_stamp_duty_bps=0.0,
        ),
    )

    assert result["metrics"]["trade_count"] == 0
    assert result["diagnostics"]["skipped_technical_stop_too_wide"] == 1


def test_open_limit_up_does_not_assume_a_fill() -> None:
    dates = pd.to_datetime(["2026-08-03", "2026-08-04"])
    panel = pd.DataFrame({
        "date": dates,
        "code": ["600001"] * 2,
        "name": ["Example"] * 2,
        "open": [10.0, 11.0],
        "close": [10.0, 11.0],
        "previous_close": [9.9, 10.0],
        "ema21": [5.0, 5.0],
        "change_pct": [1.0, 10.0],
    })
    signal = pd.DataFrame([{
        "date": dates[0],
        "code": "600001",
        "name": "Example",
        "screen_score": 80.0,
        "close": 10.0,
        "ema21": 5.0,
        "previous_high": 9.9,
        "trigger_distance_pct": 1.0,
        "adr_20_pct": 4.0,
        "anchored_vwap_60d_high": 9.5,
    }])

    result = run_martin_portfolio_backtest(
        panel,
        signal_book={dates[0]: signal},
        config=MartinBacktestConfig(
            start_date=date(2026, 8, 3),
            end_date=date(2026, 8, 4),
        ),
    )

    assert result["metrics"]["trade_count"] == 0
    assert result["diagnostics"]["skipped_limit_up"] == 1


def test_opening_auction_gap_outside_guardrail_cancels_entry() -> None:
    dates = pd.to_datetime(["2026-08-03", "2026-08-04"])
    panel = pd.DataFrame({
        "date": dates,
        "code": ["000001"] * 2,
        "name": ["Example"] * 2,
        "open": [10.0, 10.3],
        "close": [10.0, 10.2],
        "previous_close": [9.9, 10.0],
        "ema21": [9.5, 9.6],
        "change_pct": [1.0, 2.0],
    })
    signal = pd.DataFrame([{
        "date": dates[0],
        "code": "000001",
        "name": "Example",
        "screen_score": 80.0,
        "close": 10.0,
        "ema21": 9.5,
        "previous_high": 9.9,
        "trigger_distance_pct": 1.0,
        "adr_20_pct": 4.0,
        "anchored_vwap_60d_high": 9.7,
    }])

    result = run_martin_portfolio_backtest(
        panel,
        signal_book={dates[0]: signal},
        config=MartinBacktestConfig(
            start_date=date(2026, 8, 3),
            end_date=date(2026, 8, 4),
            entry_gap_max_pct=2.0,
        ),
    )

    assert result["metrics"]["trade_count"] == 0
    assert result["diagnostics"]["skipped_gap_up"] == 1


def test_volatility_adaptive_guardrail_accepts_orderly_open_without_chasing() -> None:
    dates = pd.to_datetime(["2026-08-03", "2026-08-04"])
    panel = pd.DataFrame({
        "date": dates,
        "code": ["000001"] * 2,
        "name": ["Example"] * 2,
        "open": [10.0, 10.18],
        "close": [10.0, 10.4],
        "previous_close": [9.9, 10.0],
        "ema21": [9.8, 9.9],
        "change_pct": [1.0, 4.0],
    })
    signal = pd.DataFrame([{
        "date": dates[0],
        "code": "000001",
        "name": "Example",
        "screen_score": 80.0,
        "close": 10.0,
        "ema21": 9.8,
        "previous_high": 9.9,
        "trigger_distance_pct": 1.0,
        "adr_20_pct": 4.0,
        "atr_20_pct": 6.0,
        "anchored_vwap_60d_high": 10.0,
    }])

    result = run_martin_portfolio_backtest(
        panel,
        signal_book={dates[0]: signal},
        config=MartinBacktestConfig(
            start_date=date(2026, 8, 3),
            end_date=date(2026, 8, 4),
        ),
    )

    assert result["metrics"]["trade_count"] == 1
    trade = result["trades"][0]
    assert trade["entry_gap_pct"] == 1.8
    assert trade["entry_gap_min_pct"] == -1.5
    assert trade["entry_gap_max_pct"] == 2.0
    assert trade["entry_gap_volatility_reference_pct"] == 5.0
    assert trade["entry_gap_volatility_source"] == "atr20_adr20_mean"


def test_opening_fill_at_or_below_initial_stop_cancels_entry() -> None:
    dates = pd.to_datetime(["2026-08-03", "2026-08-04"])
    panel = pd.DataFrame({
        "date": dates,
        "code": ["000001"] * 2,
        "name": ["Example"] * 2,
        "open": [10.0, 9.9],
        "close": [10.0, 9.9],
        "previous_close": [9.9, 10.0],
        "ema21": [9.0, 9.0],
        "change_pct": [1.0, -1.0],
    })
    signal = pd.DataFrame([{
        "date": dates[0],
        "code": "000001",
        "name": "Example",
        "screen_score": 80.0,
        "close": 10.0,
        "ema21": 9.0,
        "previous_high": 9.8,
        "trigger_distance_pct": 2.0,
        "adr_20_pct": 4.0,
        "anchored_vwap_60d_high": 9.99,
    }])

    result = run_martin_portfolio_backtest(
        panel,
        signal_book={dates[0]: signal},
        config=MartinBacktestConfig(
            start_date=date(2026, 8, 3),
            end_date=date(2026, 8, 4),
        ),
    )

    assert result["metrics"]["trade_count"] == 0
    assert result["diagnostics"]["skipped_broken_support"] == 1


def test_backtest_report_describes_single_adaptive_entry_rule() -> None:
    report = render_backtest_markdown({
        "period": {"start": "2025-04-01", "end": "2026-10-17"},
        "results": {
            "strategy": {
                "metrics": {},
                "config": {
                    "entry_gap_min_pct": None,
                    "entry_gap_max_pct": None,
                    "entry_gap_down_volatility_fraction": 0.30,
                    "entry_gap_down_min_pct": 1.0,
                    "entry_gap_down_max_pct": 1.5,
                    "entry_gap_up_volatility_fraction": 0.50,
                    "entry_gap_up_min_pct": 1.5,
                    "entry_gap_up_max_pct": 2.0,
                },
                "trades": [],
                "diagnostics": {},
            }
        },
    })

    assert "高开容忍为 50%" in report
    assert "突破幅度不设固定上限" in report
    assert "B组" not in report
    assert "高开边界固定为 +3.00%" not in report
