# -*- coding: utf-8 -*-
"""Contracts for the Martin multi-day market-state overlay."""

from datetime import date

import pandas as pd

from src.services.screening.market_regime import (
    MarketRegimePolicy,
    REGIME_RISK_OFF,
    REGIME_STRONG_RISK_ON,
    classify_market_regimes,
    load_market_regime_history,
    resolve_market_regime_limits,
    update_market_regime_history,
)


def _observation(day: str, *, positive_ratio: float = 70.0) -> dict[str, object]:
    return {
        "date": pd.Timestamp(day),
        "universe_size": 1000,
        "positive_ratio": positive_ratio,
        "median_change_pct": 1.0 if positive_ratio > 20 else -3.5,
        "above_ema21_ratio": 70.0,
        "above_ema50_ratio": 65.0,
        "ema21_rising_ratio": 70.0,
        "ema50_rising_ratio": 65.0,
        "median_change_20d_pct": 12.0,
        "leader_count": 30,
        "leader_ratio": 3.0,
        "leader_median_change_pct": 2.0,
        "leader_median_change_20d_pct": 35.0,
        "severe_decline_ratio": 0.5 if positive_ratio > 20 else 10.0,
    }


def test_market_regime_downgrades_immediately_and_confirms_upgrades() -> None:
    observations = pd.DataFrame([
        _observation("2026-10-12", positive_ratio=10.0),
        _observation("2026-10-13"),
        _observation("2026-10-14"),
        _observation("2026-10-15", positive_ratio=10.0),
    ])

    result = classify_market_regimes(
        observations,
        policy=MarketRegimePolicy(upgrade_confirmation_days=2),
    )

    assert result["raw_state"].tolist()[:3] == [
        REGIME_RISK_OFF,
        REGIME_STRONG_RISK_ON,
        REGIME_STRONG_RISK_ON,
    ]
    assert result["state"].tolist() == [
        REGIME_RISK_OFF,
        REGIME_RISK_OFF,
        REGIME_STRONG_RISK_ON,
        REGIME_RISK_OFF,
    ]


def test_fresh_strong_history_starts_neutral_until_second_confirmation() -> None:
    result = classify_market_regimes(
        pd.DataFrame([
            _observation("2026-10-12"),
            _observation("2026-10-13"),
        ]),
        policy=MarketRegimePolicy(upgrade_confirmation_days=2),
    )

    assert result["state"].tolist() == ["neutral", REGIME_STRONG_RISK_ON]
    assert "升级需连续 2 日确认" in result.iloc[0]["reason"]


def test_market_regime_limits_scale_new_entry_capacity() -> None:
    limits = resolve_market_regime_limits(
        {"state": "neutral", "score": 48.0},
        base_max_positions=5,
        base_risk_per_trade_pct=0.5,
        base_max_portfolio_heat_pct=2.5,
        base_max_position_weight_pct=20.0,
        base_min_position_weight_pct=12.0,
    )

    assert limits["target_exposure_pct"] == 40.0
    assert limits["max_positions"] == 2
    assert limits["risk_per_trade_pct"] == 0.25
    assert limits["max_portfolio_heat_pct"] == 1.0
    assert limits["max_position_weight_pct"] == 15.0
    assert limits["min_position_weight_pct"] == 6.0
    assert limits["new_entries_allowed"] is True


def test_live_market_regime_history_can_be_reloaded(tmp_path) -> None:
    rows = []
    for index in range(120):
        rows.append({
            "code": f"{index:06d}",
            "price": 10.0 + index / 100.0,
            "change_pct": 1.0,
            "change_20d": 15.0,
            "change_60d": 30.0,
            "relative_strength_60d_pctile": (index + 1) / 120 * 100.0,
            "ema21": 9.0,
            "ema50": 8.0,
            "ema_stack_bullish": True,
            "ema21_rising": True,
            "ema50_rising": True,
            "distance_to_120d_high_pct": -5.0,
        })
    path = tmp_path / "market_regime_history.json"

    current = update_market_regime_history(
        pd.DataFrame(rows),
        history_path=path,
        as_of=date(2026, 10, 16),
    )
    loaded = load_market_regime_history(path)

    assert path.exists()
    assert current["date"] == "2026-10-16"
    assert loaded["state"] == current["state"]
    assert loaded["score"] == current["score"]
    assert loaded["measurements"]["universe_size"] == 120
