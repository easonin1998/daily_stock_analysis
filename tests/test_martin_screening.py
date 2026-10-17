# -*- coding: utf-8 -*-
"""Focused contracts for the sole built-in Martin screening strategy."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.services.screening.daily import (
    _fetch_daily_sina,
    _fetch_daily_tencent,
    compute_daily_features,
)
from src.services.screening.filter import apply_hard_filters
from src.services.screening.models import Pick, ScreeningConfig
from src.services.screening.pipeline import (
    _attach_market_regime_features,
    _attach_martin_cross_sectional_features,
    _attach_martin_cross_sectional_features_from_universe,
)
from src.services.screening.risk import apply_sector_quota, assess_pick_risk
from src.services.screening.scorer import compute_screen_scores, factor_score_columns
from src.services.screening.strategy import load_all_strategies


STRATEGIES_DIR = Path(__file__).resolve().parents[1] / "src" / "services" / "screening" / "strategies"


def _momentum_pullback_history() -> pd.DataFrame:
    periods = 140
    close = np.linspace(10.0, 18.0, periods)
    close[-10:] = [18.5, 18.8, 19.0, 18.9, 18.7, 18.5, 18.4, 18.45, 18.5, 18.65]
    spread = np.r_[np.full(periods - 3, 0.025), np.full(3, 0.012)]
    volume = np.r_[np.full(periods - 3, 10_000_000.0), np.full(3, 5_500_000.0)]
    frame = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=periods),
        "open": close * 0.998,
        "high": close * (1.0 + spread),
        "low": close * (1.0 - spread),
        "close": close,
        "volume": volume,
        "amount": volume * close,
    })
    frame.loc[frame.index[-1], "high"] = close[-1] * 1.003
    frame.loc[frame.index[-1], "low"] = close[-1] * 0.988
    return frame


def test_only_martin_strategy_is_bundled() -> None:
    strategies = load_all_strategies(STRATEGIES_DIR)

    assert set(strategies) == {"martin_momentum_pullback"}
    strategy = strategies["martin_momentum_pullback"]
    assert strategy.version == "2.6"
    assert strategy.screening.hard_filters.price_min == 1.5
    assert strategy.screening.hard_filters.relative_strength_60d_pctile_min == 95.0
    assert strategy.screening.hard_filters.trigger_distance_pct_max is None
    assert strategy.screening.hard_filters.trigger_close_location_pct_min is None


def test_daily_features_capture_momentum_pullback_setup() -> None:
    features = compute_daily_features(_momentum_pullback_history())

    assert features["daily_data_points"] == 140
    assert 3.5 <= features["adr_20_pct"] <= 8.0
    assert features["change_20d"] > 3.0
    assert features["change_60d"] > 20.0
    assert features["ema_stack_bullish"] is True
    assert features["ema21_rising"] is True
    assert features["ema50_rising"] is True
    assert features["distance_to_120d_high_pct"] >= -15.0
    assert -12.0 <= features["pullback_depth_10d_pct"] <= -1.0
    assert features["range_contraction_ratio"] <= 0.95
    assert features["volume_contraction_ratio"] <= 0.95
    assert features["support_confluence_count"] >= 2
    assert features["trigger_distance_pct"] >= -4.0
    assert 70.0 <= features["trigger_close_location_pct"] <= 90.0
    assert 0.8 <= features["trigger_volume_vs_previous"] <= 2.0
    assert features["one_word_limit_up"] is False


def test_tencent_daily_history_estimates_missing_amount_from_lot_volume(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "code": 0,
                "data": {
                    "sh600519": {
                        "qfqday": [["2026-10-16", "100", "102", "103", "99", "2000"]]
                    }
                },
            }

    monkeypatch.setattr("src.services.screening.daily.requests.get", lambda *args, **kwargs: Response())

    frame = _fetch_daily_tencent("600519", lookback_days=30)

    assert float(frame.iloc[0]["amount"]) == 2000 * 102 * 100
    assert frame.attrs["amount_estimation"] == "tencent_volume_lots_x_close_x_100"


def test_sina_daily_history_estimates_missing_amount_from_share_volume(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "result": {
                    "data": [{
                        "day": "2026-10-16",
                        "open": "100",
                        "close": "102",
                        "high": "103",
                        "low": "99",
                        "volume": "200000",
                    }]
                }
            }

    monkeypatch.setattr("src.services.screening.daily.requests.get", lambda *args, **kwargs: Response())

    frame = _fetch_daily_sina("600519", lookback_days=30)

    assert float(frame.iloc[0]["amount"]) == 200000 * 102
    assert frame.attrs["amount_estimation"] == "sina_volume_shares_x_close"


def test_cached_sina_history_estimates_missing_amount_before_feature_calculation() -> None:
    frame = _momentum_pullback_history().drop(columns=["amount"])
    frame["amount"] = pd.NA
    frame.attrs["daily_source"] = "sina"

    features = compute_daily_features(frame)

    expected = float((frame["volume"].tail(20) * frame["close"].tail(20)).mean())
    assert features["avg_amount_20d"] == round(expected, 4)


def test_market_regime_gate_rejects_broad_risk_off_snapshots() -> None:
    strategy = load_all_strategies(STRATEGIES_DIR)["martin_momentum_pullback"]
    frame = pd.DataFrame([
        {"code": "000001", "name": "A", "price": 10.0, "amount": 200_000_000, "total_mv": 5_000_000_000, "change_pct": -2.0},
        {"code": "000002", "name": "B", "price": 10.0, "amount": 200_000_000, "total_mv": 5_000_000_000, "change_pct": -1.8},
        {"code": "000003", "name": "C", "price": 10.0, "amount": 200_000_000, "total_mv": 5_000_000_000, "change_pct": 0.2},
    ])

    gated = _attach_market_regime_features(frame)
    snapshot_filters = strategy.screening.hard_filters

    assert float(gated.iloc[0]["market_positive_ratio"]) < 35.0
    assert apply_hard_filters(gated, snapshot_filters).empty


def test_martin_hard_filters_and_seven_factor_score_form_one_pipeline() -> None:
    strategy = load_all_strategies(STRATEGIES_DIR)["martin_momentum_pullback"]
    features = compute_daily_features(_momentum_pullback_history())
    frame = pd.DataFrame([{
        "code": "000001",
        "name": "示例股份",
        "price": 18.65,
        "change_pct": 0.8,
        "amount": 200_000_000.0,
        "total_mv": 10_000_000_000.0,
        "turnover_rate": 3.0,
        "industry": "示例行业",
        **features,
    }])

    ranked = _attach_martin_cross_sectional_features(_attach_market_regime_features(frame))
    filtered = apply_hard_filters(ranked, strategy.screening.hard_filters)
    scored = compute_screen_scores(filtered, strategy.screening)

    assert len(filtered) == 1
    assert set(factor_score_columns()) == {
        "trend_quality",
        "relative_strength",
        "sector_resonance",
        "contraction",
        "support_confluence",
        "trigger_quality",
        "liquidity",
    }
    assert 0.0 <= float(scored.iloc[0]["screen_score"]) <= 100.0


def test_trigger_quality_softly_prefers_strong_close_and_orderly_volume() -> None:
    frame = pd.DataFrame([
        {
            "code": "000001",
            "trigger_distance_pct": 0.2,
            "trigger_close_location_pct": 80.0,
            "trigger_volume_vs_previous": 1.2,
            "change_pct": 2.0,
        },
        {
            "code": "000002",
            "trigger_distance_pct": 0.2,
            "trigger_close_location_pct": 50.0,
            "trigger_volume_vs_previous": 3.0,
            "change_pct": 2.0,
        },
    ])

    scored = compute_screen_scores(
        frame,
        ScreeningConfig(factor_weights={"trigger_quality": 1.0}),
    )

    assert scored.loc[0, "factor_trigger_quality_score"] > scored.loc[
        1, "factor_trigger_quality_score"
    ]


def test_relative_strength_percentile_uses_full_eligible_universe() -> None:
    universe = pd.DataFrame({
        "code": [f"00000{index}" for index in range(1, 10)],
        "change_20d": [float(index) for index in range(1, 10)],
        "change_60d": [float(index * 2) for index in range(1, 10)],
        "ema_stack_bullish": [True] * 9,
        "industry": ["示例行业"] * 9,
    })
    enriched = universe.loc[universe["code"].eq("000005")].copy()

    ranked = _attach_martin_cross_sectional_features_from_universe(
        enriched,
        universe=universe,
    )

    assert round(float(ranked.iloc[0]["relative_strength_20d_pctile"]), 2) == 55.56
    assert round(float(ranked.iloc[0]["relative_strength_60d_pctile"]), 2) == 55.56
    assert int(ranked.iloc[0]["sector_breadth_count"]) == 3
    assert float(ranked.iloc[0]["market_median_change_20d_pct"]) == 5.0


def test_risk_overlay_uses_martin_execution_risks_not_legacy_value_signals() -> None:
    pick = Pick(
        rank=1,
        code="000001",
        name="示例股份",
        final_score=80.0,
        screen_score=80.0,
        price=100.0,
        change_pct=1.0,
        pe_ratio=-5.0,
        pb_ratio=20.0,
        signal_score=10.0,
        macd_status="bearish",
        rsi_status="overbought",
        adr_20_pct=4.5,
        ema21=99.0,
        trigger_distance_pct=0.0,
        factor_scores={"sector_resonance": 60.0},
    )

    points, flags = assess_pick_risk(pick)

    assert points == 0.0
    assert flags == []

    pick.adr_20_pct = 7.5
    pick.ema21 = 90.0
    pick.trigger_distance_pct = -3.0
    pick.factor_scores = {"sector_resonance": 20.0}
    points, flags = assess_pick_risk(pick)

    assert points > 0.0
    assert set(flags) == {
        "high_adr_execution_risk",
        "extended_from_ema21",
        "weak_sector_resonance",
        "trigger_not_confirmed",
    }


def test_low_nominal_price_is_a_soft_risk_instead_of_a_four_yuan_veto() -> None:
    pick = Pick(
        rank=1,
        code="000001",
        name="低价但可交易",
        final_score=80.0,
        screen_score=80.0,
        price=3.5,
        change_pct=0.0,
        factor_scores={"sector_resonance": 60.0},
    )

    points, flags = assess_pick_risk(pick)

    assert points == 0.75
    assert flags == ["low_nominal_price"]

    pick.price = 1.8
    points, flags = assess_pick_risk(pick)

    assert points == 2.0
    assert flags == ["near_par_price_risk"]


def test_sector_quota_uses_comparable_reserve_without_sacrificing_quality() -> None:
    picks = [
        Pick(rank=1, code="000001", name="A", final_score=90, screen_score=90, industry="半导体"),
        Pick(rank=2, code="000002", name="B", final_score=89, screen_score=89, industry="半导体"),
        Pick(rank=3, code="000003", name="C", final_score=88, screen_score=88, industry="半导体"),
        Pick(rank=4, code="000004", name="D", final_score=87, screen_score=87, industry="医药"),
        Pick(rank=5, code="000005", name="E", final_score=86, screen_score=86, industry="新能源"),
    ]

    selected, notes, replacements = apply_sector_quota(
        picks,
        max_output=3,
        max_same_sector=2,
        max_score_gap=3,
    )

    assert {pick.code for pick in selected} == {"000001", "000002", "000004"}
    assert replacements == 1
    assert any("000003" in note and "000004" in note for note in notes)


def test_sector_quota_keeps_stronger_pick_when_reserve_gap_is_too_large() -> None:
    picks = [
        Pick(rank=1, code="000001", name="A", final_score=90, screen_score=90, industry="半导体"),
        Pick(rank=2, code="000002", name="B", final_score=89, screen_score=89, industry="半导体"),
        Pick(rank=3, code="000003", name="C", final_score=88, screen_score=88, industry="半导体"),
        Pick(rank=4, code="000004", name="D", final_score=80, screen_score=80, industry="医药"),
    ]

    selected, notes, replacements = apply_sector_quota(
        picks,
        max_output=3,
        max_same_sector=2,
        max_score_gap=3,
    )

    assert {pick.code for pick in selected} == {"000001", "000002", "000003"}
    assert replacements == 0
    assert any("retained 000003" in note for note in notes)
