# -*- coding: utf-8 -*-
# Derived from AlphaSift revision 9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf.
# Licensed under Apache-2.0 and modified for daily_stock_analysis.
"""Deterministic scoring for the Martin momentum-pullback strategy."""

import pandas as pd

from src.services.screening.models import ScreeningConfig


_FACTOR_COLUMNS = {
    "trend_quality": "factor_trend_quality_score",
    "relative_strength": "factor_relative_strength_score",
    "sector_resonance": "factor_sector_resonance_score",
    "contraction": "factor_contraction_score",
    "support_confluence": "factor_support_confluence_score",
    "trigger_quality": "factor_trigger_quality_score",
    "liquidity": "factor_liquidity_score",
}

_DEFAULT_WEIGHTS = {
    "trend_quality": 0.20,
    "relative_strength": 0.20,
    "sector_resonance": 0.15,
    "contraction": 0.15,
    "support_confluence": 0.15,
    "trigger_quality": 0.10,
    "liquidity": 0.05,
}


def compute_screen_scores(df: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    """Attach the seven Martin factor scores and their weighted total."""
    result = df.copy()
    factors = _compute_factor_scores(result)
    for name, series in factors.items():
        result[_FACTOR_COLUMNS[name]] = series.round(4)

    result["screen_score"] = 0.0
    for factor, weight in _normalized_factor_weights(config).items():
        result["screen_score"] += factors[factor] * weight
    result["screen_score"] = result["screen_score"].clip(0, 100).round(4)
    return result


def factor_score_columns() -> dict[str, str]:
    """Return stable factor-score columns exposed through the API."""
    return dict(_FACTOR_COLUMNS)


def _normalized_factor_weights(config: ScreeningConfig) -> dict[str, float]:
    raw = config.factor_weights or _DEFAULT_WEIGHTS
    unknown = sorted(set(raw) - set(_FACTOR_COLUMNS))
    if unknown:
        raise ValueError(f"Unsupported Martin scoring factors: {', '.join(unknown)}")
    weights = {name: max(float(raw.get(name, 0.0)), 0.0) for name in _FACTOR_COLUMNS}
    total = sum(weights.values())
    if total <= 0:
        return dict(_DEFAULT_WEIGHTS)
    return {name: value / total for name, value in weights.items()}


def _compute_factor_scores(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "trend_quality": _trend_quality_score(df),
        "relative_strength": _relative_strength_score(df),
        "sector_resonance": _sector_resonance_score(df),
        "contraction": _contraction_score(df),
        "support_confluence": _support_confluence_score(df),
        "trigger_quality": _trigger_quality_score(df),
        "liquidity": _liquidity_score(df),
    }


def _trend_quality_score(df: pd.DataFrame) -> pd.Series:
    score = _bool_score(df, "ema_stack_bullish") * 0.40
    score += _bool_score(df, "ema21_rising") * 0.20
    score += _bool_score(df, "ema50_rising") * 0.20
    distance = _numeric(df, "distance_to_120d_high_pct")
    proximity = ((distance + 15.0) / 15.0 * 100.0).clip(0, 100).fillna(50.0)
    return (score + proximity * 0.20).clip(0, 100)


def _relative_strength_score(df: pd.DataFrame) -> pd.Series:
    rs20 = _numeric(df, "relative_strength_20d_pctile")
    rs60 = _numeric(df, "relative_strength_60d_pctile")
    return pd.concat([rs20, rs60], axis=1).mean(axis=1, skipna=True).fillna(50.0).clip(0, 100)


def _sector_resonance_score(df: pd.DataFrame) -> pd.Series:
    normalized = _numeric(df, "sector_strength_score")
    heat = _numeric(df, "board_heat_score")
    breadth = _rank_score(_numeric(df, "sector_breadth_count"), neutral=50.0)
    legacy = (
        (heat.fillna(50.0).clip(0, 100) * 0.65 + breadth * 0.35).clip(0, 100)
        if heat.notna().any()
        else breadth
    )
    return normalized.where(normalized.notna(), legacy).fillna(50.0).clip(0, 100)


def _contraction_score(df: pd.DataFrame) -> pd.Series:
    range_score = _descending_band(_numeric(df, "range_contraction_ratio"), best=0.55, worst=1.10)
    volume_score = _descending_band(_numeric(df, "volume_contraction_ratio"), best=0.55, worst=1.10)
    pullback = _numeric(df, "pullback_depth_10d_pct")
    pullback_score = (100.0 - (pullback + 5.0).abs() * 14.0).clip(0, 100).fillna(50.0)
    return (range_score * 0.35 + volume_score * 0.35 + pullback_score * 0.30).clip(0, 100)


def _support_confluence_score(df: pd.DataFrame) -> pd.Series:
    return (_numeric(df, "support_confluence_count") / 3.0 * 100.0).clip(0, 100).fillna(0.0)


def _trigger_quality_score(df: pd.DataFrame) -> pd.Series:
    distance = _numeric(df, "trigger_distance_pct")
    trigger = (100.0 - distance.abs() * 22.0).clip(0, 100).fillna(40.0)
    close_location = _numeric(df, "trigger_close_location_pct")
    close_location_score = (
        100.0 - (close_location - 80.0).abs() * 4.0
    ).clip(0, 100).fillna(50.0)
    volume_ratio = _numeric(df, "trigger_volume_vs_previous")
    volume_score = (
        100.0 - (volume_ratio - 1.2).abs() * 70.0
    ).clip(0, 100).fillna(50.0)
    chase_penalty = ((_numeric(df, "change_pct") - 5.0).clip(lower=0) * 12.0).fillna(0.0)
    return (
        trigger * 0.45
        + close_location_score * 0.35
        + volume_score * 0.20
        - chase_penalty
    ).clip(0, 100)


def _liquidity_score(df: pd.DataFrame) -> pd.Series:
    amount = _numeric(df, "avg_amount_20d")
    amount = amount.where(amount.notna(), _numeric(df, "amount"))
    amount_score = _rank_score(amount, neutral=50.0)
    turnover = _numeric(df, "turnover_rate")
    turnover_score = (100.0 - (turnover - 5.0).abs() * 12.0).clip(0, 100).fillna(50.0)
    return (amount_score * 0.75 + turnover_score * 0.25).clip(0, 100)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _bool_score(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return (df[column] == True).astype(float) * 100.0  # noqa: E712


def _rank_score(series: pd.Series, *, neutral: float) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(neutral, index=series.index, dtype=float)
    if valid.nunique() <= 1:
        ranked = pd.Series(neutral, index=valid.index, dtype=float)
    else:
        ranked = valid.rank(method="average", pct=True) * 100.0
    return ranked.reindex(series.index).fillna(neutral).clip(0, 100)


def _descending_band(series: pd.Series, *, best: float, worst: float) -> pd.Series:
    score = (worst - series) / (worst - best) * 100.0
    return score.clip(0, 100).fillna(50.0)
