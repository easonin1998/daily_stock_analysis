# -*- coding: utf-8 -*-
# Derived from AlphaSift revision 9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf.
# Licensed under Apache-2.0 and modified for daily_stock_analysis.
"""Build a diversified, explainable seed pool before daily-history enrichment."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.services.screening.normalize import normalize_code, safe_text


@dataclass(frozen=True)
class SeedPoolSelection:
    dataframe: pd.DataFrame
    diagnostics: dict[str, Any]


def build_seed_pool(
    df: pd.DataFrame,
    *,
    limit: int,
    period: str,
    core_ratio: float = 0.55,
    strength_ratio: float = 0.20,
    activity_ratio: float = 0.15,
    max_sector_ratio: float = 0.20,
) -> SeedPoolSelection:
    """Select a broad but bounded daily-enrichment universe.

    The pool deliberately combines several independent sleeves. Fresh local
    daily features lead the strength sleeve when available, while deterministic
    rotation keeps stocks without local history eligible for future enrichment.
    """
    source_count = len(df)
    resolved_limit = min(max(int(limit), 1), source_count)
    if source_count == 0:
        return SeedPoolSelection(
            dataframe=df.copy(),
            diagnostics={
                "source_count": 0,
                "selected_count": 0,
                "limit": max(int(limit), 1),
                "sleeve_counts": {},
                "sector_cap": 0,
                "available_signals": [],
            },
        )

    work = df.copy().reset_index(drop=True)
    work["__seed_row"] = range(len(work))
    work["__seed_code"] = _code_series(work)
    signals = _attach_seed_scores(work)

    sleeve_counts = _allocate_sleeves(
        resolved_limit,
        core_ratio=core_ratio,
        strength_ratio=strength_ratio,
        activity_ratio=activity_ratio,
    )
    sector_cap = max(1, int(math.ceil(resolved_limit * _bounded_ratio(max_sector_ratio))))
    industry_col = _first_column(work, ("industry", "行业", "所属行业"))

    selected_rows: list[int] = []
    selected_set: set[int] = set()
    selected_reason: dict[int, str] = {}
    sector_counts: dict[str, int] = {}
    sector_cap_skips = 0

    sleeve_orders = {
        "liquidity_core": _ordered_rows(
            work,
            ["seed_pool_score", "__seed_amount_score", "__seed_code"],
            [False, False, True],
        ),
        "daily_strength": _ordered_rows(
            work,
            ["__seed_strength_score", "__seed_amount_score", "__seed_code"],
            [False, False, True],
        ),
        "trading_activity": _ordered_rows(
            work,
            ["__seed_activity_score", "__seed_amount_score", "__seed_code"],
            [False, False, True],
        ),
        "coverage_rotation": _rotation_order(work, period=period),
    }

    def append_from(order: list[int], count: int, reason: str, *, enforce_cap: bool = True) -> int:
        nonlocal sector_cap_skips
        added = 0
        for row_index in order:
            if added >= count or len(selected_rows) >= resolved_limit:
                break
            if row_index in selected_set:
                continue
            sector = _row_sector(work, row_index, industry_col)
            if enforce_cap and sector and sector_counts.get(sector, 0) >= sector_cap:
                sector_cap_skips += 1
                continue
            selected_rows.append(row_index)
            selected_set.add(row_index)
            selected_reason[row_index] = reason
            if sector:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            added += 1
        return added

    actual_sleeves: dict[str, int] = {}
    for sleeve, requested in sleeve_counts.items():
        actual_sleeves[sleeve] = append_from(sleeve_orders[sleeve], requested, sleeve)

    remaining = resolved_limit - len(selected_rows)
    if remaining > 0:
        actual_sleeves["sector_backfill"] = append_from(
            sleeve_orders["liquidity_core"],
            remaining,
            "sector_backfill",
            enforce_cap=False,
        )

    selected = work.loc[selected_rows].copy()
    selected["seed_pool_reason"] = [selected_reason[index] for index in selected_rows]
    selected = selected.drop(
        columns=[column for column in selected.columns if column.startswith("__seed_")],
        errors="ignore",
    ).reset_index(drop=True)

    diagnostics = {
        "source_count": source_count,
        "selected_count": len(selected),
        "limit": resolved_limit,
        "requested_sleeve_counts": sleeve_counts,
        "sleeve_counts": {key: value for key, value in actual_sleeves.items() if value},
        "sector_cap": sector_cap if industry_col else 0,
        "sector_cap_skips": sector_cap_skips if industry_col else 0,
        "sector_count": len(sector_counts),
        "available_signals": signals,
    }
    return SeedPoolSelection(dataframe=selected, diagnostics=diagnostics)


def _attach_seed_scores(df: pd.DataFrame) -> list[str]:
    amount = _numeric_series(df, ("amount", "成交额"))
    change = _numeric_series(df, ("change_pct", "涨跌幅"))
    turnover = _numeric_series(df, ("turnover_rate", "换手率"))
    volume_ratio = _numeric_series(df, ("volume_ratio", "量比"))
    change_20d = _numeric_series(df, ("change_20d",))
    change_60d = _numeric_series(df, ("change_60d",))
    board_heat = _numeric_series(
        df,
        ("board_heat_score", "industry_heat_score", "concept_heat_score"),
    )

    amount_score = _percentile_score(amount)
    intraday_strength_score = _target_score(change, target=2.0, falloff=18.0)
    trend_score = _weighted_available_score(
        [
            (_boolean_signal_score(df, "ema_stack_bullish"), 0.50),
            (_boolean_signal_score(df, "ema21_rising"), 0.25),
            (_boolean_signal_score(df, "ema50_rising"), 0.25),
        ],
        fallback=_missing_score(df.index),
    )
    historical_score = _weighted_available_score(
        [
            (_optional_percentile_score(change_20d), 0.35),
            (_optional_percentile_score(change_60d), 0.45),
            (trend_score, 0.20),
        ],
        fallback=_missing_score(df.index),
    )
    strength_score = _weighted_available_score(
        [(historical_score, 0.85), (intraday_strength_score, 0.15)],
        fallback=intraday_strength_score,
    )
    turnover_score = _target_score(turnover, target=5.0, falloff=10.0)
    volume_score = _target_score(volume_ratio, target=1.4, falloff=45.0)
    activity_score = _weighted_available_score(
        [(turnover_score, 0.60), (volume_score, 0.40)],
        fallback=amount_score,
    )
    heat_score = board_heat.clip(lower=0.0, upper=100.0)

    df["__seed_amount_score"] = amount_score
    df["__seed_strength_score"] = strength_score.fillna(0.0)
    df["__seed_historical_score"] = historical_score.fillna(0.0)
    df["__seed_activity_score"] = activity_score.fillna(0.0)
    df["seed_pool_score"] = _weighted_available_score(
        [
            (amount_score, 0.35),
            (historical_score, 0.30),
            (intraday_strength_score, 0.10),
            (activity_score, 0.15),
            (heat_score, 0.10),
        ],
        fallback=amount_score,
    ).round(4)

    available = ["amount"] if amount.notna().any() else []
    if change.notna().any():
        available.append("change_pct")
    if turnover.notna().any():
        available.append("turnover_rate")
    if volume_ratio.notna().any():
        available.append("volume_ratio")
    if change_20d.notna().any():
        available.append("change_20d")
    if change_60d.notna().any():
        available.append("change_60d")
    if any(
        column in df.columns and df[column].notna().any()
        for column in ("ema_stack_bullish", "ema21_rising", "ema50_rising")
    ):
        available.append("ema_trend")
    if board_heat.notna().any():
        available.append("board_heat")
    return available


def _allocate_sleeves(
    limit: int,
    *,
    core_ratio: float,
    strength_ratio: float,
    activity_ratio: float,
) -> dict[str, int]:
    ratios = [
        _bounded_ratio(core_ratio),
        _bounded_ratio(strength_ratio),
        _bounded_ratio(activity_ratio),
    ]
    total = sum(ratios)
    if total > 0.90:
        ratios = [ratio * 0.90 / total for ratio in ratios]

    counts = [int(limit * ratio) for ratio in ratios]
    if limit >= 4:
        counts = [max(count, 1) if ratio > 0 else 0 for count, ratio in zip(counts, ratios)]
    used = min(sum(counts), limit)
    coverage = max(limit - used, 0)
    return {
        "liquidity_core": counts[0],
        "daily_strength": counts[1],
        "trading_activity": counts[2],
        "coverage_rotation": coverage,
    }


def _weighted_available_score(
    components: list[tuple[pd.Series, float]],
    *,
    fallback: pd.Series,
) -> pd.Series:
    numerator = pd.Series(0.0, index=fallback.index, dtype="float64")
    denominator = pd.Series(0.0, index=fallback.index, dtype="float64")
    for series, weight in components:
        values = pd.to_numeric(series, errors="coerce")
        valid = values.notna()
        numerator = numerator.add(values.fillna(0.0) * weight, fill_value=0.0)
        denominator = denominator.add(valid.astype(float) * weight, fill_value=0.0)
    scored = numerator.div(denominator.where(denominator.gt(0)))
    return scored.fillna(fallback).clip(lower=0.0, upper=100.0)


def _percentile_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().any():
        return pd.Series(0.0, index=values.index, dtype="float64")
    return numeric.rank(method="average", pct=True).mul(100.0)


def _optional_percentile_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().any():
        return _missing_score(values.index)
    return numeric.rank(method="average", pct=True).mul(100.0)


def _boolean_signal_score(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return _missing_score(df.index)

    def convert(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return 100.0
            if normalized in {"false", "0", "no", "n"}:
                return 0.0
            return pd.NA
        return 100.0 if bool(value) else 0.0

    return pd.to_numeric(df[column].map(convert), errors="coerce")


def _missing_score(index: pd.Index) -> pd.Series:
    return pd.Series(float("nan"), index=index, dtype="float64")


def _target_score(values: pd.Series, *, target: float, falloff: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    score = 100.0 - (numeric - target).abs() * falloff
    return score.clip(lower=0.0, upper=100.0)


def _ordered_rows(df: pd.DataFrame, columns: list[str], ascending: list[bool]) -> list[int]:
    existing = [column for column in columns if column in df.columns]
    directions = [ascending[columns.index(column)] for column in existing]
    if not existing:
        return list(df.index)
    return list(df.sort_values(existing, ascending=directions, kind="mergesort").index)


def _rotation_order(df: pd.DataFrame, *, period: str) -> list[int]:
    keyed = []
    for row_index, code in zip(df.index, df["__seed_code"].tolist()):
        digest = hashlib.sha256(f"{period}|{code}|{row_index}".encode("utf-8")).hexdigest()
        keyed.append((digest, int(row_index)))
    keyed.sort()
    return [row_index for _digest, row_index in keyed]


def _numeric_series(df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    column = _first_column(df, aliases)
    if column is None:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce")


def _code_series(df: pd.DataFrame) -> pd.Series:
    column = _first_column(df, ("code", "代码", "symbol"))
    if column is None:
        return pd.Series([f"row-{index}" for index in df.index], index=df.index)
    return df[column].map(lambda value: normalize_code(value, allow_ticker=True))


def _row_sector(df: pd.DataFrame, row_index: int, industry_col: str | None) -> str:
    if industry_col is None:
        return ""
    return safe_text(df.at[row_index, industry_col], max_len=80)


def _first_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    return next((column for column in aliases if column in df.columns), None)


def _bounded_ratio(value: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0
