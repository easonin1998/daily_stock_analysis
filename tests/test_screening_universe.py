# -*- coding: utf-8 -*-
"""Tests for the pre-enrichment Screening seed pool."""

from __future__ import annotations

import pandas as pd

from src.services.screening.universe import build_seed_pool


def _snapshot(rows: int = 24) -> pd.DataFrame:
    industries = ["半导体", "医药", "银行", "新能源"]
    return pd.DataFrame(
        {
            "code": [f"{index + 1:06d}" for index in range(rows)],
            "name": [f"stock-{index}" for index in range(rows)],
            "amount": [float((rows - index) * 100_000_000) for index in range(rows)],
            "change_pct": [float((index % 9) - 3) for index in range(rows)],
            "turnover_rate": [float(1 + index % 8) for index in range(rows)],
            "volume_ratio": [float(0.7 + (index % 7) * 0.2) for index in range(rows)],
            "industry": [industries[index % len(industries)] for index in range(rows)],
        }
    )


def test_seed_pool_is_bounded_diversified_and_explainable() -> None:
    selected = build_seed_pool(
        _snapshot(),
        limit=12,
        period="2026-10-17:cn:martin",
        max_sector_ratio=0.25,
    )

    assert len(selected.dataframe) == 12
    assert selected.dataframe["code"].is_unique
    assert selected.dataframe["seed_pool_score"].notna().all()
    assert selected.dataframe["seed_pool_reason"].ne("").all()
    assert selected.dataframe["industry"].value_counts().max() <= 3
    assert selected.diagnostics["source_count"] == 24
    assert selected.diagnostics["selected_count"] == 12
    assert selected.diagnostics["sector_cap"] == 3
    assert set(selected.diagnostics["available_signals"]) == {
        "amount",
        "change_pct",
        "turnover_rate",
        "volume_ratio",
    }


def test_coverage_rotation_is_deterministic_for_the_same_period() -> None:
    snapshot = _snapshot(30)
    first = build_seed_pool(
        snapshot,
        limit=8,
        period="2026-10-17",
        core_ratio=0,
        strength_ratio=0,
        activity_ratio=0,
        max_sector_ratio=1,
    )
    second = build_seed_pool(
        snapshot,
        limit=8,
        period="2026-10-17",
        core_ratio=0,
        strength_ratio=0,
        activity_ratio=0,
        max_sector_ratio=1,
    )

    assert first.dataframe["code"].tolist() == second.dataframe["code"].tolist()
    assert set(first.dataframe["seed_pool_reason"]) == {"coverage_rotation"}


def test_seed_pool_handles_missing_optional_activity_fields() -> None:
    snapshot = _snapshot(10).drop(columns=["turnover_rate", "volume_ratio", "industry"])

    selected = build_seed_pool(snapshot, limit=5, period="2026-10-17")

    assert len(selected.dataframe) == 5
    assert selected.diagnostics["sector_cap"] == 0
    assert selected.diagnostics["available_signals"] == ["amount", "change_pct"]


def test_seed_pool_prefers_real_historical_momentum_when_library_is_available() -> None:
    snapshot = _snapshot(12)
    snapshot["change_20d"] = [-5.0] * 11 + [30.0]
    snapshot["change_60d"] = [-10.0] * 11 + [80.0]
    snapshot["ema_stack_bullish"] = [False] * 11 + [True]
    snapshot["ema21_rising"] = [False] * 11 + [True]
    snapshot["ema50_rising"] = [False] * 11 + [True]
    snapshot.loc[11, "change_pct"] = -1.0

    selected = build_seed_pool(
        snapshot,
        limit=4,
        period="2026-10-17",
        core_ratio=0,
        strength_ratio=0.9,
        activity_ratio=0,
        max_sector_ratio=1,
    )

    assert snapshot.loc[11, "code"] in selected.dataframe["code"].tolist()
    assert {"change_20d", "change_60d", "ema_trend"}.issubset(
        selected.diagnostics["available_signals"]
    )
