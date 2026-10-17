# -*- coding: utf-8 -*-
"""Tests for the persistent screening daily-feature library."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from src.services.screening import daily as daily_module
from src.services.screening.daily import (
    fetch_daily_history,
    load_cached_daily_history,
    save_cached_daily_history,
)
from src.services.screening.daily_library import (
    completed_snapshot_bar_date,
    load_daily_feature_library,
    merge_daily_feature_library,
    refresh_daily_feature_library_from_cache,
    select_daily_library_batch,
    upsert_daily_feature_library,
)


def _history(periods: int = 140) -> pd.DataFrame:
    close = np.linspace(10.0, 18.0, periods)
    volume = np.full(periods, 10_000_000.0)
    frame = pd.DataFrame({
        "date": pd.date_range("2026-05-29", periods=periods, freq="D"),
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": volume,
        "amount": volume * close,
    })
    frame.attrs["daily_source"] = "sina"
    return frame


def test_library_upsert_load_and_merge_only_fresh_entries(tmp_path) -> None:
    path = tmp_path / "features.json"
    enriched = pd.DataFrame([
        {
            "code": "000001",
            "daily_latest_date": "2026-10-16",
            "daily_data_points": 140,
            "change_20d": 12.5,
            "change_60d": 31.0,
            "ema_stack_bullish": True,
            "daily_source": "sina",
        },
        {
            "code": "000002",
            "daily_latest_date": "2026-09-01",
            "daily_data_points": 140,
            "change_20d": 5.0,
            "daily_source": "sina",
        },
    ])

    update = upsert_daily_feature_library(enriched, library_path=path)
    loaded = load_daily_feature_library(
        path,
        max_age_days=7,
        now=datetime(2026, 10, 17, 8, 0),
    )
    merged, diagnostics = merge_daily_feature_library(
        pd.DataFrame({"code": ["000001", "000002", "000003"]}),
        library_path=path,
        max_age_days=7,
        now=datetime(2026, 10, 17, 8, 0),
    )

    assert update["updated_count"] == 2
    assert loaded["code"].tolist() == ["000001"]
    assert diagnostics["entry_count"] == 2
    assert diagnostics["fresh_count"] == 1
    assert diagnostics["stale_count"] == 1
    assert diagnostics["matched_count"] == 1
    assert float(merged.loc[0, "change_20d"]) == 12.5
    assert bool(merged.loc[0, "daily_library_hit"]) is True
    assert bool(merged.loc[1, "daily_library_hit"]) is False


def test_cache_refresh_appends_completed_sina_snapshot_bar(tmp_path) -> None:
    cache_dir = tmp_path / "daily"
    library_path = tmp_path / "features.json"
    history = _history()
    history["date"] = pd.date_range("2026-05-30", "2026-10-16", periods=len(history))
    save_cached_daily_history(
        "000001",
        history,
        source="sina",
        lookback_days=160,
        cache_dir=cache_dir,
    )
    snapshot = pd.DataFrame([{
        "code": "000001",
        "price": 18.8,
        "open": 18.4,
        "high": 19.0,
        "low": 18.2,
        "volume": 12_000_000,
        "amount": 225_600_000,
    }])

    result = refresh_daily_feature_library_from_cache(
        snapshot,
        cache_dir=cache_dir,
        library_path=library_path,
        source="sina",
        lookback_days=160,
        append_snapshot_bar=True,
        snapshot_bar_date=date(2026, 10, 17),
    )
    cached = load_cached_daily_history(
        "000001",
        source="sina",
        lookback_days=160,
        cache_dir=cache_dir,
    )
    loaded = load_daily_feature_library(
        library_path,
        max_age_days=1,
        now=datetime(2026, 10, 17, 18, 0),
    )
    repeated = refresh_daily_feature_library_from_cache(
        snapshot,
        cache_dir=cache_dir,
        library_path=library_path,
        source="sina",
        lookback_days=160,
        append_snapshot_bar=True,
        snapshot_bar_date=date(2026, 10, 17),
    )

    assert result["cache_count"] == 1
    assert result["appended_count"] == 1
    assert result["updated_count"] == 1
    assert cached is not None
    assert str(cached.iloc[-1]["date"])[:10] == "2026-10-17"
    assert loaded.iloc[0]["daily_latest_date"] == "2026-10-17"
    assert repeated["appended_count"] == 0


def test_cache_loader_reuses_a_longer_compatible_window(tmp_path) -> None:
    cache_dir = tmp_path / "daily"
    history = _history()
    save_cached_daily_history(
        "000001",
        history,
        source="auto",
        lookback_days=220,
        cache_dir=cache_dir,
    )

    cached = load_cached_daily_history(
        "000001",
        source="auto",
        lookback_days=180,
        cache_dir=cache_dir,
    )

    assert cached is not None
    assert len(cached) == len(history)
    assert cached.attrs["daily_cache_compatible_fallback"] is True


def test_cache_loader_rejects_unadjusted_history_for_qfq_research(tmp_path) -> None:
    cache_dir = tmp_path / "daily"
    unadjusted = _history()
    save_cached_daily_history(
        "000001",
        unadjusted,
        source="auto",
        lookback_days=220,
        cache_dir=cache_dir,
    )
    adjusted = _history()
    adjusted.attrs["daily_source"] = "tencent"
    save_cached_daily_history(
        "000002",
        adjusted,
        source="auto",
        lookback_days=220,
        cache_dir=cache_dir,
    )

    assert load_cached_daily_history(
        "000001",
        source="auto",
        lookback_days=180,
        cache_dir=cache_dir,
        price_adjustment="qfq",
    ) is None
    accepted = load_cached_daily_history(
        "000002",
        source="auto",
        lookback_days=180,
        cache_dir=cache_dir,
        price_adjustment="qfq",
    )
    assert accepted is not None
    assert accepted.attrs["daily_price_adjustment"] == "qfq"


def test_qfq_auto_fetch_excludes_unadjusted_sina(monkeypatch) -> None:
    calls: list[str] = []

    def fake_tencent(code: str, *, lookback_days: int) -> pd.DataFrame:
        calls.append(f"tencent:{code}:{lookback_days}")
        return _history(periods=130)

    def unexpected_sina(code: str, *, lookback_days: int) -> pd.DataFrame:
        raise AssertionError(f"Sina must not serve qfq research data: {code}:{lookback_days}")

    monkeypatch.setattr(daily_module, "_has_tushare_token", lambda: False)
    monkeypatch.setattr(
        daily_module,
        "_rank_daily_sources_by_health",
        lambda sources: (sources, []),
    )
    monkeypatch.setattr(daily_module, "_source_disabled_reason", lambda source: None)
    monkeypatch.setattr(daily_module, "_fetch_daily_tencent", fake_tencent)
    monkeypatch.setattr(daily_module, "_fetch_daily_sina", unexpected_sina)

    result = fetch_daily_history(
        "000001",
        source="auto",
        lookback_days=120,
        retries=0,
        price_adjustment="qfq",
    )

    assert calls == ["tencent:000001:120"]
    assert result.attrs["daily_source"] == "tencent"
    assert result.attrs["daily_price_adjustment"] == "qfq"
    assert "sina" not in result.attrs["daily_source_order"]


def test_batch_selection_skips_fresh_coverage_and_is_deterministic(tmp_path) -> None:
    path = tmp_path / "features.json"
    upsert_daily_feature_library(
        pd.DataFrame([{
            "code": "000001",
            "daily_latest_date": "2026-10-16",
            "daily_data_points": 140,
        }]),
        library_path=path,
    )
    candidates = pd.DataFrame({"code": ["000001", "000002", "000003", "000004"]})

    first = select_daily_library_batch(
        candidates,
        library_path=path,
        limit=2,
        period="2026-10-17:0",
        max_age_days=7,
    )
    second = select_daily_library_batch(
        candidates,
        library_path=path,
        limit=2,
        period="2026-10-17:0",
        max_age_days=7,
    )

    assert "000001" not in first["code"].tolist()
    assert first["code"].tolist() == second["code"].tolist()
    assert len(first) == 2
    assert upsert_daily_feature_library(
        pd.DataFrame(),
        library_path=path,
    )["entry_count"] == 1


def test_completed_snapshot_bar_date_requires_live_completed_session() -> None:
    live = pd.DataFrame({"code": ["000001"]})
    live.attrs["fallback_used"] = False

    assert completed_snapshot_bar_date(
        live,
        market="cn",
        now=datetime(2026, 10, 17, 9, 0),
    ) == date(2026, 10, 16)

    live.attrs["fallback_used"] = True
    assert completed_snapshot_bar_date(
        live,
        market="cn",
        now=datetime(2026, 10, 17, 9, 0),
    ) is None
