# -*- coding: utf-8 -*-
"""Daily free-sector collection contracts."""

from __future__ import annotations

import pandas as pd

from scripts.collect_screening_sector_data import collect_sector_data
from src.services.screening.daily_library import upsert_daily_feature_library
from src.services.screening.sector_strength import load_sector_strength_history


def test_collect_sector_data_uses_completed_library_and_normalized_storage(tmp_path) -> None:
    library_path = tmp_path / "daily.json"
    history_path = tmp_path / "sector.sqlite3"
    map_path = tmp_path / "industry.csv"
    rows = []
    for index, (code, industry) in enumerate(
        (("000001", "光通信"), ("000002", "光通信"), ("000003", "银行"), ("000004", "银行"))
    ):
        rows.append({
            "code": code,
            "daily_latest_date": "2026-10-16",
            "daily_data_points": 160,
            "change_20d": 20.0 - index * 5.0,
            "change_60d": 60.0 - index * 15.0,
            "adr_20_pct": 5.0 - index * 0.5,
            "ema_stack_bullish": index < 2,
            "ema21_rising": index < 2,
        })
    upsert_daily_feature_library(pd.DataFrame(rows), library_path=library_path)
    pd.DataFrame([
        {"code": code, "industry": industry}
        for code, industry in (("000001", "光通信"), ("000002", "光通信"), ("000003", "银行"), ("000004", "银行"))
    ]).to_csv(map_path, index=False)

    result = collect_sector_data(
        library_path=library_path,
        history_path=history_path,
        provider="none",
        max_boards=20,
        provider_cache_dir=tmp_path / "cache",
        provider_cache_ttl_hours=24,
        map_files=[map_path],
    )
    loaded = load_sector_strength_history(history_path)

    assert result["status"] == "ok"
    assert result["coverage_pct"] == 100.0
    assert result["industry_members"] == 4
    assert result["sector_count"] == 2
    assert set(loaded["industry"]) == {"光通信", "银行"}
    assert len(loaded.attrs["membership"]) == 4
