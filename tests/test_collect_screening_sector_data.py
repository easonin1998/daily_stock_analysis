# -*- coding: utf-8 -*-
"""Daily free-sector collection contracts."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from scripts.collect_screening_sector_data import (
    _attach_qmt_concepts,
    collect_sector_data,
)
from src.services.screening.daily_library import upsert_daily_feature_library
from src.services.screening.sector_heat import load_latest_board_heat_snapshot
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


def test_collect_sector_data_can_persist_market_heat_from_injected_quotes(tmp_path) -> None:
    library_path = tmp_path / "daily.json"
    history_path = tmp_path / "sector.sqlite3"
    map_path = tmp_path / "industry.csv"
    rows = [
        {
            "code": f"00000{index}",
            "daily_latest_date": "2026-10-16",
            "daily_data_points": 160,
            "change_20d": 20.0 - index,
            "change_60d": 40.0 - index,
            "adr_20_pct": 4.0,
            "ema_stack_bullish": index < 3,
            "ema21_rising": index < 3,
            "avg_amount_20d": 100.0,
        }
        for index in range(1, 5)
    ]
    upsert_daily_feature_library(pd.DataFrame(rows), library_path=library_path)
    pd.DataFrame([
        {"code": "000001", "industry": "通信", "concepts": "算力"},
        {"code": "000002", "industry": "通信", "concepts": "算力"},
        {"code": "000003", "industry": "银行", "concepts": "高股息"},
        {"code": "000004", "industry": "银行", "concepts": "高股息"},
    ]).to_csv(map_path, index=False)
    quotes = pd.DataFrame([
        {"code": "000001", "price": 10, "change_pct": 7, "amount": 200},
        {"code": "000002", "price": 11, "change_pct": 4, "amount": 150},
        {"code": "000003", "price": 9, "change_pct": -1, "amount": 80},
        {"code": "000004", "price": 8, "change_pct": -2, "amount": 70},
    ])
    quotes.attrs["snapshot_source"] = "fixture"
    quotes.attrs["stale"] = True

    with patch(
        "scripts.collect_screening_sector_data.fetch_akshare_board_fund_flow_snapshot",
        return_value=(pd.DataFrame(), ["fixture: no external flow"]),
    ):
        result = collect_sector_data(
            library_path=library_path,
            history_path=history_path,
            provider="none",
            max_boards=20,
            provider_cache_dir=tmp_path / "cache",
            provider_cache_ttl_hours=24,
            map_files=[map_path],
            collect_market_heat=True,
            quote_snapshot=quotes,
            observed_at="2026-10-19T09:05:00+08:00",
        )
    stored = load_latest_board_heat_snapshot(history_path)

    assert result["board_heat"]["status"] == "available"
    assert result["board_heat"]["quote_source"] == "fixture"
    assert result["board_heat"]["trading_date"] == "2026-10-16"
    assert result["board_heat"]["feature_date"] == "2026-10-16"
    assert result["board_heat"]["industry"][0]["name"] == "通信"
    assert set(stored["board_type"]) == {"industry", "concept"}


def test_attach_qmt_concepts_preserves_existing_labels() -> None:
    enriched = pd.DataFrame([
        {"code": "000001", "industry": "银行", "concepts": "高股息"},
        {"code": "000002", "industry": "通信", "concepts": ""},
    ])
    membership = pd.DataFrame([
        {"code": "000001", "concepts": "高股息,AI"},
        {"code": "000002", "concepts": "算力"},
    ])

    result = _attach_qmt_concepts(enriched, membership)

    assert result.loc[0, "concepts"] == "高股息,AI"
    assert result.loc[1, "concepts"] == "算力"
