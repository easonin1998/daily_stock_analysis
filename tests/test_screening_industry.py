# -*- coding: utf-8 -*-
"""Industry provider contracts for the Screening universe."""

from __future__ import annotations

from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

from src.services.screening.industry import (
    _parse_board_fund_flow,
    enrich_industry_concepts,
    fetch_baostock_industry_map,
)


def test_baostock_industry_provider_is_cached_and_applied() -> None:
    raw_mapping = {"600519": "食品饮料", "000001": "银行"}
    with tempfile.TemporaryDirectory(dir=Path.cwd() / "data") as temp_dir:
        with patch(
            "data_provider.baostock_fetcher.BaostockFetcher.get_industry_map",
            return_value=raw_mapping,
        ) as provider:
            mapping, notes = fetch_baostock_industry_map(
                cache_dir=temp_dir,
                cache_ttl_hours=24,
            )
            cached_mapping, cached_notes = fetch_baostock_industry_map(
                cache_dir=temp_dir,
                cache_ttl_hours=24,
            )

    assert mapping == cached_mapping
    assert mapping["600519"]["industry"] == "食品饮料"
    assert provider.call_count == 1
    assert any("rows loaded: 2" in note for note in notes)
    assert any("cache hit" in note for note in cached_notes)

    snapshot = pd.DataFrame(
        [
            {"code": "600519", "name": "贵州茅台"},
            {"code": "000001", "name": "平安银行"},
        ]
    )
    with patch(
        "src.services.screening.industry.fetch_baostock_industry_map",
        return_value=(mapping, ["cached"]),
    ):
        enriched, _ = enrich_industry_concepts(snapshot, provider="baostock")

    assert enriched["industry"].tolist() == ["食品饮料", "银行"]


def test_free_provider_keeps_stable_industry_and_adds_akshare_heat() -> None:
    snapshot = pd.DataFrame([{"code": "600519", "name": "贵州茅台"}])
    stable = {"600519": {"industry": "食品饮料", "concepts": ""}}
    heat = {
        "600519": {
            "industry": "白酒",
            "concepts": "国企改革",
            "board_heat_score": 88.0,
        }
    }
    with (
        patch(
            "src.services.screening.industry.fetch_baostock_industry_map",
            return_value=(stable, ["stable"]),
        ),
        patch(
            "src.services.screening.industry.fetch_akshare_board_map",
            return_value=(heat, ["heat"]),
        ) as akshare_provider,
    ):
        enriched, notes = enrich_industry_concepts(
            snapshot,
            provider="free",
            include_fund_flow=True,
        )

    assert enriched.loc[0, "industry"] == "食品饮料"
    assert enriched.loc[0, "concepts"] == "国企改革"
    assert enriched.loc[0, "board_heat_score"] == 88.0
    assert notes == ["stable", "heat", "industry/concepts enrichment applied: industry=1, concepts=1, heat=1"]
    assert akshare_provider.call_args.kwargs["include_fund_flow"] is True


def test_board_fund_flow_parser_ranks_net_amount_cross_sectionally() -> None:
    parsed = _parse_board_fund_flow(pd.DataFrame([
        {"板块名称": "半导体", "主力净流入-净额": 200.0},
        {"板块名称": "银行", "主力净流入-净额": -100.0},
    ]))

    assert parsed["半导体"]["score"] == 100.0
    assert parsed["银行"]["score"] == 50.0
    assert parsed["银行"]["net_amount"] == -100.0


def test_baostock_provider_uses_stale_cache_after_refresh_failure(tmp_path) -> None:
    raw_mapping = {"600519": "食品饮料"}
    with patch(
        "data_provider.baostock_fetcher.BaostockFetcher.get_industry_map",
        return_value=raw_mapping,
    ):
        fetch_baostock_industry_map(cache_dir=tmp_path, cache_ttl_hours=24)
    with patch(
        "data_provider.baostock_fetcher.BaostockFetcher.get_industry_map",
        side_effect=RuntimeError("offline"),
    ):
        mapping, notes = fetch_baostock_industry_map(
            cache_dir=tmp_path,
            cache_ttl_seconds=0,
        )

    assert mapping["600519"]["industry"] == "食品饮料"
    assert any("stale cache fallback" in note for note in notes)


def test_free_cached_provider_never_calls_live_akshare() -> None:
    snapshot = pd.DataFrame([{"code": "600519"}])
    stable = {"600519": {"industry": "食品饮料", "concepts": ""}}
    heat = {"600519": {"industry": "", "concepts": "白酒", "board_heat_score": 80.0}}
    with (
        patch(
            "src.services.screening.industry.load_cached_baostock_industry_map",
            return_value=(stable, ["stable cache"]),
        ),
        patch(
            "src.services.screening.industry.load_cached_akshare_board_map",
            return_value=(heat, ["heat cache"]),
        ),
        patch(
            "src.services.screening.industry.fetch_akshare_board_map",
            side_effect=AssertionError("live provider must not run"),
        ),
    ):
        enriched, notes = enrich_industry_concepts(snapshot, provider="free_cached")

    assert enriched.loc[0, "industry"] == "食品饮料"
    assert enriched.loc[0, "concepts"] == "白酒"
    assert enriched.loc[0, "board_heat_score"] == 80.0
    assert notes[:2] == ["stable cache", "heat cache"]
