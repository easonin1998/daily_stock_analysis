# -*- coding: utf-8 -*-
"""Industry provider contracts for the Screening universe."""

from __future__ import annotations

from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

from src.services.screening.industry import (
    _normalize_board_fund_flow_snapshot,
    _parse_board_fund_flow,
    attach_qmt_industry_overlays,
    enrich_industry_concepts,
    fetch_akshare_board_map,
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


def test_ths_concept_flow_columns_are_normalized_with_price_and_members() -> None:
    frame = pd.DataFrame([
        {
            "序号": 9,
            "行业": "人形机器人",
            "行业-涨跌幅": 1.46,
            "净额": 72.22,
            "公司家数": 473,
        },
        {
            "序号": 43,
            "行业": "机器人概念",
            "行业-涨跌幅": 0.62,
            "净额": 86.05,
            "公司家数": 1227,
        },
    ])

    normalized = _normalize_board_fund_flow_snapshot(frame, board_type="concept")
    indexed = normalized.set_index("board_name")

    assert set(indexed.index) == {"人形机器人", "机器人概念"}
    assert indexed.loc["人形机器人", "weighted_change_pct"] == 1.46
    assert indexed.loc["机器人概念", "member_count"] == 1227
    assert indexed.loc["机器人概念", "fund_flow_score"] == 100.0
    assert indexed.loc["机器人概念", "board_heat_score"] == 50.0


def test_qmt_thy_membership_bridges_ths_metrics_to_baostock_industries() -> None:
    universe = pd.DataFrame([
        {"code": "000001", "industry": "J66货币金融服务"},
        {"code": "000002", "industry": "J66货币金融服务"},
        {"code": "000003", "industry": "C38电气机械和器材制造业"},
    ])
    membership = pd.DataFrame([
        {"code": "000001", "concepts": "1银行,2银行,3股份制银行"},
        {"code": "000002", "concepts": "1非银金融,2证券,3证券"},
        {"code": "000003", "concepts": "1电力设备,2电池,3电池化学品"},
    ])
    overlays = pd.DataFrame([
        {"board_type": "industry", "board_name": "银行", "board_heat_score": 90.0, "fund_flow_score": 100.0, "main_net_inflow": 48.74},
        {"board_type": "industry", "board_name": "证券", "board_heat_score": 70.0, "fund_flow_score": 80.0, "main_net_inflow": 12.0},
        {"board_type": "industry", "board_name": "电池", "board_heat_score": 60.0, "fund_flow_score": 50.0, "main_net_inflow": -2.0},
        {"board_type": "concept", "board_name": "人形机器人", "board_heat_score": 99.0, "fund_flow_score": 99.0, "main_net_inflow": 72.22},
    ])

    result, diagnostics = attach_qmt_industry_overlays(
        universe,
        membership,
        overlays,
        include_fund_flow=True,
    )

    assert result["board_heat_score"].tolist() == [90.0, 70.0, 60.0]
    assert result["board_fund_flow_score"].tolist() == [100.0, 80.0, 50.0]
    assert result["board_fund_flow_net_amount"].tolist() == [48.74, 12.0, -2.0]
    assert diagnostics == {
        "membership_rows": 3,
        "matched_stock_count": 3,
        "external_heat_stock_count": 3,
        "fund_flow_stock_count": 3,
        "matched_board_count": 3,
    }


def test_akshare_board_map_falls_back_to_eastmoney_http() -> None:
    import akshare as ak

    industry_boards = pd.DataFrame([
        {"name": "半导体", "code": "BK1001", "change_pct": 2, "rank": 1}
    ])
    concept_boards = pd.DataFrame([
        {"name": "算力", "code": "BK1002", "change_pct": 3, "rank": 1}
    ])
    constituents = pd.DataFrame([{"code": "000001", "name": "示例"}])
    with (
        patch.object(ak, "stock_board_industry_name_em", side_effect=RuntimeError("offline")),
        patch.object(ak, "stock_board_concept_name_em", side_effect=RuntimeError("offline")),
        patch(
            "src.services.screening.industry._fetch_eastmoney_http_boards",
            side_effect=[industry_boards, concept_boards],
        ),
        patch(
            "src.services.screening.industry._fetch_eastmoney_http_constituents",
            return_value=constituents,
        ) as http_members,
    ):
        mapping, notes = fetch_akshare_board_map(max_boards=1, cache_dir=None)

    assert mapping["000001"]["industry"] == "半导体"
    assert mapping["000001"]["concepts"] == "算力"
    assert mapping["000001"]["board_heat_score"] > 50
    assert http_members.call_count == 2
    assert any("concepts eastmoney_http boards loaded: 1/1" in note for note in notes)


def test_akshare_partial_cache_retries_only_missing_board_field(tmp_path) -> None:
    import akshare as ak

    industry_boards = pd.DataFrame([
        {"name": "半导体", "code": "BK1001", "change_pct": 2, "rank": 1}
    ])
    concept_boards = pd.DataFrame([
        {"name": "算力", "code": "BK1002", "change_pct": 3, "rank": 1}
    ])
    constituents = pd.DataFrame([{"code": "000001", "name": "示例"}])
    with (
        patch.object(ak, "stock_board_industry_name_em", return_value=industry_boards),
        patch.object(ak, "stock_board_industry_cons_em", return_value=constituents),
        patch.object(ak, "stock_board_concept_name_em", side_effect=RuntimeError("offline")),
        patch(
            "src.services.screening.industry._fetch_eastmoney_http_boards",
            side_effect=RuntimeError("offline"),
        ),
    ):
        partial, _ = fetch_akshare_board_map(max_boards=1, cache_dir=tmp_path)

    with (
        patch.object(
            ak,
            "stock_board_industry_name_em",
            side_effect=AssertionError("cached industry must not be fetched again"),
        ),
        patch.object(ak, "stock_board_concept_name_em", return_value=concept_boards),
        patch.object(ak, "stock_board_concept_cons_em", return_value=constituents),
    ):
        completed, notes = fetch_akshare_board_map(max_boards=1, cache_dir=tmp_path)

    assert partial["000001"]["industry"] == "半导体"
    assert not partial["000001"]["concepts"]
    assert completed["000001"]["industry"] == "半导体"
    assert completed["000001"]["concepts"] == "算力"
    assert any("cache is partial" in note for note in notes)
    assert any("industry retained from partial cache" in note for note in notes)


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
