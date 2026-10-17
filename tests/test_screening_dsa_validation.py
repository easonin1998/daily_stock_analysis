# -*- coding: utf-8 -*-
"""Final-pick AkShare/Baostock validation audit contracts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.services.screening.candidate_context import fetch_stock_news_items
from src.services.screening_service import get_dsa_daily_validation, search_dsa_stock_news


def test_final_pick_daily_validation_exposes_provider_audit() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-10-15", "2026-10-16"],
            "close": [10.0, 10.5],
        }
    )
    frame.attrs["data_source_mode"] = "akshare_baostock"
    frame.attrs["source_validation"] = {
        "mode": "akshare_baostock",
        "status": "ok",
        "validator": "BaostockFetcher",
        "overlap_rows": 2,
    }
    manager = MagicMock()
    manager.get_daily_data.return_value = (frame, "AkshareFetcher")

    with patch("src.services.screening_service._get_dsa_fetcher_manager", return_value=manager):
        result = get_dsa_daily_validation("300473", lookback_days=160)

    assert result["status"] == "ok"
    assert result["primary_source"] == "AkshareFetcher"
    assert result["rows"] == 2
    assert result["latest_date"] == "2026-10-16"
    assert result["validation"]["validator"] == "BaostockFetcher"
    manager.get_daily_data.assert_called_once_with("300473", days=160)


def test_akshare_stock_news_items_reject_undated_stale_and_future_rows() -> None:
    frame = pd.DataFrame([
        {"新闻标题": "最新消息", "发布时间": "2026-10-16 10:00:00", "文章来源": "财联社", "新闻链接": "https://example.com/new"},
        {"新闻标题": "旧消息", "发布时间": "2026-10-13 10:00:00", "文章来源": "旧来源", "新闻链接": "https://example.com/old"},
        {"新闻标题": "无日期消息", "发布时间": "", "文章来源": "未知", "新闻链接": "https://example.com/undated"},
        {"新闻标题": "未来消息", "发布时间": "2026-10-18 10:00:00", "文章来源": "未知", "新闻链接": "https://example.com/future"},
    ])

    with patch("akshare.stock_news_em", return_value=frame):
        results = fetch_stock_news_items(
            "300473",
            limit=3,
            max_age_days=2,
            now=pd.Timestamp("2026-10-17 12:00:00").to_pydatetime(),
        )

    assert [item["title"] for item in results] == ["最新消息"]
    assert results[0]["published_date"] == "2026-10-16 10:00:00"


def test_stock_news_uses_akshare_fallback_when_primary_search_is_empty() -> None:
    service = MagicMock()
    service.is_available = True
    service.search_stock_news.return_value = MagicMock(
        query="query",
        provider="searxng",
        success=False,
        error_message="rate limited",
        results=[],
    )
    fallback_items = [{
        "title": "两日内消息",
        "snippet": "内容",
        "url": "https://example.com/latest",
        "source": "财联社",
        "published_date": "2026-10-16 10:00:00",
    }]

    with (
        patch("src.services.screening_service._get_dsa_search_service", return_value=service),
        patch(
            "src.services.screening.candidate_context.fetch_stock_news_items",
            return_value=fallback_items,
        ) as fallback,
    ):
        result = search_dsa_stock_news("300473", "德尔股份", max_results=3)

    assert result["success"] is True
    assert result["provider"] == "akshare_stock_news_em"
    assert result["results"] == fallback_items
    fallback.assert_called_once_with("300473", limit=3, max_age_days=2)
