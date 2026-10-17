# -*- coding: utf-8 -*-
"""Industry provider contracts for the Screening universe."""

from __future__ import annotations

from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

from src.services.screening.industry import enrich_industry_concepts, fetch_baostock_industry_map


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
