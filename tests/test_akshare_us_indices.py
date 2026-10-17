"""Offline contracts for the AkShare Sina US-index fallback."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from data_provider.akshare_fetcher import AkshareFetcher


def test_get_main_indices_uses_dated_sina_us_index_fallback() -> None:
    latest = pd.DataFrame([
        {"date": "2026-08-12", "open": 6400.0, "high": 6420.0, "low": 6380.0, "close": 6405.0, "volume": 100, "amount": 0},
        {"date": "2026-08-13", "open": 6410.0, "high": 6460.0, "low": 6400.0, "close": 6450.0, "volume": 120, "amount": 0},
    ])
    fake_akshare = SimpleNamespace(index_us_stock_sina=Mock(return_value=latest))
    fetcher = AkshareFetcher()

    with patch.dict("sys.modules", {"akshare": fake_akshare}), \
         patch.object(fetcher, "_set_random_user_agent"), \
         patch.object(fetcher, "_enforce_rate_limit"):
        results = fetcher.get_main_indices(region="us")

    assert results is not None
    assert len(results) == 4
    assert results[0]["code"] == "SPX"
    assert results[0]["as_of_date"] == "2026-08-13"
    assert results[0]["change"] == 45.0
    assert round(results[0]["change_pct"], 2) == 0.7
