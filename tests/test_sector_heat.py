# -*- coding: utf-8 -*-
"""Point-in-time board heat and read-only QMT quote contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from src.services.screening.qmt_quotes import (
    fetch_qmt_quotes,
    fetch_qmt_sector_membership,
    overlay_qmt_quotes,
)
from src.services.screening.config import Config
from src.services.screening.sector_heat import (
    compute_board_heat_snapshot,
    load_latest_board_heat_snapshot,
    summarize_board_heat_snapshot,
    upsert_board_heat_history,
)


def _universe() -> pd.DataFrame:
    return pd.DataFrame([
        {"code": "000001", "industry": "通信", "concepts": "AI,算力", "avg_amount_20d": 100, "ema_stack_bullish": True, "ema21_rising": True, "relative_strength_60d_pctile": 96},
        {"code": "000002", "industry": "通信", "concepts": "算力", "avg_amount_20d": 100, "ema_stack_bullish": True, "ema21_rising": True, "relative_strength_60d_pctile": 85},
        {"code": "000003", "industry": "银行", "concepts": "高股息", "avg_amount_20d": 100, "ema_stack_bullish": False, "ema21_rising": False, "relative_strength_60d_pctile": 30},
    ])


def _quotes() -> pd.DataFrame:
    frame = pd.DataFrame([
        {"code": "000001", "price": 10, "change_pct": 8, "amount": 200},
        {"code": "000002", "price": 11, "change_pct": 3, "amount": 150},
        {"code": "000003", "price": 9, "change_pct": -2, "amount": 50},
    ])
    frame.attrs["snapshot_source"] = "fixture"
    frame.attrs["stale"] = False
    return frame


def test_board_heat_keeps_multi_concept_membership_and_objective_rank() -> None:
    overlays = pd.DataFrame([
        {"board_type": "concept", "board_name": "算力", "fund_flow_score": 90, "main_net_inflow": 1_000_000},
    ])

    snapshot, membership = compute_board_heat_snapshot(
        _universe(),
        quote_snapshot=_quotes(),
        board_overlays=overlays,
        observed_at="2026-10-19T15:01:00+08:00",
        trading_date="2026-10-19",
        feature_date="2026-10-16",
        min_members=1,
    )
    industry = snapshot[snapshot["board_type"].eq("industry")].set_index("board_name")
    concept = snapshot[snapshot["board_type"].eq("concept")].set_index("board_name")

    assert industry.loc["通信", "heat_rank"] == 1
    assert industry.loc["通信", "weighted_change_pct"] == 5.5
    assert concept.loc["算力", "fund_flow_score"] == 90
    assert concept.loc["算力", "data_quality_status"] == "available"
    assert set(membership[membership["code"].eq("000001")]["board_name"]) == {"通信", "AI", "算力"}
    assert set(snapshot["feature_date"]) == {"2026-10-16"}


def test_board_heat_includes_provider_only_concepts_and_fund_flow_in_score() -> None:
    overlays = pd.DataFrame([
        {
            "board_type": "concept",
            "board_name": "人形机器人",
            "weighted_change_pct": 1.46,
            "fund_flow_score": 90.0,
            "main_net_inflow": 72.22,
            "member_count": 473,
            "data_source": "akshare_ths_fund_flow",
        },
        {
            "board_type": "concept",
            "board_name": "弱势题材",
            "weighted_change_pct": -1.0,
            "fund_flow_score": 10.0,
            "main_net_inflow": -5.0,
            "member_count": 30,
            "data_source": "akshare_ths_fund_flow",
        },
    ])

    snapshot, _ = compute_board_heat_snapshot(
        _universe().drop(columns=["concepts"]),
        quote_snapshot=_quotes(),
        board_overlays=overlays,
        observed_at="2026-10-19T15:01:00+08:00",
        trading_date="2026-10-19",
        min_members=1,
    )
    concepts = snapshot[snapshot["board_type"].eq("concept")].set_index("board_name")
    summary = summarize_board_heat_snapshot(snapshot)

    assert concepts.loc["人形机器人", "heat_rank"] == 1
    assert concepts.loc["人形机器人", "heat_score"] == pytest.approx(64.5)
    assert concepts.loc["人形机器人", "data_quality_status"] == "provider_only"
    assert concepts.loc["人形机器人", "data_source"] == "akshare_ths_fund_flow"
    assert summary["concept"][0]["name"] == "人形机器人"
    assert summary["concept_source"] == "akshare_ths_fund_flow"


def test_board_heat_matches_provider_concept_aliases_without_duplicate_rows() -> None:
    universe = _universe().iloc[[0]].copy()
    universe["concepts"] = "出海50,DRGDIP"
    overlays = pd.DataFrame([
        {
            "board_type": "concept",
            "board_name": "同花顺出海50",
            "fund_flow_score": 90.0,
        },
        {
            "board_type": "concept",
            "board_name": "DRG/DIP",
            "fund_flow_score": 80.0,
        },
    ])

    snapshot, _ = compute_board_heat_snapshot(
        universe,
        quote_snapshot=_quotes().iloc[[0]],
        board_overlays=overlays,
        observed_at="2026-10-19T15:01:00+08:00",
        trading_date="2026-10-19",
        min_members=1,
    )
    concepts = snapshot[snapshot["board_type"].eq("concept")].set_index("board_name")

    assert set(concepts.index) == {"出海50", "DRGDIP"}
    assert concepts.loc["出海50", "fund_flow_score"] == 90.0
    assert concepts.loc["DRGDIP", "fund_flow_score"] == 80.0
    assert set(concepts["data_quality_status"]) == {"available"}


def test_board_heat_sqlite_is_idempotent_and_selects_prior_snapshot(tmp_path) -> None:
    path = tmp_path / "sector.sqlite3"
    first, membership = compute_board_heat_snapshot(
        _universe(),
        quote_snapshot=_quotes(),
        observed_at="2026-10-19T10:00:00+08:00",
        trading_date="2026-10-19",
        min_members=1,
    )
    update = upsert_board_heat_history(first, membership, path=path)
    repeated = upsert_board_heat_history(first, membership, path=path)
    reduced_membership = membership[membership["code"].ne("000002")]
    replaced = upsert_board_heat_history(first, reduced_membership, path=path)
    prior = load_latest_board_heat_snapshot(
        path,
        before="2026-10-19T11:00:00+08:00",
    )

    assert update["record_count"] == len(first)
    assert repeated["record_count"] == len(first)
    assert repeated["membership_count"] == len(membership)
    assert replaced["membership_count"] == len(reduced_membership)
    assert len(prior) == len(first)
    assert summarize_board_heat_snapshot(prior)["quote_source"] == "fixture"


def test_latest_board_heat_load_keeps_one_complete_observation(tmp_path) -> None:
    path = tmp_path / "sector.sqlite3"
    first, membership = compute_board_heat_snapshot(
        _universe(),
        quote_snapshot=_quotes(),
        observed_at="2026-10-19T10:00:00+08:00",
        trading_date="2026-10-19",
        min_members=1,
    )
    second = first[first["board_name"].ne("机器人")].copy()
    second["observed_at"] = "2026-10-19T11:00:00+08:00"
    second["heat_rank"] = 1
    upsert_board_heat_history(first, membership, path=path)
    upsert_board_heat_history(second, membership, path=path)

    latest = load_latest_board_heat_snapshot(path)

    assert set(latest["observed_at"]) == {"2026-10-19T11:00:00+08:00"}
    assert len(latest) == len(second)
    assert "机器人" not in set(latest["board_name"])


def test_qmt_adapter_uses_worker_output_and_only_returns_requested_a_shares() -> None:
    payload = {
        "requested_count": 2,
        "row_count": 2,
        "rows": [
            {"code": "600000", "qmt_code": "600000.SH", "price": 10, "previous_close": 9.5, "change_pct": 5.2632, "amount": 1000, "quote_time": "2026-10-19T14:30:00"},
            {"code": "000001", "qmt_code": "000001.SZ", "price": 11, "previous_close": 10, "change_pct": 10, "amount": 2000, "quote_time": "2026-10-19T14:30:01"},
        ],
    }
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="QMT log\nDSA_QMT_QUOTES=" + json.dumps(payload),
        stderr="",
    )
    with patch("src.services.screening.qmt_quotes.subprocess.run", return_value=completed) as run:
        frame = fetch_qmt_quotes(
            ["600000", "000001", "510300"],
            python_executable=sys.executable,
            port=58610,
        )

    request = json.loads(run.call_args.kwargs["input"])
    assert request["codes"] == ["000001.SZ", "600000.SH"]
    assert set(frame["code"]) == {"600000", "000001"}
    assert frame.attrs["snapshot_source"] == "qmt_readonly"
    assert frame.attrs["trading_date"] == "2026-10-19"


def test_qmt_adapter_rejects_rows_without_usable_prices() -> None:
    payload = {
        "requested_count": 1,
        "row_count": 1,
        "rows": [{"code": "600000", "qmt_code": "600000.SH", "price": 0}],
    }
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="DSA_QMT_QUOTES=" + json.dumps(payload),
        stderr="",
    )
    with (
        patch("src.services.screening.qmt_quotes.subprocess.run", return_value=completed),
        pytest.raises(RuntimeError, match="usable last price"),
    ):
        fetch_qmt_quotes(["600000"], python_executable=sys.executable)


def test_qmt_overlay_replaces_market_fields_and_preserves_metadata() -> None:
    snapshot = pd.DataFrame([
        {
            "code": "600000",
            "name": "浦发银行",
            "price": 9.5,
            "change_pct": -1.0,
            "amount": 100.0,
            "total_mv": 1_000.0,
        },
        {
            "code": "000001",
            "name": "平安银行",
            "price": 10.0,
            "change_pct": 0.0,
            "amount": 200.0,
            "total_mv": 2_000.0,
        },
    ])
    snapshot.attrs["snapshot_source"] = "sina"
    quotes = pd.DataFrame([
        {
            "code": "600000",
            "qmt_code": "600000.SH",
            "price": 10.5,
            "previous_close": 10.0,
            "change_pct": 5.0,
            "open": 10.1,
            "high": 10.8,
            "low": 10.0,
            "amount": 500.0,
            "volume": 50.0,
            "quote_time": "2026-10-19T15:00:00",
        },
    ])
    quotes.attrs.update({
        "trading_date": "2026-10-19",
        "quote_as_of": "2026-10-19T15:00:00",
    })

    result = overlay_qmt_quotes(
        snapshot,
        quotes,
        min_coverage_pct=50.0,
        expected_trading_date="2026-10-19",
    ).set_index("code")

    assert result.loc["600000", "price"] == 10.5
    assert result.loc["600000", "amount"] == 500.0
    assert result.loc["600000", "total_mv"] == 1_000.0
    assert result.loc["000001", "price"] == 10.0
    assert result.attrs["snapshot_source"] == "qmt_readonly"
    assert result.attrs["snapshot_reference_source"] == "sina"
    assert result.attrs["qmt_coverage_pct"] == 50.0


def test_qmt_overlay_rejects_sparse_or_wrong_date_quotes() -> None:
    snapshot = pd.DataFrame([
        {"code": "600000", "price": 10.0},
        {"code": "000001", "price": 11.0},
    ])
    quotes = pd.DataFrame([
        {"code": "600000", "price": 10.5},
    ])
    quotes.attrs["trading_date"] = "2026-10-16"

    with pytest.raises(RuntimeError, match="coverage below threshold"):
        overlay_qmt_quotes(snapshot, quotes, min_coverage_pct=90.0)

    with pytest.raises(RuntimeError, match="trading date mismatch"):
        overlay_qmt_quotes(
            snapshot,
            quotes,
            min_coverage_pct=50.0,
            expected_trading_date="2026-10-19",
        )

    quotes.attrs.pop("trading_date")
    with pytest.raises(RuntimeError, match="received <missing>"):
        overlay_qmt_quotes(
            snapshot,
            quotes,
            min_coverage_pct=50.0,
            expected_trading_date="2026-10-19",
        )


def test_qmt_sector_adapter_returns_only_requested_memberships() -> None:
    payload = {
        "requested_count": 2,
        "row_count": 2,
        "sector_count": 2,
        "membership_count": 3,
        "rows": [
            {
                "code": "600000",
                "qmt_code": "600000.SH",
                "concepts": "高股息",
                "concept_count": 1,
            },
            {
                "code": "000001",
                "qmt_code": "000001.SZ",
                "concepts": "AI,算力",
                "concept_count": 2,
            },
        ],
    }
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="DSA_QMT_QUOTES=" + json.dumps(payload),
        stderr="",
    )
    with patch(
        "src.services.screening.qmt_quotes.subprocess.run",
        return_value=completed,
    ) as run:
        frame = fetch_qmt_sector_membership(
            ["600000", "000001", "510300"],
            sector_prefixes=["TGN"],
            python_executable=sys.executable,
        )

    request = json.loads(run.call_args.kwargs["input"])
    assert request["mode"] == "sectors"
    assert request["codes"] == ["000001.SZ", "600000.SH"]
    assert request["sector_prefixes"] == ["TGN"]
    assert frame.set_index("code").loc["000001", "concepts"] == "AI,算力"
    assert frame.attrs["sector_count"] == 2
    assert frame.attrs["membership_count"] == 3


def test_screening_config_reads_qmt_and_sector_heat_settings() -> None:
    with patch.dict(os.environ, {
        "SCREENING_SECTOR_HEAT_ENABLED": "false",
        "SCREENING_QMT_QUOTES_ENABLED": "true",
        "SCREENING_QMT_PYTHON": "F:/QMT/python.exe",
        "SCREENING_QMT_XTQUANT_PATH": "F:/QMT/site-packages",
        "SCREENING_QMT_PORT": "58610",
        "SCREENING_QMT_TIMEOUT_SEC": "30",
        "SCREENING_QMT_PRIMARY_MIN_COVERAGE_PCT": "95",
        "SCREENING_QMT_SECTOR_PREFIXES": "TGN,TDGN",
        "SCREENING_QMT_INDUSTRY_SECTOR_PREFIXES": "THY",
    }, clear=False):
        config = Config.from_env()

    assert config.sector_heat_enabled is False
    assert config.qmt_quotes_enabled is True
    assert config.qmt_python.as_posix() == "F:/QMT/python.exe"
    assert config.qmt_port == 58610
    assert config.qmt_timeout_sec == 30.0
    assert config.qmt_primary_min_coverage_pct == 95.0
    assert config.qmt_sector_prefixes == ["TGN", "TDGN"]
    assert config.qmt_industry_sector_prefixes == ["THY"]
