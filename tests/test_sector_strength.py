# -*- coding: utf-8 -*-

import pandas as pd

from src.services.screening.sector_strength import (
    SECTOR_STATUS_AVAILABLE,
    SECTOR_STATUS_UNAVAILABLE,
    attach_point_in_time_sector_history,
    attach_sector_strength_features,
    compute_sector_strength_snapshot,
    load_sector_strength_history,
    select_sector_strength_snapshot,
    summarize_sector_strength_snapshot,
    upsert_sector_strength_history,
)


def _universe() -> pd.DataFrame:
    return pd.DataFrame([
        {"code": "000001", "industry": "光通信", "relative_strength_20d_pctile": 98, "relative_strength_60d_pctile": 99, "change_20d": 25, "change_60d": 80, "adr_20_pct": 5.0, "ema_stack_bullish": True, "ema21_rising": True, "board_heat_score": 90},
        {"code": "000002", "industry": "光通信", "relative_strength_20d_pctile": 90, "relative_strength_60d_pctile": 94, "change_20d": 18, "change_60d": 55, "adr_20_pct": 4.5, "ema_stack_bullish": True, "ema21_rising": True, "board_heat_score": 85},
        {"code": "000003", "industry": "银行", "relative_strength_20d_pctile": 40, "relative_strength_60d_pctile": 35, "change_20d": 1, "change_60d": 3, "adr_20_pct": 1.5, "ema_stack_bullish": False, "ema21_rising": False, "board_heat_score": 35},
        {"code": "000004", "industry": "银行", "relative_strength_20d_pctile": 45, "relative_strength_60d_pctile": 40, "change_20d": 2, "change_60d": 4, "adr_20_pct": 1.8, "ema_stack_bullish": False, "ema21_rising": True, "board_heat_score": 40},
    ])


def test_sector_snapshot_ranks_hot_high_adr_leader_group_first() -> None:
    snapshot = compute_sector_strength_snapshot(
        _universe(),
        as_of="2026-10-16",
        min_sector_members=2,
    ).set_index("industry")

    assert snapshot.loc["光通信", "sector_strength_rank"] == 1
    assert snapshot.loc["光通信", "sector_high_adr_leader_count"] == 2
    assert snapshot.loc["光通信", "sector_trend_state"] == "hot_leading"
    assert bool(snapshot.loc["光通信", "sector_leadership_priority"]) is True
    assert snapshot.loc["光通信", "sector_data_status"] == SECTOR_STATUS_AVAILABLE


def test_sector_overview_keeps_local_heat_when_external_layers_are_missing() -> None:
    universe = _universe().drop(columns="board_heat_score")
    snapshot = compute_sector_strength_snapshot(
        universe,
        as_of="2026-10-16",
        min_sector_members=2,
    )

    overview = summarize_sector_strength_snapshot(snapshot, top=2)

    assert overview["status"] == SECTOR_STATUS_AVAILABLE
    assert overview["local_heat_status"] == SECTOR_STATUS_AVAILABLE
    assert overview["external_heat_status"] == SECTOR_STATUS_UNAVAILABLE
    assert overview["fund_flow_status"] == SECTOR_STATUS_UNAVAILABLE
    assert overview["as_of"] == "2026-10-16"
    assert overview["top_sectors"][0]["industry"] == "光通信"


def test_sector_snapshot_selection_never_uses_a_future_date() -> None:
    first = compute_sector_strength_snapshot(_universe(), as_of="2026-10-15")
    first["date"] = "2026-10-15"
    second = compute_sector_strength_snapshot(_universe(), as_of="2026-10-17")
    second["date"] = "2026-10-17"

    selected = select_sector_strength_snapshot(
        pd.concat([first, second], ignore_index=True),
        as_of="2026-10-16",
    )

    assert selected.attrs["selected_as_of"] == "2026-10-15"
    assert selected.attrs["stale"] is True
    assert set(selected["sector_data_as_of"]) == {"2026-10-15"}


def test_sector_snapshot_derives_relative_strength_when_percentiles_are_missing() -> None:
    universe = _universe().drop(
        columns=["relative_strength_20d_pctile", "relative_strength_60d_pctile"]
    )

    snapshot = compute_sector_strength_snapshot(universe, as_of="2026-10-16")

    assert not snapshot.empty
    assert snapshot["sector_strength_score"].notna().all()
    assert snapshot.set_index("industry").loc["光通信", "sector_strength_rank"] == 1


def test_sector_attachment_marks_missing_membership_unavailable() -> None:
    snapshot = compute_sector_strength_snapshot(_universe(), as_of="2026-10-16")
    candidates = pd.DataFrame([
        {"code": "000001", "industry": "光通信"},
        {"code": "000005", "industry": "未知行业"},
    ])

    attached = attach_sector_strength_features(candidates, snapshot).set_index("code")

    assert attached.loc["000001", "sector_data_status"] == SECTOR_STATUS_AVAILABLE
    assert attached.loc["000005", "sector_data_status"] == SECTOR_STATUS_UNAVAILABLE
    assert pd.isna(attached.loc["000005", "sector_strength_score"])


def test_historical_sector_join_requires_exact_date_and_never_backfills_future() -> None:
    panel = pd.DataFrame([
        {"date": pd.Timestamp("2026-10-15"), "code": "000001", "close": 10.0},
        {"date": pd.Timestamp("2026-10-16"), "code": "000001", "close": 10.5},
    ])
    history = pd.DataFrame([{
        "date": pd.Timestamp("2026-10-16"),
        "code": "000001",
        "industry": "光通信",
        "sector_strength_score": 88.0,
        "sector_data_status": SECTOR_STATUS_AVAILABLE,
        "sector_data_source": "fixture",
        "sector_data_as_of": "2026-10-16",
    }])

    attached = attach_point_in_time_sector_history(panel, history)

    assert attached.loc[0, "sector_data_status"] == SECTOR_STATUS_UNAVAILABLE
    assert attached.loc[1, "sector_strength_score"] == 88.0


def test_historical_sector_join_preserves_existing_industry_column() -> None:
    panel = pd.DataFrame([{
        "date": pd.Timestamp("2026-10-16"),
        "code": "000001",
        "industry": "光通信",
        "close": 10.5,
    }])
    history = pd.DataFrame([{
        "date": pd.Timestamp("2026-10-16"),
        "code": "000001",
        "industry": "通信设备",
        "sector_strength_score": 88.0,
        "sector_data_status": SECTOR_STATUS_AVAILABLE,
    }])

    attached = attach_point_in_time_sector_history(panel, history)

    assert "industry" in attached.columns
    assert "industry_x" not in attached.columns
    assert "industry_y" not in attached.columns
    assert attached.loc[0, "industry"] == "光通信"
    assert attached.loc[0, "sector_strength_score"] == 88.0


def test_invalid_sector_history_degrades_to_empty_frame(tmp_path) -> None:
    path = tmp_path / "sector-history.json"
    path.write_text("{invalid", encoding="utf-8")

    loaded = load_sector_strength_history(path)

    assert loaded.empty
    assert {"date", "industry", "sector_strength_score"}.issubset(loaded.columns)


def test_sector_history_persists_exact_date_stock_membership(tmp_path) -> None:
    universe = _universe()
    snapshot = compute_sector_strength_snapshot(universe, as_of="2026-10-16")
    path = tmp_path / "sector_strength_history.json"

    update = upsert_sector_strength_history(
        snapshot,
        universe,
        path=path,
        as_of="2026-10-16",
    )
    loaded = load_sector_strength_history(path)

    assert update["updated_count"] == 4
    assert set(loaded["code"]) == {"000001", "000002", "000003", "000004"}
    assert set(loaded["date"].dt.strftime("%Y-%m-%d")) == {"2026-10-16"}


def test_sqlite_sector_history_normalizes_membership_and_joins_exact_date(tmp_path) -> None:
    universe = _universe()
    snapshot = compute_sector_strength_snapshot(universe, as_of="2026-10-16")
    path = tmp_path / "sector_strength.sqlite3"

    first = upsert_sector_strength_history(
        snapshot,
        universe,
        path=path,
        as_of="2026-10-16",
    )
    repeated = upsert_sector_strength_history(
        snapshot,
        universe,
        path=path,
        as_of="2026-10-16",
    )
    loaded = load_sector_strength_history(path)
    panel = pd.DataFrame([
        {"date": pd.Timestamp("2026-10-15"), "code": "000001", "industry": "", "close": 10.0},
        {"date": pd.Timestamp("2026-10-16"), "code": "000001", "industry": "", "close": 10.5},
    ])
    attached = attach_point_in_time_sector_history(panel, loaded)

    assert first["membership_changes"] == 4
    assert repeated["membership_changes"] == 0
    assert first["record_count"] == 2
    assert len(loaded.attrs["membership"]) == 4
    assert attached.loc[0, "sector_data_status"] == SECTOR_STATUS_UNAVAILABLE
    assert attached.loc[1, "industry"] == "光通信"
    expected = snapshot.set_index("industry").loc["光通信", "sector_strength_score"]
    assert attached.loc[1, "sector_strength_score"] == expected
