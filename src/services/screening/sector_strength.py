# -*- coding: utf-8 -*-
"""Point-in-time sector strength contract shared by live and research flows."""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sqlite3
from typing import Protocol

import numpy as np
import pandas as pd

SECTOR_STATUS_AVAILABLE = "available"
SECTOR_STATUS_UNAVAILABLE = "unavailable"

SECTOR_FEATURE_COLUMNS = (
    "sector_strength_score",
    "sector_strength_rank",
    "sector_member_count",
    "sector_breadth_ratio",
    "sector_leader_count",
    "sector_high_adr_leader_count",
    "sector_median_change_20d_pct",
    "sector_median_change_60d_pct",
    "sector_heat_score",
    "sector_fund_flow_score",
    "sector_main_net_inflow",
    "sector_trend_state",
    "sector_leadership_priority",
    "sector_data_status",
    "sector_data_source",
    "sector_data_as_of",
)

_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


class SectorStrengthProvider(Protocol):
    """Adapter boundary for future AkShare/Baostock or vendor sector feeds."""

    def snapshot(self, *, as_of: date | datetime | str | None = None) -> pd.DataFrame:
        """Return the normalized sector-strength schema for one completed day."""


def compute_sector_strength_snapshot(
    universe: pd.DataFrame,
    *,
    as_of: date | datetime | str | None = None,
    source: str = "local_daily_industry",
    min_sector_members: int = 5,
    shrinkage_members: int = 5,
) -> pd.DataFrame:
    """Aggregate a completed cross-section into ranked sector strength rows."""
    industry_column = _industry_column(universe)
    if universe.empty or industry_column is None:
        return pd.DataFrame(columns=("industry", *SECTOR_FEATURE_COLUMNS))

    work = universe.copy()
    work["industry"] = work[industry_column].fillna("").astype(str).str.strip()
    work = work[work["industry"].ne("")].copy()
    if work.empty:
        return pd.DataFrame(columns=("industry", *SECTOR_FEATURE_COLUMNS))

    change20 = _numeric(work, "change_20d")
    change60 = _numeric(work, "change_60d")
    rs20 = _numeric(work, "relative_strength_20d_pctile")
    rs60 = _numeric(work, "relative_strength_60d_pctile")
    if not rs20.notna().any():
        rs20 = change20.rank(method="average", pct=True).mul(100.0)
    if not rs60.notna().any():
        rs60 = change60.rank(method="average", pct=True).mul(100.0)
    adr = _numeric(work, "adr_20_pct")
    heat = _numeric(work, "board_heat_score")
    fund_flow_score = _numeric(work, "board_fund_flow_score")
    fund_flow_amount = _numeric(work, "board_fund_flow_net_amount")
    trend = _boolean(work, "ema_stack_bullish") & _boolean(work, "ema21_rising")
    leaders = trend & rs60.ge(80.0)
    high_adr_leaders = leaders & rs60.ge(90.0) & adr.ge(3.5)
    work = work.assign(
        __rs20=rs20,
        __rs60=rs60,
        __change20=change20,
        __change60=change60,
        __heat=heat,
        __fund_flow_score=fund_flow_score,
        __fund_flow_amount=fund_flow_amount,
        __trend=trend.astype(float),
        __leader=leaders.astype(int),
        __high_adr_leader=high_adr_leaders.astype(int),
    )

    grouped = work.groupby("industry", sort=True)
    snapshot = grouped.agg(
        sector_member_count=("industry", "size"),
        sector_breadth_ratio=("__trend", "mean"),
        sector_leader_count=("__leader", "sum"),
        sector_high_adr_leader_count=("__high_adr_leader", "sum"),
        sector_median_rs20=("__rs20", "median"),
        sector_median_rs60=("__rs60", "median"),
        sector_median_change_20d_pct=("__change20", "median"),
        sector_median_change_60d_pct=("__change60", "median"),
        sector_heat_score=("__heat", "median"),
        sector_fund_flow_score=("__fund_flow_score", "median"),
        sector_main_net_inflow=("__fund_flow_amount", "median"),
    ).reset_index()
    snapshot["sector_breadth_ratio"] *= 100.0
    prior_members = max(int(shrinkage_members), 0)
    if prior_members:
        market_breadth = float(work["__trend"].mean() * 100.0)
        snapshot["sector_breadth_ratio"] = (
            snapshot["sector_breadth_ratio"].mul(snapshot["sector_member_count"])
            + market_breadth * prior_members
        ) / (snapshot["sector_member_count"] + prior_members)
    leader_share = (
        snapshot["sector_leader_count"]
        / snapshot["sector_member_count"].clip(lower=1)
        * 100.0
    )
    if prior_members:
        market_leader_share = float(work["__leader"].mean() * 100.0)
        leader_share = (
            leader_share.mul(snapshot["sector_member_count"])
            + market_leader_share * prior_members
        ) / (snapshot["sector_member_count"] + prior_members)
    external_heat = snapshot["sector_heat_score"].fillna(50.0).clip(0, 100)
    flow_available = snapshot["sector_fund_flow_score"].notna()
    external_heat = external_heat.where(
        ~flow_available,
        external_heat.mul(0.60)
        + snapshot["sector_fund_flow_score"].fillna(50.0).clip(0, 100).mul(0.40),
    )
    snapshot["sector_strength_score"] = (
        _cross_rank(snapshot["sector_median_rs20"]) * 0.20
        + _cross_rank(snapshot["sector_median_rs60"]) * 0.25
        + snapshot["sector_breadth_ratio"].fillna(0.0).clip(0, 100) * 0.25
        + _cross_rank(leader_share) * 0.20
        + external_heat * 0.10
    ).clip(0, 100)
    minimum_members = max(int(min_sector_members), 1)
    sample_reliability = (
        snapshot["sector_member_count"].div(minimum_members).clip(lower=0.0, upper=1.0)
    )
    snapshot["sector_strength_score"] = (
        50.0
        + snapshot["sector_strength_score"].sub(50.0).mul(sample_reliability)
    ).clip(0, 100)
    snapshot["sector_strength_rank"] = snapshot["sector_strength_score"].rank(
        method="min", ascending=False
    ).astype(int)
    sample_eligible = snapshot["sector_member_count"].ge(minimum_members)
    snapshot["sector_trend_state"] = np.select(
        [
            ~sample_eligible,
            snapshot["sector_strength_score"].ge(70)
            & snapshot["sector_high_adr_leader_count"].ge(1),
            snapshot["sector_strength_score"].ge(55),
            snapshot["sector_strength_score"].ge(40),
        ],
        ["insufficient_sample", "hot_leading", "improving", "neutral"],
        default="weak",
    )
    snapshot["sector_leadership_priority"] = (
        sample_eligible
        & snapshot["sector_trend_state"].isin(["hot_leading", "improving"])
        & snapshot["sector_high_adr_leader_count"].ge(1)
    )
    snapshot["sector_data_status"] = SECTOR_STATUS_AVAILABLE
    snapshot["sector_data_source"] = str(source or "unknown")
    snapshot["sector_data_as_of"] = _as_of_text(as_of)

    integer_columns = (
        "sector_member_count",
        "sector_leader_count",
        "sector_high_adr_leader_count",
        "sector_strength_rank",
    )
    for column in integer_columns:
        snapshot[column] = snapshot[column].fillna(0).astype(int)
    numeric_columns = (
        "sector_strength_score",
        "sector_breadth_ratio",
        "sector_median_change_20d_pct",
        "sector_median_change_60d_pct",
        "sector_heat_score",
        "sector_fund_flow_score",
        "sector_main_net_inflow",
    )
    for column in numeric_columns:
        snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce").round(4)
    return snapshot[["industry", *SECTOR_FEATURE_COLUMNS]]


def attach_sector_strength_features(
    candidates: pd.DataFrame,
    snapshot: pd.DataFrame,
    *,
    unavailable_source: str = "not_provided",
) -> pd.DataFrame:
    """Map a normalized snapshot to candidates without pretending missing data is neutral."""
    result = candidates.copy()
    industry_column = _industry_column(result)
    if result.empty:
        return _ensure_unavailable_columns(result, source=unavailable_source)
    if industry_column is None or snapshot.empty or "industry" not in snapshot.columns:
        return _ensure_unavailable_columns(result, source=unavailable_source)

    normalized = snapshot.drop_duplicates("industry", keep="last").copy()
    normalized["industry"] = normalized["industry"].fillna("").astype(str).str.strip()
    normalized = normalized.set_index("industry")
    keys = result[industry_column].fillna("").astype(str).str.strip()
    for column in SECTOR_FEATURE_COLUMNS:
        if column in normalized.columns:
            result[column] = keys.map(normalized[column])
    mapped = result.get(
        "sector_strength_score", pd.Series(np.nan, index=result.index)
    ).notna()
    result["sector_data_status"] = result.get(
        "sector_data_status", pd.Series("", index=result.index)
    ).where(mapped, SECTOR_STATUS_UNAVAILABLE)
    result["sector_data_source"] = result.get(
        "sector_data_source", pd.Series("", index=result.index)
    ).where(mapped, unavailable_source)
    result["sector_data_as_of"] = result.get(
        "sector_data_as_of", pd.Series("", index=result.index)
    ).where(mapped, "")
    return result


def load_sector_strength_history(path: str | Path) -> pd.DataFrame:
    """Load normalized point-in-time sector history from JSON or CSV."""
    resolved = Path(path)
    if not resolved.exists():
        return pd.DataFrame(columns=("date", "industry", *SECTOR_FEATURE_COLUMNS))
    if resolved.suffix.lower() in _SQLITE_SUFFIXES:
        return _load_sqlite_sector_history(resolved)
    try:
        if resolved.suffix.lower() == ".csv":
            frame = pd.read_csv(resolved)
        else:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            rows = payload.get("records", []) if isinstance(payload, dict) else payload
            frame = pd.DataFrame(rows if isinstance(rows, list) else [])
    except (OSError, ValueError, json.JSONDecodeError):
        return pd.DataFrame(columns=("date", "industry", *SECTOR_FEATURE_COLUMNS))
    if frame.empty or not {"date", "industry"}.issubset(frame.columns):
        return pd.DataFrame(columns=("date", "industry", *SECTOR_FEATURE_COLUMNS))
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["industry"] = frame["industry"].fillna("").astype(str).str.strip()
    duplicate_keys = ["date", "code"] if "code" in frame.columns else ["date", "industry"]
    if "code" in frame.columns:
        frame["code"] = frame["code"].map(_normalize_code)
    return frame.dropna(subset=["date"]).drop_duplicates(
        duplicate_keys, keep="last"
    )


def select_sector_strength_snapshot(
    history: pd.DataFrame,
    *,
    as_of: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Select the latest sector snapshot that is not newer than ``as_of``."""
    if history.empty or "date" not in history.columns:
        return pd.DataFrame(columns=("industry", *SECTOR_FEATURE_COLUMNS))

    work = history.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date"])
    requested = pd.to_datetime(as_of, errors="coerce") if as_of is not None else None
    if requested is not None and not pd.isna(requested):
        requested = requested.normalize()
        work = work[work["date"].le(requested)]
    if work.empty:
        return pd.DataFrame(columns=("industry", *SECTOR_FEATURE_COLUMNS))

    selected_date = work["date"].max()
    selected = work[work["date"].eq(selected_date)].copy()
    if "industry" not in selected.columns:
        return pd.DataFrame(columns=("industry", *SECTOR_FEATURE_COLUMNS))
    selected["industry"] = selected["industry"].fillna("").astype(str).str.strip()
    selected = selected[selected["industry"].ne("")]
    selected = selected.drop_duplicates("industry", keep="last")
    for column in SECTOR_FEATURE_COLUMNS:
        if column not in selected.columns:
            selected[column] = np.nan
    selected.attrs["selected_as_of"] = selected_date.strftime("%Y-%m-%d")
    selected.attrs["requested_as_of"] = (
        requested.strftime("%Y-%m-%d")
        if requested is not None and not pd.isna(requested)
        else ""
    )
    selected.attrs["stale"] = bool(
        requested is not None
        and not pd.isna(requested)
        and selected_date < requested
    )
    return selected[["industry", *SECTOR_FEATURE_COLUMNS]]


def summarize_sector_strength_snapshot(
    snapshot: pd.DataFrame,
    *,
    top: int = 8,
) -> dict[str, object]:
    """Build a notification-safe overview without conflating local and external heat."""
    if snapshot.empty or "industry" not in snapshot.columns:
        return {
            "status": SECTOR_STATUS_UNAVAILABLE,
            "as_of": "",
            "sector_count": 0,
            "source": "",
            "stale": False,
            "local_heat_status": SECTOR_STATUS_UNAVAILABLE,
            "external_heat_status": SECTOR_STATUS_UNAVAILABLE,
            "fund_flow_status": SECTOR_STATUS_UNAVAILABLE,
            "external_heat_coverage_count": 0,
            "fund_flow_coverage_count": 0,
            "external_heat_coverage_pct": 0.0,
            "fund_flow_coverage_pct": 0.0,
            "top_sectors": [],
        }

    work = snapshot.copy()
    local_scores = pd.to_numeric(work.get("sector_strength_score"), errors="coerce")
    external_scores = pd.to_numeric(work.get("sector_heat_score"), errors="coerce")
    flow_scores = pd.to_numeric(work.get("sector_fund_flow_score"), errors="coerce")
    flow_amounts = pd.to_numeric(work.get("sector_main_net_inflow"), errors="coerce")
    external_count = int(external_scores.notna().sum())
    flow_count = int((flow_scores.notna() | flow_amounts.notna()).sum())
    local_available = bool(local_scores.notna().any())

    work["__local_score"] = local_scores
    rank = pd.to_numeric(work.get("sector_strength_rank"), errors="coerce")
    work["__sort_rank"] = rank.fillna(10_000)
    work = work.sort_values(
        ["__sort_rank", "__local_score"],
        ascending=[True, False],
        kind="stable",
    )
    rows: list[dict[str, object]] = []
    for row in work.head(max(1, int(top or 8))).to_dict(orient="records"):
        rows.append({
            "industry": str(row.get("industry") or "").strip(),
            "rank": _optional_int(row.get("sector_strength_rank")),
            "local_heat_score": _optional_float(row.get("sector_strength_score")),
            "breadth_ratio": _optional_float(row.get("sector_breadth_ratio")),
            "high_adr_leader_count": _optional_int(
                row.get("sector_high_adr_leader_count")
            ),
            "change_20d_pct": _optional_float(
                row.get("sector_median_change_20d_pct")
            ),
            "change_60d_pct": _optional_float(
                row.get("sector_median_change_60d_pct")
            ),
            "trend_state": str(row.get("sector_trend_state") or "").strip(),
            "external_heat_score": _optional_float(row.get("sector_heat_score")),
            "fund_flow_score": _optional_float(row.get("sector_fund_flow_score")),
            "main_net_inflow": _optional_float(row.get("sector_main_net_inflow")),
        })

    as_of_values = [
        str(value).strip()
        for value in work.get("sector_data_as_of", pd.Series(dtype=object)).dropna()
        if str(value).strip()
    ]
    source_values = [
        str(value).strip()
        for value in work.get("sector_data_source", pd.Series(dtype=object)).dropna()
        if str(value).strip()
    ]
    selected_as_of = str(snapshot.attrs.get("selected_as_of") or "").strip()
    sector_count = int(work["industry"].nunique())
    return {
        "status": SECTOR_STATUS_AVAILABLE if local_available else SECTOR_STATUS_UNAVAILABLE,
        "as_of": as_of_values[0] if as_of_values else selected_as_of,
        "sector_count": sector_count,
        "source": source_values[0] if source_values else "",
        "stale": bool(snapshot.attrs.get("stale", False)),
        "local_heat_status": (
            SECTOR_STATUS_AVAILABLE if local_available else SECTOR_STATUS_UNAVAILABLE
        ),
        "external_heat_status": (
            SECTOR_STATUS_AVAILABLE if external_count else SECTOR_STATUS_UNAVAILABLE
        ),
        "fund_flow_status": (
            SECTOR_STATUS_AVAILABLE if flow_count else SECTOR_STATUS_UNAVAILABLE
        ),
        "external_heat_coverage_count": external_count,
        "fund_flow_coverage_count": flow_count,
        "external_heat_coverage_pct": round(
            external_count / sector_count * 100.0, 2
        ) if sector_count else 0.0,
        "fund_flow_coverage_pct": round(
            flow_count / sector_count * 100.0, 2
        ) if sector_count else 0.0,
        "top_sectors": rows,
    }


def upsert_sector_strength_history(
    snapshot: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    path: str | Path,
    as_of: date | datetime | str,
) -> dict[str, object]:
    """Persist exact-date stock membership plus normalized sector metrics."""
    resolved = Path(path)
    if resolved.suffix.lower() in _SQLITE_SUFFIXES:
        return _upsert_sqlite_sector_history(
            snapshot,
            membership,
            path=resolved,
            as_of=as_of,
        )
    industry_column = _industry_column(membership)
    as_of_text = _as_of_text(as_of)
    if snapshot.empty or membership.empty or industry_column is None or not as_of_text:
        return {"updated_count": 0, "record_count": 0, "path": str(resolved)}

    members = membership.copy()
    members["code"] = members["code"].map(_normalize_code)
    members["industry"] = members[industry_column].fillna("").astype(str).str.strip()
    members = members[members["code"].ne("") & members["industry"].ne("")]
    sector_rows = snapshot.drop_duplicates("industry", keep="last").copy()
    rows = members[["code", "industry"]].drop_duplicates("code", keep="last").merge(
        sector_rows,
        on="industry",
        how="inner",
        validate="many_to_one",
    )
    if rows.empty:
        return {"updated_count": 0, "record_count": 0, "path": str(resolved)}
    rows["date"] = as_of_text

    existing = load_sector_strength_history(resolved)
    if not existing.empty:
        existing = existing.copy()
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        combined = pd.concat([existing, rows], ignore_index=True, sort=False)
    else:
        combined = rows
    combined["code"] = combined["code"].map(_normalize_code)
    combined = combined.drop_duplicates(["date", "code"], keep="last")
    records = [
        {
            str(key): _json_value(value)
            for key, value in record.items()
        }
        for record in combined.to_dict(orient="records")
    ]
    payload = {
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "record_count": len(records),
        "records": records,
    }
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(resolved)
    return {
        "updated_count": len(rows),
        "record_count": len(records),
        "path": str(resolved),
    }


def attach_point_in_time_sector_history(
    panel: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact-date sector history; future observations are never carried back."""
    result = panel.copy()
    industry_column = _industry_column(result)
    if result.empty or history.empty:
        return _ensure_unavailable_columns(result, source="historical_sector_file_missing")
    membership = history.attrs.get("membership")
    if isinstance(membership, pd.DataFrame) and not membership.empty:
        return _attach_normalized_sector_history(result, history, membership)
    if "code" in result.columns and "code" in history.columns:
        left = result.copy()
        left["__sector_code"] = left["code"].map(_normalize_code)
        right = history.copy().rename(columns={"code": "__sector_code"})
        include_history_industry = industry_column is None and "industry" in right.columns
        keep = [
            "date",
            "__sector_code",
            *(["industry"] if include_history_industry else []),
            *[c for c in SECTOR_FEATURE_COLUMNS if c in right.columns],
        ]
        merged = left.merge(
            right[keep],
            on=["date", "__sector_code"],
            how="left",
            sort=False,
        ).drop(columns="__sector_code")
        return _ensure_unavailable_columns(merged, source="historical_sector_file")
    if industry_column is None:
        return _ensure_unavailable_columns(result, source="historical_sector_membership_missing")
    left = result.copy()
    left["__sector_industry"] = left[industry_column].fillna("").astype(str).str.strip()
    right = history.copy().rename(columns={"industry": "__sector_industry"})
    keep = ["date", "__sector_industry", *[c for c in SECTOR_FEATURE_COLUMNS if c in right.columns]]
    merged = left.merge(
        right[keep],
        on=["date", "__sector_industry"],
        how="left",
        sort=False,
        suffixes=("", "__sector"),
    ).drop(columns="__sector_industry")
    return _ensure_unavailable_columns(merged, source="historical_sector_file")


def _load_sqlite_sector_history(path: Path) -> pd.DataFrame:
    try:
        with sqlite3.connect(path) as connection:
            sectors = pd.read_sql_query(
                "SELECT * FROM sector_daily ORDER BY date, industry",
                connection,
            )
            membership = pd.read_sql_query(
                "SELECT code, industry, effective_from, effective_to, source "
                "FROM sector_membership ORDER BY code, effective_from",
                connection,
            )
    except (OSError, sqlite3.Error):
        return pd.DataFrame(columns=("date", "industry", *SECTOR_FEATURE_COLUMNS))
    if sectors.empty:
        sectors = pd.DataFrame(columns=("date", "industry", *SECTOR_FEATURE_COLUMNS))
    else:
        sectors["date"] = pd.to_datetime(sectors["date"], errors="coerce").dt.normalize()
        sectors = sectors.dropna(subset=["date"])
        if "sector_leadership_priority" in sectors.columns:
            sectors["sector_leadership_priority"] = (
                pd.to_numeric(sectors["sector_leadership_priority"], errors="coerce")
                .fillna(0)
                .astype(bool)
            )
    if not membership.empty:
        membership["code"] = membership["code"].map(_normalize_code)
        membership["effective_from"] = pd.to_datetime(
            membership["effective_from"], errors="coerce"
        ).dt.normalize()
        membership["effective_to"] = pd.to_datetime(
            membership["effective_to"], errors="coerce"
        ).dt.normalize()
        membership = membership.dropna(subset=["effective_from"])
    sectors.attrs["membership"] = membership
    sectors.attrs["storage"] = "sqlite_normalized"
    return sectors


def _upsert_sqlite_sector_history(
    snapshot: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    path: Path,
    as_of: date | datetime | str,
) -> dict[str, object]:
    industry_column = _industry_column(membership)
    as_of_text = _as_of_text(as_of)
    if snapshot.empty or membership.empty or industry_column is None or not as_of_text:
        return {
            "updated_count": 0,
            "record_count": 0,
            "membership_count": 0,
            "path": str(path),
        }

    members = membership.copy()
    members["code"] = members["code"].map(_normalize_code)
    members["industry"] = members[industry_column].fillna("").astype(str).str.strip()
    members = members[members["code"].ne("") & members["industry"].ne("")]
    members = members[["code", "industry"]].drop_duplicates("code", keep="last")

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        _create_sector_tables(connection)
        for row in snapshot.drop_duplicates("industry", keep="last").to_dict(orient="records"):
            values = {
                "date": as_of_text,
                "industry": str(row.get("industry") or "").strip(),
                **{column: _sqlite_value(row.get(column)) for column in SECTOR_FEATURE_COLUMNS},
            }
            if not values["industry"]:
                continue
            columns = list(values)
            placeholders = ", ".join("?" for _ in columns)
            assignments = ", ".join(
                f"{column}=excluded.{column}" for column in columns if column not in {"date", "industry"}
            )
            connection.execute(
                f"INSERT INTO sector_daily ({', '.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(date, industry) DO UPDATE SET {assignments}",
                [values[column] for column in columns],
            )

        membership_changes = 0
        for row in members.itertuples(index=False):
            current = connection.execute(
                "SELECT industry, effective_from FROM sector_membership "
                "WHERE code=? AND effective_to IS NULL ORDER BY effective_from DESC LIMIT 1",
                (row.code,),
            ).fetchone()
            if current is not None and current[0] == row.industry:
                continue
            if current is not None:
                connection.execute(
                    "UPDATE sector_membership SET effective_to=date(?, '-1 day') "
                    "WHERE code=? AND effective_from=?",
                    (as_of_text, row.code, current[1]),
                )
            connection.execute(
                "INSERT INTO sector_membership "
                "(code, industry, effective_from, effective_to, source) VALUES (?, ?, ?, NULL, ?) "
                "ON CONFLICT(code, effective_from) DO UPDATE SET "
                "industry=excluded.industry, effective_to=NULL, source=excluded.source",
                (row.code, row.industry, as_of_text, "free_composite"),
            )
            membership_changes += 1
        connection.commit()
        sector_count = int(connection.execute("SELECT COUNT(*) FROM sector_daily").fetchone()[0])
        membership_count = int(
            connection.execute("SELECT COUNT(*) FROM sector_membership").fetchone()[0]
        )
    return {
        "updated_count": len(members),
        "membership_changes": membership_changes,
        "record_count": sector_count,
        "membership_count": membership_count,
        "path": str(path),
    }


def _create_sector_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sector_daily (
            date TEXT NOT NULL,
            industry TEXT NOT NULL,
            sector_strength_score REAL,
            sector_strength_rank INTEGER,
            sector_member_count INTEGER,
            sector_breadth_ratio REAL,
            sector_leader_count INTEGER,
            sector_high_adr_leader_count INTEGER,
            sector_median_change_20d_pct REAL,
            sector_median_change_60d_pct REAL,
            sector_heat_score REAL,
            sector_fund_flow_score REAL,
            sector_main_net_inflow REAL,
            sector_trend_state TEXT,
            sector_leadership_priority INTEGER,
            sector_data_status TEXT,
            sector_data_source TEXT,
            sector_data_as_of TEXT,
            PRIMARY KEY (date, industry)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sector_membership (
            code TEXT NOT NULL,
            industry TEXT NOT NULL,
            effective_from TEXT NOT NULL,
            effective_to TEXT,
            source TEXT NOT NULL,
            PRIMARY KEY (code, effective_from)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_sector_membership_effective "
        "ON sector_membership(code, effective_from, effective_to)"
    )


def _attach_normalized_sector_history(
    panel: pd.DataFrame,
    sectors: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    if "code" not in panel.columns:
        return _ensure_unavailable_columns(
            panel,
            source="historical_sector_membership_missing",
        )
    left = panel.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    left["__sector_code"] = left["code"].map(_normalize_code)
    left["__sector_row"] = np.arange(len(left))

    right = membership.copy()
    right["__sector_code"] = right["code"].map(_normalize_code)
    right = right.rename(columns={"industry": "__membership_industry"})
    expanded = left.merge(
        right[["__sector_code", "__membership_industry", "effective_from", "effective_to"]],
        on="__sector_code",
        how="left",
        sort=False,
    )
    valid = expanded["effective_from"].notna() & expanded["date"].ge(
        expanded["effective_from"]
    )
    valid &= expanded["effective_to"].isna() | expanded["date"].le(expanded["effective_to"])
    matched = (
        expanded.loc[valid]
        .sort_values(["__sector_row", "effective_from"], kind="stable")
        .drop_duplicates("__sector_row", keep="last")
    )
    membership_by_row = matched.set_index("__sector_row")["__membership_industry"]
    if "industry" not in left.columns:
        left["industry"] = left["__sector_row"].map(membership_by_row).fillna("")
    else:
        incoming = left["__sector_row"].map(membership_by_row)
        current = left["industry"].fillna("").astype(str).str.strip()
        left["industry"] = current.where(current.ne(""), incoming)

    sector_rows = sectors.drop_duplicates(["date", "industry"], keep="last")
    keep = [
        "date",
        "industry",
        *[column for column in SECTOR_FEATURE_COLUMNS if column in sector_rows.columns],
    ]
    merged = left.merge(
        sector_rows[keep],
        on=["date", "industry"],
        how="left",
        sort=False,
        suffixes=("", "__sector"),
    )
    merged = merged.sort_values("__sector_row", kind="stable").drop(
        columns=["__sector_code", "__sector_row"],
        errors="ignore",
    )
    return _ensure_unavailable_columns(merged, source="historical_sector_sqlite")


def _sqlite_value(value: object) -> object:
    converted = _json_value(value)
    if isinstance(converted, bool):
        return int(converted)
    if isinstance(converted, (dict, list)):
        return json.dumps(converted, ensure_ascii=False)
    return converted


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, 4)


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


def _ensure_unavailable_columns(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    result = frame.copy()
    for column in SECTOR_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = "" if column.startswith("sector_data_") or column == "sector_trend_state" else np.nan
    score = pd.to_numeric(result["sector_strength_score"], errors="coerce")
    available = score.notna()
    result["sector_data_status"] = result["sector_data_status"].where(
        available, SECTOR_STATUS_UNAVAILABLE
    )
    result["sector_data_source"] = result["sector_data_source"].where(
        available, source
    )
    result["sector_data_as_of"] = result["sector_data_as_of"].where(available, "")
    return result


def _industry_column(frame: pd.DataFrame) -> str | None:
    return next(
        (column for column in ("industry", "行业", "所属行业") if column in frame.columns),
        None,
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _cross_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index, dtype=float)
    return values.rank(method="average", pct=True).mul(100.0).fillna(50.0)


def _as_of_text(value: date | datetime | str | None) -> str:
    if value is None:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _normalize_code(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def _json_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
