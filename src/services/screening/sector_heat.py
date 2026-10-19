# -*- coding: utf-8 -*-
"""Point-in-time industry and concept heat snapshots for A-share research."""

from __future__ import annotations

from contextlib import closing
from datetime import date, datetime
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

BOARD_HEAT_COLUMNS = (
    "observed_at",
    "trading_date",
    "feature_date",
    "board_type",
    "board_name",
    "heat_score",
    "heat_rank",
    "heat_delta",
    "rank_delta",
    "lifecycle",
    "member_count",
    "quote_coverage_count",
    "quote_coverage_pct",
    "weighted_change_pct",
    "median_change_pct",
    "advance_ratio",
    "decline_ratio",
    "limit_up_count",
    "limit_down_count",
    "leader_count",
    "trend_breadth_ratio",
    "total_amount",
    "amount_share_pct",
    "activity_ratio_median",
    "turnover_rate_median",
    "volume_ratio_median",
    "fund_flow_score",
    "main_net_inflow",
    "data_quality_status",
    "data_source",
    "quote_source",
    "quote_stale",
)


def compute_board_heat_snapshot(
    universe: pd.DataFrame,
    *,
    quote_snapshot: pd.DataFrame | None = None,
    board_overlays: pd.DataFrame | None = None,
    previous_snapshot: pd.DataFrame | None = None,
    observed_at: date | datetime | str | None = None,
    trading_date: date | datetime | str | None = None,
    feature_date: date | datetime | str | None = None,
    source: str = "local_daily",
    min_members: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate stock-level membership and quotes into objective board heat."""
    empty_snapshot = pd.DataFrame(columns=BOARD_HEAT_COLUMNS)
    empty_membership = pd.DataFrame(
        columns=("trading_date", "board_type", "board_name", "code", "source")
    )
    if universe.empty or "code" not in universe.columns:
        return empty_snapshot, empty_membership

    observed_text = _observed_at_text(observed_at)
    trading_text = _date_text(trading_date or observed_text)
    feature_text = _date_text(feature_date or trading_text)
    work = universe.copy()
    work["code"] = work["code"].map(_normalize_code)
    work = work[work["code"].ne("")].drop_duplicates("code", keep="last")
    work = _attach_quotes(work, quote_snapshot)
    members = _build_membership(work, trading_date=trading_text, source=source)
    if members.empty:
        return empty_snapshot, empty_membership

    feature_columns = [
        "code", "__price", "__change_pct", "__amount", "__turnover_rate",
        "__volume_ratio", "__market_cap", "avg_amount_20d", "ema_stack_bullish",
        "ema21_rising", "relative_strength_60d_pctile",
    ]
    for column in feature_columns:
        if column not in work.columns:
            work[column] = pd.NA
    expanded = members.merge(
        work[feature_columns],
        on="code",
        how="left",
        validate="many_to_one",
    )
    market_amount = pd.to_numeric(work["__amount"], errors="coerce").clip(lower=0).sum(
        min_count=1
    )
    rows = [
        _aggregate_board(group, market_amount=market_amount)
        for _, group in expanded.groupby(["board_type", "board_name"], sort=True)
    ]
    snapshot = pd.DataFrame(rows)
    if snapshot.empty:
        return empty_snapshot, members

    snapshot = _attach_board_overlays(snapshot, board_overlays)
    snapshot = _score_board_heat(snapshot, min_members=max(int(min_members), 1))
    snapshot["observed_at"] = observed_text
    snapshot["trading_date"] = trading_text
    snapshot["feature_date"] = feature_text
    if "data_source" not in snapshot.columns:
        snapshot["data_source"] = str(source or "unknown")
    else:
        snapshot["data_source"] = snapshot["data_source"].fillna(
            str(source or "unknown")
        )
    quote_attrs = getattr(quote_snapshot, "attrs", {}) if quote_snapshot is not None else {}
    snapshot["quote_source"] = str(quote_attrs.get("snapshot_source") or "unavailable")
    snapshot["quote_stale"] = bool(quote_attrs.get("stale", False))
    snapshot = _attach_lifecycle(snapshot, previous_snapshot)
    for column in BOARD_HEAT_COLUMNS:
        if column not in snapshot.columns:
            snapshot[column] = pd.NA
    return snapshot[list(BOARD_HEAT_COLUMNS)], members


def upsert_board_heat_history(
    snapshot: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    path: str | Path,
) -> dict[str, object]:
    """Persist exact observations and daily board membership in SQLite."""
    resolved = Path(path)
    if snapshot.empty:
        return {
            "updated_count": 0,
            "record_count": 0,
            "membership_count": 0,
            "path": str(resolved),
        }
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(resolved)) as connection:
        _create_tables(connection)
        columns = list(BOARD_HEAT_COLUMNS)
        placeholders = ", ".join("?" for _ in columns)
        assignments = ", ".join(
            f"{column}=excluded.{column}"
            for column in columns
            if column not in {"observed_at", "board_type", "board_name"}
        )
        for row in snapshot.to_dict(orient="records"):
            connection.execute(
                f"INSERT INTO board_heat_snapshot ({', '.join(columns)}) "
                f"VALUES ({placeholders}) ON CONFLICT(observed_at, board_type, board_name) "
                f"DO UPDATE SET {assignments}",
                [_sqlite_value(row.get(column)) for column in columns],
            )

        if not membership.empty:
            partitions = sorted({
                (str(row.trading_date), str(row.board_type))
                for row in membership.itertuples(index=False)
            })
            connection.executemany(
                "DELETE FROM board_membership_daily "
                "WHERE trading_date=? AND board_type=?",
                partitions,
            )
            connection.executemany(
                "INSERT INTO board_membership_daily "
                "(trading_date, board_type, board_name, code, source) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(trading_date, board_type, board_name, code) "
                "DO UPDATE SET source=excluded.source",
                [
                    (
                        str(row.trading_date),
                        str(row.board_type),
                        str(row.board_name),
                        str(row.code),
                        str(row.source),
                    )
                    for row in membership.itertuples(index=False)
                ],
            )
        connection.commit()
        record_count = int(
            connection.execute("SELECT COUNT(*) FROM board_heat_snapshot").fetchone()[0]
        )
        membership_count = int(
            connection.execute("SELECT COUNT(*) FROM board_membership_daily").fetchone()[0]
        )
    return {
        "updated_count": len(snapshot),
        "record_count": record_count,
        "membership_count": membership_count,
        "path": str(resolved),
    }


def load_latest_board_heat_snapshot(
    path: str | Path,
    *,
    before: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Load one complete latest observation, optionally before a cutoff."""
    resolved = Path(path)
    if not resolved.is_file():
        return pd.DataFrame(columns=BOARD_HEAT_COLUMNS)
    where = "WHERE observed_at < ?" if before is not None else ""
    parameters = (_observed_at_text(before),) if before is not None else ()
    query = f"""
        SELECT *
        FROM board_heat_snapshot
        WHERE observed_at = (
            SELECT MAX(observed_at)
            FROM board_heat_snapshot
            {where}
        )
        ORDER BY board_type, heat_rank, board_name
    """
    try:
        with closing(sqlite3.connect(resolved)) as connection:
            frame = pd.read_sql_query(query, connection, params=parameters)
    except (OSError, sqlite3.Error):
        return pd.DataFrame(columns=BOARD_HEAT_COLUMNS)
    return frame


def summarize_board_heat_snapshot(
    snapshot: pd.DataFrame,
    *,
    top: int = 10,
) -> dict[str, object]:
    """Return a stable compact payload for logs, reports, and later model context."""
    if snapshot.empty:
        return {
            "status": "unavailable",
            "observed_at": "",
            "trading_date": "",
            "feature_date": "",
            "quote_source": "",
            "industry": [],
            "concept": [],
        }
    result: dict[str, object] = {
        "status": "available",
        "observed_at": str(snapshot.iloc[0].get("observed_at") or ""),
        "trading_date": str(snapshot.iloc[0].get("trading_date") or ""),
        "feature_date": str(snapshot.iloc[0].get("feature_date") or ""),
        "quote_source": str(snapshot.iloc[0].get("quote_source") or ""),
        "industry_source": _board_type_sources(snapshot, "industry"),
        "concept_source": _board_type_sources(snapshot, "concept"),
    }
    for board_type in ("industry", "concept"):
        selected = snapshot[snapshot["board_type"].eq(board_type)].sort_values(
            ["heat_rank", "board_name"], kind="stable"
        ).head(max(1, int(top)))
        result[board_type] = [
            {
                "name": str(row.get("board_name") or ""),
                "rank": _optional_int(row.get("heat_rank")),
                "score": _optional_float(row.get("heat_score")),
                "change_pct": _optional_float(row.get("weighted_change_pct")),
                "advance_ratio": _optional_float(row.get("advance_ratio")),
                "amount_share_pct": _optional_float(row.get("amount_share_pct")),
                "fund_flow_score": _optional_float(row.get("fund_flow_score")),
                "main_net_inflow": _optional_float(row.get("main_net_inflow")),
                "member_count": _optional_int(row.get("member_count")),
                "source": str(row.get("data_source") or ""),
                "lifecycle": str(row.get("lifecycle") or ""),
                "quality": str(row.get("data_quality_status") or ""),
            }
            for row in selected.to_dict(orient="records")
        ]
    return result


def _attach_quotes(universe: pd.DataFrame, quote_snapshot: pd.DataFrame | None) -> pd.DataFrame:
    work = universe.copy()
    quote_fields = ("price", "change_pct", "amount", "turnover_rate", "volume_ratio")
    quotes = quote_snapshot.copy() if isinstance(quote_snapshot, pd.DataFrame) else pd.DataFrame()
    if not quotes.empty and "code" in quotes.columns:
        quotes["code"] = quotes["code"].map(_normalize_code)
        keep = ["code", *[column for column in quote_fields if column in quotes.columns]]
        quotes = quotes[keep].drop_duplicates("code", keep="last")
        quotes = quotes.rename(columns={column: f"__quote_{column}" for column in keep[1:]})
        work = work.merge(quotes, on="code", how="left", validate="one_to_one")
    for column in quote_fields:
        quote_column = f"__quote_{column}"
        quote_values = (
            pd.to_numeric(work[quote_column], errors="coerce")
            if quote_column in work.columns
            else pd.Series(np.nan, index=work.index)
        )
        existing = (
            pd.to_numeric(work[column], errors="coerce")
            if column in work.columns
            else pd.Series(np.nan, index=work.index)
        )
        work[f"__{column}"] = quote_values.where(quote_values.notna(), existing)
    cap = pd.to_numeric(work.get("circ_mv"), errors="coerce")
    if not isinstance(cap, pd.Series):
        cap = pd.Series(np.nan, index=work.index)
    total_cap = pd.to_numeric(work.get("total_mv"), errors="coerce")
    if not isinstance(total_cap, pd.Series):
        total_cap = pd.Series(np.nan, index=work.index)
    work["__market_cap"] = cap.where(cap.gt(0), total_cap)
    return work


def _build_membership(
    universe: pd.DataFrame,
    *,
    trading_date: str,
    source: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if "industry" in universe.columns:
        industry = universe[["code", "industry"]].copy()
        industry["board_name"] = industry["industry"].fillna("").astype(str).str.strip()
        industry = industry[industry["board_name"].ne("")]
        industry["board_type"] = "industry"
        frames.append(industry[["code", "board_type", "board_name"]])
    if "concepts" in universe.columns:
        concepts = universe[["code", "concepts"]].copy()
        concepts["board_name"] = concepts["concepts"].map(_split_labels)
        concepts = concepts.explode("board_name")
        concepts["board_name"] = concepts["board_name"].fillna("").astype(str).str.strip()
        concepts = concepts[concepts["board_name"].ne("")]
        concepts["board_type"] = "concept"
        frames.append(concepts[["code", "board_type", "board_name"]])
    if not frames:
        return pd.DataFrame(
            columns=("trading_date", "board_type", "board_name", "code", "source")
        )
    members = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["board_type", "board_name", "code"], keep="last"
    )
    members["trading_date"] = trading_date
    members["source"] = str(source or "unknown")
    return members[["trading_date", "board_type", "board_name", "code", "source"]]


def _aggregate_board(group: pd.DataFrame, *, market_amount: float) -> dict[str, object]:
    change = pd.to_numeric(group["__change_pct"], errors="coerce")
    price = pd.to_numeric(group["__price"], errors="coerce")
    amount = pd.to_numeric(group["__amount"], errors="coerce").clip(lower=0)
    cap = pd.to_numeric(group["__market_cap"], errors="coerce").clip(lower=0)
    turnover = pd.to_numeric(group["__turnover_rate"], errors="coerce")
    volume_ratio = pd.to_numeric(group["__volume_ratio"], errors="coerce")
    avg_amount = pd.to_numeric(group["avg_amount_20d"], errors="coerce")
    activity_ratio = amount.div(avg_amount.where(avg_amount.gt(0)))
    trend = (
        group["ema_stack_bullish"].fillna(False).astype(bool)
        & group["ema21_rising"].fillna(False).astype(bool)
    )
    rs60 = pd.to_numeric(group["relative_strength_60d_pctile"], errors="coerce")
    leader = change.ge(5.0) | (trend & rs60.ge(90.0))
    valid_change = change.notna()
    weights = cap.where(valid_change & cap.gt(0))
    weighted_change = (
        float(np.average(change[weights.notna()], weights=weights.dropna()))
        if weights.notna().any()
        else _optional_float(change.median())
    )
    valid_count = int(valid_change.sum())
    member_count = int(group["code"].nunique())
    total_amount = amount.sum(min_count=1)
    return {
        "board_type": str(group.iloc[0]["board_type"]),
        "board_name": str(group.iloc[0]["board_name"]),
        "member_count": member_count,
        "quote_coverage_count": int(price.gt(0).sum()),
        "quote_coverage_pct": round(float(price.gt(0).mean() * 100.0), 4),
        "weighted_change_pct": _optional_float(weighted_change),
        "median_change_pct": _optional_float(change.median()),
        "advance_ratio": _optional_float(change.gt(0).sum() / valid_count * 100.0) if valid_count else None,
        "decline_ratio": _optional_float(change.lt(0).sum() / valid_count * 100.0) if valid_count else None,
        "limit_up_count": int(change.ge(9.5).sum()),
        "limit_down_count": int(change.le(-9.5).sum()),
        "leader_count": int(leader.sum()),
        "trend_breadth_ratio": round(float(trend.mean() * 100.0), 4),
        "total_amount": _optional_float(total_amount),
        "amount_share_pct": (
            _optional_float(float(total_amount) / float(market_amount) * 100.0)
            if pd.notna(total_amount) and pd.notna(market_amount) and market_amount > 0
            else None
        ),
        "activity_ratio_median": _optional_float(activity_ratio.median()),
        "turnover_rate_median": _optional_float(turnover.median()),
        "volume_ratio_median": _optional_float(volume_ratio.median()),
    }


def _attach_board_overlays(
    snapshot: pd.DataFrame,
    overlays: pd.DataFrame | None,
) -> pd.DataFrame:
    result = snapshot.copy()
    result["fund_flow_score"] = np.nan
    result["main_net_inflow"] = np.nan
    result["data_source"] = pd.NA
    result["__provider_only"] = False
    if not isinstance(overlays, pd.DataFrame) or overlays.empty:
        return result
    required = {"board_type", "board_name"}
    if not required.issubset(overlays.columns):
        return result
    result["__board_key"] = [
        _board_match_key(board_type, board_name)
        for board_type, board_name in zip(
            result["board_type"], result["board_name"]
        )
    ]
    available = overlays.copy()
    available["__board_key"] = [
        _board_match_key(board_type, board_name)
        for board_type, board_name in zip(
            available["board_type"], available["board_name"]
        )
    ]
    available = available.drop_duplicates(
        ["board_type", "__board_key"], keep="last"
    )
    value_columns = [
        column
        for column in (
            "fund_flow_score",
            "main_net_inflow",
            "weighted_change_pct",
            "member_count",
            "data_source",
        )
        if column in available.columns
    ]
    incoming = available[
        ["board_type", "board_name", "__board_key", *value_columns]
    ].rename(
        columns={
            "board_name": "__overlay_board_name",
            **{
                column: f"__overlay_{column}"
                for column in value_columns
            },
        }
    )
    merged = result.merge(
        incoming,
        on=["board_type", "__board_key"],
        how="outer",
        validate="many_to_one",
        indicator="__overlay_merge",
    )
    merged["board_name"] = merged["board_name"].where(
        merged["board_name"].notna(), merged["__overlay_board_name"]
    )
    # Baostock and THS use different industry taxonomies.  Do not mix the THS
    # industry-only rows into the Baostock industry ranking; external-only
    # concept rows are intentional because they restore the missing theme tier.
    merged = merged[
        ~(
            merged["__overlay_merge"].eq("right_only")
            & ~merged["board_type"].eq("concept")
        )
    ].copy()
    merged["__provider_only"] = merged["__overlay_merge"].eq("right_only")
    for column in value_columns:
        overlay_column = f"__overlay_{column}"
        if column not in merged.columns:
            merged[column] = pd.NA
        merged[column] = merged[column].where(
            merged[column].notna(), merged[overlay_column]
        )
        merged = merged.drop(columns=[overlay_column])
    return merged.drop(
        columns=["__overlay_merge", "__overlay_board_name", "__board_key"]
    )


def _board_match_key(board_type: object, board_name: object) -> str:
    text = str(board_name or "").strip()
    if str(board_type).strip().lower() == "concept":
        text = re.sub(r"^同花顺", "", text)
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).casefold()


def _score_board_heat(snapshot: pd.DataFrame, *, min_members: int) -> pd.DataFrame:
    result = snapshot.copy()
    numeric_inputs = (
        "weighted_change_pct",
        "advance_ratio",
        "activity_ratio_median",
        "leader_count",
        "member_count",
        "trend_breadth_ratio",
        "amount_share_pct",
        "fund_flow_score",
        "quote_coverage_pct",
    )
    for column in numeric_inputs:
        if column not in result.columns:
            result[column] = np.nan
    if "__provider_only" not in result.columns:
        result["__provider_only"] = False
    scored: list[pd.DataFrame] = []
    for _, board_rows in result.groupby("board_type", sort=False):
        part = board_rows.copy()
        components = {
            "price": (_cross_rank(part["weighted_change_pct"]), 0.25),
            "breadth": (pd.to_numeric(part["advance_ratio"], errors="coerce"), 0.20),
            "activity": (_cross_rank(part["activity_ratio_median"]), 0.15),
            "attention": (_cross_rank(part["amount_share_pct"]), 0.15),
            "leaders": (
                _cross_rank(
                    pd.to_numeric(part["leader_count"], errors="coerce")
                    .div(pd.to_numeric(part["member_count"], errors="coerce").clip(lower=1))
                ),
                0.10,
            ),
            "trend": (pd.to_numeric(part["trend_breadth_ratio"], errors="coerce"), 0.10),
            "fund_flow": (pd.to_numeric(part["fund_flow_score"], errors="coerce"), 0.05),
        }
        score = pd.Series(0.0, index=part.index)
        has_signal = pd.Series(False, index=part.index)
        for values, weight in components.values():
            available = values.notna()
            has_signal |= available
            # Missing dimensions are neutral, rather than redistributing their
            # weights to the few available fields. This keeps sparse provider-
            # only boards comparable with boards backed by full membership.
            score = score.add(values.fillna(50.0).mul(weight), fill_value=0.0)
        score = score.where(has_signal)
        reliability = pd.to_numeric(part["member_count"], errors="coerce").div(
            min_members
        ).clip(0.0, 1.0)
        part["heat_score"] = (50.0 + score.sub(50.0).mul(reliability)).clip(0, 100).round(4)
        part["heat_rank"] = part["heat_score"].rank(method="min", ascending=False).astype("Int64")
        coverage = pd.to_numeric(part["quote_coverage_pct"], errors="coerce")
        provider_only = part["__provider_only"].fillna(False).astype(bool)
        part["data_quality_status"] = np.select(
            [provider_only, coverage.ge(70), coverage.ge(30), coverage.eq(0) & part["heat_score"].notna()],
            ["provider_only", "available", "partial", "trend_only"],
            default="sparse",
        )
        scored.append(part)
    return pd.concat(scored, ignore_index=True) if scored else result


def _board_type_sources(snapshot: pd.DataFrame, board_type: str) -> str:
    if "data_source" not in snapshot.columns:
        return ""
    sources = sorted({
        str(value).strip()
        for value in snapshot.loc[
            snapshot["board_type"].eq(board_type), "data_source"
        ].dropna()
        if str(value).strip()
    })
    return ", ".join(sources)


def _attach_lifecycle(
    snapshot: pd.DataFrame,
    previous_snapshot: pd.DataFrame | None,
) -> pd.DataFrame:
    result = snapshot.copy()
    previous = previous_snapshot if isinstance(previous_snapshot, pd.DataFrame) else pd.DataFrame()
    if previous.empty or not {"board_type", "board_name"}.issubset(previous.columns):
        result["heat_delta"] = np.nan
        result["rank_delta"] = np.nan
    else:
        prior = previous[[
            "board_type", "board_name", "heat_score", "heat_rank",
        ]].drop_duplicates(["board_type", "board_name"], keep="last")
        prior = prior.rename(columns={"heat_score": "__previous_score", "heat_rank": "__previous_rank"})
        result = result.merge(prior, on=["board_type", "board_name"], how="left")
        result["heat_delta"] = (
            pd.to_numeric(result["heat_score"], errors="coerce")
            - pd.to_numeric(result["__previous_score"], errors="coerce")
        ).round(4)
        result["rank_delta"] = (
            pd.to_numeric(result["__previous_rank"], errors="coerce")
            - pd.to_numeric(result["heat_rank"], errors="coerce")
        ).round(4)
        result = result.drop(columns=["__previous_score", "__previous_rank"])
    score = pd.to_numeric(result["heat_score"], errors="coerce")
    delta = pd.to_numeric(result["heat_delta"], errors="coerce")
    has_prior = delta.notna()
    result["lifecycle"] = np.select(
        [
            has_prior & score.ge(70) & delta.ge(3),
            has_prior & score.ge(65) & delta.gt(-3),
            has_prior & score.ge(55) & delta.gt(0),
            has_prior & delta.le(-5),
            ~has_prior & score.ge(65),
        ],
        ["accelerating", "leading", "warming", "cooling", "emerging"],
        default="observing",
    )
    return result


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS board_heat_snapshot (
            observed_at TEXT NOT NULL,
            trading_date TEXT NOT NULL,
            feature_date TEXT NOT NULL,
            board_type TEXT NOT NULL,
            board_name TEXT NOT NULL,
            heat_score REAL,
            heat_rank INTEGER,
            heat_delta REAL,
            rank_delta REAL,
            lifecycle TEXT,
            member_count INTEGER,
            quote_coverage_count INTEGER,
            quote_coverage_pct REAL,
            weighted_change_pct REAL,
            median_change_pct REAL,
            advance_ratio REAL,
            decline_ratio REAL,
            limit_up_count INTEGER,
            limit_down_count INTEGER,
            leader_count INTEGER,
            trend_breadth_ratio REAL,
            total_amount REAL,
            amount_share_pct REAL,
            activity_ratio_median REAL,
            turnover_rate_median REAL,
            volume_ratio_median REAL,
            fund_flow_score REAL,
            main_net_inflow REAL,
            data_quality_status TEXT,
            data_source TEXT,
            quote_source TEXT,
            quote_stale INTEGER,
            PRIMARY KEY (observed_at, board_type, board_name)
        )
        """
    )
    existing_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(board_heat_snapshot)")
    }
    if "feature_date" not in existing_columns:
        connection.execute("ALTER TABLE board_heat_snapshot ADD COLUMN feature_date TEXT")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS board_membership_daily (
            trading_date TEXT NOT NULL,
            board_type TEXT NOT NULL,
            board_name TEXT NOT NULL,
            code TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (trading_date, board_type, board_name, code)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_board_heat_lookup "
        "ON board_heat_snapshot(board_type, board_name, observed_at)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_board_membership_lookup "
        "ON board_membership_daily(code, trading_date, board_type)"
    )


def _cross_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return values.where(values.isna(), 50.0)
    return values.rank(method="average", pct=True).mul(100.0)


def _split_labels(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return []
    seen: set[str] = set()
    labels: list[str] = []
    for item in re.split(r"[,，、;；|]", text):
        label = item.strip()
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _normalize_code(value: object) -> str:
    text = str(value or "").strip().upper()
    if "." in text:
        text = text.split(".", 1)[0]
    return text.zfill(6) if text.isdigit() else ""


def _observed_at_text(value: date | datetime | str | None) -> str:
    parsed = pd.Timestamp.now(tz="Asia/Shanghai") if value is None else pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid board heat observation time: {value}")
    return parsed.isoformat(timespec="seconds")


def _date_text(value: date | datetime | str | None) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid board heat trading date: {value}")
    return parsed.strftime("%Y-%m-%d")


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if np.isfinite(number) else None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


def _sqlite_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value
