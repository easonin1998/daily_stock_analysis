# -*- coding: utf-8 -*-
"""Build one completed-day sector-strength observation from the local A-share library."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.screening.config import Config  # noqa: E402
from src.services.screening.daily_library import load_daily_feature_library  # noqa: E402
from src.services.screening.filter import exclude_beijing_stock_exchange_rows  # noqa: E402
from src.services.screening.industry import (  # noqa: E402
    attach_qmt_industry_overlays,
    enrich_industry_concepts,
    fetch_akshare_board_fund_flow_snapshot,
)
from src.services.screening.qmt_quotes import (  # noqa: E402
    fetch_qmt_quotes,
    fetch_qmt_sector_membership,
)
from src.services.screening.sector_heat import (  # noqa: E402
    compute_board_heat_snapshot,
    load_latest_board_heat_snapshot,
    summarize_board_heat_snapshot,
    upsert_board_heat_history,
)
from src.services.screening.sector_strength import (  # noqa: E402
    compute_sector_strength_snapshot,
    upsert_sector_strength_history,
)
from src.services.screening.snapshot import fetch_snapshot_with_fallback  # noqa: E402


def collect_sector_data(
    *,
    library_path: str | Path,
    history_path: str | Path,
    provider: str,
    max_boards: int,
    provider_cache_dir: str | Path | None,
    provider_cache_ttl_hours: float,
    map_files: list[str | Path] | None = None,
    as_of: str | date | None = None,
    min_coverage_pct: float = 80.0,
    include_fund_flow: bool = False,
    collect_market_heat: bool = False,
    quote_snapshot: pd.DataFrame | None = None,
    quote_sources: list[str] | None = None,
    fallback_snapshot_path: str | Path | None = None,
    snapshot_cache_ttl_seconds: float = 0.0,
    qmt_enabled: bool = False,
    qmt_python: str | Path | None = None,
    qmt_xtquant_path: str | Path | None = None,
    qmt_port: int | None = None,
    qmt_timeout_sec: float = 45.0,
    qmt_sector_prefixes: list[str] | None = None,
    qmt_industry_sector_prefixes: list[str] | None = None,
    observed_at: str | datetime | None = None,
) -> dict[str, Any]:
    """Persist one exact-date sector snapshot using only completed local bars."""
    library = exclude_beijing_stock_exchange_rows(
        load_daily_feature_library(library_path, max_age_days=36_500)
    )
    if library.empty:
        raise RuntimeError(f"Daily feature library is empty: {library_path}")

    dates = pd.to_datetime(library.get("daily_latest_date"), errors="coerce").dt.normalize()
    requested = pd.to_datetime(as_of, errors="coerce") if as_of is not None else dates.max()
    if pd.isna(requested):
        raise RuntimeError("No valid completed date is available in the daily feature library")
    completed_date = pd.Timestamp(requested).normalize()
    universe = library.loc[dates.eq(completed_date)].copy()
    total_entries = len(library)
    coverage_pct = len(universe) / max(total_entries, 1) * 100.0
    if coverage_pct < float(min_coverage_pct):
        raise RuntimeError(
            "Sector snapshot coverage is below the configured floor: "
            f"{len(universe)}/{total_entries} ({coverage_pct:.2f}% < {min_coverage_pct:.2f}%)"
        )

    enriched, notes = enrich_industry_concepts(
        universe,
        map_files=list(map_files or []),
        provider=provider,
        max_boards=max_boards,
        provider_cache_dir=provider_cache_dir,
        provider_cache_ttl_hours=provider_cache_ttl_hours,
        include_fund_flow=include_fund_flow and not collect_market_heat,
    )
    qmt_membership_source = ""
    qmt_industry_membership = pd.DataFrame()
    qmt_industry_membership_source = ""
    if collect_market_heat and qmt_enabled and qmt_sector_prefixes:
        try:
            qmt_membership = fetch_qmt_sector_membership(
                enriched["code"],
                sector_prefixes=qmt_sector_prefixes,
                python_executable=qmt_python,
                xtquant_path=qmt_xtquant_path,
                port=qmt_port,
                timeout_sec=qmt_timeout_sec,
            )
            enriched = _attach_qmt_concepts(enriched, qmt_membership)
            qmt_membership_source = str(
                qmt_membership.attrs.get("membership_source") or "qmt_sector"
            )
            notes.append(
                "qmt concept memberships loaded: "
                f"stocks={len(qmt_membership)}, "
                f"boards={qmt_membership.attrs.get('sector_count', 0)}, "
                f"memberships={qmt_membership.attrs.get('membership_count', 0)}"
            )
        except Exception as exc:  # noqa: BLE001 - optional membership overlay.
            notes.append(f"qmt concept memberships failed: {exc}")
    if collect_market_heat and qmt_enabled and qmt_industry_sector_prefixes:
        try:
            qmt_industry_membership = fetch_qmt_sector_membership(
                enriched["code"],
                sector_prefixes=qmt_industry_sector_prefixes,
                python_executable=qmt_python,
                xtquant_path=qmt_xtquant_path,
                port=qmt_port,
                timeout_sec=qmt_timeout_sec,
            )
            qmt_industry_membership_source = str(
                qmt_industry_membership.attrs.get("membership_source")
                or "qmt_sector:THY"
            )
            notes.append(
                "qmt industry memberships loaded: "
                f"stocks={len(qmt_industry_membership)}, "
                f"boards={qmt_industry_membership.attrs.get('sector_count', 0)}, "
                f"memberships={qmt_industry_membership.attrs.get('membership_count', 0)}"
            )
        except Exception as exc:  # noqa: BLE001 - optional industry bridge.
            notes.append(f"qmt industry memberships failed: {exc}")
    industry = enriched.get("industry", pd.Series("", index=enriched.index)).fillna("")
    industry_coverage = int(industry.astype(str).str.strip().ne("").sum())
    if industry_coverage == 0:
        raise RuntimeError("No industry membership was available from files or free providers")

    board_overlays = pd.DataFrame()
    industry_bridge = {
        "matched_board_count": 0,
        "matched_stock_count": 0,
        "external_heat_stock_count": 0,
        "fund_flow_stock_count": 0,
    }
    if collect_market_heat:
        board_overlays, flow_notes = fetch_akshare_board_fund_flow_snapshot()
        notes.extend(flow_notes)
        if not qmt_industry_membership.empty:
            enriched, industry_bridge = attach_qmt_industry_overlays(
                enriched,
                qmt_industry_membership,
                board_overlays,
                include_fund_flow=include_fund_flow,
            )
            notes.append(
                "qmt THY industry bridge applied: "
                f"boards={industry_bridge['matched_board_count']}, "
                f"stocks={industry_bridge['matched_stock_count']}, "
                f"external_heat={industry_bridge['external_heat_stock_count']}, "
                f"fund_flow={industry_bridge['fund_flow_stock_count']}"
            )
        elif include_fund_flow:
            enriched = _attach_industry_fund_flow(enriched, board_overlays)

    base_source = "local_daily+" + str(provider or "files")
    board_source = base_source
    if qmt_membership_source:
        board_source += "+" + qmt_membership_source
    sector_source = base_source
    if qmt_industry_membership_source:
        sector_source += "+" + qmt_industry_membership_source
    if industry_bridge["external_heat_stock_count"]:
        sector_source += "+akshare_ths_industry"
    snapshot = compute_sector_strength_snapshot(
        enriched,
        as_of=completed_date,
        source=sector_source,
    )
    if snapshot.empty:
        raise RuntimeError("Sector-strength calculation produced no rows")
    update = upsert_sector_strength_history(
        snapshot,
        enriched,
        path=history_path,
        as_of=completed_date,
    )
    board_heat_result: dict[str, Any] = {
        "status": "disabled",
        "quote_notes": [],
        "storage": {},
    }
    if collect_market_heat:
        quotes, quote_notes = _resolve_quote_snapshot(
            enriched["code"],
            provided=quote_snapshot,
            sources=list(quote_sources or []),
            fallback_snapshot_path=fallback_snapshot_path,
            cache_ttl_seconds=snapshot_cache_ttl_seconds,
            qmt_enabled=qmt_enabled,
            qmt_python=qmt_python,
            qmt_xtquant_path=qmt_xtquant_path,
            qmt_port=qmt_port,
            qmt_timeout_sec=qmt_timeout_sec,
        )
        observation_time = observed_at or datetime.now().astimezone()
        observation_date = pd.to_datetime(observation_time, errors="coerce").date()
        quote_trading_date = str(quotes.attrs.get("trading_date") or "").strip()
        heat_trading_date: object = (
            quote_trading_date
            if quote_trading_date
            else (
                completed_date
                if quotes.empty or bool(quotes.attrs.get("stale", False))
                else observation_date
            )
        )
        previous = load_latest_board_heat_snapshot(history_path, before=observation_time)
        heat_snapshot, board_membership = compute_board_heat_snapshot(
            enriched,
            quote_snapshot=quotes,
            board_overlays=board_overlays,
            previous_snapshot=previous,
            observed_at=observation_time,
            trading_date=heat_trading_date,
            feature_date=completed_date,
            source=board_source,
        )
        heat_storage = upsert_board_heat_history(
            heat_snapshot,
            board_membership,
            path=history_path,
        )
        board_heat_result = {
            **summarize_board_heat_snapshot(heat_snapshot),
            "quote_notes": quote_notes,
            "storage": heat_storage,
        }
    leaders = snapshot.sort_values(
        ["sector_strength_score", "industry"],
        ascending=[False, True],
        kind="stable",
    ).head(10)
    return {
        "status": "ok",
        "as_of": completed_date.strftime("%Y-%m-%d"),
        "library_path": str(Path(library_path)),
        "history_path": str(Path(history_path)),
        "library_entries": total_entries,
        "completed_date_entries": len(universe),
        "coverage_pct": round(coverage_pct, 4),
        "industry_members": industry_coverage,
        "industry_coverage_pct": round(industry_coverage / max(len(universe), 1) * 100.0, 4),
        "sector_count": len(snapshot),
        "fund_flow_requested": bool(include_fund_flow),
        "provider_notes": notes,
        "storage": update,
        "board_heat": board_heat_result,
        "leaders": [
            {
                key: (None if pd.isna(value) else value)
                for key, value in row.items()
            }
            for row in leaders[
                [
                    "industry",
                    "sector_strength_score",
                    "sector_breadth_ratio",
                    "sector_high_adr_leader_count",
                    "sector_heat_score",
                    "sector_fund_flow_score",
                ]
            ].to_dict(orient="records")
        ],
    }


def main() -> int:
    runtime = Config.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-path", default=str(runtime.daily_feature_library_path or ""))
    parser.add_argument("--history-path", default=str(runtime.sector_strength_history_path or ""))
    parser.add_argument("--provider", default="free")
    parser.add_argument("--max-boards", type=int, default=runtime.industry_provider_max_boards)
    parser.add_argument("--as-of", default="")
    parser.add_argument("--min-coverage-pct", type=float, default=80.0)
    parser.add_argument(
        "--with-fund-flow",
        action="store_true",
        default=runtime.sector_fund_flow_enabled,
    )
    parser.add_argument(
        "--without-market-heat",
        action="store_true",
        help="Skip industry/concept point-in-time heat collection.",
    )
    args = parser.parse_args()

    result = collect_sector_data(
        library_path=args.library_path,
        history_path=args.history_path,
        provider=args.provider,
        max_boards=max(args.max_boards, 1),
        provider_cache_dir=runtime.industry_provider_cache_dir,
        provider_cache_ttl_hours=runtime.industry_provider_cache_ttl_hours,
        map_files=runtime.industry_map_files,
        as_of=args.as_of or None,
        min_coverage_pct=args.min_coverage_pct,
        include_fund_flow=args.with_fund_flow,
        collect_market_heat=runtime.sector_heat_enabled and not args.without_market_heat,
        quote_sources=runtime.snapshot_source_priority,
        fallback_snapshot_path=runtime.fallback_snapshot_path,
        snapshot_cache_ttl_seconds=runtime.snapshot_cache_ttl_seconds,
        qmt_enabled=runtime.qmt_quotes_enabled,
        qmt_python=runtime.qmt_python,
        qmt_xtquant_path=runtime.qmt_xtquant_path,
        qmt_port=runtime.qmt_port,
        qmt_timeout_sec=runtime.qmt_timeout_sec,
        qmt_sector_prefixes=runtime.qmt_sector_prefixes,
        qmt_industry_sector_prefixes=runtime.qmt_industry_sector_prefixes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return 0


def _resolve_quote_snapshot(
    codes: pd.Series,
    *,
    provided: pd.DataFrame | None,
    sources: list[str],
    fallback_snapshot_path: str | Path | None,
    cache_ttl_seconds: float,
    qmt_enabled: bool,
    qmt_python: str | Path | None,
    qmt_xtquant_path: str | Path | None,
    qmt_port: int | None,
    qmt_timeout_sec: float,
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    if isinstance(provided, pd.DataFrame):
        frame = provided.copy()
        frame.attrs.update(provided.attrs)
        frame.attrs.setdefault("snapshot_source", "provided")
        return frame, notes
    if qmt_enabled:
        try:
            return fetch_qmt_quotes(
                codes,
                python_executable=qmt_python,
                xtquant_path=qmt_xtquant_path,
                port=qmt_port,
                timeout_sec=qmt_timeout_sec,
            ), notes
        except Exception as exc:  # noqa: BLE001 - explicit fallback contract.
            notes.append(f"qmt_readonly failed: {exc}")
    if sources:
        try:
            frame = fetch_snapshot_with_fallback(
                sources,
                required_columns=["code", "price", "change_pct", "amount"],
                fallback_snapshot_path=fallback_snapshot_path,
                cache_ttl_seconds=max(float(cache_ttl_seconds), 0.0),
                market="cn",
            )
            return frame, notes
        except Exception as exc:  # noqa: BLE001 - trend-only heat remains valid.
            notes.append(f"snapshot fallback failed: {exc}")
    empty = pd.DataFrame()
    empty.attrs["snapshot_source"] = "unavailable"
    empty.attrs["stale"] = False
    return empty, notes


def _attach_industry_fund_flow(
    enriched: pd.DataFrame,
    overlays: pd.DataFrame,
) -> pd.DataFrame:
    if enriched.empty or overlays.empty or "industry" not in enriched.columns:
        return enriched
    industry = overlays[overlays["board_type"].eq("industry")].drop_duplicates(
        "board_name", keep="last"
    )
    if industry.empty:
        return enriched
    score_by_name = industry.set_index("board_name")["fund_flow_score"]
    amount_by_name = industry.set_index("board_name")["main_net_inflow"]
    result = enriched.copy()
    keys = result["industry"].fillna("").astype(str).str.strip()
    result["board_fund_flow_score"] = keys.map(score_by_name)
    result["board_fund_flow_net_amount"] = keys.map(amount_by_name)
    return result


def _attach_qmt_concepts(
    enriched: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    if enriched.empty or membership.empty or "code" not in enriched.columns:
        return enriched
    if not {"code", "concepts"}.issubset(membership.columns):
        return enriched
    concept_by_code = membership.drop_duplicates("code", keep="last").set_index(
        "code"
    )["concepts"]
    result = enriched.copy()
    current = result.get("concepts", pd.Series("", index=result.index)).fillna("")
    incoming = result["code"].map(concept_by_code).fillna("")
    result["concepts"] = [
        _merge_concept_text(left, right)
        for left, right in zip(current, incoming)
    ]
    return result


def _merge_concept_text(*values: object) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in re.split(r"[,，、;；|]", str(value or "")):
            label = item.strip()
            if label and label.lower() not in {"nan", "none", "<na>"} and label not in seen:
                seen.add(label)
                labels.append(label)
    return ",".join(labels)


if __name__ == "__main__":
    raise SystemExit(main())
