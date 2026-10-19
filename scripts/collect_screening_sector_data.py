# -*- coding: utf-8 -*-
"""Build one completed-day sector-strength observation from the local A-share library."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.screening.config import Config  # noqa: E402
from src.services.screening.daily_library import load_daily_feature_library  # noqa: E402
from src.services.screening.industry import enrich_industry_concepts  # noqa: E402
from src.services.screening.sector_strength import (  # noqa: E402
    compute_sector_strength_snapshot,
    upsert_sector_strength_history,
)


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
) -> dict[str, Any]:
    """Persist one exact-date sector snapshot using only completed local bars."""
    library = load_daily_feature_library(library_path, max_age_days=36_500)
    if library.empty:
        raise RuntimeError(f"Daily feature library is empty: {library_path}")

    dates = pd.to_datetime(library.get("daily_latest_date"), errors="coerce").dt.normalize()
    requested = pd.to_datetime(as_of, errors="coerce") if as_of is not None else dates.max()
    if pd.isna(requested):
        raise RuntimeError("No valid completed date is available in the daily feature library")
    completed_date = pd.Timestamp(requested).normalize()
    universe = library.loc[dates.eq(completed_date)].copy()
    total_entries = int(library.attrs.get("diagnostics", {}).get("entry_count", len(library)))
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
        include_fund_flow=include_fund_flow,
    )
    industry = enriched.get("industry", pd.Series("", index=enriched.index)).fillna("")
    industry_coverage = int(industry.astype(str).str.strip().ne("").sum())
    if industry_coverage == 0:
        raise RuntimeError("No industry membership was available from files or free providers")

    source = "local_daily+" + str(provider or "files")
    snapshot = compute_sector_strength_snapshot(
        enriched,
        as_of=completed_date,
        source=source,
    )
    if snapshot.empty:
        raise RuntimeError("Sector-strength calculation produced no rows")
    update = upsert_sector_strength_history(
        snapshot,
        enriched,
        path=history_path,
        as_of=completed_date,
    )
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
