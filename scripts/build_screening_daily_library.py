# -*- coding: utf-8 -*-
"""Build or resume the local A-share daily-feature library.

Examples:
    python scripts/build_screening_daily_library.py
    python scripts/build_screening_daily_library.py --batch-size 250 --all
    python scripts/build_screening_daily_library.py --universe-mode full-a-share --all
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.screening.config import Config, _load_env_file  # noqa: E402
from src.services.screening.daily import enrich_daily_features  # noqa: E402
from src.services.screening.daily_library import (  # noqa: E402
    completed_snapshot_bar_date,
    daily_feature_library_date_coverage,
    latest_daily_feature_date,
    refresh_daily_feature_library_from_cache,
    select_daily_feature_library_date_missing,
    select_daily_library_batch,
    upsert_daily_feature_library,
)
from src.services.screening.filter import apply_hard_filters, without_daily_filters  # noqa: E402
from src.services.screening.normalize import is_beijing_stock_code, normalize_code  # noqa: E402
from src.services.screening.pipeline import _attach_market_regime_features  # noqa: E402
from src.services.screening.snapshot import fetch_snapshot_with_fallback  # noqa: E402
from src.services.screening.strategy import load_all_strategies  # noqa: E402

STRATEGY_NAME = "martin_momentum_pullback"


def main() -> int:
    args = _parse_args()
    _load_env_file()
    os.environ.setdefault("SCREENING_DATA_DIR", str(REPO_ROOT / "data" / "screening"))
    config = Config.from_env()
    library_path = Path(args.library_path or config.daily_feature_library_path or config.data_dir / "daily_feature_library.json")
    cache_dir = Path(args.cache_dir or config.daily_history_cache_dir or config.data_dir / "daily_history")
    source = args.source or config.daily_source
    price_adjustment = args.price_adjustment
    lookback_days = max(int(args.lookback_days or config.daily_lookback_days), 120)
    workers = max(int(args.workers or config.daily_fetch_max_workers), 1)

    strategy = load_all_strategies(config.strategies_dir)[STRATEGY_NAME]
    snapshot = fetch_snapshot_with_fallback(
        config.snapshot_source_priority,
        required_columns=["name", "price", "change_pct", "amount", "total_mv"],
        fallback_snapshot_path=config.fallback_snapshot_path,
        fallback_max_age_hours=config.snapshot_fallback_max_age_hours,
        cache_ttl_seconds=config.snapshot_cache_ttl_seconds,
        market="cn",
    )
    snapshot = _attach_market_regime_features(snapshot)
    if args.universe_mode == "full-a-share":
        candidates = snapshot.copy()
        universe_definition = "current full A-share snapshot; no strategy prefilter"
    else:
        bootstrap_filters = replace(
            without_daily_filters(strategy.screening.hard_filters),
            change_pct_min=None,
            change_pct_max=None,
            market_positive_ratio_min=None,
            market_median_change_pct_min=None,
        )
        candidates = apply_hard_filters(snapshot, bootstrap_filters).copy()
        universe_definition = (
            "current strategy-eligible snapshot: non-ST, price>=1.5, "
            "amount>=1e8, market_cap>=3e9"
        )
    candidates["code"] = candidates["code"].map(
        lambda value: normalize_code(value, allow_ticker=True)
    )
    candidates = candidates[
        candidates["code"].map(_is_a_share_code)
    ].drop_duplicates("code", keep="last")

    completed_trade_date = (
        completed_snapshot_bar_date(snapshot, market="cn")
        if args.append_snapshot_bar
        else None
    )
    can_append_snapshot = bool(
        completed_trade_date
        and not snapshot.attrs.get("fallback_used")
        and source.strip().lower() in {"auto", "sina", "tencent"}
    )
    library_latest_date = latest_daily_feature_date(library_path)
    date_coverage = (
        daily_feature_library_date_coverage(
            candidates,
            library_path=library_path,
            as_of=completed_trade_date,
        )
        if completed_trade_date is not None
        else None
    )
    cache_refresh_candidates = (
        select_daily_feature_library_date_missing(
            candidates,
            library_path=library_path,
            as_of=completed_trade_date,
        )
        if completed_trade_date is not None
        else candidates
    )
    should_refresh_cache = library_latest_date is None or (
        completed_trade_date is not None
        and (
            library_latest_date < completed_trade_date
            or bool(date_coverage and date_coverage["missing_count"])
        )
    )
    if should_refresh_cache:
        cache_refresh = refresh_daily_feature_library_from_cache(
            cache_refresh_candidates,
            cache_dir=cache_dir,
            library_path=library_path,
            source=source,
            lookback_days=lookback_days,
            price_adjustment=price_adjustment,
            append_snapshot_bar=can_append_snapshot,
            snapshot_bar_date=completed_trade_date,
        )
    else:
        cache_refresh = {
            "cache_count": 0,
            "updated_count": 0,
            "entry_count": 0,
            "appended_count": 0,
            "error_count": 0,
            "errors": [],
            "skipped": "library_already_current",
        }

    attempted: set[str] = set()
    batches: list[dict[str, object]] = []
    batch_index = 0
    while True:
        remaining = candidates[~candidates["code"].isin(attempted)].copy()
        batch = select_daily_library_batch(
            remaining,
            library_path=library_path,
            limit=max(int(args.batch_size), 1),
            period=f"{date.today().isoformat()}:{batch_index}",
            max_age_days=max(int(args.max_age_days), 0),
        )
        if batch.empty:
            break
        batch_index += 1
        attempted.update(batch["code"].astype(str).tolist())
        enriched = enrich_daily_features(
            batch,
            max_rows=len(batch),
            lookback_days=lookback_days,
            source=source,
            price_adjustment=price_adjustment,
            fetch_retries=config.daily_fetch_retries,
            cache_dir=cache_dir,
            cache_ttl_seconds=config.daily_history_cache_ttl_hours * 3600,
            max_workers=workers,
        )
        update = upsert_daily_feature_library(enriched, library_path=library_path)
        errors = [str(item) for item in enriched.attrs.get("daily_errors", [])]
        batches.append({
            "batch": batch_index,
            "attempted": len(batch),
            "succeeded": int(enriched.attrs.get("daily_success_count", 0)),
            "updated": int(update.get("updated_count", 0)),
            "library_entries": int(update.get("entry_count", 0)),
            "error_count": len(errors),
            "error_samples": errors[:3],
        })
        _print_status(batches[-1])
        if not args.all and batch_index >= max(int(args.max_batches), 1):
            break

    summary = {
        "snapshot_source": snapshot.attrs.get("snapshot_source", ""),
        "snapshot_rows": len(snapshot),
        "eligible_rows": len(candidates),
        "universe_mode": args.universe_mode,
        "universe_definition": universe_definition,
        "library_path": str(library_path),
        "cache_dir": str(cache_dir),
        "source": source,
        "price_adjustment": price_adjustment,
        "lookback_days": lookback_days,
        "cache_refresh": cache_refresh,
        "date_coverage_before_refresh": date_coverage,
        "cache_refresh_candidate_count": len(cache_refresh_candidates),
        "completed_trade_date": completed_trade_date.isoformat() if completed_trade_date else "",
        "snapshot_bar_append_enabled": can_append_snapshot,
        "batch_count": len(batches),
        "attempted_count": len(attempted),
        "batches": batches,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the resumable A-share daily feature library")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Continue until every eligible stock has been attempted")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--source", choices=["auto", "sina", "tencent", "akshare", "baostock", "tushare"])
    parser.add_argument(
        "--price-adjustment",
        choices=["qfq", "hfq", "none"],
        default="qfq",
        help="Require one consistent price-adjustment convention; research defaults to qfq",
    )
    parser.add_argument("--lookback-days", type=int)
    parser.add_argument("--max-age-days", type=int, default=7)
    parser.add_argument("--library-path")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--universe-mode",
        choices=["strategy-eligible", "full-a-share"],
        default="strategy-eligible",
        help=(
            "Build the normal strategy-prefiltered library or a separate full "
            "current A-share universe for research backtests"
        ),
    )
    parser.add_argument(
        "--no-snapshot-append",
        dest="append_snapshot_bar",
        action="store_false",
        help="Do not update compatible caches from a completed live snapshot",
    )
    parser.set_defaults(append_snapshot_bar=True)
    return parser.parse_args()


def _is_a_share_code(value: object) -> bool:
    """Keep Shanghai and Shenzhen common-share code families used by the strategy."""
    code = str(value or "").strip()
    if len(code) != 6 or not code.isdigit():
        return False
    return code.startswith(("0", "3", "6")) and not is_beijing_stock_code(code)


def _print_status(status: dict[str, object]) -> None:
    print(
        "batch={batch} attempted={attempted} succeeded={succeeded} "
        "updated={updated} library_entries={library_entries} errors={error_count}".format(**status),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
