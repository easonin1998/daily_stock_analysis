# -*- coding: utf-8 -*-
"""Run the Martin A-share daily portfolio backtest from local history caches."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.stock_index_loader import get_stock_name_index_map  # noqa: E402
from src.services.screening.config import Config, _load_env_file  # noqa: E402
from src.services.screening.strategy import load_all_strategies  # noqa: E402
from src.services.screening.strategy_backtest import (  # noqa: E402
    MartinBacktestConfig,
    build_equal_weight_benchmark,
    generate_martin_signals,
    load_local_backtest_panel,
    render_backtest_markdown,
    run_martin_portfolio_backtest,
)

STRATEGY_NAME = "martin_momentum_pullback"


def main() -> int:
    args = _parse_args()
    _load_env_file()
    runtime = Config.from_env()
    # Strategy-research assets intentionally live under data/screening even
    # when the main DSA database uses the broader data directory.
    data_dir = Path(args.data_dir or REPO_ROOT / "data" / "screening")
    cache_dir = Path(
        args.cache_dir
        or data_dir / "daily_history"
    )
    library_path = Path(
        args.library_path
        or data_dir / "daily_feature_library.json"
    )
    output_dir = Path(args.output_dir or data_dir / "backtests")
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    if start_date > end_date:
        raise ValueError("--start-date cannot be after --end-date")
    estimated_period_sessions = len(pd.bdate_range(start_date, end_date))
    minimum_lookback_days = 120 + estimated_period_sessions + 15
    lookback_days = max(
        int(args.lookback_days or runtime.daily_lookback_days),
        minimum_lookback_days,
    )

    strategies = load_all_strategies(runtime.strategies_dir)
    strategy = strategies.get(STRATEGY_NAME)
    if strategy is None:
        raise RuntimeError(f"Strategy is unavailable: {STRATEGY_NAME}")

    names = get_stock_name_index_map()
    panel, data_diagnostics = load_local_backtest_panel(
        cache_dir=cache_dir,
        library_path=library_path,
        source=args.source or runtime.daily_source,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        names=names,
        price_adjustment=args.price_adjustment,
    )
    signal_book, signal_diagnostics = generate_martin_signals(
        panel,
        strategy.screening,
        current_static_prefiltered=args.universe_mode == "strategy-eligible",
    )

    common = {
        "start_date": start_date,
        "end_date": end_date,
        "initial_cash": args.initial_cash,
        "max_positions": args.max_positions,
        "risk_per_trade_pct": args.risk_per_trade_pct,
        "max_portfolio_heat_pct": args.max_portfolio_heat_pct,
        "max_position_weight_pct": args.max_position_weight_pct,
        "max_technical_stop_pct": args.max_technical_stop_pct,
        "min_position_weight_pct": args.min_position_weight_pct,
        "adr_stress_fraction": args.adr_stress_fraction,
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "sell_stamp_duty_bps": args.sell_stamp_duty_bps,
        "trailing_drawdown_pct": args.trailing_drawdown_pct,
        "follow_through_days": args.follow_through_days,
        "follow_through_pct": args.follow_through_pct,
        "max_holding_days": args.max_holding_days,
        "reentry_cooldown_days": args.reentry_cooldown_days,
        "entry_gap_min_pct": args.entry_gap_min_pct,
        "entry_gap_max_pct": args.entry_gap_max_pct,
        "entry_gap_down_volatility_fraction": args.entry_gap_down_volatility_fraction,
        "entry_gap_up_volatility_fraction": args.entry_gap_up_volatility_fraction,
        "entry_gap_down_min_pct": args.entry_gap_down_min_pct,
        "entry_gap_down_max_pct": args.entry_gap_down_max_pct,
        "entry_gap_up_min_pct": args.entry_gap_up_min_pct,
        "entry_gap_up_max_pct": args.entry_gap_up_max_pct,
        "entry_gap_fallback_min_pct": args.entry_gap_fallback_min_pct,
        "entry_gap_fallback_max_pct": args.entry_gap_fallback_max_pct,
    }
    result = run_martin_portfolio_backtest(
        panel,
        signal_book=signal_book,
        config=MartinBacktestConfig(**common),
    )
    results = {"strategy": result}
    benchmark = build_equal_weight_benchmark(panel)
    result_metrics = result.get("metrics", {})
    benchmark_return = benchmark.get("metrics", {}).get("total_return_pct")
    strategy_return = result_metrics.get("total_return_pct")
    if benchmark_return is not None and strategy_return is not None:
        result_metrics["excess_return_pct"] = round(
            float(strategy_return) - float(benchmark_return),
            4,
        )

    source_validation = _validate_trade_samples(
        panel,
        results,
        start_date=start_date,
        end_date=end_date,
        limit=max(int(args.validate_samples), 0),
    )

    universe_limitation = (
        "Current-eligible universe introduces survivorship and current-liquidity bias."
        if args.universe_mode == "strategy-eligible"
        else (
            "Universe covers the current full A-share snapshot, but stocks delisted before "
            "the snapshot remain absent and introduce residual survivorship bias."
        )
    )
    limitations = [
        universe_limitation,
        (
            "Historical ST status and market capitalization are not replayed point in time; "
            "the full-market mode excludes names currently marked ST on every historical date."
            if args.universe_mode == "full-a-share"
            else "Historical ST status and market capitalization are not replayed point in time."
        ),
        "Historical sector membership and sector heat are unavailable; sector score is neutral and quota is disabled.",
        "Daily bars cannot reproduce order queues or intraday execution probability.",
    ]
    if float(data_diagnostics.get("start_history_coverage_pct") or 0.0) < 90.0:
        limitations.append(
            "Fewer than 90% of symbols available at the period start have 120 warm-up bars; early-period signals are incomplete."
        )

    payload = {
        "schema_version": 5,
        "strategy": {
            "name": strategy.name,
            "display_name": strategy.display_name,
            "version": strategy.version,
        },
        "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "universe_mode": args.universe_mode,
        "data_diagnostics": data_diagnostics,
        "signal_diagnostics": signal_diagnostics,
        "source_validation": source_validation,
        "benchmark": benchmark,
        "results": results,
        "limitations": limitations,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    universe_suffix = "_full_a_share" if args.universe_mode == "full-a-share" else ""
    stem = (
        f"martin_backtest_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
        f"{universe_suffix}"
    )
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_backtest_markdown(payload), encoding="utf-8")
    pd.DataFrame(result.get("trades", [])).to_csv(
        output_dir / f"{stem}_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(result.get("equity_curve", [])).to_csv(
        output_dir / f"{stem}_equity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "report": str(markdown_path),
        "json": str(json_path),
        "data": data_diagnostics,
        "signals": {
            key: signal_diagnostics.get(key)
            for key in (
                "setup_count",
                "entry_signal_count",
                "unconfirmed_setup_count",
                "first_signal_date",
            )
        },
        "source_validation": source_validation.get("summary", {}),
        "benchmark": benchmark.get("metrics", {}),
        "strategy": result.get("metrics", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the Martin momentum-pullback strategy on cached A-share daily bars"
    )
    parser.add_argument("--start-date", default="2026-08-01")
    parser.add_argument("--end-date", default="2026-10-16")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.5)
    parser.add_argument("--max-portfolio-heat-pct", type=float, default=2.5)
    parser.add_argument("--max-position-weight-pct", type=float, default=20.0)
    parser.add_argument("--max-technical-stop-pct", type=float, default=3.0)
    parser.add_argument("--min-position-weight-pct", type=float, default=12.0)
    parser.add_argument(
        "--adr-stress-fraction",
        "--gap-risk-adr-fraction",
        dest="adr_stress_fraction",
        type=float,
        default=1.0,
        help=(
            "Report this fraction of ADR as a gap-stress scenario without using it "
            "to reduce position size; the old option name remains an alias"
        ),
    )
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--sell-stamp-duty-bps", type=float, default=5.0)
    parser.add_argument("--trailing-drawdown-pct", type=float, default=8.0)
    parser.add_argument("--follow-through-days", type=int, default=10)
    parser.add_argument("--follow-through-pct", type=float, default=3.0)
    parser.add_argument("--max-holding-days", type=int, default=20)
    parser.add_argument("--reentry-cooldown-days", type=int, default=3)
    parser.add_argument(
        "--entry-gap-min-pct",
        type=float,
        help="Exact lower-gap override; omit to use the volatility-adaptive rule",
    )
    parser.add_argument(
        "--entry-gap-max-pct",
        type=float,
        help="Exact upper-gap override; omit to use the volatility-adaptive rule",
    )
    parser.add_argument(
        "--entry-gap-down-volatility-fraction",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--entry-gap-up-volatility-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument("--entry-gap-down-min-pct", type=float, default=1.0)
    parser.add_argument("--entry-gap-down-max-pct", type=float, default=1.5)
    parser.add_argument("--entry-gap-up-min-pct", type=float, default=1.5)
    parser.add_argument("--entry-gap-up-max-pct", type=float, default=2.0)
    parser.add_argument("--entry-gap-fallback-min-pct", type=float, default=-1.25)
    parser.add_argument("--entry-gap-fallback-max-pct", type=float, default=1.75)
    parser.add_argument("--lookback-days", type=int)
    parser.add_argument("--source", default="auto")
    parser.add_argument(
        "--price-adjustment",
        choices=["qfq"],
        default="qfq",
        help="Require homogeneous forward-adjusted A-share history",
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--library-path")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--universe-mode",
        choices=["strategy-eligible", "full-a-share"],
        default="strategy-eligible",
        help="Describe the library universe and apply the matching historical ST policy",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=0,
        help="Validate the largest absolute-PnL trades online with AkShare and Baostock",
    )
    return parser.parse_args()


def _validate_trade_samples(
    panel: pd.DataFrame,
    results: dict[str, dict],
    *,
    start_date: date,
    end_date: date,
    limit: int,
) -> dict[str, object]:
    if limit <= 0:
        return {
            "enabled": False,
            "summary": {"requested": 0, "validated": 0, "ok": 0, "warning": 0, "failed": 0},
            "samples": [],
        }

    from data_provider.akshare_fetcher import AkshareFetcher
    from data_provider.baostock_fetcher import BaostockFetcher

    ranked: list[dict] = []
    for variant, result in results.items():
        for trade in result.get("trades", []):
            ranked.append({**trade, "strategy_variant": variant})
    ranked.sort(key=lambda trade: abs(float(trade.get("pnl", 0.0))), reverse=True)
    selected: list[dict] = []
    seen: set[str] = set()
    for trade in ranked:
        code = str(trade.get("code") or "")
        if not code or code in seen:
            continue
        selected.append(trade)
        seen.add(code)
        if len(selected) >= limit:
            break

    primary = AkshareFetcher()
    validator = BaostockFetcher()
    samples: list[dict[str, object]] = []
    for trade in selected:
        code = str(trade["code"])
        record: dict[str, object] = {
            "code": code,
            "name": trade.get("name", ""),
            "strategy_variant": trade.get("strategy_variant", ""),
            "absolute_pnl": round(abs(float(trade.get("pnl", 0.0))), 4),
            "primary": "AkshareFetcher",
            "validator": "BaostockFetcher",
        }
        try:
            primary_frame = primary.get_daily_data(
                code,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                days=160,
            )
            validator_frame = validator.get_daily_data(
                code,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                days=160,
            )
            primary_check = _compare_close_series(primary_frame, validator_frame)
            local_frame = panel.loc[panel["code"].astype(str).eq(code), ["date", "close"]]
            local_check = _compare_close_series(local_frame, validator_frame)
            warnings: list[str] = []
            if primary_check["overlap_rows"] == 0:
                warnings.append("primary_no_overlap")
            if local_check["overlap_rows"] == 0:
                warnings.append("local_no_overlap")
            if float(primary_check.get("max_normalized_close_delta_pct") or 0.0) > 0.5:
                warnings.append("primary_price_path_mismatch")
            if float(local_check.get("max_normalized_close_delta_pct") or 0.0) > 0.5:
                warnings.append("local_price_path_mismatch")
            if float(primary_check.get("max_daily_return_delta_pct") or 0.0) > 0.5:
                warnings.append("primary_daily_return_mismatch")
            if float(local_check.get("max_daily_return_delta_pct") or 0.0) > 0.5:
                warnings.append("local_daily_return_mismatch")
            record.update({
                "status": "warning" if warnings else "ok",
                "warnings": warnings,
                "akshare_vs_baostock": primary_check,
                "local_vs_baostock": local_check,
            })
        except Exception as exc:  # noqa: BLE001 - online audit is fail-open.
            record.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "reason": " ".join(str(exc).split())[:500],
            })
        samples.append(record)

    counts = {
        status: sum(sample.get("status") == status for sample in samples)
        for status in ("ok", "warning", "failed")
    }
    return {
        "enabled": True,
        "summary": {
            "requested": limit,
            "validated": len(samples),
            **counts,
        },
        "selection": "largest absolute-PnL unique traded symbols",
        "price_path_tolerance_pct": 0.5,
        "daily_return_tolerance_pct": 0.5,
        "samples": samples,
    }


def _compare_close_series(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, object]:
    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty or not {"date", "close"}.issubset(frame.columns):
            return pd.DataFrame(columns=["close"])
        normalized = frame[["date", "close"]].copy()
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
        normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
        return normalized.dropna().drop_duplicates("date", keep="last").set_index("date")

    left_norm = normalize(left).rename(columns={"close": "left"})
    right_norm = normalize(right).rename(columns={"close": "right"})
    overlap = left_norm.join(right_norm, how="inner").dropna()
    if overlap.empty:
        return {
            "overlap_rows": 0,
            "max_close_delta_pct": None,
            "median_close_delta_pct": None,
            "max_normalized_close_delta_pct": None,
            "median_normalized_close_delta_pct": None,
            "max_daily_return_delta_pct": None,
            "median_daily_return_delta_pct": None,
        }
    denominator = overlap[["left", "right"]].abs().max(axis=1).replace(0, pd.NA)
    delta = ((overlap["left"] - overlap["right"]).abs() / denominator * 100.0).dropna()
    left_base = float(overlap["left"].iloc[0])
    right_base = float(overlap["right"].iloc[0])
    if left_base and right_base:
        normalized_left = overlap["left"] / left_base
        normalized_right = overlap["right"] / right_base
        normalized_denominator = pd.concat(
            [normalized_left.abs(), normalized_right.abs()], axis=1
        ).max(axis=1).replace(0, pd.NA)
        normalized_delta = (
            (normalized_left - normalized_right).abs() / normalized_denominator * 100.0
        ).dropna()
    else:
        normalized_delta = pd.Series(dtype=float)
    return_delta = (
        overlap["left"].pct_change() - overlap["right"].pct_change()
    ).abs().mul(100.0).dropna()
    return {
        "overlap_rows": int(len(overlap)),
        "first_date": overlap.index.min().strftime("%Y-%m-%d"),
        "last_date": overlap.index.max().strftime("%Y-%m-%d"),
        "max_close_delta_pct": round(float(delta.max()), 4) if not delta.empty else None,
        "median_close_delta_pct": round(float(delta.median()), 4) if not delta.empty else None,
        "max_normalized_close_delta_pct": (
            round(float(normalized_delta.max()), 4) if not normalized_delta.empty else None
        ),
        "median_normalized_close_delta_pct": (
            round(float(normalized_delta.median()), 4) if not normalized_delta.empty else None
        ),
        "max_daily_return_delta_pct": (
            round(float(return_delta.max()), 4) if not return_delta.empty else None
        ),
        "median_daily_return_delta_pct": (
            round(float(return_delta.median()), 4) if not return_delta.empty else None
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
