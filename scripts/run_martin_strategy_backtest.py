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
from src.services.screening.martin_setups import (  # noqa: E402
    ALL_SETUP_TYPES,
    SETUP_FIRST_PULLBACK,
    normalize_tradable_setup_types,
)
from src.services.screening.strategy import load_all_strategies  # noqa: E402
from src.services.screening.sector_strength import load_sector_strength_history  # noqa: E402
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

    execution_profile = _resolve_execution_profile(
        args,
        strategy.screening.execution_profile,
    )
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
        market_regime_min_universe_size=args.market_regime_min_universe_size,
        market_regime_upgrade_confirmation_days=(
            args.market_regime_upgrade_confirmation_days
        ),
    )
    sector_history_path = (
        Path(args.sector_strength_path)
        if args.sector_strength_path
        else runtime.sector_strength_history_path
    )
    sector_history = (
        load_sector_strength_history(sector_history_path)
        if sector_history_path is not None
        else pd.DataFrame()
    )
    signal_book, signal_diagnostics = generate_martin_signals(
        panel,
        strategy.screening,
        current_static_prefiltered=args.universe_mode == "strategy-eligible",
        entry_setup_mode=str(execution_profile["entry_setup_mode"]),
        tradable_setup_types=execution_profile["tradable_setup_types"],
        sector_history=sector_history,
        sector_leadership_enabled=bool(
            execution_profile["sector_leadership_enabled"]
        ),
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
        "min_position_weight_pct": (
            args.min_position_weight_pct
            if args.min_position_weight_pct is not None
            else (12.0 if args.strategy_profile == "2.8" else 8.0)
        ),
        "adr_stress_fraction": args.adr_stress_fraction,
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "sell_stamp_duty_bps": args.sell_stamp_duty_bps,
        "trailing_drawdown_pct": args.trailing_drawdown_pct,
        "follow_through_days": args.follow_through_days,
        "follow_through_pct": args.follow_through_pct,
        "max_holding_days": args.max_holding_days,
        "strategy_profile_version": str(execution_profile["profile_version"]),
        "entry_setup_mode": str(execution_profile["entry_setup_mode"]),
        "reentry_cooldown_days": int(execution_profile["reentry_cooldown_days"]),
        "sector_leadership_enabled": bool(execution_profile["sector_leadership_enabled"]),
        "max_same_sector_positions": args.max_same_sector_positions,
        "progressive_exposure_enabled": bool(execution_profile["progressive_exposure_enabled"]),
        "pyramiding_enabled": bool(execution_profile["pyramiding_enabled"]),
        "adaptive_exit_enabled": bool(execution_profile["adaptive_exit_enabled"]),
        "initial_position_weight_pct": float(execution_profile["initial_position_weight_pct"]),
        "add_on_position_weight_pct": float(execution_profile["add_on_position_weight_pct"]),
        "max_add_ons": int(execution_profile["max_add_ons"]),
        "first_add_on_r_multiple": float(execution_profile["first_add_on_r_multiple"]),
        "second_add_on_r_multiple": float(execution_profile["second_add_on_r_multiple"]),
        "first_partial_r_multiple": float(execution_profile["first_partial_r_multiple"]),
        "second_partial_r_multiple": float(execution_profile["second_partial_r_multiple"]),
        "partial_exit_fraction": float(execution_profile["partial_exit_fraction"]),
        "adaptive_trailing_drawdown_pct": float(execution_profile["adaptive_trailing_drawdown_pct"]),
        "mature_trailing_drawdown_pct": float(execution_profile["mature_trailing_drawdown_pct"]),
        "adaptive_exit_model": str(execution_profile["adaptive_exit_model"]),
        "developing_exit_activation_r": float(execution_profile["developing_exit_activation_r"]),
        "developing_trailing_adr_multiple": float(execution_profile["developing_trailing_adr_multiple"]),
        "developing_trailing_min_pct": float(execution_profile["developing_trailing_min_pct"]),
        "developing_trailing_max_pct": float(execution_profile["developing_trailing_max_pct"]),
        "established_trailing_adr_multiple": float(execution_profile["established_trailing_adr_multiple"]),
        "established_trailing_min_pct": float(execution_profile["established_trailing_min_pct"]),
        "established_trailing_max_pct": float(execution_profile["established_trailing_max_pct"]),
        "mature_trailing_adr_multiple": float(execution_profile["mature_trailing_adr_multiple"]),
        "mature_trailing_min_pct": float(execution_profile["mature_trailing_min_pct"]),
        "mature_trailing_max_pct": float(execution_profile["mature_trailing_max_pct"]),
        "established_profit_floor_r": float(execution_profile["established_profit_floor_r"]),
        "mature_profit_floor_r": float(execution_profile["mature_profit_floor_r"]),
        "first_partial_exit_fraction": float(execution_profile["first_partial_exit_fraction"]),
        "second_partial_exit_fraction": float(execution_profile["second_partial_exit_fraction"]),
        "partial_exit_requires_exhaustion": bool(execution_profile["partial_exit_requires_exhaustion"]),
        "exhaustion_volume_ratio_min": float(execution_profile["exhaustion_volume_ratio_min"]),
        "exhaustion_close_location_max_pct": float(execution_profile["exhaustion_close_location_max_pct"]),
        "exhaustion_range_adr_multiple_min": float(execution_profile["exhaustion_range_adr_multiple_min"]),
        "weak_sector_score_max": float(execution_profile["weak_sector_score_max"]),
        "ema9_confirmation_days": int(execution_profile["ema9_confirmation_days"]),
        "risk_off_ema9_confirmation_days": int(execution_profile["risk_off_ema9_confirmation_days"]),
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
        "market_regime_enabled": args.market_regime_enabled,
        "market_regime_min_universe_size": args.market_regime_min_universe_size,
        "market_regime_upgrade_confirmation_days": (
            args.market_regime_upgrade_confirmation_days
        ),
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
        (
            "Point-in-time sector strength was loaded from the supplied history file."
            if not sector_history.empty
            else "Point-in-time sector history was not supplied; sector fields are explicitly unavailable rather than treated as confirmed neutral data."
        ),
        "Daily bars cannot reproduce order queues or intraday execution probability.",
    ]
    if float(data_diagnostics.get("start_history_coverage_pct") or 0.0) < 90.0:
        limitations.append(
            "Fewer than 90% of symbols available at the period start have 120 warm-up bars; early-period signals are incomplete."
        )

    payload = {
        "schema_version": 7,
        "strategy": {
            "name": strategy.name,
            "display_name": strategy.display_name,
            "version": strategy.version,
            "execution_profile": execution_profile,
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
        f"_v{str(execution_profile['profile_version']).replace('.', '')}"
        f"{universe_suffix}"
    )
    if args.ablation_label:
        stem += "_" + "".join(
            char if char.isalnum() or char in {"-", "_"} else "-"
            for char in args.ablation_label.strip()
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
    parser.add_argument(
        "--strategy-profile",
        choices=["2.8", "3.0", "3.1", "3.2", "3.3"],
        default="3.3",
        help="Run frozen v2.8-v3.2 baselines or the current v3.3 profile",
    )
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-same-sector-positions", type=int, default=2)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.5)
    parser.add_argument("--max-portfolio-heat-pct", type=float, default=2.5)
    parser.add_argument("--max-position-weight-pct", type=float, default=20.0)
    parser.add_argument("--max-technical-stop-pct", type=float, default=3.0)
    parser.add_argument(
        "--min-position-weight-pct",
        type=float,
        default=None,
        help="Override the profile minimum; v3.x defaults to 8%% and v2.8 to 12%%",
    )
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
    parser.add_argument(
        "--max-holding-days",
        type=int,
        default=None,
        help=(
            "Optional research-only holding cap; omit it to use the production "
            "trend-following exit without a fixed holding period"
        ),
    )
    parser.add_argument(
        "--reentry-cooldown-days",
        type=int,
        default=None,
        help="Override the selected profile; v3.x defaults to zero fixed cooldown",
    )
    parser.add_argument("--disable-multi-setup", action="store_true")
    parser.add_argument("--disable-sector-leadership", action="store_true")
    parser.add_argument("--disable-progressive-exposure", action="store_true")
    parser.add_argument("--disable-pyramiding", action="store_true")
    parser.add_argument("--disable-adaptive-exit", action="store_true")
    parser.add_argument(
        "--ablation-label",
        default="",
        help="Append a stable label to output filenames for module-by-module comparisons",
    )
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
    parser.add_argument(
        "--no-market-regime",
        dest="market_regime_enabled",
        action="store_false",
        help="Disable multi-day market-state position adjustment for comparison runs",
    )
    parser.set_defaults(market_regime_enabled=True)
    parser.add_argument("--market-regime-min-universe-size", type=int, default=100)
    parser.add_argument(
        "--market-regime-upgrade-confirmation-days",
        type=int,
        default=2,
    )
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
    parser.add_argument(
        "--sector-strength-path",
        help="Optional exact-date normalized sector-strength JSON/CSV history",
    )
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


def _resolve_execution_profile(
    args: argparse.Namespace,
    declared: dict[str, object],
) -> dict[str, object]:
    if args.strategy_profile == "2.8":
        profile: dict[str, object] = {
            "profile_version": "2.8",
            "entry_setup_mode": "legacy_pullback",
            "tradable_setup_types": [SETUP_FIRST_PULLBACK],
            "reentry_cooldown_days": 3,
            "sector_leadership_enabled": False,
            "progressive_exposure_enabled": False,
            "pyramiding_enabled": False,
            "adaptive_exit_enabled": False,
            "initial_position_weight_pct": args.max_position_weight_pct,
            "add_on_position_weight_pct": 5.0,
            "max_add_ons": 0,
            "first_add_on_r_multiple": 1.0,
            "second_add_on_r_multiple": 2.0,
            "first_partial_r_multiple": 3.0,
            "second_partial_r_multiple": 5.0,
            "partial_exit_fraction": 0.20,
            "adaptive_trailing_drawdown_pct": 6.0,
            "mature_trailing_drawdown_pct": 5.0,
            "adaptive_exit_model": "legacy_tightening",
        }
    elif args.strategy_profile == "3.0":
        profile = {
            "profile_version": "3.0",
            "entry_setup_mode": "multi",
            "tradable_setup_types": list(ALL_SETUP_TYPES),
            "reentry_cooldown_days": 0,
            "sector_leadership_enabled": True,
            "progressive_exposure_enabled": True,
            "pyramiding_enabled": True,
            "adaptive_exit_enabled": True,
            "initial_position_weight_pct": 10.0,
            "add_on_position_weight_pct": 5.0,
            "max_add_ons": 2,
            "first_add_on_r_multiple": 1.0,
            "second_add_on_r_multiple": 2.0,
            "first_partial_r_multiple": 3.0,
            "second_partial_r_multiple": 5.0,
            "partial_exit_fraction": 0.20,
            "adaptive_trailing_drawdown_pct": 6.0,
            "mature_trailing_drawdown_pct": 5.0,
            "adaptive_exit_model": "legacy_tightening",
        }
    else:
        current_setup_mode = str(declared.get("entry_setup_mode") or "multi")
        profile = {
            "profile_version": str(declared.get("profile_version") or "3.3"),
            "entry_setup_mode": current_setup_mode,
            "tradable_setup_types": list(
                normalize_tradable_setup_types(
                    declared.get("tradable_setup_types"),
                    mode=current_setup_mode,
                )
            ),
            "reentry_cooldown_days": int(declared.get("reentry_cooldown_days", 0)),
            "sector_leadership_enabled": bool(declared.get("sector_leadership_enabled", True)),
            "progressive_exposure_enabled": bool(declared.get("progressive_exposure_enabled", True)),
            "pyramiding_enabled": bool(declared.get("pyramiding_enabled", True)),
            "adaptive_exit_enabled": bool(declared.get("adaptive_exit_enabled", True)),
            "initial_position_weight_pct": float(declared.get("initial_position_weight_pct", 10.0)),
            "add_on_position_weight_pct": float(declared.get("add_on_position_weight_pct", 5.0)),
            "max_add_ons": int(declared.get("max_add_ons", 2)),
            "first_add_on_r_multiple": float(declared.get("first_add_on_r_multiple", 1.0)),
            "second_add_on_r_multiple": float(declared.get("second_add_on_r_multiple", 2.0)),
            "first_partial_r_multiple": float(declared.get("first_partial_r_multiple", 3.0)),
            "second_partial_r_multiple": float(declared.get("second_partial_r_multiple", 5.0)),
            "partial_exit_fraction": float(declared.get("partial_exit_fraction", 0.20)),
            "adaptive_trailing_drawdown_pct": float(declared.get("adaptive_trailing_drawdown_pct", 6.0)),
            "mature_trailing_drawdown_pct": float(declared.get("mature_trailing_drawdown_pct", 5.0)),
            "adaptive_exit_model": str(declared.get("adaptive_exit_model") or "profit_floor_volatility"),
        }
        if args.strategy_profile == "3.1":
            profile["profile_version"] = "3.1"
            profile["adaptive_exit_enabled"] = True
            profile["adaptive_exit_model"] = "profit_floor_volatility"
        elif args.strategy_profile == "3.2":
            profile["profile_version"] = "3.2"
            profile["adaptive_exit_enabled"] = False
            profile["adaptive_exit_model"] = "profit_floor_volatility"
        if args.disable_multi_setup:
            profile["entry_setup_mode"] = "legacy_pullback"
            profile["tradable_setup_types"] = [SETUP_FIRST_PULLBACK]
        if args.disable_sector_leadership:
            profile["sector_leadership_enabled"] = False
        if args.disable_progressive_exposure:
            profile["progressive_exposure_enabled"] = False
        if args.disable_pyramiding:
            profile["pyramiding_enabled"] = False
        if args.disable_adaptive_exit:
            profile["adaptive_exit_enabled"] = False
    if args.reentry_cooldown_days is not None:
        profile["reentry_cooldown_days"] = int(args.reentry_cooldown_days)
    profile.update({
        "developing_exit_activation_r": float(declared.get("developing_exit_activation_r", 1.0)),
        "developing_trailing_adr_multiple": float(declared.get("developing_trailing_adr_multiple", 1.0)),
        "developing_trailing_min_pct": float(declared.get("developing_trailing_min_pct", 5.0)),
        "developing_trailing_max_pct": float(declared.get("developing_trailing_max_pct", 8.0)),
        "established_trailing_adr_multiple": float(declared.get("established_trailing_adr_multiple", 1.5)),
        "established_trailing_min_pct": float(declared.get("established_trailing_min_pct", 7.0)),
        "established_trailing_max_pct": float(declared.get("established_trailing_max_pct", 12.0)),
        "mature_trailing_adr_multiple": float(declared.get("mature_trailing_adr_multiple", 1.75)),
        "mature_trailing_min_pct": float(declared.get("mature_trailing_min_pct", 9.0)),
        "mature_trailing_max_pct": float(declared.get("mature_trailing_max_pct", 15.0)),
        "established_profit_floor_r": float(declared.get("established_profit_floor_r", 1.0)),
        "mature_profit_floor_r": float(declared.get("mature_profit_floor_r", 2.5)),
        "first_partial_exit_fraction": float(declared.get("first_partial_exit_fraction", 0.0)),
        "second_partial_exit_fraction": float(declared.get("second_partial_exit_fraction", 0.15)),
        "partial_exit_requires_exhaustion": bool(declared.get("partial_exit_requires_exhaustion", True)),
        "exhaustion_volume_ratio_min": float(declared.get("exhaustion_volume_ratio_min", 1.5)),
        "exhaustion_close_location_max_pct": float(declared.get("exhaustion_close_location_max_pct", 60.0)),
        "exhaustion_range_adr_multiple_min": float(declared.get("exhaustion_range_adr_multiple_min", 1.0)),
        "weak_sector_score_max": float(declared.get("weak_sector_score_max", 40.0)),
        "ema9_confirmation_days": int(declared.get("ema9_confirmation_days", 2)),
        "risk_off_ema9_confirmation_days": int(declared.get("risk_off_ema9_confirmation_days", 1)),
    })
    return profile


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
