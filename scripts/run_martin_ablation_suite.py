# -*- coding: utf-8 -*-
"""Run the formal Martin 2.8-3.3 ablation suite on one shared panel."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

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
from src.services.screening.sector_strength import load_sector_strength_history  # noqa: E402
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
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    if start_date > end_date:
        raise ValueError("--start-date cannot be after --end-date")

    data_dir = Path(args.data_dir or REPO_ROOT / "data" / "screening")
    cache_dir = Path(args.cache_dir or data_dir / "daily_history")
    library_path = Path(
        args.library_path
        or data_dir / "backtests" / "full_a_share_daily_feature_library.json"
    )
    output_dir = Path(
        args.output_dir or data_dir / "backtests" / "ablation_2024_2026"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    estimated_sessions = len(pd.bdate_range(start_date, end_date))
    lookback_days = max(
        int(args.lookback_days or runtime.daily_lookback_days),
        120 + estimated_sessions + 15,
    )
    strategy = load_all_strategies(runtime.strategies_dir).get(STRATEGY_NAME)
    if strategy is None:
        raise RuntimeError(f"Strategy is unavailable: {STRATEGY_NAME}")

    print("[1/4] Loading the shared point-in-time panel...", flush=True)
    panel, data_diagnostics = load_local_backtest_panel(
        cache_dir=cache_dir,
        library_path=library_path,
        source=args.source or runtime.daily_source,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        names=get_stock_name_index_map(),
        price_adjustment="qfq",
        market_regime_min_universe_size=args.market_regime_min_universe_size,
        market_regime_upgrade_confirmation_days=(
            args.market_regime_upgrade_confirmation_days
        ),
    )
    benchmark = build_equal_weight_benchmark(panel)
    sector_path = (
        Path(args.sector_strength_path)
        if args.sector_strength_path
        else runtime.sector_strength_history_path
    )
    sector_history = (
        load_sector_strength_history(sector_path)
        if sector_path is not None
        else pd.DataFrame()
    )
    variants = _build_variants(strategy.screening.execution_profile)

    print("[2/4] Building distinct signal books...", flush=True)
    signal_cache: dict[
        tuple[str, bool, tuple[str, ...]],
        tuple[dict, dict[str, Any]],
    ] = {}
    for variant in variants:
        profile = variant["profile"]
        key = (
            str(profile["entry_setup_mode"]),
            bool(profile["sector_leadership_enabled"]),
            tuple(str(item) for item in profile["tradable_setup_types"]),
        )
        if key in signal_cache:
            continue
        signal_cache[key] = generate_martin_signals(
            panel,
            strategy.screening,
            current_static_prefiltered=False,
            entry_setup_mode=key[0],
            tradable_setup_types=key[2],
            sector_history=sector_history,
            sector_leadership_enabled=key[1],
        )

    print(f"[3/4] Executing {len(variants)} portfolio variants...", flush=True)
    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for index, variant in enumerate(variants, start=1):
        label = str(variant["label"])
        profile = variant["profile"]
        signal_book, signal_diagnostics = signal_cache[
            (
                str(profile["entry_setup_mode"]),
                bool(profile["sector_leadership_enabled"]),
                tuple(str(item) for item in profile["tradable_setup_types"]),
            )
        ]
        print(f"  [{index}/{len(variants)}] {label}", flush=True)
        config = _backtest_config(
            profile,
            start_date=start_date,
            end_date=end_date,
            initial_cash=args.initial_cash,
            args=args,
        )
        result = run_martin_portfolio_backtest(
            panel,
            signal_book=signal_book,
            config=config,
        )
        strategy_return = result["metrics"].get("total_return_pct")
        benchmark_return = benchmark["metrics"].get("total_return_pct")
        if strategy_return is not None and benchmark_return is not None:
            result["metrics"]["excess_return_pct"] = round(
                float(strategy_return) - float(benchmark_return), 4
            )
        payload = _payload(
            strategy=strategy,
            profile=profile,
            start_date=start_date,
            end_date=end_date,
            data_diagnostics=data_diagnostics,
            signal_diagnostics=signal_diagnostics,
            benchmark=benchmark,
            result=result,
            sector_history=sector_history,
        )
        payloads[label] = payload
        _write_variant(output_dir, label, payload, result)
        rows.append(_summary_row(label, variant["module"], result))
        annual_rows.extend(
            _annual_summary_rows(label, result, initial_cash=args.initial_cash)
        )

    print("[4/4] Writing the comparison summary...", flush=True)
    summary = _attach_full_variant_deltas(pd.DataFrame(rows))
    summary.to_csv(
        output_dir / "martin_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annual = pd.DataFrame(annual_rows).sort_values(["label", "year"])
    annual.to_csv(
        output_dir / "martin_ablation_annual.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_payload = {
        "schema_version": 1,
        "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "initial_cash": args.initial_cash,
        "universe_mode": "full-a-share",
        "sector_history_available": not sector_history.empty,
        "data_diagnostics": data_diagnostics,
        "benchmark": benchmark.get("metrics", {}),
        "variants": json.loads(summary.to_json(orient="records")),
        "annual": json.loads(annual.to_json(orient="records")),
    }
    (output_dir / "martin_ablation_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "martin_ablation_summary.md").write_text(
        _render_summary(summary_payload),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Summary: {output_dir / 'martin_ablation_summary.md'}", flush=True)
    return 0


def _build_variants(declared: dict[str, object]) -> list[dict[str, Any]]:
    current_setup_mode = str(declared.get("entry_setup_mode") or "multi")
    full = {
        "profile_version": str(declared.get("profile_version") or "3.3"),
        "entry_setup_mode": current_setup_mode,
        "tradable_setup_types": list(
            normalize_tradable_setup_types(
                declared.get("tradable_setup_types"),
                mode=current_setup_mode,
            )
        ),
        "reentry_cooldown_days": int(declared.get("reentry_cooldown_days", 0)),
        "sector_leadership_enabled": bool(
            declared.get("sector_leadership_enabled", True)
        ),
        "progressive_exposure_enabled": bool(
            declared.get("progressive_exposure_enabled", True)
        ),
        "pyramiding_enabled": bool(declared.get("pyramiding_enabled", True)),
        "adaptive_exit_enabled": bool(declared.get("adaptive_exit_enabled", True)),
        "initial_position_weight_pct": float(
            declared.get("initial_position_weight_pct", 10.0)
        ),
        "add_on_position_weight_pct": float(
            declared.get("add_on_position_weight_pct", 5.0)
        ),
        "max_add_ons": int(declared.get("max_add_ons", 2)),
        "first_add_on_r_multiple": float(
            declared.get("first_add_on_r_multiple", 1.0)
        ),
        "second_add_on_r_multiple": float(
            declared.get("second_add_on_r_multiple", 2.0)
        ),
        "first_partial_r_multiple": float(
            declared.get("first_partial_r_multiple", 3.0)
        ),
        "second_partial_r_multiple": float(
            declared.get("second_partial_r_multiple", 5.0)
        ),
        "partial_exit_fraction": float(declared.get("partial_exit_fraction", 0.20)),
        "adaptive_trailing_drawdown_pct": float(
            declared.get("adaptive_trailing_drawdown_pct", 6.0)
        ),
        "mature_trailing_drawdown_pct": float(
            declared.get("mature_trailing_drawdown_pct", 5.0)
        ),
        "adaptive_exit_model": str(
            declared.get("adaptive_exit_model") or "profit_floor_volatility"
        ),
        "developing_exit_activation_r": float(
            declared.get("developing_exit_activation_r", 1.0)
        ),
        "developing_trailing_adr_multiple": float(
            declared.get("developing_trailing_adr_multiple", 1.0)
        ),
        "developing_trailing_min_pct": float(
            declared.get("developing_trailing_min_pct", 5.0)
        ),
        "developing_trailing_max_pct": float(
            declared.get("developing_trailing_max_pct", 8.0)
        ),
        "established_trailing_adr_multiple": float(
            declared.get("established_trailing_adr_multiple", 1.5)
        ),
        "established_trailing_min_pct": float(
            declared.get("established_trailing_min_pct", 7.0)
        ),
        "established_trailing_max_pct": float(
            declared.get("established_trailing_max_pct", 12.0)
        ),
        "mature_trailing_adr_multiple": float(
            declared.get("mature_trailing_adr_multiple", 1.75)
        ),
        "mature_trailing_min_pct": float(
            declared.get("mature_trailing_min_pct", 9.0)
        ),
        "mature_trailing_max_pct": float(
            declared.get("mature_trailing_max_pct", 15.0)
        ),
        "established_profit_floor_r": float(
            declared.get("established_profit_floor_r", 1.0)
        ),
        "mature_profit_floor_r": float(
            declared.get("mature_profit_floor_r", 2.5)
        ),
        "first_partial_exit_fraction": float(
            declared.get("first_partial_exit_fraction", 0.0)
        ),
        "second_partial_exit_fraction": float(
            declared.get("second_partial_exit_fraction", 0.15)
        ),
        "partial_exit_requires_exhaustion": bool(
            declared.get("partial_exit_requires_exhaustion", True)
        ),
        "exhaustion_volume_ratio_min": float(
            declared.get("exhaustion_volume_ratio_min", 1.5)
        ),
        "exhaustion_close_location_max_pct": float(
            declared.get("exhaustion_close_location_max_pct", 60.0)
        ),
        "exhaustion_range_adr_multiple_min": float(
            declared.get("exhaustion_range_adr_multiple_min", 1.0)
        ),
        "weak_sector_score_max": float(
            declared.get("weak_sector_score_max", 40.0)
        ),
        "ema9_confirmation_days": int(declared.get("ema9_confirmation_days", 2)),
        "risk_off_ema9_confirmation_days": int(
            declared.get("risk_off_ema9_confirmation_days", 1)
        ),
    }
    v28 = {
        **deepcopy(full),
        "profile_version": "2.8",
        "entry_setup_mode": "legacy_pullback",
        "tradable_setup_types": [SETUP_FIRST_PULLBACK],
        "reentry_cooldown_days": 3,
        "sector_leadership_enabled": False,
        "progressive_exposure_enabled": False,
        "pyramiding_enabled": False,
        "adaptive_exit_enabled": False,
        "initial_position_weight_pct": 20.0,
        "max_add_ons": 0,
    }
    v30 = {
        **deepcopy(full),
        "profile_version": "3.0",
        "tradable_setup_types": list(ALL_SETUP_TYPES),
        "adaptive_exit_model": "legacy_tightening",
        "partial_exit_fraction": 0.20,
        "adaptive_trailing_drawdown_pct": 6.0,
        "mature_trailing_drawdown_pct": 5.0,
    }
    v31 = {
        **deepcopy(full),
        "profile_version": "3.1",
        "adaptive_exit_enabled": True,
        "adaptive_exit_model": "profit_floor_volatility",
    }
    v32 = {
        **deepcopy(full),
        "profile_version": "3.2",
        "adaptive_exit_enabled": False,
        "adaptive_exit_model": "profit_floor_volatility",
    }
    definitions = [
        ("baseline-v28", "historical_baseline", v28),
        ("baseline-v30", "previous_baseline", v30),
        ("baseline-v31", "legacy_adaptive_exit", v31),
        ("baseline-v32", "adaptive_exit", v32),
        ("full-v33", "full_v33", full),
        (
            "all-setup-trading",
            "setup_trade_permission",
            {"tradable_setup_types": list(ALL_SETUP_TYPES)},
        ),
        (
            "no-multi-setup",
            "multi_setup",
            {
                "entry_setup_mode": "legacy_pullback",
                "tradable_setup_types": [SETUP_FIRST_PULLBACK],
            },
        ),
        ("no-sector-leadership", "sector_leadership", {"sector_leadership_enabled": False}),
        ("no-progressive-exposure", "progressive_exposure", {"progressive_exposure_enabled": False}),
        ("no-pyramiding", "pyramiding", {"pyramiding_enabled": False}),
        ("cooldown-3d", "zero_fixed_cooldown", {"reentry_cooldown_days": 3}),
    ]
    variants: list[dict[str, Any]] = []
    for label, module, changes in definitions:
        profile = (
            deepcopy(changes)
            if label in {
                "baseline-v28",
                "baseline-v30",
                "baseline-v31",
                "baseline-v32",
            }
            else {**deepcopy(full), **changes}
        )
        variants.append({"label": label, "module": module, "profile": profile})
    return variants


def _backtest_config(
    profile: dict[str, Any],
    *,
    start_date: date,
    end_date: date,
    initial_cash: float,
    args: argparse.Namespace,
) -> MartinBacktestConfig:
    is_v28 = str(profile["profile_version"]) == "2.8"
    return MartinBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        max_positions=5,
        risk_per_trade_pct=0.5,
        max_portfolio_heat_pct=2.5,
        max_position_weight_pct=20.0,
        max_technical_stop_pct=3.0,
        min_position_weight_pct=12.0 if is_v28 else 8.0,
        adr_stress_fraction=1.0,
        commission_bps=3.0,
        slippage_bps=5.0,
        sell_stamp_duty_bps=5.0,
        trailing_drawdown_pct=8.0,
        follow_through_days=10,
        follow_through_pct=3.0,
        max_holding_days=None,
        strategy_profile_version=str(profile["profile_version"]),
        entry_setup_mode=str(profile["entry_setup_mode"]),
        reentry_cooldown_days=int(profile["reentry_cooldown_days"]),
        sector_leadership_enabled=bool(profile["sector_leadership_enabled"]),
        max_same_sector_positions=2,
        progressive_exposure_enabled=bool(profile["progressive_exposure_enabled"]),
        pyramiding_enabled=bool(profile["pyramiding_enabled"]),
        adaptive_exit_enabled=bool(profile["adaptive_exit_enabled"]),
        initial_position_weight_pct=float(profile["initial_position_weight_pct"]),
        add_on_position_weight_pct=float(profile["add_on_position_weight_pct"]),
        max_add_ons=int(profile["max_add_ons"]),
        first_add_on_r_multiple=float(profile["first_add_on_r_multiple"]),
        second_add_on_r_multiple=float(profile["second_add_on_r_multiple"]),
        first_partial_r_multiple=float(profile["first_partial_r_multiple"]),
        second_partial_r_multiple=float(profile["second_partial_r_multiple"]),
        partial_exit_fraction=float(profile["partial_exit_fraction"]),
        adaptive_trailing_drawdown_pct=float(
            profile["adaptive_trailing_drawdown_pct"]
        ),
        mature_trailing_drawdown_pct=float(profile["mature_trailing_drawdown_pct"]),
        adaptive_exit_model=str(profile["adaptive_exit_model"]),
        developing_exit_activation_r=float(profile["developing_exit_activation_r"]),
        developing_trailing_adr_multiple=float(profile["developing_trailing_adr_multiple"]),
        developing_trailing_min_pct=float(profile["developing_trailing_min_pct"]),
        developing_trailing_max_pct=float(profile["developing_trailing_max_pct"]),
        established_trailing_adr_multiple=float(profile["established_trailing_adr_multiple"]),
        established_trailing_min_pct=float(profile["established_trailing_min_pct"]),
        established_trailing_max_pct=float(profile["established_trailing_max_pct"]),
        mature_trailing_adr_multiple=float(profile["mature_trailing_adr_multiple"]),
        mature_trailing_min_pct=float(profile["mature_trailing_min_pct"]),
        mature_trailing_max_pct=float(profile["mature_trailing_max_pct"]),
        established_profit_floor_r=float(profile["established_profit_floor_r"]),
        mature_profit_floor_r=float(profile["mature_profit_floor_r"]),
        first_partial_exit_fraction=float(profile["first_partial_exit_fraction"]),
        second_partial_exit_fraction=float(profile["second_partial_exit_fraction"]),
        partial_exit_requires_exhaustion=bool(profile["partial_exit_requires_exhaustion"]),
        exhaustion_volume_ratio_min=float(profile["exhaustion_volume_ratio_min"]),
        exhaustion_close_location_max_pct=float(profile["exhaustion_close_location_max_pct"]),
        exhaustion_range_adr_multiple_min=float(profile["exhaustion_range_adr_multiple_min"]),
        weak_sector_score_max=float(profile["weak_sector_score_max"]),
        ema9_confirmation_days=int(profile["ema9_confirmation_days"]),
        risk_off_ema9_confirmation_days=int(profile["risk_off_ema9_confirmation_days"]),
        market_regime_enabled=True,
        market_regime_min_universe_size=args.market_regime_min_universe_size,
        market_regime_upgrade_confirmation_days=(
            args.market_regime_upgrade_confirmation_days
        ),
    )


def _payload(
    *,
    strategy: Any,
    profile: dict[str, Any],
    start_date: date,
    end_date: date,
    data_diagnostics: dict[str, Any],
    signal_diagnostics: dict[str, Any],
    benchmark: dict[str, Any],
    result: dict[str, Any],
    sector_history: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema_version": 7,
        "strategy": {
            "name": strategy.name,
            "display_name": strategy.display_name,
            "version": strategy.version,
            "execution_profile": profile,
        },
        "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "universe_mode": "full-a-share",
        "data_diagnostics": data_diagnostics,
        "signal_diagnostics": signal_diagnostics,
        "source_validation": {
            "enabled": False,
            "summary": {
                "requested": 0,
                "validated": 0,
                "ok": 0,
                "warning": 0,
                "failed": 0,
            },
            "samples": [],
        },
        "benchmark": benchmark,
        "results": {"strategy": result},
        "limitations": [
            "Universe covers the current full A-share snapshot; delisted stocks remain absent.",
            "Historical ST status and market capitalization are not replayed point in time.",
            (
                "Point-in-time sector strength was loaded from the configured history file."
                if not sector_history.empty
                else "Point-in-time sector history is unavailable; the sector ablation is a contract-control run and cannot estimate historical sector alpha."
            ),
            "Daily bars cannot reproduce order queues or intraday execution probability.",
        ],
    }


def _write_variant(
    output_dir: Path,
    label: str,
    payload: dict[str, Any],
    result: dict[str, Any],
) -> None:
    period = payload["period"]
    profile = payload["strategy"]["execution_profile"]
    version = str(profile["profile_version"]).replace(".", "")
    stem = (
        f"martin_backtest_{period['start'].replace('-', '')}_"
        f"{period['end'].replace('-', '')}_v{version}_full_a_share_{label}"
    )
    (output_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / f"{stem}.md").write_text(
        render_backtest_markdown(payload),
        encoding="utf-8",
    )
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


def _summary_row(label: str, module: str, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    diagnostics = result.get("diagnostics", {})
    return {
        "label": label,
        "module": module,
        "total_return_pct": metrics.get("total_return_pct"),
        "annualized_return_pct": metrics.get("annualized_return_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "trade_count": metrics.get("trade_count"),
        "win_rate_pct": metrics.get("win_rate_pct"),
        "profit_factor": metrics.get("profit_factor"),
        "average_exposure_pct": metrics.get("average_exposure_pct"),
        "estimated_total_cost": metrics.get("estimated_total_cost"),
        "initial_stop_exit_count": metrics.get("initial_stop_exit_count"),
        "add_on_count": diagnostics.get("add_on_count", 0),
        "partial_exit_count": diagnostics.get("partial_exit_count", 0),
        "final_equity": metrics.get("final_equity"),
        "excess_return_pct": metrics.get("excess_return_pct"),
    }


def _attach_full_variant_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    full = summary.loc[summary["label"].eq("full-v33")]
    if full.empty:
        return summary
    reference = full.iloc[0]
    summary = summary.copy()
    summary["module_return_contribution_pct"] = (
        float(reference["total_return_pct"])
        - pd.to_numeric(summary["total_return_pct"], errors="coerce")
    ).round(4)
    summary["module_drawdown_improvement_pct"] = (
        float(reference["max_drawdown_pct"])
        - pd.to_numeric(summary["max_drawdown_pct"], errors="coerce")
    ).round(4)
    summary["module_sharpe_contribution"] = (
        float(reference["sharpe_ratio"])
        - pd.to_numeric(summary["sharpe_ratio"], errors="coerce")
    ).round(4)
    summary.loc[
        summary["label"].isin([
            "baseline-v28",
            "baseline-v30",
            "baseline-v31",
            "full-v33",
        ]),
        [
            "module_return_contribution_pct",
            "module_drawdown_improvement_pct",
            "module_sharpe_contribution",
        ],
    ] = pd.NA
    return summary


def _annual_summary_rows(
    label: str,
    result: dict[str, Any],
    *,
    initial_cash: float,
) -> list[dict[str, Any]]:
    curve = pd.DataFrame(result.get("equity_curve", []))
    if curve.empty:
        return []
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    curve["equity"] = pd.to_numeric(curve["equity"], errors="coerce")
    curve = curve.dropna(subset=["date", "equity"]).sort_values("date")
    prior_equity = float(initial_cash)
    rows: list[dict[str, Any]] = []
    for year, year_curve in curve.groupby(curve["date"].dt.year, sort=True):
        end_equity = float(year_curve.iloc[-1]["equity"])
        values = pd.concat(
            [pd.Series([prior_equity]), year_curve["equity"].reset_index(drop=True)],
            ignore_index=True,
        )
        drawdown = (values / values.cummax() - 1.0).min() * 100.0
        rows.append({
            "label": label,
            "year": int(year),
            "return_pct": round((end_equity / prior_equity - 1.0) * 100.0, 4),
            "max_drawdown_pct": round(float(drawdown), 4),
            "start_equity": round(prior_equity, 4),
            "end_equity": round(end_equity, 4),
        })
        prior_equity = end_equity
    return rows


def _render_summary(payload: dict[str, Any]) -> str:
    period = payload["period"]
    benchmark = payload["benchmark"]
    rows = payload["variants"]
    lines = [
        "# Martin 2024-2026 正式消融回测",
        "",
        f"> 区间：{period['start']} 至 {period['end']}；本金：{payload['initial_cash']:,.0f} 元；全市场当前 A 股快照。",
        "",
        "## 对照口径",
        "",
        "- 所有组共用同一份点时行情面板、市场状态、交易成本和风险参数。",
        "- `full-v33` 是当前策略；模块贡献 = 当前 v3.3 指标减去对照组指标。",
        "- `baseline-v31` 保留旧首笔R口径；`baseline-v32` 关闭自适应退出，是v3.3最重要的单变量对照。",
        (
            "- 本次存在点时板块历史，可评估板块模块。"
            if payload["sector_history_available"]
            else "- 本次没有 2024-2026 点时板块历史；板块组只能验证缺失数据契约，不能估计板块因子的历史超额收益。"
        ),
        "",
        "## 核心结果",
        "",
        "| 组别 | 收益 | 最大回撤 | 夏普 | 交易数 | 胜率 | 利润因子 | 平均仓位 | 模块收益贡献 | 模块回撤改善 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {ret:+.2f}% | {dd:.2f}% | {sharpe:.2f} | {trades} | {win:.2f}% | {pf} | {exposure:.2f}% | {ret_delta} | {dd_delta} |".format(
                label=row["label"],
                ret=float(row["total_return_pct"] or 0.0),
                dd=float(row["max_drawdown_pct"] or 0.0),
                sharpe=float(row["sharpe_ratio"] or 0.0),
                trades=int(row["trade_count"] or 0),
                win=float(row["win_rate_pct"] or 0.0),
                pf=(
                    f"{float(row['profit_factor']):.2f}"
                    if row.get("profit_factor") is not None
                    else "--"
                ),
                exposure=float(row["average_exposure_pct"] or 0.0),
                ret_delta=_delta(row.get("module_return_contribution_pct"), "%"),
                dd_delta=_delta(row.get("module_drawdown_improvement_pct"), "%"),
            )
        )
    annual = pd.DataFrame(payload.get("annual", []))
    if not annual.empty:
        return_table = annual.pivot(index="label", columns="year", values="return_pct")
        lines.extend([
            "",
            "## 年度拆分",
            "",
            "| 组别 | 2024 | 2025 | 2026截至期末 |",
            "| --- | ---: | ---: | ---: |",
        ])
        for label, values in return_table.iterrows():
            lines.append(
                "| {label} | {y2024} | {y2025} | {y2026} |".format(
                    label=label,
                    y2024=_delta(values.get(2024), "%"),
                    y2025=_delta(values.get(2025), "%"),
                    y2026=_delta(values.get(2026), "%"),
                )
            )
    lines.extend([
        "",
        "## 基准与数据",
        "",
        "- 同池等权基准：收益 {ret:+.2f}%，最大回撤 {dd:.2f}%，夏普 {sharpe:.2f}。".format(
            ret=float(benchmark.get("total_return_pct") or 0.0),
            dd=float(benchmark.get("max_drawdown_pct") or 0.0),
            sharpe=float(benchmark.get("sharpe_ratio") or 0.0),
        ),
        "- 当前快照会遗漏历史退市股票，且历史 ST、市值和板块成分未完整点时重放，结果仍存在幸存者偏差。",
        "",
    ])
    return "\n".join(lines)


def _delta(value: Any, suffix: str) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):+.2f}{suffix}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Martin 2.8-3.3 formal ablation suite on one shared panel"
    )
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-10-16")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--data-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--library-path")
    parser.add_argument("--sector-strength-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--lookback-days", type=int)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--market-regime-min-universe-size", type=int, default=100)
    parser.add_argument(
        "--market-regime-upgrade-confirmation-days",
        type=int,
        default=2,
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
