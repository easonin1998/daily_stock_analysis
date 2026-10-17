# -*- coding: utf-8 -*-
# Derived from AlphaSift revision 9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf.
# Licensed under Apache-2.0 and modified for daily_stock_analysis.
"""Daily-bar portfolio backtest for the Martin momentum-pullback strategy.

This simulator is intentionally separate from ``BacktestService``.  The
existing service evaluates stored AI reports; this module evaluates a
deterministic screening strategy over a historical cross-section.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.services.screening.daily import (
    _history_price_adjustment,
    load_cached_daily_history,
)
from src.services.screening.daily_library import load_daily_feature_library
from src.services.screening.filter import apply_hard_filters
from src.services.screening.models import ScreeningConfig
from src.services.screening.scorer import compute_screen_scores
from src.services.screening.trade_plan import (
    MartinTradePlanConfig,
    build_martin_trade_plan,
    rebase_martin_trade_plan_after_fill,
)


@dataclass(frozen=True)
class MartinBacktestConfig:
    """Execution assumptions for the single Martin auction strategy."""

    start_date: date
    end_date: date
    initial_cash: float = 1_000_000.0
    max_positions: int = 5
    risk_per_trade_pct: float = 0.5
    max_portfolio_heat_pct: float = 2.5
    max_position_weight_pct: float = 20.0
    max_technical_stop_pct: float = 3.0
    min_position_weight_pct: float = 12.0
    adr_stress_fraction: float = 1.0
    lot_size: int = 100
    commission_bps: float = 3.0
    slippage_bps: float = 5.0
    sell_stamp_duty_bps: float = 5.0
    trailing_drawdown_pct: float = 8.0
    follow_through_days: int = 10
    follow_through_pct: float = 3.0
    max_holding_days: int = 20
    reentry_cooldown_days: int = 3
    entry_gap_min_pct: float | None = None
    entry_gap_max_pct: float | None = None
    entry_gap_down_volatility_fraction: float = 0.30
    entry_gap_up_volatility_fraction: float = 0.50
    entry_gap_down_min_pct: float = 1.0
    entry_gap_down_max_pct: float = 1.5
    entry_gap_up_min_pct: float = 1.5
    entry_gap_up_max_pct: float = 2.0
    entry_gap_fallback_min_pct: float = -1.25
    entry_gap_fallback_max_pct: float = 1.75

    def validate(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.max_positions <= 0 or self.lot_size <= 0:
            raise ValueError("max_positions and lot_size must be positive")
        if self.risk_per_trade_pct <= 0 or self.max_portfolio_heat_pct <= 0:
            raise ValueError("risk budgets must be positive")
        if not 0 < self.max_position_weight_pct <= 100:
            raise ValueError("max_position_weight_pct must be within (0, 100]")
        if not 0 < self.max_technical_stop_pct <= 100:
            raise ValueError("max_technical_stop_pct must be within (0, 100]")
        if not 0 <= self.min_position_weight_pct <= self.max_position_weight_pct:
            raise ValueError(
                "min_position_weight_pct must be within [0, max_position_weight_pct]"
            )
        if self.adr_stress_fraction < 0:
            raise ValueError("adr_stress_fraction cannot be negative")
        if self.follow_through_days <= 0 or self.max_holding_days <= 0:
            raise ValueError("holding-day limits must be positive")
        if self.follow_through_days > self.max_holding_days:
            raise ValueError("follow_through_days cannot exceed max_holding_days")
        if (
            self.entry_gap_min_pct is not None
            and self.entry_gap_max_pct is not None
            and self.entry_gap_min_pct > self.entry_gap_max_pct
        ):
            raise ValueError("entry_gap_min_pct cannot exceed entry_gap_max_pct")
        for field_name in (
            "entry_gap_down_volatility_fraction",
            "entry_gap_up_volatility_fraction",
            "entry_gap_down_min_pct",
            "entry_gap_down_max_pct",
            "entry_gap_up_min_pct",
            "entry_gap_up_max_pct",
        ):
            if float(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.entry_gap_down_min_pct > self.entry_gap_down_max_pct:
            raise ValueError("entry gap down minimum cannot exceed maximum")
        if self.entry_gap_up_min_pct > self.entry_gap_up_max_pct:
            raise ValueError("entry gap up minimum cannot exceed maximum")
        if self.entry_gap_fallback_min_pct > self.entry_gap_fallback_max_pct:
            raise ValueError("entry gap fallback minimum cannot exceed maximum")
        for field_name in (
            "commission_bps",
            "slippage_bps",
            "sell_stamp_duty_bps",
            "trailing_drawdown_pct",
            "follow_through_pct",
        ):
            if float(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass
class _Position:
    code: str
    name: str
    shares: int
    entry_date: pd.Timestamp
    entry_raw_price: float
    entry_price: float
    entry_notional: float
    entry_fee: float
    entry_total: float
    entry_score: float
    signal_date: pd.Timestamp
    signal_close: float
    signal_trigger_price: float | None
    entry_gap_pct: float
    entry_gap_min_pct: float
    entry_gap_max_pct: float
    entry_gap_volatility_reference_pct: float | None
    entry_gap_volatility_source: str
    initial_stop_price: float
    breakeven_activation_price: float
    two_r_reference_price: float
    planned_risk_per_share: float
    effective_risk_per_share: float
    adr_stress_per_share: float
    planned_account_risk_pct: float
    effective_account_risk_pct: float
    adr_stress_account_risk_pct: float
    position_weight_pct: float
    signal_context: dict[str, float | None]
    holding_days: int = 0
    highest_close: float = 0.0


def load_local_backtest_panel(
    *,
    cache_dir: str | Path,
    library_path: str | Path,
    source: str,
    lookback_days: int,
    start_date: date,
    end_date: date,
    names: Mapping[str, str] | None = None,
    price_adjustment: str = "qfq",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load cached histories and build point-in-time Martin features."""
    library = load_daily_feature_library(
        library_path,
        max_age_days=36_500,
    )
    codes = sorted(set(library.get("code", pd.Series(dtype=str)).astype(str)))
    histories: dict[str, pd.DataFrame] = {}
    source_counts: dict[str, int] = {}
    adjustment_counts: dict[str, int] = {}
    incompatible_adjustment_codes: list[str] = []
    incompatible_adjustment_counts: dict[str, int] = {}
    missing_codes: list[str] = []
    for code in codes:
        history = load_cached_daily_history(
            code,
            source=source,
            lookback_days=lookback_days,
            cache_dir=cache_dir,
            price_adjustment=price_adjustment,
        )
        if history is None or history.empty:
            fallback = load_cached_daily_history(
                code,
                source=source,
                lookback_days=lookback_days,
                cache_dir=cache_dir,
            )
            if fallback is not None and not fallback.empty:
                adjustment = _history_price_adjustment(fallback)
                incompatible_adjustment_codes.append(code)
                incompatible_adjustment_counts[adjustment] = (
                    incompatible_adjustment_counts.get(adjustment, 0) + 1
                )
            else:
                missing_codes.append(code)
            continue
        histories[code] = history
        source_name = str(history.attrs.get("daily_source") or "unknown")
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        adjustment = _history_price_adjustment(history)
        adjustment_counts[adjustment] = adjustment_counts.get(adjustment, 0) + 1

    panel = build_historical_feature_panel(
        histories,
        start_date=start_date,
        end_date=end_date,
        names=names,
    )
    available_at_start = _count_ready_codes(
        histories,
        start_date=start_date,
        minimum=1,
    )
    ready_at_start = _count_ready_codes(
        histories,
        start_date=start_date,
        minimum=120,
    )
    diagnostics = {
        "library_codes": len(codes),
        "loaded_codes": len(histories),
        "lookback_days": int(lookback_days),
        "missing_cache_codes": len(missing_codes),
        "missing_cache_samples": missing_codes[:10],
        "panel_rows": len(panel),
        "panel_dates": int(panel["date"].nunique()) if not panel.empty else 0,
        "panel_start": _date_text(panel["date"].min()) if not panel.empty else "",
        "panel_end": _date_text(panel["date"].max()) if not panel.empty else "",
        "source_counts": source_counts,
        "required_price_adjustment": price_adjustment,
        "price_adjustment_counts": adjustment_counts,
        "incompatible_price_adjustment_codes": len(incompatible_adjustment_codes),
        "incompatible_price_adjustment_counts": incompatible_adjustment_counts,
        "incompatible_price_adjustment_samples": incompatible_adjustment_codes[:10],
        "codes_available_at_period_start": available_at_start,
        "codes_with_120_bars_at_period_start": ready_at_start,
        "start_history_coverage_pct": round(
            ready_at_start / available_at_start * 100.0 if available_at_start else 0.0,
            4,
        ),
    }
    return panel, diagnostics


def build_historical_feature_panel(
    histories: Mapping[str, pd.DataFrame],
    *,
    start_date: date,
    end_date: date,
    names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compute each row using only bars available on or before that row's date."""
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    frames: list[pd.DataFrame] = []
    name_map = names or {}
    for code, raw in histories.items():
        prepared = _prepare_stock_history(
            str(code),
            raw,
            name=str(name_map.get(str(code), "")),
        )
        if prepared.empty:
            continue
        mask = prepared["date"].dt.date.between(start_date, end_date)
        selected = prepared.loc[mask].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    return _attach_cross_sectional_features(panel)


def generate_martin_signals(
    panel: pd.DataFrame,
    screening: ScreeningConfig,
    *,
    current_static_prefiltered: bool = True,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], dict[str, Any]]:
    """Build confirmed Martin entries from point-in-time daily setup rows."""
    if panel.empty:
        return {}, {
            "setup_count": 0,
            "entry_signal_count": 0,
            "unconfirmed_setup_count": 0,
            "first_signal_date": "",
        }

    # Historical point-in-time ST and market-cap values are unavailable.  The
    # normal research library already excludes current ST names; a full-market
    # library does not, so conservatively apply the current name flag there.
    historical_filters = replace(
        screening.hard_filters,
        exclude_st=not current_static_prefiltered,
        market_cap_min=None,
        market_cap_max=None,
    )
    signal_book: dict[pd.Timestamp, pd.DataFrame] = {}
    setup_count = 0
    entry_signal_count = 0
    first_signal: pd.Timestamp | None = None
    daily_counts: list[dict[str, Any]] = []

    for signal_date, day in panel.groupby("date", sort=True):
        exact = apply_hard_filters(day, historical_filters)
        scored = compute_screen_scores(exact, screening) if not exact.empty else exact.copy()
        trigger_distance = pd.to_numeric(scored.get("trigger_distance_pct"), errors="coerce")
        confirmed = _sort_signals(scored[trigger_distance.ge(0)])
        if not confirmed.empty:
            timestamp = pd.Timestamp(signal_date).normalize()
            signal_book[timestamp] = confirmed
            if first_signal is None:
                first_signal = timestamp
        setup_count += len(scored)
        entry_signal_count += len(confirmed)
        daily_counts.append({
            "date": _date_text(signal_date),
            "eligible_rows": int(len(day)),
            "setup_count": int(len(scored)),
            "entry_signal_count": int(len(confirmed)),
            "unconfirmed_setup_count": int(len(scored) - len(confirmed)),
        })

    return signal_book, {
        "setup_count": setup_count,
        "entry_signal_count": entry_signal_count,
        "unconfirmed_setup_count": setup_count - entry_signal_count,
        "first_signal_date": _date_text(first_signal),
        "daily_counts": daily_counts,
        "entry_trigger": "signal close at or above the previous trading day's high",
        "execution_session": "next trading day's opening call auction",
        "historical_static_filters": (
            "current strategy-eligible universe; no historical ST/market-cap replay"
            if current_static_prefiltered
            else "full current A-share snapshot; current ST names excluded, no historical ST/market-cap replay"
        ),
        "sector_scoring": "neutral because point-in-time industry/sector breadth is unavailable",
    }


def run_martin_portfolio_backtest(
    panel: pd.DataFrame,
    *,
    signal_book: Mapping[pd.Timestamp, pd.DataFrame],
    config: MartinBacktestConfig,
) -> dict[str, Any]:
    """Simulate the long-only strategy at the next opening call auction."""
    config.validate()
    if panel.empty:
        return _empty_result(config, "historical panel is empty")

    work = panel[
        panel["date"].dt.date.between(config.start_date, config.end_date)
    ].copy()
    if work.empty:
        return _empty_result(config, "no bars in requested date range")

    work = work.sort_values(["date", "code"], kind="mergesort")
    dates = [pd.Timestamp(item).normalize() for item in sorted(work["date"].unique())]
    bars = {
        (pd.Timestamp(row.date).normalize(), str(row.code)): row
        for row in work.itertuples(index=False)
    }
    cash = float(config.initial_cash)
    positions: dict[str, _Position] = {}
    pending_entries = pd.DataFrame()
    pending_exits: dict[str, str] = {}
    last_exit_index: dict[str, int] = {}
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    entry_count = 0
    skipped_limit_up = 0
    skipped_gap_up = 0
    skipped_gap_down = 0
    skipped_broken_support = 0
    skipped_technical_stop_too_wide = 0
    skipped_min_position_weight = 0
    skipped_risk_budget = 0
    delayed_limit_down = 0
    skipped_cooldown = 0
    skipped_missing_bar = 0

    for date_index, trading_date in enumerate(dates):
        exited_today: set[str] = set()

        for code, reason in list(pending_exits.items()):
            position = positions.get(code)
            bar = bars.get((trading_date, code))
            if position is None:
                pending_exits.pop(code, None)
                continue
            if bar is None or not _positive(bar.open):
                skipped_missing_bar += 1
                continue
            if _is_open_limit_down(code, float(bar.open), _number(bar.previous_close)):
                delayed_limit_down += 1
                continue
            cash_delta, trade = _close_position(
                position,
                raw_price=float(bar.open),
                exit_date=trading_date,
                reason=reason,
                config=config,
            )
            cash += cash_delta
            trades.append(trade)
            positions.pop(code, None)
            pending_exits.pop(code, None)
            exited_today.add(code)
            last_exit_index[code] = date_index

        if not pending_entries.empty and len(positions) < config.max_positions:
            for row in pending_entries.itertuples(index=False):
                code = str(row.code)
                if len(positions) >= config.max_positions:
                    break
                if code in positions or code in exited_today:
                    continue
                previous_exit = last_exit_index.get(code)
                if previous_exit is not None and date_index - previous_exit <= config.reentry_cooldown_days:
                    skipped_cooldown += 1
                    continue
                bar = bars.get((trading_date, code))
                if bar is None or not _positive(bar.open):
                    skipped_missing_bar += 1
                    continue
                if _is_open_limit_up(code, float(bar.open), _number(bar.previous_close)):
                    skipped_limit_up += 1
                    continue

                signal_close = _number(getattr(row, "close", None)) or _number(
                    getattr(row, "price", None)
                )
                if signal_close is None or signal_close <= 0:
                    skipped_missing_bar += 1
                    continue
                raw_price = float(bar.open)
                entry_gap_pct = (raw_price / signal_close - 1.0) * 100.0
                trade_plan = build_martin_trade_plan(
                    price=signal_close,
                    trigger_distance_pct=getattr(row, "trigger_distance_pct", None),
                    adr_20_pct=getattr(row, "adr_20_pct", None),
                    ema21=getattr(row, "ema21", None),
                    atr_20_pct=getattr(row, "atr_20_pct", None),
                    anchored_vwap=getattr(row, "anchored_vwap_60d_high", None),
                    config=MartinTradePlanConfig(
                        entry_gap_min_pct=config.entry_gap_min_pct,
                        entry_gap_max_pct=config.entry_gap_max_pct,
                        entry_gap_down_volatility_fraction=(
                            config.entry_gap_down_volatility_fraction
                        ),
                        entry_gap_up_volatility_fraction=(
                            config.entry_gap_up_volatility_fraction
                        ),
                        entry_gap_down_min_pct=config.entry_gap_down_min_pct,
                        entry_gap_down_max_pct=config.entry_gap_down_max_pct,
                        entry_gap_up_min_pct=config.entry_gap_up_min_pct,
                        entry_gap_up_max_pct=config.entry_gap_up_max_pct,
                        entry_gap_fallback_min_pct=config.entry_gap_fallback_min_pct,
                        entry_gap_fallback_max_pct=config.entry_gap_fallback_max_pct,
                        trailing_drawdown_pct=config.trailing_drawdown_pct,
                        follow_through_days=config.follow_through_days,
                        follow_through_pct=config.follow_through_pct,
                        max_holding_days=config.max_holding_days,
                        max_portfolio_risk_per_trade_pct=config.risk_per_trade_pct,
                        max_portfolio_heat_pct=config.max_portfolio_heat_pct,
                        max_position_weight_pct=config.max_position_weight_pct,
                        max_technical_stop_pct=config.max_technical_stop_pct,
                        min_position_weight_pct=config.min_position_weight_pct,
                        adr_stress_fraction=config.adr_stress_fraction,
                    ),
                )
                if trade_plan.get("signal_status") == "risk_rejected":
                    skipped_technical_stop_too_wide += 1
                    continue
                entry_plan = trade_plan.get("entry") if isinstance(trade_plan.get("entry"), dict) else {}
                exit_plan = trade_plan.get("exit") if isinstance(trade_plan.get("exit"), dict) else {}
                max_entry_price = _number(entry_plan.get("max_price"))
                min_entry_price = _number(entry_plan.get("min_price"))
                min_entry_gap_pct = _number(entry_plan.get("min_gap_pct"))
                max_entry_gap_pct = _number(entry_plan.get("max_gap_pct"))
                entry_gap_volatility_reference_pct = _number(
                    entry_plan.get("gap_volatility_reference_pct")
                )
                entry_gap_volatility_source = str(
                    entry_plan.get("gap_volatility_source") or "unknown"
                )
                initial_stop_price = _number(exit_plan.get("initial_stop_price"))
                if (
                    max_entry_price is None
                    or min_entry_price is None
                    or min_entry_gap_pct is None
                    or max_entry_gap_pct is None
                    or initial_stop_price is None
                ):
                    skipped_missing_bar += 1
                    continue
                if raw_price > max_entry_price:
                    skipped_gap_up += 1
                    continue
                if raw_price < min_entry_price:
                    skipped_gap_down += 1
                    continue
                if raw_price <= initial_stop_price:
                    skipped_broken_support += 1
                    continue
                signal_ema21 = _number(getattr(row, "ema21", None))
                if signal_ema21 is not None and raw_price < signal_ema21:
                    skipped_broken_support += 1
                    continue

                execution_price = raw_price * (1.0 + config.slippage_bps / 10_000.0)
                per_share_total = execution_price * (1.0 + config.commission_bps / 10_000.0)
                filled_plan = rebase_martin_trade_plan_after_fill(
                    trade_plan,
                    fill_price=execution_price,
                )
                if filled_plan.get("fill_status") == "rejected":
                    skipped_technical_stop_too_wide += 1
                    continue
                if filled_plan.get("fill_status") != "confirmed":
                    skipped_missing_bar += 1
                    continue
                filled_exit = (
                    filled_plan.get("exit")
                    if isinstance(filled_plan.get("exit"), dict)
                    else {}
                )
                breakeven_activation_price = _number(
                    filled_exit.get("breakeven_activation_price")
                )
                two_r_reference_price = _number(
                    filled_exit.get("two_r_reference_price")
                )
                if breakeven_activation_price is None or two_r_reference_price is None:
                    skipped_missing_bar += 1
                    continue
                planned_risk_per_share = execution_price - initial_stop_price
                technical_risk_per_share = _number(
                    filled_exit.get("sizing_risk_per_share")
                    or filled_exit.get("effective_risk_per_share")
                )
                adr_stress_per_share = _number(
                    filled_exit.get("adr_stress_per_share")
                ) or 0.0
                if technical_risk_per_share is None or technical_risk_per_share <= 0:
                    skipped_risk_budget += 1
                    continue

                equity_at_open = cash + sum(
                    position.shares * _position_open_value(position, trading_date, bars)
                    for position in positions.values()
                )
                current_heat = sum(
                    position.shares * position.effective_risk_per_share
                    for position in positions.values()
                )
                heat_capacity = max(
                    equity_at_open * config.max_portfolio_heat_pct / 100.0 - current_heat,
                    0.0,
                )
                risk_budget = min(
                    equity_at_open * config.risk_per_trade_pct / 100.0,
                    heat_capacity,
                )
                sizing = (
                    filled_plan.get("position_sizing")
                    if isinstance(filled_plan.get("position_sizing"), dict)
                    else {}
                )
                planned_weight_cap = _number(sizing.get("max_position_weight_pct"))
                weight_cap_pct = min(
                    planned_weight_cap or config.max_position_weight_pct,
                    config.max_position_weight_pct,
                )
                max_shares = min(
                    cash / per_share_total,
                    equity_at_open * weight_cap_pct / 100.0 / execution_price,
                    risk_budget / technical_risk_per_share,
                )
                shares = int(
                    math.floor(max_shares / config.lot_size) * config.lot_size
                )
                if shares <= 0:
                    skipped_risk_budget += 1
                    continue
                notional = shares * execution_price
                fee = notional * config.commission_bps / 10_000.0
                total = notional + fee
                if total > cash + 1e-8:
                    skipped_risk_budget += 1
                    continue
                planned_account_risk_pct = (
                    shares * planned_risk_per_share / equity_at_open * 100.0
                )
                technical_account_risk_pct = (
                    shares * technical_risk_per_share / equity_at_open * 100.0
                )
                adr_stress_account_risk_pct = (
                    shares * adr_stress_per_share / equity_at_open * 100.0
                )
                position_weight_pct = notional / equity_at_open * 100.0
                if position_weight_pct + 1e-8 < config.min_position_weight_pct:
                    skipped_min_position_weight += 1
                    continue
                cash -= total
                close_value = _number(bar.close) or raw_price
                positions[code] = _Position(
                    code=code,
                    name=str(getattr(row, "name", "") or code),
                    shares=shares,
                    entry_date=trading_date,
                    entry_raw_price=raw_price,
                    entry_price=execution_price,
                    entry_notional=notional,
                    entry_fee=fee,
                    entry_total=total,
                    entry_score=float(getattr(row, "screen_score", 0.0) or 0.0),
                    signal_date=pd.Timestamp(getattr(row, "date")).normalize(),
                    signal_close=signal_close,
                    signal_trigger_price=_number(getattr(row, "previous_high", None)),
                    entry_gap_pct=entry_gap_pct,
                    entry_gap_min_pct=min_entry_gap_pct,
                    entry_gap_max_pct=max_entry_gap_pct,
                    entry_gap_volatility_reference_pct=(
                        entry_gap_volatility_reference_pct
                    ),
                    entry_gap_volatility_source=entry_gap_volatility_source,
                    initial_stop_price=initial_stop_price,
                    breakeven_activation_price=breakeven_activation_price,
                    two_r_reference_price=two_r_reference_price,
                    planned_risk_per_share=planned_risk_per_share,
                    effective_risk_per_share=technical_risk_per_share,
                    adr_stress_per_share=adr_stress_per_share,
                    planned_account_risk_pct=planned_account_risk_pct,
                    effective_account_risk_pct=technical_account_risk_pct,
                    adr_stress_account_risk_pct=adr_stress_account_risk_pct,
                    position_weight_pct=position_weight_pct,
                    signal_context=_signal_context(row),
                    highest_close=close_value,
                )
                entry_count += 1

        market_value = 0.0
        for code, position in positions.items():
            bar = bars.get((trading_date, code))
            if bar is None or not _positive(bar.close):
                close_value = position.highest_close or position.entry_raw_price
            else:
                close_value = float(bar.close)
            position.holding_days += 1
            position.highest_close = max(position.highest_close, close_value)
            market_value += position.shares * close_value

            if code in pending_exits:
                continue
            exit_reason = _exit_reason(position, close_value, bar, config=config)
            if exit_reason:
                pending_exits[code] = exit_reason

        equity = cash + market_value
        risk_heat = sum(
            position.shares * position.effective_risk_per_share
            for position in positions.values()
        )
        adr_stress_heat = sum(
            position.shares * position.adr_stress_per_share
            for position in positions.values()
        )
        equity_curve.append({
            "date": _date_text(trading_date),
            "equity": round(equity, 4),
            "cash": round(cash, 4),
            "market_value": round(market_value, 4),
            "positions": len(positions),
            "risk_heat_pct": round(risk_heat / equity * 100.0, 4) if equity > 0 else 0.0,
            "adr_stress_heat_pct": (
                round(adr_stress_heat / equity * 100.0, 4) if equity > 0 else 0.0
            ),
        })

        pending_entries = signal_book.get(trading_date, pd.DataFrame()).copy()

    if dates:
        final_date = dates[-1]
        for code, position in list(positions.items()):
            bar = bars.get((final_date, code))
            raw_price = _number(getattr(bar, "close", None)) if bar is not None else None
            raw_price = raw_price or position.highest_close or position.entry_raw_price
            cash_delta, trade = _close_position(
                position,
                raw_price=raw_price,
                exit_date=final_date,
                reason="period_end_mark",
                config=config,
            )
            cash += cash_delta
            trades.append(trade)
            positions.pop(code, None)
        if equity_curve:
            equity_curve[-1].update({
                "equity": round(cash, 4),
                "cash": round(cash, 4),
                "market_value": 0.0,
                "positions": 0,
                "risk_heat_pct": 0.0,
                "adr_stress_heat_pct": 0.0,
            })

    metrics = _portfolio_metrics(
        equity_curve,
        trades,
        initial_cash=config.initial_cash,
    )
    return {
        "execution_model": "close_signal_next_open_call_auction",
        "config": _config_dict(config),
        "metrics": metrics,
        "diagnostics": {
            "entry_count": entry_count,
            "skipped_limit_up": skipped_limit_up,
            "skipped_gap_up": skipped_gap_up,
            "skipped_gap_down": skipped_gap_down,
            "skipped_broken_support": skipped_broken_support,
            "skipped_technical_stop_too_wide": skipped_technical_stop_too_wide,
            "skipped_min_position_weight": skipped_min_position_weight,
            "skipped_risk_budget": skipped_risk_budget,
            "delayed_limit_down": delayed_limit_down,
            "skipped_reentry_cooldown": skipped_cooldown,
            "skipped_missing_bar": skipped_missing_bar,
            "open_positions_at_end": 0,
            "final_valuation": "last close with estimated sell slippage, commission and stamp duty",
        },
        "equity_curve": equity_curve,
        "trades": trades,
    }


def build_equal_weight_benchmark(panel: pd.DataFrame) -> dict[str, Any]:
    """Build a no-cost equal-weight benchmark from the same cached universe."""
    if panel.empty:
        return {"name": "eligible_universe_equal_weight", "metrics": {}, "equity_curve": []}
    daily = panel.groupby("date", sort=True)["change_pct"].mean().div(100.0)
    if daily.empty:
        return {"name": "eligible_universe_equal_weight", "metrics": {}, "equity_curve": []}
    daily.iloc[0] = 0.0
    curve = (1.0 + daily.fillna(0.0)).cumprod()
    records = [
        {"date": _date_text(index), "equity": round(float(value), 8)}
        for index, value in curve.items()
    ]
    metrics = _curve_metrics(curve)
    return {
        "name": "eligible_universe_equal_weight",
        "metrics": metrics,
        "equity_curve": records,
        "notes": "No fees; same cached universe as the strategy backtest.",
    }


def render_backtest_markdown(payload: Mapping[str, Any]) -> str:
    """Render a stable human-readable research report."""
    period = payload.get("period", {})
    results = payload.get("results", {})
    result = results.get("strategy", {})
    metrics = result.get("metrics", {})
    benchmark = payload.get("benchmark", {})
    data = payload.get("data_diagnostics", {})
    signals = payload.get("signal_diagnostics", {})
    validation = payload.get("source_validation", {})
    validation_summary = validation.get("summary", {}) if isinstance(validation, Mapping) else {}
    full_market = payload.get("universe_mode") == "full-a-share"
    universe_label = "完整当前A股快照等权" if full_market else "当前可交易池等权"
    backtest_config = result.get("config", {})
    lower_override = backtest_config.get("entry_gap_min_pct")
    upper_override = backtest_config.get("entry_gap_max_pct")
    override_note = (
        "本次使用固定覆盖值"
        if lower_override is not None or upper_override is not None
        else "本次未使用固定覆盖值"
    )
    down_fraction = (
        float(backtest_config.get("entry_gap_down_volatility_fraction", 0.30))
        * 100.0
    )
    down_min = float(backtest_config.get("entry_gap_down_min_pct", 1.0))
    down_max = float(backtest_config.get("entry_gap_down_max_pct", 1.5))
    up_fraction = (
        float(backtest_config.get("entry_gap_up_volatility_fraction", 0.50))
        * 100.0
    )
    up_min = float(backtest_config.get("entry_gap_up_min_pct", 1.5))
    up_max = float(backtest_config.get("entry_gap_up_max_pct", 2.0))
    entry_gap_rule_note = (
        "- 次日开盘区间按信号日 ATR20% 与 ADR20% 的均值自适应"
        f"（缺一用另一个）：低开容忍为波动基准的 {down_fraction:.0f}%"
        f"（限制 {down_min:.2f}%～{down_max:.2f}%），"
        f"高开容忍为 {up_fraction:.0f}%"
        f"（限制 {up_min:.2f}%～{up_max:.2f}%）；{override_note}。"
        "开盘还必须守住突破位、EMA21 与成交后技术风险上限。"
    )
    lines = [
        "# Martin A股日线策略回测",
        "",
        f"> 区间：{period.get('start', '')} 至 {period.get('end', '')}；收盘确认信号，次日开盘集合竞价成交。",
        "",
        "## 结果摘要",
        "",
        "| 策略 | 净收益 | 成本前估算 | 最大回撤 | 夏普 | 交易数 | 胜率 | 盈亏比 | 平均技术风险热度 | 平均ADR压力热度 | 估算成本 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| Martin 动量回调 | {ret} | {gross} | {dd} | {sharpe} | {count} | {win} | {wl} | {heat} | {adr_heat} | {cost} |".format(
            ret=_pct(metrics.get("total_return_pct")),
            gross=_pct(metrics.get("estimated_pre_cost_return_pct")),
            dd=_pct(metrics.get("max_drawdown_pct")),
            sharpe=_decimal(metrics.get("sharpe_ratio")),
            count=int(metrics.get("trade_count", 0) or 0),
            win=_pct(metrics.get("win_rate_pct")),
            wl=_decimal(metrics.get("win_loss_ratio")),
            heat=_pct(metrics.get("average_risk_heat_pct")),
            adr_heat=_pct(metrics.get("average_adr_stress_heat_pct")),
            cost=_money(metrics.get("estimated_total_cost")),
        ),
    ]
    benchmark_metrics = benchmark.get("metrics", {})
    lines.extend([
        "",
        "| 同口径基准 | 区间收益 | 最大回撤 | 年化波动 | 夏普 |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| {label}（不计费） | {ret} | {dd} | {vol} | {sharpe} |".format(
            label=universe_label,
            ret=_pct(benchmark_metrics.get("total_return_pct")),
            dd=_pct(benchmark_metrics.get("max_drawdown_pct")),
            vol=_pct(benchmark_metrics.get("annualized_volatility_pct")),
            sharpe=_decimal(benchmark_metrics.get("sharpe_ratio")),
        ),
        "",
        "## 策略与执行规则",
        "",
        "- Martin 形态条件：60 日相对强度前 5%、20 日强度前 30%、EMA9/21/50 多头、ADR20、缩量缩波、支撑共振、距高点与大盘宽度门槛。",
        "- 触发日收盘位置与相对前一日量能用于质量排序，不作为事后拟合的刚性删除条件。",
        "- 只有收盘重新站上前一交易日高点才形成入场信号；突破幅度不设固定上限，改由触发质量排序、次日开盘区间和成交后技术止损控制追高风险；形态合格但未触发的股票只保留观察。",
        "- 不使用分时或盘中高低点；信号日收盘确认，下一交易日开盘集合竞价成交，A股 100 股整数手。",
        entry_gap_rule_note,
        "- 仓位只按技术止损距离、单笔账户风险和组合风险热度反推，同时受单只权重与 A 股 100 股整数手约束；ADR 作为独立压力指标展示，不再压缩计仓数量。",
        "- 实际成交后的技术止损距离不得超过 {stop:.2f}%，整数手取整后的仓位不得低于账户 {minimum:.2f}%，否则放弃该次交易；单只仓位上限 {maximum:.2f}%，组合技术风险热度上限 {heat:.2f}%。".format(
            stop=float(result.get("config", {}).get("max_technical_stop_pct", 3.0)),
            minimum=float(result.get("config", {}).get("min_position_weight_pct", 12.0)),
            maximum=float(result.get("config", {}).get("max_position_weight_pct", 20.0)),
            heat=float(result.get("config", {}).get("max_portfolio_heat_pct", 2.5)),
        ),
        "- 达到 1R 后启用成本防守；跌破有效防守线、10 日收益未达 3% 或持有 20 日时，下一交易日开盘退出。",
        "- 成本：买卖佣金各 3bp、买卖滑点各 5bp、卖出印花税 5bp；期末持仓按最后收盘价扣除预计卖出成本。",
        "- 同一股票退出后冷却 3 个交易日；开盘涨停不假设可以买入，开盘跌停不假设可以卖出。",
        "",
        "## 数据与信号",
        "",
        f"- 本地日线覆盖：{data.get('loaded_codes', 0)} 只，{data.get('panel_rows', 0)} 条区间记录，来源分布 `{data.get('source_counts', {})}`。",
        "- 价格口径：要求 `{required}`，实际 `{actual}`；另有 {excluded} 只不兼容缓存被排除，分布 `{excluded_counts}`。".format(
            required=data.get("required_price_adjustment", ""),
            actual=data.get("price_adjustment_counts", {}),
            excluded=data.get("incompatible_price_adjustment_codes", 0),
            excluded_counts=data.get("incompatible_price_adjustment_counts", {}),
        ),
        "- 区间起点预热覆盖：{ready}/{available} 只具备 120 根历史（{coverage}）；首个严格信号日：{first}。".format(
            ready=data.get("codes_with_120_bars_at_period_start", 0),
            available=data.get("codes_available_at_period_start", 0),
            coverage=_pct(data.get("start_history_coverage_pct")),
            first=signals.get("first_signal_date") or "无",
        ),
        f"- 形态合格 {signals.get('setup_count', 0)} 条；确认入场 {signals.get('entry_signal_count', 0)} 条；未确认观察 {signals.get('unconfirmed_setup_count', 0)} 条。",
        "- AkShare/Baostock 抽样复核：请求 {requested} 只，完成 {validated} 只，其中通过 {ok}、警告 {warning}、失败 {failed}。".format(
            requested=validation_summary.get("requested", 0),
            validated=validation_summary.get("validated", 0),
            ok=validation_summary.get("ok", 0),
            warning=validation_summary.get("warning", 0),
            failed=validation_summary.get("failed", 0),
        ),
        "- 板块历史成分和逐日行业热度尚无可靠点时数据，本次板块因子固定为中性，不执行板块配额，避免把当前板块信息倒灌到历史。",
        "",
        "## 风险执行诊断",
        "",
        "- 亏损尾部：亏损不低于 -5% / -8% / -10% 的交易分别为 {le5} / {le8} / {le10} 笔。".format(
            le5=metrics.get("loss_count_le_5pct", 0),
            le8=metrics.get("loss_count_le_8pct", 0),
            le10=metrics.get("loss_count_le_10pct", 0),
        ),
        "- 初始止损退出 {count} 笔；实际收益相对计划止损平均偏离 {average}，最差偏离 {worst}。负数表示集合竞价退出晚于图表止损。".format(
            count=metrics.get("initial_stop_exit_count", 0),
            average=_pct(metrics.get("average_initial_stop_overshoot_pct")),
            worst=_pct(metrics.get("worst_initial_stop_overshoot_pct")),
        ),
        "- 入场仓位：平均 {average}，中位数 {median}，最小 {minimum}，最大 {maximum}；仅买 100 股的交易 {one_lot} 笔。".format(
            average=_pct(metrics.get("average_entry_position_weight_pct")),
            median=_pct(metrics.get("median_entry_position_weight_pct")),
            minimum=_pct(metrics.get("min_entry_position_weight_pct")),
            maximum=_pct(metrics.get("max_entry_position_weight_pct")),
            one_lot=metrics.get("one_lot_trade_count", 0),
        ),
        "- 放弃交易：技术止损超过上限 {wide} 次，整数手后仓位不足最低有效仓位 {small} 次，风险预算或现金不足 {budget} 次。".format(
            wide=result.get("diagnostics", {}).get("skipped_technical_stop_too_wide", 0),
            small=result.get("diagnostics", {}).get("skipped_min_position_weight", 0),
            budget=result.get("diagnostics", {}).get("skipped_risk_budget", 0),
        ),
        "",
        "## 逐笔交易",
        "",
        "| 代码 | 名称 | 信号日 | 买入日 | 卖出日 | 股数 | 买入价 | 买入总支出 | 卖出价 | 卖出净回款 | 仓位 | 技术风险 | ADR压力 | 计划止损 | 净盈亏 | 净收益 | 退出原因 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    trades = result.get("trades", [])
    if not trades:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 无成交 |")
    for trade in trades:
        shares = float(trade.get("shares") or 0)
        entry_price = float(trade.get("entry_price") or 0)
        exit_price = float(trade.get("exit_price") or 0)
        entry_total = float(
            trade.get("entry_total")
            or shares * entry_price + float(trade.get("entry_fee") or 0)
        )
        exit_proceeds = float(
            trade.get("exit_proceeds")
            or shares * exit_price - float(trade.get("exit_fee") or 0)
        )
        lines.append(
            "| {code} | {name} | {signal} | {entry} | {exit} | {shares} | {entry_price:.3f} | {entry_total:,.2f} | "
            "{exit_price:.3f} | {exit_proceeds:,.2f} | {weight} | {risk} | {adr_stress} | {stop} | {pnl:+,.2f} | {ret} | {reason} |".format(
                code=trade.get("code", ""),
                name=str(trade.get("name", "")).replace("|", "/"),
                signal=trade.get("signal_date", ""),
                entry=trade.get("entry_date", ""),
                exit=trade.get("exit_date", ""),
                shares=int(shares),
                entry_price=entry_price,
                entry_total=entry_total,
                exit_price=exit_price,
                exit_proceeds=exit_proceeds,
                weight=_pct(trade.get("position_weight_pct")),
                risk=_pct(
                    trade.get("technical_account_risk_pct")
                    or trade.get("effective_account_risk_pct")
                ),
                adr_stress=_pct(trade.get("adr_stress_account_risk_pct")),
                stop=_pct(trade.get("planned_stop_return_pct")),
                pnl=float(trade.get("pnl") or 0),
                ret=_pct(trade.get("return_pct")),
                reason=trade.get("exit_reason", ""),
            )
        )
    lines.append("")

    lines.extend([
        "## 解释边界",
        "",
        "- 这是研究回测，不是收益承诺。区间很短，结果对少数交易、成本和退出参数敏感。",
        (
            "- 股票池覆盖期末附近的完整当前 A 股快照，不使用当前成交额、市值和价格预筛；"
            "但历史已退市股票仍缺失，当前 ST 名称按全区间保守排除，历史 ST、总市值和停复牌状态未按日重建。"
            if full_market
            else "- 股票池来自期末附近的当前可交易池，因此存在幸存者偏差；历史 ST、历史总市值、停复牌和退市样本未完整重建。"
        ),
        "- 回测只接受同一前复权口径；不复权或未知口径缓存会被明确排除，不再与前复权序列混算。",
        "- 开盘集合竞价用日线开盘价近似；涨跌停只按开盘价和昨收判断，无法重建委托、排队、撤单和实际成交概率。",
        "- 收盘只用于生成信号和退出判定，不假设能在已经知道收盘结果后按同一收盘价成交。",
        "- 先把这版作为可重复基线；扩大历史窗口并建立点时股票池、板块成分和财务快照后，才能用于更可靠的策略判断。",
        "",
    ])
    return "\n".join(lines)


def _prepare_stock_history(code: str, raw: pd.DataFrame, *, name: str) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = raw.rename(columns=rename_map).copy()
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10), errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date", kind="mergesort")
    df = df.drop_duplicates("date", keep="last").reset_index(drop=True)
    if df.empty:
        return df
    for column in ("open", "high", "low"):
        if column not in df.columns:
            df[column] = df["close"]
        else:
            df[column] = df[column].fillna(df["close"])
    if "volume" not in df.columns:
        df["volume"] = np.nan
    if "amount" not in df.columns:
        df["amount"] = np.nan

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    amount = df["amount"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    daily_range = (high / low - 1.0).replace([np.inf, -np.inf], np.nan) * 100.0
    adr = daily_range.rolling(20, min_periods=1).mean()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(20, min_periods=1).mean() / close * 100.0
    range_latest = daily_range.rolling(3, min_periods=1).mean()
    range_prior = daily_range.shift(3).rolling(10, min_periods=1).mean()
    volume_latest = volume.rolling(3, min_periods=1).mean()
    volume_prior = volume.shift(3).rolling(20, min_periods=1).mean()
    anchored_vwap = _rolling_anchored_vwap(df)
    support_count = (
        _within_pct(close, ema9, 3.0).astype(int)
        + _within_pct(close, ema21, 3.0).astype(int)
        + _within_pct(close, anchored_vwap, 3.0).astype(int)
    )
    previous_high = high.shift(1)

    df["code"] = str(code)
    df["name"] = name or str(code)
    df["price"] = close
    df["previous_close"] = previous_close
    df["previous_high"] = previous_high
    df["change_pct"] = (close / previous_close - 1.0) * 100.0
    df["daily_data_points"] = np.arange(1, len(df) + 1)
    df["change_20d"] = (close / close.shift(20) - 1.0) * 100.0
    df["change_60d"] = (close / close.shift(60) - 1.0) * 100.0
    df["avg_amount_20d"] = amount.rolling(20, min_periods=1).mean()
    df["adr_20_pct"] = adr
    df["atr_20_pct"] = atr
    df["ema9"] = ema9
    df["ema21"] = ema21
    df["ema50"] = ema50
    df["ema_stack_bullish"] = (ema9 > ema21) & (ema21 > ema50)
    df["ema21_rising"] = ema21 > ema21.shift(5)
    df["ema50_rising"] = ema50 > ema50.shift(5)
    df["distance_to_120d_high_pct"] = (close / high.rolling(120, min_periods=1).max() - 1.0) * 100.0
    df["pullback_depth_10d_pct"] = (close / high.rolling(10, min_periods=1).max() - 1.0) * 100.0
    df["range_contraction_ratio"] = range_latest / range_prior.replace(0, np.nan)
    df["volume_contraction_ratio"] = volume_latest / volume_prior.replace(0, np.nan)
    df["anchored_vwap_60d_high"] = anchored_vwap
    df["support_confluence_count"] = support_count
    df["trigger_distance_pct"] = (close / previous_high - 1.0) * 100.0
    intraday_span = (high - low).replace(0, np.nan)
    df["trigger_close_location_pct"] = ((close - low) / intraday_span * 100.0).clip(0, 100)
    df["trigger_volume_vs_previous"] = volume / volume.shift(1).replace(0, np.nan)
    df["one_word_limit_up"] = (
        (high - low).abs().le(high.clip(lower=1.0) * 1e-6)
        & df["change_pct"].ge(9.5)
    )
    df["turnover_rate"] = np.nan
    df["board_heat_score"] = np.nan
    df["sector_breadth_count"] = 0
    return df


def _rolling_anchored_vwap(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    typical = (high + low + close) / 3.0
    output = np.full(len(df), np.nan, dtype=float)
    for end in range(len(df)):
        start = max(0, end - 59)
        window_high = high[start : end + 1]
        if len(window_high) == 0 or np.isnan(window_high).all():
            continue
        anchor = start + int(np.nanargmax(window_high))
        valid = np.isfinite(typical[anchor : end + 1]) & (volume[anchor : end + 1] > 0)
        if not valid.any():
            continue
        window_volume = volume[anchor : end + 1][valid]
        output[end] = float(
            np.sum(typical[anchor : end + 1][valid] * window_volume) / np.sum(window_volume)
        )
    return pd.Series(output, index=df.index, dtype=float)


def _within_pct(price: pd.Series, level: pd.Series, tolerance_pct: float) -> pd.Series:
    ratio = (price / level - 1.0).abs() * 100.0
    return level.gt(0) & ratio.le(tolerance_pct)


def _attach_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["relative_strength_20d_pctile"] = result.groupby("date")["change_20d"].rank(
        method="average", pct=True
    ) * 100.0
    result["relative_strength_60d_pctile"] = result.groupby("date")["change_60d"].rank(
        method="average", pct=True
    ) * 100.0
    grouped_change = result.groupby("date")["change_pct"]
    positive_ratio = grouped_change.transform(lambda values: values.dropna().gt(0).mean() * 100.0)
    median_change = grouped_change.transform("median")
    result["market_positive_ratio"] = positive_ratio
    result["market_median_change_pct"] = median_change
    result["market_above_ema21_ratio"] = (
        result["close"].gt(result["ema21"]).groupby(result["date"]).transform("mean") * 100.0
    )
    result["market_above_ema50_ratio"] = (
        result["close"].gt(result["ema50"]).groupby(result["date"]).transform("mean") * 100.0
    )
    result["market_median_change_20d_pct"] = result.groupby("date")["change_20d"].transform(
        "median"
    )
    return result


def _sort_signals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        ["screen_score", "relative_strength_60d_pctile", "code"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _exit_reason(
    position: _Position,
    close_value: float,
    bar: Any,
    *,
    config: MartinBacktestConfig,
) -> str:
    ema21 = _number(getattr(bar, "ema21", None)) if bar is not None else None
    if close_value <= position.initial_stop_price:
        return "close_below_initial_stop"
    if ema21 is not None and close_value < ema21:
        return "close_below_ema21"
    if (
        position.highest_close >= position.breakeven_activation_price
        and close_value <= position.entry_price
    ):
        return "close_below_breakeven"
    if position.highest_close > 0:
        drawdown_pct = (close_value / position.highest_close - 1.0) * 100.0
        if drawdown_pct <= -config.trailing_drawdown_pct:
            return "trailing_close_drawdown"
    return_pct = (close_value / (position.entry_total / position.shares) - 1.0) * 100.0
    if position.holding_days >= config.follow_through_days and return_pct < config.follow_through_pct:
        return "no_follow_through"
    if position.holding_days >= config.max_holding_days:
        return "max_holding_days"
    return ""


def _close_position(
    position: _Position,
    *,
    raw_price: float,
    exit_date: pd.Timestamp,
    reason: str,
    config: MartinBacktestConfig,
) -> tuple[float, dict[str, Any]]:
    execution_price = raw_price * (1.0 - config.slippage_bps / 10_000.0)
    notional = position.shares * execution_price
    fee_rate = (config.commission_bps + config.sell_stamp_duty_bps) / 10_000.0
    exit_fee = notional * fee_rate
    proceeds = notional - exit_fee
    pnl = proceeds - position.entry_total
    entry_notional = position.shares * position.entry_price
    return_pct = pnl / position.entry_total * 100.0 if position.entry_total else 0.0
    planned_stop_return_pct = (
        (position.initial_stop_price / position.entry_price - 1.0) * 100.0
        if position.entry_price > 0
        else 0.0
    )
    return proceeds, {
        "code": position.code,
        "name": position.name,
        "signal_date": _date_text(position.signal_date),
        "entry_date": _date_text(position.entry_date),
        "exit_date": _date_text(exit_date),
        "holding_days": position.holding_days,
        "shares": position.shares,
        "entry_raw_price": round(position.entry_raw_price, 6),
        "entry_price": round(position.entry_price, 6),
        "exit_raw_price": round(raw_price, 6),
        "exit_price": round(execution_price, 6),
        "entry_notional": round(entry_notional, 4),
        "entry_total": round(position.entry_total, 4),
        "exit_notional": round(notional, 4),
        "exit_proceeds": round(proceeds, 4),
        "entry_fee": round(position.entry_fee, 4),
        "exit_fee": round(exit_fee, 4),
        "pnl": round(pnl, 4),
        "return_pct": round(return_pct, 4),
        "entry_score": round(position.entry_score, 4),
        "signal_close": round(position.signal_close, 6),
        "signal_trigger_price": (
            round(position.signal_trigger_price, 6)
            if position.signal_trigger_price is not None
            else None
        ),
        "entry_gap_pct": round(position.entry_gap_pct, 4),
        "entry_gap_min_pct": round(position.entry_gap_min_pct, 4),
        "entry_gap_max_pct": round(position.entry_gap_max_pct, 4),
        "entry_gap_volatility_reference_pct": (
            round(position.entry_gap_volatility_reference_pct, 4)
            if position.entry_gap_volatility_reference_pct is not None
            else None
        ),
        "entry_gap_volatility_source": position.entry_gap_volatility_source,
        "initial_stop_price": round(position.initial_stop_price, 6),
        "planned_stop_return_pct": round(planned_stop_return_pct, 4),
        "stop_overshoot_pct": round(return_pct - planned_stop_return_pct, 4),
        "planned_risk_per_share": round(position.planned_risk_per_share, 6),
        "sizing_risk_per_share": round(position.effective_risk_per_share, 6),
        "effective_risk_per_share": round(position.effective_risk_per_share, 6),
        "adr_stress_per_share": round(position.adr_stress_per_share, 6),
        "planned_account_risk_pct": round(position.planned_account_risk_pct, 4),
        "technical_account_risk_pct": round(position.effective_account_risk_pct, 4),
        "effective_account_risk_pct": round(position.effective_account_risk_pct, 4),
        "adr_stress_account_risk_pct": round(
            position.adr_stress_account_risk_pct,
            4,
        ),
        "position_weight_pct": round(position.position_weight_pct, 4),
        "breakeven_activation_price": round(position.breakeven_activation_price, 6),
        "two_r_reference_price": round(position.two_r_reference_price, 6),
        "entry_session": "opening_call_auction",
        "exit_session": (
            "period_end_valuation"
            if reason == "period_end_mark"
            else "opening_call_auction"
        ),
        "exit_reason": reason,
        **position.signal_context,
    }


def _signal_context(row: Any) -> dict[str, float | None]:
    fields = (
        "adr_20_pct",
        "atr_20_pct",
        "change_20d",
        "change_60d",
        "relative_strength_20d_pctile",
        "relative_strength_60d_pctile",
        "distance_to_120d_high_pct",
        "pullback_depth_10d_pct",
        "range_contraction_ratio",
        "volume_contraction_ratio",
        "support_confluence_count",
        "trigger_distance_pct",
        "trigger_close_location_pct",
        "trigger_volume_vs_previous",
        "market_positive_ratio",
        "market_median_change_pct",
        "market_above_ema21_ratio",
        "market_above_ema50_ratio",
        "market_median_change_20d_pct",
    )
    return {
        f"signal_{field}": (
            round(value, 6)
            if (value := _number(getattr(row, field, None))) is not None
            else None
        )
        for field in fields
    }


def _position_open_value(
    position: _Position,
    trading_date: pd.Timestamp,
    bars: Mapping[tuple[pd.Timestamp, str], Any],
) -> float:
    bar = bars.get((trading_date, position.code))
    open_price = _number(getattr(bar, "open", None)) if bar is not None else None
    return open_price or position.highest_close or position.entry_raw_price


def _is_open_limit_up(code: str, open_price: float, previous_close: float | None) -> bool:
    if previous_close is None or previous_close <= 0:
        return False
    return (open_price / previous_close - 1.0) * 100.0 >= _limit_threshold_pct(code)


def _is_open_limit_down(code: str, open_price: float, previous_close: float | None) -> bool:
    if previous_close is None or previous_close <= 0:
        return False
    return (open_price / previous_close - 1.0) * 100.0 <= -_limit_threshold_pct(code)


def _limit_threshold_pct(code: str) -> float:
    normalized = str(code).zfill(6)
    if normalized.startswith(("4", "8", "920")):
        return 29.5
    if normalized.startswith(("300", "301", "688", "689")):
        return 19.5
    return 9.5


def _portfolio_metrics(
    equity_curve: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    *,
    initial_cash: float,
) -> dict[str, Any]:
    if not equity_curve:
        return {}
    equity = pd.Series([float(row["equity"]) for row in equity_curve], dtype=float)
    curve_metrics = _curve_metrics(equity / initial_cash)
    returns = [float(trade["return_pct"]) for trade in trades]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    pnl_values = [float(trade["pnl"]) for trade in trades]
    explicit_fees = sum(
        float(trade.get("entry_fee", 0.0)) + float(trade.get("exit_fee", 0.0))
        for trade in trades
    )
    estimated_slippage = sum(
        float(trade.get("shares", 0))
        * (
            float(trade.get("entry_price", 0.0)) - float(trade.get("entry_raw_price", 0.0))
            + float(trade.get("exit_raw_price", 0.0)) - float(trade.get("exit_price", 0.0))
        )
        for trade in trades
    )
    positive_pnl = sum(value for value in pnl_values if value > 0)
    negative_pnl = abs(sum(value for value in pnl_values if value < 0))
    exposure = [
        float(row["market_value"]) / float(row["equity"]) * 100.0
        if float(row["equity"]) > 0
        else 0.0
        for row in equity_curve
    ]
    risk_heat = [float(row.get("risk_heat_pct", 0.0)) for row in equity_curve]
    adr_stress_heat = [
        float(row.get("adr_stress_heat_pct", 0.0)) for row in equity_curve
    ]
    position_weights = [
        float(trade.get("position_weight_pct", 0.0)) for trade in trades
    ]
    initial_stop_trades = [
        trade for trade in trades if trade.get("exit_reason") == "close_below_initial_stop"
    ]
    stop_overshoots = [
        float(trade.get("stop_overshoot_pct", 0.0)) for trade in initial_stop_trades
    ]
    total_cost = explicit_fees + estimated_slippage
    curve_metrics.update({
        "trade_count": len(trades),
        "win_rate_pct": round(len(wins) / len(returns) * 100.0, 4) if returns else 0.0,
        "average_trade_pct": round(float(np.mean(returns)), 4) if returns else 0.0,
        "average_win_pct": round(float(np.mean(wins)), 4) if wins else 0.0,
        "average_loss_pct": round(float(np.mean(losses)), 4) if losses else 0.0,
        "win_loss_ratio": round(float(np.mean(wins) / abs(np.mean(losses))), 4) if wins and losses else None,
        "profit_factor": round(positive_pnl / negative_pnl, 4) if negative_pnl > 0 else None,
        "average_exposure_pct": round(float(np.mean(exposure)), 4) if exposure else 0.0,
        "average_risk_heat_pct": round(float(np.mean(risk_heat)), 4) if risk_heat else 0.0,
        "max_risk_heat_pct": round(float(np.max(risk_heat)), 4) if risk_heat else 0.0,
        "average_adr_stress_heat_pct": (
            round(float(np.mean(adr_stress_heat)), 4) if adr_stress_heat else 0.0
        ),
        "max_adr_stress_heat_pct": (
            round(float(np.max(adr_stress_heat)), 4) if adr_stress_heat else 0.0
        ),
        "average_entry_position_weight_pct": (
            round(float(np.mean(position_weights)), 4) if position_weights else 0.0
        ),
        "median_entry_position_weight_pct": (
            round(float(np.median(position_weights)), 4) if position_weights else 0.0
        ),
        "min_entry_position_weight_pct": (
            round(float(np.min(position_weights)), 4) if position_weights else 0.0
        ),
        "max_entry_position_weight_pct": (
            round(float(np.max(position_weights)), 4) if position_weights else 0.0
        ),
        "one_lot_trade_count": sum(
            int(trade.get("shares", 0)) == 100 for trade in trades
        ),
        "explicit_fees": round(explicit_fees, 4),
        "estimated_slippage_cost": round(estimated_slippage, 4),
        "estimated_total_cost": round(total_cost, 4),
        "estimated_pre_cost_return_pct": round(
            (sum(pnl_values) + total_cost) / initial_cash * 100.0,
            4,
        ),
        "loss_count_le_5pct": sum(value <= -5.0 for value in losses),
        "loss_count_le_8pct": sum(value <= -8.0 for value in losses),
        "loss_count_le_10pct": sum(value <= -10.0 for value in losses),
        "initial_stop_exit_count": len(initial_stop_trades),
        "average_initial_stop_overshoot_pct": round(
            float(np.mean(stop_overshoots)), 4
        ) if stop_overshoots else 0.0,
        "worst_initial_stop_overshoot_pct": round(
            float(np.min(stop_overshoots)), 4
        ) if stop_overshoots else 0.0,
        "final_equity": round(float(equity.iloc[-1]), 4),
    })
    return curve_metrics


def _curve_metrics(curve: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(curve, errors="coerce").dropna()
    if values.empty:
        return {}
    returns = values.pct_change().dropna()
    running_high = values.cummax()
    drawdown = values / running_high - 1.0
    total_return = (float(values.iloc[-1]) / float(values.iloc[0]) - 1.0) * 100.0
    volatility = float(returns.std(ddof=1) * math.sqrt(252) * 100.0) if len(returns) > 1 else 0.0
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else None
    )
    annualized = (
        (float(values.iloc[-1]) / float(values.iloc[0])) ** (252.0 / max(len(values) - 1, 1)) - 1.0
    ) * 100.0
    return {
        "total_return_pct": round(total_return, 4),
        "annualized_return_pct": round(annualized, 4),
        "max_drawdown_pct": round(float(drawdown.min() * 100.0), 4),
        "annualized_volatility_pct": round(volatility, 4),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "trading_days": len(values),
    }


def _count_ready_codes(
    histories: Mapping[str, pd.DataFrame],
    *,
    start_date: date,
    minimum: int,
) -> int:
    ready = 0
    cutoff = pd.Timestamp(start_date)
    for history in histories.values():
        if "date" not in history.columns:
            continue
        dates = pd.to_datetime(history["date"], errors="coerce")
        if int(dates.le(cutoff).sum()) >= minimum:
            ready += 1
    return ready


def _empty_result(config: MartinBacktestConfig, reason: str) -> dict[str, Any]:
    return {
        "execution_model": "close_signal_next_open_call_auction",
        "config": _config_dict(config),
        "metrics": {},
        "diagnostics": {"empty_reason": reason},
        "equity_curve": [],
        "trades": [],
    }


def _config_dict(config: MartinBacktestConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["start_date"] = config.start_date.isoformat()
    payload["end_date"] = config.end_date.isoformat()
    return payload


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> bool:
    number = _number(value)
    return number is not None and number > 0


def _date_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _pct(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.2f}%"


def _decimal(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.2f}"


def _money(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:,.0f}元"
