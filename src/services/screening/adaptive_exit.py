# -*- coding: utf-8 -*-
"""Shared adaptive-exit calculations for Martin holding plans and backtests."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


LEGACY_TIGHTENING_EXIT = "legacy_tightening"
PROFIT_FLOOR_VOLATILITY_EXIT = "profit_floor_volatility"
TRANCHE_COST_PROFIT_FLOOR_EXIT = "tranche_cost_profit_floor_volatility"
PROFIT_FLOOR_EXIT_MODELS = {
    PROFIT_FLOOR_VOLATILITY_EXIT,
    TRANCHE_COST_PROFIT_FLOOR_EXIT,
}
SUPPORTED_ADAPTIVE_EXIT_MODELS = {
    LEGACY_TIGHTENING_EXIT,
    *PROFIT_FLOOR_EXIT_MODELS,
}


@dataclass(frozen=True)
class AdaptiveExitState:
    """Resolved defenses for one end-of-day holding snapshot."""

    stage: str
    peak_r_multiple: float
    trailing_drawdown_pct: float | None
    trailing_stop_price: float | None
    profit_floor_r_multiple: float | None
    profit_floor_price: float | None
    current_r_multiple: float = 0.0
    average_cost_price: float | None = None
    net_breakeven_price: float | None = None
    position_high_water_price: float | None = None
    tranche_states: tuple[dict[str, Any], ...] = ()


def resolve_profit_floor_volatility_exit(
    *,
    entry_price: float,
    initial_stop_price: float,
    highest_close: float,
    adr_20_pct: Any,
    developing_activation_r: float,
    established_activation_r: float,
    mature_activation_r: float,
    developing_adr_multiple: float,
    developing_drawdown_min_pct: float,
    developing_drawdown_max_pct: float,
    established_adr_multiple: float,
    established_drawdown_min_pct: float,
    established_drawdown_max_pct: float,
    mature_adr_multiple: float,
    mature_drawdown_min_pct: float,
    mature_drawdown_max_pct: float,
    established_profit_floor_r: float,
    mature_profit_floor_r: float,
    fallback_drawdown_pct: float,
) -> AdaptiveExitState:
    """Widen peak tolerance as a winner matures while ratcheting its R floor."""
    initial_risk = max(float(entry_price) - float(initial_stop_price), 0.0)
    peak_r = (
        (float(highest_close) - float(entry_price)) / initial_risk
        if initial_risk > 0
        else 0.0
    )
    if peak_r < developing_activation_r:
        return AdaptiveExitState(
            stage="initial",
            peak_r_multiple=peak_r,
            trailing_drawdown_pct=None,
            trailing_stop_price=None,
            profit_floor_r_multiple=None,
            profit_floor_price=None,
        )

    adr = _positive_number(adr_20_pct)
    if peak_r >= mature_activation_r:
        stage = "mature"
        adr_multiple = mature_adr_multiple
        lower = mature_drawdown_min_pct
        upper = mature_drawdown_max_pct
        floor_r = mature_profit_floor_r
    elif peak_r >= established_activation_r:
        stage = "established"
        adr_multiple = established_adr_multiple
        lower = established_drawdown_min_pct
        upper = established_drawdown_max_pct
        floor_r = established_profit_floor_r
    else:
        stage = "developing"
        adr_multiple = developing_adr_multiple
        lower = developing_drawdown_min_pct
        upper = developing_drawdown_max_pct
        floor_r = 0.0

    raw_drawdown = adr * adr_multiple if adr is not None else fallback_drawdown_pct
    trailing_pct = _clip(raw_drawdown, lower, upper)
    trailing_stop = float(highest_close) * (1.0 - trailing_pct / 100.0)
    profit_floor = float(entry_price) + floor_r * initial_risk
    return AdaptiveExitState(
        stage=stage,
        peak_r_multiple=peak_r,
        trailing_drawdown_pct=trailing_pct,
        trailing_stop_price=trailing_stop,
        profit_floor_r_multiple=floor_r,
        profit_floor_price=profit_floor,
    )


def resolve_tranche_cost_profit_floor_exit(
    *,
    tranches: Any,
    current_close: float,
    highest_close: float,
    highest_position_r_multiple: Any,
    adr_20_pct: Any,
    developing_activation_r: float,
    established_activation_r: float,
    mature_activation_r: float,
    developing_adr_multiple: float,
    developing_drawdown_min_pct: float,
    developing_drawdown_max_pct: float,
    established_adr_multiple: float,
    established_drawdown_min_pct: float,
    established_drawdown_max_pct: float,
    mature_adr_multiple: float,
    mature_drawdown_min_pct: float,
    mature_drawdown_max_pct: float,
    established_profit_floor_r: float,
    mature_profit_floor_r: float,
    fallback_drawdown_pct: float,
    exit_retention_rate: float = 1.0,
) -> AdaptiveExitState:
    """Resolve an adaptive exit from fee-aware, remaining tranche economics.

    Each tranche contributes its remaining cost and original technical risk.
    Position R is therefore based on the whole live position, not the first
    fill. The profit floor is the raw exit price required to preserve the
    requested aggregate R after estimated exit friction.
    """
    close = _positive_number(current_close)
    peak_close = _positive_number(highest_close)
    retention = _number(exit_retention_rate)
    if retention is None or not 0 < retention <= 1:
        retention = 1.0

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(tranches or []):
        if not isinstance(item, dict):
            continue
        shares = _positive_number(
            item.get("remaining_shares", item.get("shares"))
        )
        entry_price = _positive_number(item.get("price", item.get("entry_price")))
        cost_per_share = _positive_number(
            item.get("cost_per_share", item.get("entry_cost_per_share"))
        ) or entry_price
        stop_price = _positive_number(
            item.get("protective_stop_price", item.get("initial_stop_price"))
        )
        risk_per_share = _positive_number(item.get("risk_per_share"))
        if risk_per_share is None and cost_per_share is not None and stop_price is not None:
            risk_per_share = max(cost_per_share - stop_price, 0.01)
        if None in (shares, entry_price, cost_per_share, risk_per_share):
            continue
        tranche_high = _positive_number(item.get("highest_close")) or peak_close
        tranche_high = max(tranche_high or entry_price, entry_price)
        tranche_current_r = (
            (close - cost_per_share) / risk_per_share
            if close is not None
            else 0.0
        )
        tranche_peak_r = (tranche_high - cost_per_share) / risk_per_share
        normalized.append({
            "index": int(item.get("tranche_index", index)),
            "type": str(item.get("type") or "entry"),
            "shares": shares,
            "entry_price": entry_price,
            "cost_per_share": cost_per_share,
            "protective_stop_price": stop_price,
            "risk_per_share": risk_per_share,
            "highest_close": tranche_high,
            "current_r_multiple": tranche_current_r,
            "peak_r_multiple": tranche_peak_r,
        })

    if not normalized or close is None:
        return AdaptiveExitState(
            stage="initial",
            peak_r_multiple=0.0,
            trailing_drawdown_pct=None,
            trailing_stop_price=None,
            profit_floor_r_multiple=None,
            profit_floor_price=None,
        )

    total_shares = sum(item["shares"] for item in normalized)
    total_cost = sum(
        item["shares"] * item["cost_per_share"] for item in normalized
    )
    total_risk = sum(
        item["shares"] * item["risk_per_share"] for item in normalized
    )
    average_cost = total_cost / total_shares
    net_breakeven = average_cost / retention
    current_r = (
        (close * total_shares - total_cost) / total_risk
        if total_risk > 0
        else 0.0
    )
    recorded_peak_r = _number(highest_position_r_multiple)
    fallback_peak_r = (
        (peak_close * total_shares - total_cost) / total_risk
        if peak_close is not None and total_risk > 0
        else current_r
    )
    peak_r = (
        recorded_peak_r
        if recorded_peak_r is not None
        else max(fallback_peak_r, current_r)
    )
    position_high_water = (
        (total_cost + peak_r * total_risk) / total_shares
        if total_risk > 0
        else close
    )

    if peak_r < developing_activation_r:
        return AdaptiveExitState(
            stage="initial",
            peak_r_multiple=peak_r,
            trailing_drawdown_pct=None,
            trailing_stop_price=None,
            profit_floor_r_multiple=None,
            profit_floor_price=None,
            current_r_multiple=current_r,
            average_cost_price=average_cost,
            net_breakeven_price=net_breakeven,
            position_high_water_price=position_high_water,
            tranche_states=tuple(normalized),
        )

    adr = _positive_number(adr_20_pct)
    if peak_r >= mature_activation_r:
        stage = "mature"
        adr_multiple = mature_adr_multiple
        lower = mature_drawdown_min_pct
        upper = mature_drawdown_max_pct
        floor_r = mature_profit_floor_r
    elif peak_r >= established_activation_r:
        stage = "established"
        adr_multiple = established_adr_multiple
        lower = established_drawdown_min_pct
        upper = established_drawdown_max_pct
        floor_r = established_profit_floor_r
    else:
        stage = "developing"
        adr_multiple = developing_adr_multiple
        lower = developing_drawdown_min_pct
        upper = developing_drawdown_max_pct
        floor_r = None

    raw_drawdown = adr * adr_multiple if adr is not None else fallback_drawdown_pct
    trailing_pct = _clip(raw_drawdown, lower, upper)
    trailing_stop = position_high_water * (1.0 - trailing_pct / 100.0)
    profit_floor = None
    if floor_r is not None:
        floor_value = total_cost + floor_r * total_risk
        profit_floor = floor_value / total_shares / retention
    return AdaptiveExitState(
        stage=stage,
        peak_r_multiple=peak_r,
        trailing_drawdown_pct=trailing_pct,
        trailing_stop_price=trailing_stop,
        profit_floor_r_multiple=floor_r,
        profit_floor_price=profit_floor,
        current_r_multiple=current_r,
        average_cost_price=average_cost,
        net_breakeven_price=net_breakeven,
        position_high_water_price=position_high_water,
        tranche_states=tuple(normalized),
    )


def is_profit_exhaustion_bar(
    bar: Any,
    *,
    adr_20_pct: Any,
    volume_ratio_min: float,
    close_location_max_pct: float,
    range_adr_multiple_min: float,
    weak_sector_score_max: float,
) -> bool:
    """Detect a conservative daily-bar exhaustion proxy without intraday data."""
    if bar is None:
        return False
    sector_status = str(getattr(bar, "sector_data_status", "") or "")
    sector_score = _number(getattr(bar, "sector_strength_score", None))
    if (
        sector_status == "available"
        and sector_score is not None
        and sector_score < weak_sector_score_max
    ):
        return True

    high = _positive_number(getattr(bar, "high", None))
    low = _positive_number(getattr(bar, "low", None))
    close = _positive_number(getattr(bar, "close", None))
    volume_ratio = _number(getattr(bar, "trigger_volume_vs_previous", None))
    adr = _positive_number(adr_20_pct)
    if None in (high, low, close, volume_ratio, adr) or high <= low:
        return False
    close_location = (close - low) / (high - low) * 100.0
    daily_range_pct = (high / low - 1.0) * 100.0
    return bool(
        volume_ratio >= volume_ratio_min
        and close_location <= close_location_max_pct
        and daily_range_pct >= adr * range_adr_multiple_min
    )


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_number(value: Any) -> float | None:
    result = _number(value)
    return result if result is not None and result > 0 else None


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))
