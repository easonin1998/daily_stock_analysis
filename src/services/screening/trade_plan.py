# -*- coding: utf-8 -*-
"""Deterministic Martin trade plans for daily, auction-only execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any

from src.services.screening.adaptive_exit import (
    LEGACY_TIGHTENING_EXIT,
    PROFIT_FLOOR_EXIT_MODELS,
    PROFIT_FLOOR_VOLATILITY_EXIT,
    TRANCHE_COST_PROFIT_FLOOR_EXIT,
    resolve_profit_floor_volatility_exit,
    resolve_tranche_cost_profit_floor_exit,
)


@dataclass(frozen=True)
class MartinTradePlanConfig:
    """Risk and execution rules shared by live plans and research backtests."""

    # Exact overrides remain available for controlled research comparisons.
    # Normal execution derives asymmetric bounds from recent volatility.
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
    support_buffer_adr_fraction: float = 0.15
    support_buffer_min_pct: float = 0.5
    support_buffer_max_pct: float = 1.0
    initial_risk_adr_fraction: float = 0.75
    initial_risk_min_pct: float = 3.0
    initial_risk_max_pct: float = 6.0
    breakeven_r_multiple: float = 1.0
    first_profit_r_multiple: float = 2.0
    trailing_drawdown_pct: float = 8.0
    follow_through_days: int = 10
    follow_through_pct: float = 3.0
    # Disabled by default: winners are held until the trend-based defenses fail.
    # A positive value is retained only for controlled research comparisons.
    max_holding_days: int | None = None
    max_portfolio_risk_per_trade_pct: float = 0.5
    max_portfolio_heat_pct: float = 2.5
    max_position_weight_pct: float = 20.0
    max_technical_stop_pct: float = 3.0
    min_position_weight_pct: float = 12.0
    pyramiding_enabled: bool = False
    adaptive_exit_enabled: bool = False
    initial_position_weight_pct: float = 20.0
    add_on_position_weight_pct: float = 5.0
    max_add_ons: int = 2
    first_add_on_r_multiple: float = 1.0
    second_add_on_r_multiple: float = 2.0
    first_partial_r_multiple: float = 3.0
    second_partial_r_multiple: float = 5.0
    partial_exit_fraction: float = 0.20
    adaptive_trailing_drawdown_pct: float = 6.0
    mature_trailing_drawdown_pct: float = 5.0
    adaptive_exit_model: str = LEGACY_TIGHTENING_EXIT
    developing_exit_activation_r: float = 1.0
    developing_trailing_adr_multiple: float = 1.0
    developing_trailing_min_pct: float = 5.0
    developing_trailing_max_pct: float = 8.0
    established_trailing_adr_multiple: float = 1.5
    established_trailing_min_pct: float = 7.0
    established_trailing_max_pct: float = 12.0
    mature_trailing_adr_multiple: float = 1.75
    mature_trailing_min_pct: float = 9.0
    mature_trailing_max_pct: float = 15.0
    established_profit_floor_r: float = 1.0
    mature_profit_floor_r: float = 2.5
    first_partial_exit_fraction: float = 0.0
    second_partial_exit_fraction: float = 0.15
    partial_exit_requires_exhaustion: bool = True
    ema9_confirmation_days: int = 2
    risk_off_ema9_confirmation_days: int = 1
    adr_stress_fraction: float = 1.0
    lot_size: int = 100


def build_martin_trade_plan(
    *,
    price: Any,
    trigger_distance_pct: Any,
    adr_20_pct: Any,
    ema21: Any,
    atr_20_pct: Any = None,
    anchored_vwap: Any = None,
    config: MartinTradePlanConfig | None = None,
) -> dict[str, Any]:
    """Build an actionable plan without using data after the signal close."""
    rules = config or MartinTradePlanConfig()
    close = _positive_number(price)
    trigger_distance = _number(trigger_distance_pct)
    adr = _positive_number(adr_20_pct)
    atr = _positive_number(atr_20_pct)
    ema21_value = _positive_number(ema21)
    anchored_vwap_value = _positive_number(anchored_vwap)
    if close is None or trigger_distance is None or adr is None or ema21_value is None:
        return {
            "schema_version": 1,
            "signal_status": "data_insufficient",
            "action": "wait",
            "reason": "缺少收盘价、触发距离、ADR20 或 EMA21，不能生成条件单价位。",
            "execution_model": "daily_close_decision_next_open_call_auction",
        }

    trigger_price = close / (1.0 + trigger_distance / 100.0)
    confirmed = trigger_distance >= 0.0
    entry_gap = resolve_martin_entry_gap_bounds(
        adr_20_pct=adr,
        atr_20_pct=atr,
        config=rules,
    )
    min_entry = max(
        trigger_price,
        ema21_value,
        close * (1.0 + entry_gap["min_gap_pct"] / 100.0),
    )
    max_entry = close * (1.0 + entry_gap["max_gap_pct"] / 100.0)

    support_candidates = [trigger_price, ema21_value]
    if anchored_vwap_value is not None and anchored_vwap_value <= close * 1.01:
        support_candidates.append(anchored_vwap_value)
    structural_support = max(level for level in support_candidates if level > 0)
    support_buffer_pct = _clip(
        adr * rules.support_buffer_adr_fraction,
        rules.support_buffer_min_pct,
        rules.support_buffer_max_pct,
    )
    structure_stop = structural_support * (1.0 - support_buffer_pct / 100.0)
    initial_risk_pct = _clip(
        adr * rules.initial_risk_adr_fraction,
        rules.initial_risk_min_pct,
        rules.initial_risk_max_pct,
    )
    volatility_stop = close * (1.0 - initial_risk_pct / 100.0)
    initial_stop = min(max(structure_stop, volatility_stop), close * 0.995)
    risk_per_share = max(close - initial_stop, close * 0.005)
    initial_risk_pct = risk_per_share / close * 100.0
    adr_stress_pct = adr * rules.adr_stress_fraction
    adr_stress_per_share = close * adr_stress_pct / 100.0
    sizing_risk_pct = initial_risk_pct
    sizing_risk_per_share = risk_per_share
    position_weight_cap = _position_weight_cap(sizing_risk_pct, rules)
    technical_risk_eligible = sizing_risk_pct <= rules.max_technical_stop_pct
    breakeven_activation = close + rules.breakeven_r_multiple * risk_per_share
    two_r_reference = close + rules.first_profit_r_multiple * risk_per_share

    if not confirmed:
        signal_status = "watch_wait_trigger"
        action = "no_order"
    elif not technical_risk_eligible:
        signal_status = "risk_rejected"
        action = "no_order"
    else:
        signal_status = "buy_signal"
        action = "place_next_open_auction_order"

    return {
        "schema_version": 1,
        "signal_status": signal_status,
        "action": action,
        "risk_rejection_reason": (
            f"技术止损距离 {sizing_risk_pct:.2f}% 超过 {rules.max_technical_stop_pct:.2f}% 上限。"
            if confirmed and not technical_risk_eligible
            else ""
        ),
        "execution_model": "daily_close_decision_next_open_call_auction",
        "reference_close": _price(close),
        "entry": {
            "session": "next_open_call_auction",
            "trigger_price": _price(trigger_price),
            "min_price": _price(min_entry),
            "max_price": _price(max_entry),
            "gap_model": entry_gap["model"],
            "gap_volatility_source": entry_gap["volatility_source"],
            "gap_volatility_reference_pct": entry_gap["volatility_reference_pct"],
            "min_gap_pct": entry_gap["min_gap_pct"],
            "max_gap_pct": entry_gap["max_gap_pct"],
            "cancel_if": [
                "开盘价低于最低可接受价或信号日 EMA21",
                "开盘价高于最高可接受价，避免跳空追涨",
                "开盘涨停或无法确认成交",
            ],
        },
        "exit": {
            "decision_time": "after_daily_close",
            "execution_session": "next_open_call_auction",
            "pricing_status": "provisional_until_fill",
            "reference_entry_price": _price(close),
            "initial_stop_price": _price(initial_stop),
            "initial_risk_per_share": _price(risk_per_share),
            "initial_risk_pct": round(initial_risk_pct, 4),
            "sizing_risk_per_share": _price(sizing_risk_per_share),
            "sizing_risk_pct": round(sizing_risk_pct, 4),
            "adr_stress_per_share": _price(adr_stress_per_share),
            "adr_stress_pct": round(adr_stress_pct, 4),
            "adr_stress_affects_position_size": False,
            # Compatibility aliases: effective risk now means technical sizing risk.
            "effective_risk_per_share": _price(sizing_risk_per_share),
            "effective_risk_pct": round(sizing_risk_pct, 4),
            "structural_support_price": _price(structural_support),
            "breakeven_activation_price": _price(breakeven_activation),
            "first_profit_price": _price(two_r_reference),
            "two_r_reference_price": _price(two_r_reference),
            "fixed_take_profit": False,
            "adaptive_exit_model": rules.adaptive_exit_model,
            "ema9_profit_protection_after_r": (
                rules.developing_exit_activation_r
                if rules.adaptive_exit_model in PROFIT_FLOOR_EXIT_MODELS
                else rules.first_partial_r_multiple
            ),
            "partial_profit_levels": _partial_profit_levels(rules),
            "trailing_drawdown_pct": rules.trailing_drawdown_pct,
            "profit_floor_schedule": (
                [
                    {
                        "activation_r": rules.first_partial_r_multiple,
                        "floor_r": rules.established_profit_floor_r,
                    },
                    {
                        "activation_r": rules.second_partial_r_multiple,
                        "floor_r": rules.mature_profit_floor_r,
                    },
                ]
                if rules.adaptive_exit_enabled
                and rules.adaptive_exit_model in PROFIT_FLOOR_EXIT_MODELS
                else []
            ),
            "follow_through_days": rules.follow_through_days,
            "follow_through_min_return_pct": rules.follow_through_pct,
            "holding_period_mode": "trend_following",
            "max_holding_days": rules.max_holding_days,
        },
        "holding_premises": [
            "每日收盘不低于有效防守价：初始止损、EMA21、最高收盘回撤线和R利润底线取最高",
            f"持有 {rules.follow_through_days} 个交易日时收益至少达到 {rules.follow_through_pct:.1f}%",
            "EMA21 与 EMA50 维持上行，趋势结构未明显破坏",
            "板块共振和相对强度没有持续转弱",
            "不设固定持有期限；趋势与移动防线未破坏时继续持有",
        ],
        "position_sizing": {
            "pricing_status": "provisional_until_fill",
            "reference_price": _price(close),
            "max_position_weight_pct": round(position_weight_cap, 4),
            "min_position_weight_pct": rules.min_position_weight_pct,
            "max_portfolio_risk_per_trade_pct": rules.max_portfolio_risk_per_trade_pct,
            "max_portfolio_heat_pct": rules.max_portfolio_heat_pct,
            "max_technical_stop_pct": rules.max_technical_stop_pct,
            "risk_basis": "technical_stop",
            "risk_basis_pct": round(sizing_risk_pct, 4),
            "adr_stress_pct": round(adr_stress_pct, 4),
            "adr_stress_affects_position_size": False,
            "lot_size": rules.lot_size,
            "entry_mode": "trial_then_add" if rules.pyramiding_enabled else "single_entry",
            "initial_position_weight_pct": rules.initial_position_weight_pct,
            "add_on_position_weight_pct": rules.add_on_position_weight_pct,
            "max_add_ons": rules.max_add_ons if rules.pyramiding_enabled else 0,
            "add_only_when_profitable": rules.pyramiding_enabled,
            "quantity_formula": (
                "floor(min(account_equity * max_weight_pct / 100 / fill_price, "
                "available_risk_budget / technical_risk_per_share) / lot_size) * lot_size"
            ),
        },
        "condition_orders": {
            "entry": {
                "status": "ready" if signal_status == "buy_signal" else "not_armed",
                "session": "next_open_call_auction",
                "allowed_price_range": [_price(min_entry), _price(max_entry)],
                "requires_open_price_range_condition": True,
            },
            "protective_exit": {
                "trigger_basis": "daily_close",
                "operator": "less_than_or_equal",
                "trigger_price": _price(initial_stop),
                "submit_session": "next_open_call_auction",
            },
            "profit_management": {
                "arm_breakeven_after_close_at": _price(breakeven_activation),
                "two_r_review_price": _price(two_r_reference),
                "fixed_take_profit": False,
                "adaptive_exit_model": rules.adaptive_exit_model,
                "partial_profit_levels": _partial_profit_levels(rules),
                "partial_profit_r_multiples": [
                    level["r_multiple"] for level in _partial_profit_levels(rules)
                ],
                "partial_exit_fraction": (
                    rules.second_partial_exit_fraction
                    if rules.adaptive_exit_model in PROFIT_FLOOR_EXIT_MODELS
                    else rules.partial_exit_fraction
                ) if rules.adaptive_exit_enabled else 0.0,
                "ema9_confirmation_days": rules.ema9_confirmation_days,
                "risk_off_ema9_confirmation_days": (
                    rules.risk_off_ema9_confirmation_days
                ),
            },
        },
        "condition_order_note": (
            "券商条件单必须按收盘确认、下一开盘集合竞价执行设置；"
            "若券商只能按盘中触价触发，其成交语义与本策略及回测不同。"
            "1R/2R 在实际成交前只是参考值，成交后必须按真实成交价重算。"
        ),
        "parameters": asdict(rules),
    }


def resolve_martin_entry_gap_bounds(
    *,
    adr_20_pct: Any,
    atr_20_pct: Any = None,
    config: MartinTradePlanConfig | None = None,
) -> dict[str, Any]:
    """Resolve asymmetric next-open bounds from point-in-time volatility."""
    rules = config or MartinTradePlanConfig()
    adr = _positive_number(adr_20_pct)
    atr = _positive_number(atr_20_pct)
    volatility_values = [value for value in (atr, adr) if value is not None]
    if len(volatility_values) == 2:
        volatility_reference = sum(volatility_values) / 2.0
        volatility_source = "atr20_adr20_mean"
    elif atr is not None:
        volatility_reference = atr
        volatility_source = "atr20"
    elif adr is not None:
        volatility_reference = adr
        volatility_source = "adr20"
    else:
        volatility_reference = None
        volatility_source = "fallback"

    if volatility_reference is None:
        dynamic_min = rules.entry_gap_fallback_min_pct
        dynamic_max = rules.entry_gap_fallback_max_pct
    else:
        down_tolerance = _clip(
            volatility_reference * rules.entry_gap_down_volatility_fraction,
            rules.entry_gap_down_min_pct,
            rules.entry_gap_down_max_pct,
        )
        up_tolerance = _clip(
            volatility_reference * rules.entry_gap_up_volatility_fraction,
            rules.entry_gap_up_min_pct,
            rules.entry_gap_up_max_pct,
        )
        dynamic_min = -down_tolerance
        dynamic_max = up_tolerance

    minimum = (
        float(rules.entry_gap_min_pct)
        if rules.entry_gap_min_pct is not None
        else dynamic_min
    )
    maximum = (
        float(rules.entry_gap_max_pct)
        if rules.entry_gap_max_pct is not None
        else dynamic_max
    )
    if minimum > maximum:
        raise ValueError("entry gap minimum cannot exceed maximum")
    model = (
        "fixed_override"
        if rules.entry_gap_min_pct is not None and rules.entry_gap_max_pct is not None
        else "volatility_adaptive_with_override"
        if rules.entry_gap_min_pct is not None or rules.entry_gap_max_pct is not None
        else "volatility_adaptive"
    )
    return {
        "model": model,
        "volatility_source": volatility_source,
        "volatility_reference_pct": (
            round(volatility_reference, 4)
            if volatility_reference is not None
            else None
        ),
        "min_gap_pct": round(minimum, 4),
        "max_gap_pct": round(maximum, 4),
    }


def rebase_martin_trade_plan_after_fill(
    plan: dict[str, Any],
    *,
    fill_price: Any,
) -> dict[str, Any]:
    """Finalize risk and R-multiple levels from the actual auction fill."""
    result = deepcopy(plan)
    fill = _positive_number(fill_price)
    exit_plan = result.get("exit") if isinstance(result.get("exit"), dict) else {}
    initial_stop = _positive_number(exit_plan.get("initial_stop_price"))
    if fill is None or initial_stop is None or fill <= initial_stop:
        return {
            **result,
            "fill_status": "invalid",
            "fill_action": "manual_review",
            "fill_reason": "实际成交价无效、缺少初始防守价，或成交价未高于防守价。",
        }

    parameters = result.get("parameters") if isinstance(result.get("parameters"), dict) else {}
    rules = MartinTradePlanConfig(
        **{
            key: value
            for key, value in parameters.items()
            if key in MartinTradePlanConfig.__dataclass_fields__
        }
    )
    one_r_multiple = _positive_number(parameters.get("breakeven_r_multiple")) or 1.0
    two_r_multiple = _positive_number(parameters.get("first_profit_r_multiple")) or 2.0
    risk_per_share = fill - initial_stop
    initial_risk_pct = risk_per_share / fill * 100.0
    adr_stress_pct = _number(exit_plan.get("adr_stress_pct")) or 0.0
    adr_stress_per_share = fill * adr_stress_pct / 100.0
    sizing_risk_pct = initial_risk_pct
    sizing_risk_per_share = risk_per_share
    breakeven_activation = fill + one_r_multiple * risk_per_share
    two_r_reference = fill + two_r_multiple * risk_per_share
    exit_plan.update({
        "pricing_status": "finalized_from_actual_fill",
        "reference_entry_price": _price(fill),
        "initial_risk_per_share": _price(risk_per_share),
        "initial_risk_pct": round(initial_risk_pct, 4),
        "sizing_risk_per_share": _price(sizing_risk_per_share),
        "sizing_risk_pct": round(sizing_risk_pct, 4),
        "adr_stress_per_share": _price(adr_stress_per_share),
        "effective_risk_per_share": _price(sizing_risk_per_share),
        "effective_risk_pct": round(sizing_risk_pct, 4),
        "breakeven_activation_price": _price(breakeven_activation),
        "first_profit_price": _price(two_r_reference),
        "two_r_reference_price": _price(two_r_reference),
    })
    result["exit"] = exit_plan
    technical_risk_eligible = sizing_risk_pct <= rules.max_technical_stop_pct
    result["fill_status"] = "confirmed" if technical_risk_eligible else "rejected"
    result["fill_action"] = "keep_entry" if technical_risk_eligible else "cancel_entry"
    result["fill_reason"] = (
        ""
        if technical_risk_eligible
        else (
            f"实际成交后的技术止损距离 {sizing_risk_pct:.2f}% 超过 "
            f"{rules.max_technical_stop_pct:.2f}% 上限。"
        )
    )
    result["actual_fill_price"] = _price(fill)
    sizing = (
        result.get("position_sizing")
        if isinstance(result.get("position_sizing"), dict)
        else {}
    )
    sizing.update({
        "pricing_status": "finalized_from_actual_fill",
        "reference_price": _price(fill),
        "max_position_weight_pct": round(
            _position_weight_cap(sizing_risk_pct, rules),
            4,
        ),
        "min_position_weight_pct": rules.min_position_weight_pct,
        "max_portfolio_risk_per_trade_pct": rules.max_portfolio_risk_per_trade_pct,
        "max_portfolio_heat_pct": rules.max_portfolio_heat_pct,
        "max_technical_stop_pct": rules.max_technical_stop_pct,
        "risk_basis": "technical_stop",
        "risk_basis_pct": round(sizing_risk_pct, 4),
        "adr_stress_pct": round(adr_stress_pct, 4),
        "adr_stress_affects_position_size": False,
    })
    result["position_sizing"] = sizing

    orders = result.get("condition_orders") if isinstance(result.get("condition_orders"), dict) else {}
    profit = orders.get("profit_management") if isinstance(orders.get("profit_management"), dict) else {}
    profit.update({
        "arm_breakeven_after_close_at": _price(breakeven_activation),
        "two_r_review_price": _price(two_r_reference),
    })
    orders["profit_management"] = profit
    result["condition_orders"] = orders
    return result


def update_martin_holding_plan(
    *,
    entry_price: Any,
    initial_stop_price: Any,
    current_close: Any,
    highest_close: Any,
    ema21: Any,
    ema9: Any = None,
    market_regime_state: str = "",
    partial_exit_count: int = 0,
    previous_ema9_below_streak: int = 0,
    adr_20_pct: Any = None,
    exhaustion_signal: bool = False,
    holding_days: int,
    breakeven_activation_price: Any = None,
    entry_tranches: list[dict[str, Any]] | None = None,
    highest_position_r_multiple: Any = None,
    estimated_exit_cost_bps: float = 0.0,
    config: MartinTradePlanConfig | None = None,
) -> dict[str, Any]:
    """Update the next-auction action from one end-of-day holding snapshot."""
    rules = config or MartinTradePlanConfig()
    entry = _positive_number(entry_price)
    initial_stop = _positive_number(initial_stop_price)
    close = _positive_number(current_close)
    high_close = _positive_number(highest_close)
    ema21_value = _positive_number(ema21)
    if None in (entry, initial_stop, close, high_close, ema21_value):
        return {
            "status": "data_insufficient",
            "action": "manual_review",
            "reason": "持仓更新缺少成交价、初始止损、收盘价、最高收盘价或 EMA21。",
        }

    activation = _positive_number(breakeven_activation_price)
    if activation is None:
        activation = entry + rules.breakeven_r_multiple * (entry - initial_stop)
    initial_risk = entry - initial_stop
    peak_r = (high_close - entry) / initial_risk if initial_risk > 0 else 0.0
    current_r = (close - entry) / initial_risk if initial_risk > 0 else 0.0
    average_cost_price = entry
    net_breakeven_price = entry
    position_high_water_price = high_close
    tranche_states: tuple[dict[str, Any], ...] = ()
    breakeven_active = high_close >= activation
    exit_stage = "legacy"
    profit_floor_r: float | None = None
    profit_floor_price: float | None = None
    trailing_pct: float | None = rules.trailing_drawdown_pct
    trailing_stop: float | None = high_close * (1.0 - trailing_pct / 100.0)
    if (
        rules.adaptive_exit_enabled
        and rules.adaptive_exit_model == TRANCHE_COST_PROFIT_FLOOR_EXIT
    ):
        tranches = [dict(item) for item in (entry_tranches or []) if isinstance(item, dict)]
        if not tranches:
            tranches = [{
                "tranche_index": 0,
                "type": "initial",
                "shares": 1.0,
                "remaining_shares": 1.0,
                "price": entry,
                "cost_per_share": entry,
                "protective_stop_price": initial_stop,
                "risk_per_share": max(entry - initial_stop, 0.01),
                "highest_close": high_close,
            }]
        retention = max(1.0 - max(float(estimated_exit_cost_bps), 0.0) / 10_000.0, 0.0001)
        adaptive = resolve_tranche_cost_profit_floor_exit(
            tranches=tranches,
            current_close=close,
            highest_close=high_close,
            highest_position_r_multiple=highest_position_r_multiple,
            adr_20_pct=adr_20_pct,
            developing_activation_r=rules.developing_exit_activation_r,
            established_activation_r=rules.first_partial_r_multiple,
            mature_activation_r=rules.second_partial_r_multiple,
            developing_adr_multiple=rules.developing_trailing_adr_multiple,
            developing_drawdown_min_pct=rules.developing_trailing_min_pct,
            developing_drawdown_max_pct=rules.developing_trailing_max_pct,
            established_adr_multiple=rules.established_trailing_adr_multiple,
            established_drawdown_min_pct=rules.established_trailing_min_pct,
            established_drawdown_max_pct=rules.established_trailing_max_pct,
            mature_adr_multiple=rules.mature_trailing_adr_multiple,
            mature_drawdown_min_pct=rules.mature_trailing_min_pct,
            mature_drawdown_max_pct=rules.mature_trailing_max_pct,
            established_profit_floor_r=rules.established_profit_floor_r,
            mature_profit_floor_r=rules.mature_profit_floor_r,
            fallback_drawdown_pct=rules.trailing_drawdown_pct,
            exit_retention_rate=retention,
        )
        exit_stage = adaptive.stage
        peak_r = adaptive.peak_r_multiple
        current_r = adaptive.current_r_multiple
        average_cost_price = adaptive.average_cost_price or entry
        net_breakeven_price = adaptive.net_breakeven_price or average_cost_price
        position_high_water_price = (
            adaptive.position_high_water_price or high_close
        )
        tranche_states = adaptive.tranche_states
        total_tranche_shares = sum(item["shares"] for item in tranche_states)
        total_tranche_risk = sum(
            item["shares"] * item["risk_per_share"] for item in tranche_states
        )
        if total_tranche_shares > 0:
            activation = (
                average_cost_price
                + rules.developing_exit_activation_r
                * total_tranche_risk
                / total_tranche_shares
            ) / retention
        breakeven_active = False
        trailing_pct = adaptive.trailing_drawdown_pct
        trailing_stop = adaptive.trailing_stop_price
        profit_floor_r = adaptive.profit_floor_r_multiple
        profit_floor_price = adaptive.profit_floor_price
    elif (
        rules.adaptive_exit_enabled
        and rules.adaptive_exit_model == PROFIT_FLOOR_VOLATILITY_EXIT
    ):
        adaptive = resolve_profit_floor_volatility_exit(
            entry_price=entry,
            initial_stop_price=initial_stop,
            highest_close=high_close,
            adr_20_pct=adr_20_pct,
            developing_activation_r=rules.developing_exit_activation_r,
            established_activation_r=rules.first_partial_r_multiple,
            mature_activation_r=rules.second_partial_r_multiple,
            developing_adr_multiple=rules.developing_trailing_adr_multiple,
            developing_drawdown_min_pct=rules.developing_trailing_min_pct,
            developing_drawdown_max_pct=rules.developing_trailing_max_pct,
            established_adr_multiple=rules.established_trailing_adr_multiple,
            established_drawdown_min_pct=rules.established_trailing_min_pct,
            established_drawdown_max_pct=rules.established_trailing_max_pct,
            mature_adr_multiple=rules.mature_trailing_adr_multiple,
            mature_drawdown_min_pct=rules.mature_trailing_min_pct,
            mature_drawdown_max_pct=rules.mature_trailing_max_pct,
            established_profit_floor_r=rules.established_profit_floor_r,
            mature_profit_floor_r=rules.mature_profit_floor_r,
            fallback_drawdown_pct=rules.trailing_drawdown_pct,
        )
        exit_stage = adaptive.stage
        trailing_pct = adaptive.trailing_drawdown_pct
        trailing_stop = adaptive.trailing_stop_price
        profit_floor_r = adaptive.profit_floor_r_multiple
        profit_floor_price = adaptive.profit_floor_price
    elif rules.adaptive_exit_enabled:
        if peak_r >= rules.second_partial_r_multiple:
            trailing_pct = rules.mature_trailing_drawdown_pct
        elif peak_r >= rules.first_partial_r_multiple:
            trailing_pct = rules.adaptive_trailing_drawdown_pct
        trailing_stop = high_close * (1.0 - trailing_pct / 100.0)

    active_levels = [initial_stop, ema21_value]
    if trailing_stop is not None:
        active_levels.append(trailing_stop)
    if profit_floor_price is not None:
        active_levels.append(profit_floor_price)
    if breakeven_active:
        active_levels.append(net_breakeven_price)
    active_exit = max(active_levels)
    return_pct = (close / average_cost_price - 1.0) * 100.0
    reasons: list[str] = []
    ema9_value = _positive_number(ema9)
    ema9_active = bool(
        rules.adaptive_exit_enabled
        and ema9_value is not None
        and (
            peak_r >= (
                rules.developing_exit_activation_r
                if rules.adaptive_exit_model in PROFIT_FLOOR_EXIT_MODELS
                else rules.first_partial_r_multiple
            )
            or (str(market_regime_state) == "risk_off" and current_r >= 1.0)
        )
    )
    ema9_below_streak = (
        max(int(previous_ema9_below_streak), 0) + 1
        if ema9_active and close < ema9_value
        else 0
    )
    if ema9_active:
        if rules.adaptive_exit_model in PROFIT_FLOOR_EXIT_MODELS:
            required_days = (
                rules.risk_off_ema9_confirmation_days
                if str(market_regime_state) == "risk_off"
                else rules.ema9_confirmation_days
            )
            hard_exit_armed = (
                str(market_regime_state) == "risk_off" and current_r >= 1.0
            ) or peak_r >= rules.first_partial_r_multiple
            if hard_exit_armed and ema9_below_streak >= required_days:
                reasons.append("confirmed_close_below_ema9_profit_protection")
        else:
            active_exit = max(active_exit, ema9_value)
            if close < ema9_value:
                reasons.append("close_below_ema9_profit_protection")
    if close < ema21_value:
        reasons.append("close_below_ema21")
    if close <= active_exit:
        reasons.append("close_below_active_exit")
    if holding_days >= rules.follow_through_days and return_pct < rules.follow_through_pct:
        reasons.append("no_follow_through")
    if (
        rules.max_holding_days is not None
        and holding_days >= rules.max_holding_days
    ):
        reasons.append("max_holding_days")

    exit_required = bool(reasons)
    partial_targets = _holding_partial_targets(
        rules,
        current_r=current_r,
        partial_exit_count=partial_exit_count,
        exhaustion_signal=exhaustion_signal,
    )
    return {
        "status": "exit_signal" if exit_required else "hold",
        "action": "sell_next_open_call_auction" if exit_required else "keep_and_review_after_close",
        "current_close": _price(close),
        "return_pct": round(return_pct, 4),
        "active_exit_price": _price(active_exit),
        "breakeven_activation_price": _price(activation),
        "breakeven_active": breakeven_active,
        "initial_stop_price": _price(initial_stop),
        "ema21": _price(ema21_value),
        "ema9": _price(ema9_value) if ema9_value is not None else None,
        "ema9_profit_protection_active": ema9_active,
        "ema9_below_streak": ema9_below_streak,
        "adaptive_exit_stage": exit_stage,
        "profit_floor_r_multiple": profit_floor_r,
        "profit_floor_price": (
            _price(profit_floor_price) if profit_floor_price is not None else None
        ),
        "average_entry_cost_price": _price(average_cost_price),
        "net_breakeven_price": _price(net_breakeven_price),
        "position_high_water_price": _price(position_high_water_price),
        "trailing_stop_price": _price(trailing_stop) if trailing_stop is not None else None,
        "trailing_drawdown_pct": trailing_pct,
        "current_r_multiple": round(current_r, 4),
        "peak_r_multiple": round(peak_r, 4),
        "tranche_r_states": [
            {
                **item,
                "shares": round(float(item["shares"]), 6),
                "entry_price": _price(item["entry_price"]),
                "cost_per_share": _price(item["cost_per_share"]),
                "protective_stop_price": (
                    _price(item["protective_stop_price"])
                    if item["protective_stop_price"] is not None
                    else None
                ),
                "risk_per_share": _price(item["risk_per_share"]),
                "highest_close": _price(item["highest_close"]),
                "current_r_multiple": round(float(item["current_r_multiple"]), 4),
                "peak_r_multiple": round(float(item["peak_r_multiple"]), 4),
            }
            for item in tranche_states
        ],
        "partial_exit_orders": partial_targets if not exit_required else [],
        "holding_days": int(holding_days),
        "reasons": reasons,
    }


def _partial_profit_levels(rules: MartinTradePlanConfig) -> list[dict[str, Any]]:
    if not rules.adaptive_exit_enabled:
        return []
    if rules.adaptive_exit_model not in PROFIT_FLOOR_EXIT_MODELS:
        return [
            {
                "r_multiple": rules.first_partial_r_multiple,
                "sell_fraction": rules.partial_exit_fraction,
                "requires_exhaustion": False,
            },
            {
                "r_multiple": rules.second_partial_r_multiple,
                "sell_fraction": rules.partial_exit_fraction,
                "requires_exhaustion": False,
            },
        ]
    levels = []
    for target_r, fraction in (
        (rules.first_partial_r_multiple, rules.first_partial_exit_fraction),
        (rules.second_partial_r_multiple, rules.second_partial_exit_fraction),
    ):
        if fraction > 0:
            levels.append({
                "r_multiple": target_r,
                "sell_fraction": fraction,
                "requires_exhaustion": rules.partial_exit_requires_exhaustion,
            })
    return levels


def _holding_partial_targets(
    rules: MartinTradePlanConfig,
    *,
    current_r: float,
    partial_exit_count: int,
    exhaustion_signal: bool,
) -> list[dict[str, float]]:
    levels = _partial_profit_levels(rules)
    if rules.adaptive_exit_model in PROFIT_FLOOR_EXIT_MODELS:
        if partial_exit_count > 0:
            return []
        for level in levels:
            if current_r < float(level["r_multiple"]):
                continue
            if bool(level["requires_exhaustion"]) and not exhaustion_signal:
                continue
            return [{
                "r_multiple": float(level["r_multiple"]),
                "sell_fraction": float(level["sell_fraction"]),
            }]
        return []
    return [
        {
            "r_multiple": float(level["r_multiple"]),
            "sell_fraction": float(level["sell_fraction"]),
        }
        for stage, level in enumerate(levels, start=1)
        if partial_exit_count < stage and current_r >= float(level["r_multiple"])
    ]


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _position_weight_cap(
    initial_risk_pct: float,
    rules: MartinTradePlanConfig,
) -> float:
    if initial_risk_pct <= 0:
        return 0.0
    risk_based_cap = rules.max_portfolio_risk_per_trade_pct / initial_risk_pct * 100.0
    nominal_cap = (
        min(rules.max_position_weight_pct, rules.initial_position_weight_pct)
        if rules.pyramiding_enabled
        else rules.max_position_weight_pct
    )
    return max(min(risk_based_cap, nominal_cap), 0.0)


def _price(value: float) -> float:
    return round(float(value), 2)
