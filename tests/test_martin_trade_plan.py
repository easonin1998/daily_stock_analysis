# -*- coding: utf-8 -*-
"""Contracts for Martin daily close decisions and auction condition plans."""

from src.services.screening.trade_plan import (
    MartinTradePlanConfig,
    build_martin_trade_plan,
    rebase_martin_trade_plan_after_fill,
    resolve_martin_entry_gap_bounds,
    update_martin_holding_plan,
)


def test_confirmed_signal_builds_complete_next_open_condition_plan() -> None:
    plan = build_martin_trade_plan(
        price=100.0,
        trigger_distance_pct=1.0,
        adr_20_pct=4.0,
        ema21=97.0,
        anchored_vwap=98.0,
    )

    assert plan["signal_status"] == "buy_signal"
    assert plan["action"] == "place_next_open_auction_order"
    assert plan["entry"] == {
        "session": "next_open_call_auction",
        "trigger_price": 99.01,
        "min_price": 99.01,
        "max_price": 102.0,
        "gap_model": "volatility_adaptive",
        "gap_volatility_source": "adr20",
        "gap_volatility_reference_pct": 4.0,
        "min_gap_pct": -1.2,
        "max_gap_pct": 2.0,
        "cancel_if": [
            "开盘价低于最低可接受价或信号日 EMA21",
            "开盘价高于最高可接受价，避免跳空追涨",
            "开盘涨停或无法确认成交",
        ],
    }
    assert plan["exit"]["initial_stop_price"] == 98.42
    assert plan["exit"]["breakeven_activation_price"] == 101.58
    assert plan["exit"]["first_profit_price"] == 103.17
    assert plan["exit"]["initial_risk_pct"] == 1.5842
    assert plan["exit"]["sizing_risk_pct"] == 1.5842
    assert plan["exit"]["effective_risk_pct"] == 1.5842
    assert plan["exit"]["adr_stress_pct"] == 4.0
    assert plan["exit"]["adr_stress_affects_position_size"] is False
    assert plan["exit"]["pricing_status"] == "provisional_until_fill"
    assert plan["exit"]["fixed_take_profit"] is False
    assert plan["condition_orders"]["protective_exit"]["trigger_basis"] == "daily_close"
    assert plan["position_sizing"]["max_position_weight_pct"] == 20.0
    assert plan["position_sizing"]["min_position_weight_pct"] == 12.0
    assert plan["position_sizing"]["max_portfolio_risk_per_trade_pct"] == 0.5
    assert plan["position_sizing"]["max_portfolio_heat_pct"] == 2.5
    assert plan["position_sizing"]["risk_basis"] == "technical_stop"
    assert len(plan["holding_premises"]) == 5


def test_entry_gap_bounds_are_asymmetric_and_volatility_adaptive() -> None:
    bounds = resolve_martin_entry_gap_bounds(
        adr_20_pct=4.0,
        atr_20_pct=6.0,
    )

    assert bounds == {
        "model": "volatility_adaptive",
        "volatility_source": "atr20_adr20_mean",
        "volatility_reference_pct": 5.0,
        "min_gap_pct": -1.5,
        "max_gap_pct": 2.0,
    }

    quieter_bounds = resolve_martin_entry_gap_bounds(
        adr_20_pct=3.5,
        atr_20_pct=3.5,
    )
    assert quieter_bounds["min_gap_pct"] == -1.05
    assert quieter_bounds["max_gap_pct"] == 1.75


def test_entry_gap_bounds_support_fallback_and_exact_research_override() -> None:
    fallback = resolve_martin_entry_gap_bounds(adr_20_pct=None, atr_20_pct=None)
    overridden = resolve_martin_entry_gap_bounds(
        adr_20_pct=4.0,
        atr_20_pct=6.0,
        config=MartinTradePlanConfig(
            entry_gap_min_pct=-2.0,
            entry_gap_max_pct=2.0,
        ),
    )

    assert fallback["min_gap_pct"] == -1.25
    assert fallback["max_gap_pct"] == 1.75
    assert fallback["volatility_source"] == "fallback"
    assert overridden["model"] == "fixed_override"
    assert overridden["min_gap_pct"] == -2.0
    assert overridden["max_gap_pct"] == 2.0


def test_actual_fill_rebases_r_multiple_prices() -> None:
    plan = build_martin_trade_plan(
        price=100.0,
        trigger_distance_pct=1.0,
        adr_20_pct=4.0,
        ema21=97.0,
        anchored_vwap=98.0,
    )

    filled = rebase_martin_trade_plan_after_fill(plan, fill_price=101.0)

    assert filled["fill_status"] == "confirmed"
    assert filled["actual_fill_price"] == 101.0
    assert filled["exit"]["pricing_status"] == "finalized_from_actual_fill"
    assert filled["exit"]["initial_risk_per_share"] == 2.58
    assert filled["exit"]["sizing_risk_per_share"] == 2.58
    assert filled["exit"]["effective_risk_per_share"] == 2.58
    assert filled["exit"]["effective_risk_pct"] == 2.5545
    assert filled["exit"]["adr_stress_per_share"] == 4.04
    assert filled["exit"]["breakeven_activation_price"] == 103.58
    assert filled["exit"]["two_r_reference_price"] == 106.16
    assert filled["position_sizing"]["pricing_status"] == "finalized_from_actual_fill"
    assert filled["position_sizing"]["max_position_weight_pct"] == 19.5736
    assert filled["position_sizing"]["risk_basis_pct"] == 2.5545


def test_actual_fill_rejects_trade_when_technical_stop_becomes_too_wide() -> None:
    plan = build_martin_trade_plan(
        price=100.0,
        trigger_distance_pct=1.0,
        adr_20_pct=4.0,
        ema21=97.0,
        anchored_vwap=98.0,
    )

    filled = rebase_martin_trade_plan_after_fill(plan, fill_price=102.0)

    assert filled["fill_status"] == "rejected"
    assert filled["fill_action"] == "cancel_entry"
    assert filled["exit"]["sizing_risk_pct"] > 3.0


def test_unconfirmed_setup_is_watch_only_and_has_no_order() -> None:
    plan = build_martin_trade_plan(
        price=100.0,
        trigger_distance_pct=-1.0,
        adr_20_pct=4.0,
        ema21=97.0,
    )

    assert plan["signal_status"] == "watch_wait_trigger"
    assert plan["action"] == "no_order"


def test_holding_plan_emits_next_open_exit_after_close_breaks_defense() -> None:
    update = update_martin_holding_plan(
        entry_price=100.0,
        initial_stop_price=96.0,
        current_close=97.0,
        highest_close=110.0,
        ema21=101.0,
        holding_days=6,
    )

    assert update["status"] == "exit_signal"
    assert update["action"] == "sell_next_open_call_auction"
    assert update["active_exit_price"] == 101.2
    assert update["breakeven_active"] is True
    assert set(update["reasons"]) == {"close_below_ema21", "close_below_active_exit"}
