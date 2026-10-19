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
    assert plan["exit"]["holding_period_mode"] == "trend_following"
    assert plan["exit"]["max_holding_days"] is None
    assert plan["position_sizing"]["max_position_weight_pct"] == 20.0
    assert plan["position_sizing"]["min_position_weight_pct"] == 12.0
    assert plan["position_sizing"]["max_portfolio_risk_per_trade_pct"] == 0.5
    assert plan["position_sizing"]["max_portfolio_heat_pct"] == 2.5
    assert plan["position_sizing"]["risk_basis"] == "technical_stop"
    assert len(plan["holding_premises"]) == 5
    assert "不设固定持有期限" in plan["holding_premises"][-1]


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


def test_holding_plan_does_not_exit_a_healthy_trend_after_twenty_days() -> None:
    update = update_martin_holding_plan(
        entry_price=100.0,
        initial_stop_price=96.0,
        current_close=125.0,
        highest_close=126.0,
        ema21=118.0,
        holding_days=35,
    )

    assert update["status"] == "hold"
    assert update["action"] == "keep_and_review_after_close"
    assert update["reasons"] == []


def test_holding_plan_uses_ema9_after_large_profit() -> None:
    update = update_martin_holding_plan(
        entry_price=10.0,
        initial_stop_price=9.5,
        current_close=11.4,
        highest_close=12.0,
        ema9=11.5,
        ema21=10.8,
        holding_days=8,
        partial_exit_count=1,
        config=MartinTradePlanConfig(adaptive_exit_enabled=True),
    )

    assert update["status"] == "exit_signal"
    assert update["ema9_profit_protection_active"] is True
    assert "close_below_ema9_profit_protection" in update["reasons"]
    assert update["trailing_drawdown_pct"] == 6.0


def test_profit_floor_exit_widens_peak_tolerance_as_profit_matures() -> None:
    rules = MartinTradePlanConfig(
        adaptive_exit_enabled=True,
        adaptive_exit_model="profit_floor_volatility",
    )
    initial = update_martin_holding_plan(
        entry_price=100.0,
        initial_stop_price=96.0,
        current_close=102.5,
        highest_close=103.0,
        ema9=102.0,
        ema21=98.0,
        adr_20_pct=4.0,
        holding_days=3,
        config=rules,
    )
    developing = update_martin_holding_plan(
        entry_price=100.0,
        initial_stop_price=96.0,
        current_close=107.0,
        highest_close=108.0,
        ema9=106.0,
        ema21=99.0,
        adr_20_pct=4.0,
        holding_days=6,
        config=rules,
    )
    mature = update_martin_holding_plan(
        entry_price=100.0,
        initial_stop_price=96.0,
        current_close=112.0,
        highest_close=120.0,
        ema9=111.0,
        ema21=105.0,
        adr_20_pct=4.0,
        holding_days=12,
        config=rules,
    )

    assert initial["adaptive_exit_stage"] == "initial"
    assert initial["trailing_drawdown_pct"] is None
    assert initial["trailing_stop_price"] is None
    assert developing["adaptive_exit_stage"] == "developing"
    assert developing["trailing_drawdown_pct"] == 5.0
    assert developing["profit_floor_r_multiple"] == 0.0
    assert mature["adaptive_exit_stage"] == "mature"
    assert mature["trailing_drawdown_pct"] == 9.0
    assert mature["profit_floor_r_multiple"] == 2.5
    assert mature["profit_floor_price"] == 110.0
    assert mature["active_exit_price"] == 110.0


def test_profit_floor_exit_requires_confirmed_ema9_break_outside_risk_off() -> None:
    rules = MartinTradePlanConfig(
        adaptive_exit_enabled=True,
        adaptive_exit_model="profit_floor_volatility",
    )
    first_close = update_martin_holding_plan(
        entry_price=10.0,
        initial_stop_price=9.5,
        current_close=11.4,
        highest_close=12.0,
        ema9=11.5,
        ema21=10.8,
        adr_20_pct=4.0,
        previous_ema9_below_streak=0,
        holding_days=8,
        config=rules,
    )
    second_close = update_martin_holding_plan(
        entry_price=10.0,
        initial_stop_price=9.5,
        current_close=11.4,
        highest_close=12.0,
        ema9=11.5,
        ema21=10.8,
        adr_20_pct=4.0,
        previous_ema9_below_streak=1,
        holding_days=9,
        config=rules,
    )
    risk_off_close = update_martin_holding_plan(
        entry_price=10.0,
        initial_stop_price=9.5,
        current_close=11.4,
        highest_close=12.0,
        ema9=11.5,
        ema21=10.8,
        adr_20_pct=4.0,
        market_regime_state="risk_off",
        previous_ema9_below_streak=0,
        holding_days=8,
        config=rules,
    )

    assert first_close["status"] == "hold"
    assert first_close["ema9_below_streak"] == 1
    assert second_close["status"] == "exit_signal"
    assert "confirmed_close_below_ema9_profit_protection" in second_close["reasons"]
    assert risk_off_close["status"] == "exit_signal"
    assert risk_off_close["ema9_below_streak"] == 1


def test_profit_floor_partial_exit_requires_five_r_and_exhaustion() -> None:
    rules = MartinTradePlanConfig(
        adaptive_exit_enabled=True,
        adaptive_exit_model="profit_floor_volatility",
    )
    ordinary = update_martin_holding_plan(
        entry_price=10.0,
        initial_stop_price=9.5,
        current_close=12.6,
        highest_close=12.6,
        ema9=12.0,
        ema21=11.0,
        adr_20_pct=4.0,
        exhaustion_signal=False,
        holding_days=10,
        config=rules,
    )
    exhausted = update_martin_holding_plan(
        entry_price=10.0,
        initial_stop_price=9.5,
        current_close=12.6,
        highest_close=12.6,
        ema9=12.0,
        ema21=11.0,
        adr_20_pct=4.0,
        exhaustion_signal=True,
        holding_days=10,
        config=rules,
    )

    assert ordinary["partial_exit_orders"] == []
    assert exhausted["partial_exit_orders"] == [
        {"r_multiple": 5.0, "sell_fraction": 0.15}
    ]


def test_v33_profit_floor_uses_whole_position_cost_and_tranche_risk() -> None:
    rules = MartinTradePlanConfig(
        adaptive_exit_enabled=True,
        adaptive_exit_model="tranche_cost_profit_floor_volatility",
    )
    tranches = [
        {
            "tranche_index": 0,
            "type": "initial",
            "remaining_shares": 1000,
            "price": 10.0,
            "cost_per_share": 10.0,
            "protective_stop_price": 9.5,
            "risk_per_share": 0.5,
            "highest_close": 13.0,
        },
        {
            "tranche_index": 1,
            "type": "add_on",
            "remaining_shares": 500,
            "price": 11.5,
            "cost_per_share": 11.5,
            "protective_stop_price": 10.0,
            "risk_per_share": 1.5,
            "highest_close": 13.0,
        },
    ]
    holding = update_martin_holding_plan(
        entry_price=10.0,
        initial_stop_price=9.5,
        current_close=13.0,
        highest_close=13.0,
        ema9=12.5,
        ema21=11.0,
        adr_20_pct=4.0,
        holding_days=8,
        entry_tranches=tranches,
        highest_position_r_multiple=3.0,
        config=rules,
    )
    pulled_back = update_martin_holding_plan(
        entry_price=10.0,
        initial_stop_price=9.5,
        current_close=11.3,
        highest_close=13.0,
        ema9=11.4,
        ema21=11.0,
        adr_20_pct=4.0,
        holding_days=9,
        entry_tranches=tranches,
        highest_position_r_multiple=3.0,
        config=rules,
    )

    assert holding["adaptive_exit_stage"] == "established"
    assert holding["average_entry_cost_price"] == 10.5
    assert holding["current_r_multiple"] == 3.0
    assert holding["profit_floor_price"] == 11.33
    assert [item["current_r_multiple"] for item in holding["tranche_r_states"]] == [
        6.0,
        1.0,
    ]
    assert pulled_back["status"] == "exit_signal"
    assert pulled_back["return_pct"] == 7.619
    assert "close_below_active_exit" in pulled_back["reasons"]


def test_v33_developing_position_has_trailing_room_without_zero_r_floor() -> None:
    update = update_martin_holding_plan(
        entry_price=20.16,
        initial_stop_price=19.81,
        current_close=20.70,
        highest_close=21.50,
        ema9=20.80,
        ema21=20.20,
        adr_20_pct=4.0,
        holding_days=3,
        entry_tranches=[
            {
                "tranche_index": 0,
                "type": "initial",
                "remaining_shares": 500,
                "price": 20.16,
                "cost_per_share": 20.17,
                "protective_stop_price": 19.81,
                "risk_per_share": 0.36,
                "highest_close": 21.50,
            },
            {
                "tranche_index": 1,
                "type": "add_on",
                "remaining_shares": 200,
                "price": 21.10,
                "cost_per_share": 21.11,
                "protective_stop_price": 20.16,
                "risk_per_share": 0.95,
                "highest_close": 21.50,
            },
        ],
        highest_position_r_multiple=1.7,
        config=MartinTradePlanConfig(
            adaptive_exit_enabled=True,
            adaptive_exit_model="tranche_cost_profit_floor_volatility",
        ),
    )

    assert update["adaptive_exit_stage"] == "developing"
    assert update["profit_floor_r_multiple"] is None
    assert update["profit_floor_price"] is None
    assert update["status"] == "hold"


def test_optional_research_holding_cap_remains_available() -> None:
    update = update_martin_holding_plan(
        entry_price=100.0,
        initial_stop_price=96.0,
        current_close=125.0,
        highest_close=126.0,
        ema21=118.0,
        holding_days=20,
        config=MartinTradePlanConfig(max_holding_days=20),
    )

    assert update["status"] == "exit_signal"
    assert update["reasons"] == ["max_holding_days"]
