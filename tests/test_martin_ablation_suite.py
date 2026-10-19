# -*- coding: utf-8 -*-

import pandas as pd

from scripts.run_martin_ablation_suite import (
    _annual_summary_rows,
    _attach_full_variant_deltas,
    _build_variants,
)


def _declared_profile() -> dict[str, object]:
    return {
        "profile_version": "3.3",
        "entry_setup_mode": "multi",
        "tradable_setup_types": ["first_pullback"],
        "reentry_cooldown_days": 0,
        "sector_leadership_enabled": True,
        "progressive_exposure_enabled": True,
        "pyramiding_enabled": True,
        "adaptive_exit_enabled": True,
        "adaptive_exit_model": "tranche_cost_profit_floor_volatility",
        "initial_position_weight_pct": 10.0,
        "add_on_position_weight_pct": 5.0,
        "max_add_ons": 2,
    }


def test_ablation_variants_change_only_the_named_v3_module() -> None:
    variants = {item["label"]: item["profile"] for item in _build_variants(_declared_profile())}

    assert set(variants) == {
        "baseline-v28",
        "baseline-v30",
        "baseline-v31",
        "baseline-v32",
        "full-v33",
        "all-setup-trading",
        "no-multi-setup",
        "no-sector-leadership",
        "no-progressive-exposure",
        "no-pyramiding",
        "cooldown-3d",
    }
    assert variants["full-v33"]["profile_version"] == "3.3"
    assert variants["full-v33"]["reentry_cooldown_days"] == 0
    assert variants["full-v33"]["adaptive_exit_enabled"] is True
    assert variants["full-v33"]["adaptive_exit_model"] == (
        "tranche_cost_profit_floor_volatility"
    )
    assert variants["full-v33"]["tradable_setup_types"] == ["first_pullback"]
    assert set(variants["all-setup-trading"]["tradable_setup_types"]) == {
        "first_pullback",
        "platform_breakout",
        "tight_continuation",
    }
    assert variants["baseline-v30"]["adaptive_exit_model"] == "legacy_tightening"
    assert variants["baseline-v31"]["profile_version"] == "3.1"
    assert variants["baseline-v31"]["adaptive_exit_enabled"] is True
    assert variants["baseline-v31"]["adaptive_exit_model"] == "profit_floor_volatility"
    assert variants["baseline-v32"]["profile_version"] == "3.2"
    assert variants["baseline-v32"]["adaptive_exit_enabled"] is False
    assert set(variants["baseline-v30"]["tradable_setup_types"]) == {
        "first_pullback",
        "platform_breakout",
        "tight_continuation",
    }
    assert variants["cooldown-3d"]["reentry_cooldown_days"] == 3
    assert variants["cooldown-3d"]["entry_setup_mode"] == "multi"
    assert variants["no-multi-setup"]["entry_setup_mode"] == "legacy_pullback"
    assert variants["no-multi-setup"]["pyramiding_enabled"] is True
    assert variants["baseline-v28"]["profile_version"] == "2.8"
    assert variants["baseline-v28"]["initial_position_weight_pct"] == 20.0


def test_annual_summary_uses_prior_year_end_as_next_year_start() -> None:
    rows = _annual_summary_rows(
        "fixture",
        {
            "equity_curve": [
                {"date": "2024-01-02", "equity": 110_000},
                {"date": "2024-12-31", "equity": 121_000},
                {"date": "2025-01-02", "equity": 108_900},
                {"date": "2025-12-31", "equity": 133_100},
            ]
        },
        initial_cash=100_000,
    )

    assert rows[0]["return_pct"] == 21.0
    assert rows[1]["start_equity"] == 121_000
    assert rows[1]["return_pct"] == 10.0
    assert rows[1]["max_drawdown_pct"] == -10.0


def test_ablation_delta_is_full_v33_minus_disabled_variant() -> None:
    summary = pd.DataFrame([
        {
            "label": "full-v33",
            "total_return_pct": 10.0,
            "max_drawdown_pct": -5.0,
            "sharpe_ratio": 0.8,
        },
        {
            "label": "no-module",
            "total_return_pct": 4.0,
            "max_drawdown_pct": -9.0,
            "sharpe_ratio": 0.3,
        },
    ])

    compared = _attach_full_variant_deltas(summary).set_index("label")

    assert compared.loc["no-module", "module_return_contribution_pct"] == 6.0
    assert compared.loc["no-module", "module_drawdown_improvement_pct"] == 4.0
    assert compared.loc["no-module", "module_sharpe_contribution"] == 0.5
