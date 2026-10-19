# -*- coding: utf-8 -*-

import pandas as pd

from src.services.screening.martin_setups import (
    SETUP_FIRST_PULLBACK,
    SETUP_PLATFORM_BREAKOUT,
    SETUP_TIGHT_CONTINUATION,
    attach_martin_entry_setups,
    common_martin_hard_filters,
)
from src.services.screening.models import HardFilterConfig


def test_multi_setup_classifier_keeps_three_entry_families_distinct() -> None:
    frame = pd.DataFrame([
        {
            "code": "000001",
            "close": 10.0,
            "previous_high": 9.9,
            "previous_5d_high": 10.5,
            "previous_20d_high": 11.0,
            "previous_5d_range_pct": 12.0,
            "previous_10d_range_pct": 15.0,
            "pullback_depth_10d_pct": -4.0,
            "range_contraction_ratio": 0.75,
            "volume_contraction_ratio": 0.70,
            "support_confluence_count": 2,
            "trigger_close_location_pct": 80.0,
            "trigger_volume_vs_previous": 1.0,
            "change_20d": 10.0,
            "adr_20_pct": 4.0,
        },
        {
            "code": "000002",
            "close": 10.2,
            "previous_high": 10.1,
            "previous_5d_high": 10.1,
            "previous_20d_high": 10.0,
            "previous_5d_range_pct": 9.0,
            "previous_10d_range_pct": 9.0,
            "pullback_depth_10d_pct": 0.0,
            "range_contraction_ratio": 1.0,
            "volume_contraction_ratio": 1.0,
            "support_confluence_count": 1,
            "trigger_close_location_pct": 85.0,
            "trigger_volume_vs_previous": 1.2,
            "change_20d": 18.0,
            "adr_20_pct": 4.0,
        },
        {
            "code": "000003",
            "close": 10.1,
            "previous_high": 10.0,
            "previous_5d_high": 10.0,
            "previous_20d_high": 11.0,
            "previous_5d_range_pct": 8.0,
            "previous_10d_range_pct": 16.0,
            "pullback_depth_10d_pct": -0.5,
            "range_contraction_ratio": 0.90,
            "volume_contraction_ratio": 0.90,
            "support_confluence_count": 1,
            "trigger_close_location_pct": 70.0,
            "trigger_volume_vs_previous": 1.0,
            "change_20d": 12.0,
            "adr_20_pct": 4.0,
        },
    ])

    result = attach_martin_entry_setups(frame, mode="multi").set_index("code")

    assert result.loc["000001", "setup_type"] == SETUP_FIRST_PULLBACK
    assert result.loc["000002", "setup_type"] == SETUP_PLATFORM_BREAKOUT
    assert result.loc["000003", "setup_type"] == SETUP_TIGHT_CONTINUATION
    assert result["setup_confirmed"].all()


def test_setup_trade_permission_keeps_disabled_shapes_as_watch_only() -> None:
    frame = pd.DataFrame([{
        "code": "000002",
        "close": 10.2,
        "previous_high": 10.1,
        "previous_5d_high": 10.1,
        "previous_20d_high": 10.0,
        "previous_5d_range_pct": 9.0,
        "previous_10d_range_pct": 9.0,
        "pullback_depth_10d_pct": 0.0,
        "range_contraction_ratio": 1.0,
        "volume_contraction_ratio": 1.0,
        "support_confluence_count": 1,
        "trigger_close_location_pct": 85.0,
        "trigger_volume_vs_previous": 1.2,
        "change_20d": 18.0,
        "adr_20_pct": 4.0,
    }])

    result = attach_martin_entry_setups(
        frame,
        mode="multi",
        tradable_setup_types=[SETUP_FIRST_PULLBACK],
    ).iloc[0]

    assert result["setup_type"] == SETUP_PLATFORM_BREAKOUT
    assert bool(result["setup_confirmed"]) is True
    assert bool(result["setup_trade_eligible"]) is False
    assert result["setup_execution_status"] == "watch_only"


def test_multi_setup_common_filter_preserves_trend_gates() -> None:
    source = HardFilterConfig(
        adr_20_pct_min=3.5,
        pullback_depth_10d_pct_min=-12.0,
        range_contraction_ratio_max=0.95,
        support_confluence_count_min=2,
        trigger_distance_pct_min=-4.0,
    )

    result = common_martin_hard_filters(source)

    assert result.adr_20_pct_min == 3.5
    assert result.pullback_depth_10d_pct_min is None
    assert result.range_contraction_ratio_max is None
    assert result.support_confluence_count_min is None
    assert result.trigger_distance_pct_min is None
