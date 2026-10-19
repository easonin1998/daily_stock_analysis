# -*- coding: utf-8 -*-
"""Shared entry-setup classification for the Martin A-share strategy."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.services.screening.models import HardFilterConfig

LEGACY_SETUP_MODE = "legacy_pullback"
MULTI_SETUP_MODE = "multi"

SETUP_FIRST_PULLBACK = "first_pullback"
SETUP_PLATFORM_BREAKOUT = "platform_breakout"
SETUP_TIGHT_CONTINUATION = "tight_continuation"
ALL_SETUP_TYPES = (
    SETUP_FIRST_PULLBACK,
    SETUP_PLATFORM_BREAKOUT,
    SETUP_TIGHT_CONTINUATION,
)

SETUP_EXECUTION_TRADABLE = "tradable"
SETUP_EXECUTION_WATCH_ONLY = "watch_only"
SETUP_EXECUTION_UNCONFIRMED = "unconfirmed"


def common_martin_hard_filters(filters: HardFilterConfig) -> HardFilterConfig:
    """Keep trend/liquidity gates while delegating shape gates to setup rules."""
    return replace(
        filters,
        pullback_depth_10d_pct_min=None,
        pullback_depth_10d_pct_max=None,
        range_contraction_ratio_max=None,
        volume_contraction_ratio_max=None,
        support_confluence_count_min=None,
        trigger_distance_pct_min=None,
        trigger_distance_pct_max=None,
    )


def attach_martin_entry_setups(
    frame: pd.DataFrame,
    *,
    mode: str = MULTI_SETUP_MODE,
    tradable_setup_types: object | None = None,
) -> pd.DataFrame:
    """Attach one deterministic setup plus its independent trade permission."""
    result = frame.copy()
    tradable = normalize_tradable_setup_types(
        tradable_setup_types,
        mode=mode,
    )
    if result.empty:
        return _empty_setup_columns(result)

    close = _numeric(result, "close").where(
        _numeric(result, "close").notna(), _numeric(result, "price")
    )
    previous_high = _numeric(result, "previous_high")
    if previous_high.isna().all():
        previous_high = _implied_level(close, _numeric(result, "trigger_distance_pct"))

    if mode == LEGACY_SETUP_MODE:
        legacy_distance = _numeric(result, "trigger_distance_pct")
        result["setup_type"] = SETUP_FIRST_PULLBACK
        result["setup_trigger_price"] = previous_high
        result["setup_trigger_distance_pct"] = legacy_distance
        result["setup_confirmed"] = result["setup_trigger_distance_pct"].ge(0)
        return _attach_setup_trade_permissions(result, tradable)
    if mode != MULTI_SETUP_MODE:
        raise ValueError(f"Unsupported Martin entry setup mode: {mode}")

    adr = _numeric(result, "adr_20_pct")
    pullback = _numeric(result, "pullback_depth_10d_pct")
    range_contraction = _numeric(result, "range_contraction_ratio")
    volume_contraction = _numeric(result, "volume_contraction_ratio")
    support = _numeric(result, "support_confluence_count")
    close_location = _numeric(result, "trigger_close_location_pct")
    volume_trigger = _numeric(result, "trigger_volume_vs_previous")
    change_20d = _numeric(result, "change_20d")

    previous_20d_high = _first_numeric(
        result,
        "previous_20d_high",
        "prev_high_20d",
    )
    previous_5d_high = _numeric(result, "previous_5d_high")
    previous_10d_range = _numeric(result, "previous_10d_range_pct")
    if previous_10d_range.isna().all():
        previous_10d_range = _numeric(result, "range_20d_pct")
    previous_5d_range = _numeric(result, "previous_5d_range_pct")
    if previous_5d_range.isna().all():
        previous_5d_range = _numeric(result, "range_5d_pct")

    pullback_trigger = _distance_pct(close, previous_high)
    platform_trigger = _distance_pct(close, previous_20d_high)
    continuation_trigger = _distance_pct(close, previous_5d_high)

    first_pullback = (
        pullback.between(-12.0, -1.0, inclusive="both")
        & range_contraction.le(0.95)
        & volume_contraction.le(0.95)
        & support.ge(2)
        & pullback_trigger.ge(0)
    )
    platform_limit = (adr * 3.0).clip(lower=10.0, upper=18.0)
    platform_breakout = (
        platform_trigger.ge(0)
        & previous_10d_range.le(platform_limit)
        & close_location.fillna(50.0).ge(60.0)
        & volume_trigger.fillna(1.0).ge(0.80)
    )
    continuation_limit = (adr * 2.5).clip(lower=7.0, upper=13.0)
    tight_continuation = (
        continuation_trigger.ge(0)
        & previous_5d_range.le(continuation_limit)
        & volume_contraction.le(1.10)
        & support.ge(1)
        & change_20d.ge(3.0)
    )

    setup_type = pd.Series("", index=result.index, dtype=object)
    trigger_price = pd.Series(np.nan, index=result.index, dtype=float)
    trigger_distance = pd.Series(np.nan, index=result.index, dtype=float)

    # A constructive first pullback is the most selective entry. A platform
    # breakout takes precedence over a shorter tight continuation afterwards.
    _assign_setup(
        setup_type,
        trigger_price,
        trigger_distance,
        mask=first_pullback,
        setup=SETUP_FIRST_PULLBACK,
        level=previous_high,
        distance=pullback_trigger,
    )
    _assign_setup(
        setup_type,
        trigger_price,
        trigger_distance,
        mask=platform_breakout & setup_type.eq(""),
        setup=SETUP_PLATFORM_BREAKOUT,
        level=previous_20d_high,
        distance=platform_trigger,
    )
    _assign_setup(
        setup_type,
        trigger_price,
        trigger_distance,
        mask=tight_continuation & setup_type.eq(""),
        setup=SETUP_TIGHT_CONTINUATION,
        level=previous_5d_high,
        distance=continuation_trigger,
    )

    result["setup_type"] = setup_type
    result["setup_trigger_price"] = trigger_price
    result["setup_trigger_distance_pct"] = trigger_distance
    result["setup_confirmed"] = setup_type.ne("") & trigger_distance.ge(0)
    # Downstream trade-plan and scoring code use the stable trigger field.
    result["trigger_distance_pct"] = trigger_distance.where(
        result["setup_confirmed"], _numeric(result, "trigger_distance_pct")
    )
    return _attach_setup_trade_permissions(result, tradable)


def normalize_tradable_setup_types(
    value: object | None,
    *,
    mode: str = MULTI_SETUP_MODE,
) -> tuple[str, ...]:
    """Normalize configured trade permissions without disabling recognition."""
    if mode not in {LEGACY_SETUP_MODE, MULTI_SETUP_MODE}:
        raise ValueError(f"Unsupported Martin entry setup mode: {mode}")
    defaults = (
        (SETUP_FIRST_PULLBACK,)
        if mode == LEGACY_SETUP_MODE
        else ALL_SETUP_TYPES
    )
    if value is None:
        return defaults
    if isinstance(value, str):
        requested = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        requested = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise TypeError("tradable_setup_types must be a list or comma-separated string")
    unknown = sorted(set(requested) - set(ALL_SETUP_TYPES))
    if unknown:
        raise ValueError(
            "Unsupported Martin tradable setup types: " + ", ".join(unknown)
        )
    requested_set = set(requested)
    return tuple(setup for setup in ALL_SETUP_TYPES if setup in requested_set)


def setup_diagnostics(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "setup_type" not in frame.columns:
        return {}
    values = frame.loc[frame.get("setup_confirmed", False) == True, "setup_type"]  # noqa: E712
    return {str(key): int(value) for key, value in values.value_counts().items()}


def _assign_setup(
    setup_type: pd.Series,
    trigger_price: pd.Series,
    trigger_distance: pd.Series,
    *,
    mask: pd.Series,
    setup: str,
    level: pd.Series,
    distance: pd.Series,
) -> None:
    selected = mask.fillna(False)
    setup_type.loc[selected] = setup
    trigger_price.loc[selected] = level.loc[selected]
    trigger_distance.loc[selected] = distance.loc[selected]


def _empty_setup_columns(result: pd.DataFrame) -> pd.DataFrame:
    result["setup_type"] = pd.Series(dtype=object)
    result["setup_trigger_price"] = pd.Series(dtype=float)
    result["setup_trigger_distance_pct"] = pd.Series(dtype=float)
    result["setup_confirmed"] = pd.Series(dtype=bool)
    result["setup_trade_eligible"] = pd.Series(dtype=bool)
    result["setup_execution_status"] = pd.Series(dtype=object)
    return result


def _attach_setup_trade_permissions(
    result: pd.DataFrame,
    tradable_setup_types: tuple[str, ...],
) -> pd.DataFrame:
    confirmed = result["setup_confirmed"].fillna(False).astype(bool)
    eligible = confirmed & result["setup_type"].isin(tradable_setup_types)
    status = pd.Series(
        SETUP_EXECUTION_UNCONFIRMED,
        index=result.index,
        dtype=object,
    )
    status.loc[confirmed & ~eligible] = SETUP_EXECUTION_WATCH_ONLY
    status.loc[eligible] = SETUP_EXECUTION_TRADABLE
    result["setup_trade_eligible"] = eligible
    result["setup_execution_status"] = status
    return result


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _first_numeric(frame: pd.DataFrame, *columns: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        values = _numeric(frame, column)
        result = result.where(result.notna(), values)
    return result


def _distance_pct(price: pd.Series, level: pd.Series) -> pd.Series:
    return (price / level.where(level.gt(0)) - 1.0) * 100.0


def _implied_level(price: pd.Series, distance_pct: pd.Series) -> pd.Series:
    denominator = 1.0 + distance_pct / 100.0
    return price / denominator.where(denominator.gt(0))
