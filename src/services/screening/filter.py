# -*- coding: utf-8 -*-
# Derived from AlphaSift revision 9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf.
# Licensed under Apache-2.0 and modified for daily_stock_analysis.
"""L1 hard filter — apply strategy hard_filters to snapshot DataFrame."""

import logging
from dataclasses import replace

import pandas as pd

from src.services.screening.models import HardFilterConfig
from src.services.screening.normalize import is_beijing_stock_code

logger = logging.getLogger(__name__)
_DAILY_FILTER_DEFAULTS = {
    "daily_data_points_min": None,
    "avg_amount_20d_min": None,
    "adr_20_pct_min": None,
    "adr_20_pct_max": None,
    "change_20d_min": None,
    "change_20d_max": None,
    "relative_strength_20d_pctile_min": None,
    "relative_strength_60d_pctile_min": None,
    "require_ema_stack": False,
    "require_ema21_rising": False,
    "require_ema50_rising": False,
    "distance_to_120d_high_pct_min": None,
    "pullback_depth_10d_pct_min": None,
    "pullback_depth_10d_pct_max": None,
    "range_contraction_ratio_max": None,
    "volume_contraction_ratio_max": None,
    "support_confluence_count_min": None,
    "trigger_distance_pct_min": None,
    "trigger_distance_pct_max": None,
    "trigger_close_location_pct_min": None,
    "trigger_close_location_pct_max": None,
    "trigger_volume_vs_previous_min": None,
    "trigger_volume_vs_previous_max": None,
    "market_median_change_20d_pct_min": None,
    "exclude_one_word_limit_up": False,
    "change_60d_min": None,
    "change_60d_max": None,
    "require_ma_bullish": False,
    "require_price_above_ma20": False,
    "signal_score_min": None,
    "macd_status_whitelist": None,
    "rsi_status_whitelist": None,
    "breakout_20d_pct_min": None,
    "breakout_20d_pct_max": None,
    "range_20d_pct_max": None,
    "volume_ratio_20d_min": None,
    "volume_ratio_20d_max": None,
    "body_pct_min": None,
    "body_pct_max": None,
    "pullback_to_ma20_pct_min": None,
    "pullback_to_ma20_pct_max": None,
    "consolidation_days_20d_min": None,
    "consolidation_days_20d_max": None,
    "volatility_20d_pct_min": None,
    "volatility_20d_pct_max": None,
    "max_drawdown_20d_pct_min": None,
    "max_drawdown_20d_pct_max": None,
    "atr_20_pct_min": None,
    "atr_20_pct_max": None,
}


class SnapshotFieldMissingError(ValueError):
    """Raised when a configured hard filter cannot be evaluated safely."""


def exclude_beijing_stock_exchange_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove current and legacy BSE codes while preserving frame metadata."""
    if df.empty:
        return df.copy()
    mask = pd.Series(True, index=df.index)
    filtered = df.loc[_filter_out_beijing_stocks(df, mask)].copy()
    filtered.attrs.update(df.attrs)
    return filtered


def apply_hard_filters(df: pd.DataFrame, filters: HardFilterConfig) -> pd.DataFrame:
    """Filter snapshot DataFrame by hard conditions. Returns filtered copy."""
    result = df.copy()
    if result.empty:
        return result

    mask = pd.Series(True, index=result.index)

    if filters.exclude_beijing_stock_exchange:
        mask = _filter_out_beijing_stocks(result, mask)

    if filters.exclude_st:
        name_col = _find_col(result, ["name", "股票名称", "名称"]) if mask.any() else None
        if not name_col:
            raise SnapshotFieldMissingError(
                "Missing required snapshot column for exclude_st filter: name"
            )
        mask &= ~result[name_col].str.contains(r"ST|退", na=False)

    # Numeric filters — each is optional
    mask = _filter_min(result, mask, ["amount", "成交额"], filters.amount_min)
    mask = _filter_min(result, mask, ["price", "最新价", "现价"], filters.price_min)
    mask = _filter_max(result, mask, ["price", "最新价", "现价"], filters.price_max)
    mask = _filter_min(result, mask, ["total_mv", "总市值"], filters.market_cap_min)
    mask = _filter_max(result, mask, ["total_mv", "总市值"], filters.market_cap_max)
    mask = _filter_min(result, mask, ["pe_ratio", "市盈率"], filters.pe_ttm_min)
    mask = _filter_max(result, mask, ["pe_ratio", "市盈率"], filters.pe_ttm_max)
    mask = _filter_min(result, mask, ["pb_ratio", "市净率"], filters.pb_min)
    mask = _filter_max(result, mask, ["pb_ratio", "市净率"], filters.pb_max)
    mask = _filter_min(result, mask, ["volume_ratio", "量比"], filters.volume_ratio_min)
    mask = _filter_min(result, mask, ["turnover_rate", "换手率"], filters.turnover_rate_min)
    mask = _filter_min(result, mask, ["change_pct", "涨跌幅"], filters.change_pct_min)
    mask = _filter_max(result, mask, ["change_pct", "涨跌幅"], filters.change_pct_max)
    mask = _filter_min(result, mask, ["market_positive_ratio"], filters.market_positive_ratio_min)
    mask = _filter_min(
        result, mask, ["market_median_change_pct"], filters.market_median_change_pct_min
    )

    mask = _filter_min(result, mask, ["change_60d"], filters.change_60d_min)
    mask = _filter_max(result, mask, ["change_60d"], filters.change_60d_max)
    mask = _filter_bool_true(result, mask, "ma_bullish", filters.require_ma_bullish)
    mask = _filter_bool_true(result, mask, "price_above_ma20", filters.require_price_above_ma20)
    mask = _filter_min(result, mask, ["signal_score"], filters.signal_score_min)
    mask = _filter_in(result, mask, "macd_status", filters.macd_status_whitelist)
    mask = _filter_in(result, mask, "rsi_status", filters.rsi_status_whitelist)
    mask = _filter_min(result, mask, ["breakout_20d_pct"], filters.breakout_20d_pct_min)
    mask = _filter_max(result, mask, ["breakout_20d_pct"], filters.breakout_20d_pct_max)
    mask = _filter_max(result, mask, ["range_20d_pct"], filters.range_20d_pct_max)
    mask = _filter_min(result, mask, ["volume_ratio_20d"], filters.volume_ratio_20d_min)
    mask = _filter_max(result, mask, ["volume_ratio_20d"], filters.volume_ratio_20d_max)
    mask = _filter_min(result, mask, ["body_pct"], filters.body_pct_min)
    mask = _filter_max(result, mask, ["body_pct"], filters.body_pct_max)
    mask = _filter_min(result, mask, ["pullback_to_ma20_pct"], filters.pullback_to_ma20_pct_min)
    mask = _filter_max(result, mask, ["pullback_to_ma20_pct"], filters.pullback_to_ma20_pct_max)
    mask = _filter_min(result, mask, ["consolidation_days_20d"], filters.consolidation_days_20d_min)
    mask = _filter_max(result, mask, ["consolidation_days_20d"], filters.consolidation_days_20d_max)
    mask = _filter_min(result, mask, ["volatility_20d_pct"], filters.volatility_20d_pct_min)
    mask = _filter_max(result, mask, ["volatility_20d_pct"], filters.volatility_20d_pct_max)
    mask = _filter_min(result, mask, ["max_drawdown_20d_pct"], filters.max_drawdown_20d_pct_min)
    mask = _filter_max(result, mask, ["max_drawdown_20d_pct"], filters.max_drawdown_20d_pct_max)
    mask = _filter_min(result, mask, ["atr_20_pct"], filters.atr_20_pct_min)
    mask = _filter_max(result, mask, ["atr_20_pct"], filters.atr_20_pct_max)
    mask = _filter_min(result, mask, ["daily_data_points"], filters.daily_data_points_min)
    mask = _filter_min(result, mask, ["avg_amount_20d"], filters.avg_amount_20d_min)
    mask = _filter_min(result, mask, ["adr_20_pct"], filters.adr_20_pct_min)
    mask = _filter_max(result, mask, ["adr_20_pct"], filters.adr_20_pct_max)
    mask = _filter_min(result, mask, ["change_20d"], filters.change_20d_min)
    mask = _filter_max(result, mask, ["change_20d"], filters.change_20d_max)
    mask = _filter_min(
        result, mask, ["relative_strength_20d_pctile"], filters.relative_strength_20d_pctile_min
    )
    mask = _filter_min(
        result, mask, ["relative_strength_60d_pctile"], filters.relative_strength_60d_pctile_min
    )
    mask = _filter_bool_true(result, mask, "ema_stack_bullish", filters.require_ema_stack)
    mask = _filter_bool_true(result, mask, "ema21_rising", filters.require_ema21_rising)
    mask = _filter_bool_true(result, mask, "ema50_rising", filters.require_ema50_rising)
    mask = _filter_min(
        result, mask, ["distance_to_120d_high_pct"], filters.distance_to_120d_high_pct_min
    )
    mask = _filter_min(result, mask, ["pullback_depth_10d_pct"], filters.pullback_depth_10d_pct_min)
    mask = _filter_max(result, mask, ["pullback_depth_10d_pct"], filters.pullback_depth_10d_pct_max)
    mask = _filter_max(result, mask, ["range_contraction_ratio"], filters.range_contraction_ratio_max)
    mask = _filter_max(result, mask, ["volume_contraction_ratio"], filters.volume_contraction_ratio_max)
    mask = _filter_min(
        result, mask, ["support_confluence_count"], filters.support_confluence_count_min
    )
    mask = _filter_min(result, mask, ["trigger_distance_pct"], filters.trigger_distance_pct_min)
    mask = _filter_max(result, mask, ["trigger_distance_pct"], filters.trigger_distance_pct_max)
    mask = _filter_min(
        result, mask, ["trigger_close_location_pct"], filters.trigger_close_location_pct_min
    )
    mask = _filter_max(
        result, mask, ["trigger_close_location_pct"], filters.trigger_close_location_pct_max
    )
    mask = _filter_min(
        result, mask, ["trigger_volume_vs_previous"], filters.trigger_volume_vs_previous_min
    )
    mask = _filter_max(
        result, mask, ["trigger_volume_vs_previous"], filters.trigger_volume_vs_previous_max
    )
    mask = _filter_min(
        result, mask, ["market_median_change_20d_pct"], filters.market_median_change_20d_pct_min
    )
    mask = _filter_bool_false(
        result, mask, "one_word_limit_up", filters.exclude_one_word_limit_up
    )

    return result.loc[mask].copy()


def hard_filter_rejection_summary(
    df: pd.DataFrame,
    filters: HardFilterConfig,
    *,
    limit: int = 8,
) -> list[str]:
    """Return compact sequential hard-filter rejection counts."""
    if df.empty or limit <= 0:
        return []

    mask = pd.Series(True, index=df.index)
    diagnostics: list[str] = []

    def record(label: str, next_mask: pd.Series) -> None:
        nonlocal mask
        before = int(mask.sum())
        after = int(next_mask.sum())
        removed = before - after
        if removed > 0 and len(diagnostics) < limit:
            diagnostics.append(f"{label} removed {removed} ({before}->{after})")
        mask = next_mask

    if filters.exclude_beijing_stock_exchange:
        record(
            "exclude_beijing_stock_exchange",
            _filter_out_beijing_stocks(df, mask),
        )

    if filters.exclude_st:
        name_col = _find_col(df, ["name", "股票名称", "名称"]) if mask.any() else None
        if not name_col:
            raise SnapshotFieldMissingError(
                "Missing required snapshot column for exclude_st filter: name"
            )
        record("exclude_st", mask & ~df[name_col].str.contains(r"ST|退", na=False))

    def record_min(label: str, columns: list[str], value: float | None) -> None:
        if value is not None:
            record(label, _filter_min(df, mask, columns, value))

    def record_max(label: str, columns: list[str], value: float | None) -> None:
        if value is not None:
            record(label, _filter_max(df, mask, columns, value))

    record_min("amount_min", ["amount", "成交额"], filters.amount_min)
    record_min("price_min", ["price", "最新价", "现价"], filters.price_min)
    record_max("price_max", ["price", "最新价", "现价"], filters.price_max)
    record_min("market_cap_min", ["total_mv", "总市值"], filters.market_cap_min)
    record_max("market_cap_max", ["total_mv", "总市值"], filters.market_cap_max)
    record_min("pe_ttm_min", ["pe_ratio", "市盈率"], filters.pe_ttm_min)
    record_max("pe_ttm_max", ["pe_ratio", "市盈率"], filters.pe_ttm_max)
    record_min("pb_min", ["pb_ratio", "市净率"], filters.pb_min)
    record_max("pb_max", ["pb_ratio", "市净率"], filters.pb_max)
    record_min("volume_ratio_min", ["volume_ratio", "量比"], filters.volume_ratio_min)
    record_min("turnover_rate_min", ["turnover_rate", "换手率"], filters.turnover_rate_min)
    record_min("change_pct_min", ["change_pct", "涨跌幅"], filters.change_pct_min)
    record_max("change_pct_max", ["change_pct", "涨跌幅"], filters.change_pct_max)
    record_min("market_positive_ratio_min", ["market_positive_ratio"], filters.market_positive_ratio_min)
    record_min("market_median_change_pct_min", ["market_median_change_pct"], filters.market_median_change_pct_min)
    record_min("change_60d_min", ["change_60d"], filters.change_60d_min)
    record_max("change_60d_max", ["change_60d"], filters.change_60d_max)

    if filters.require_ma_bullish:
        record("require_ma_bullish", _filter_bool_true(df, mask, "ma_bullish", True))
    if filters.require_price_above_ma20:
        record("require_price_above_ma20", _filter_bool_true(df, mask, "price_above_ma20", True))

    record_min("signal_score_min", ["signal_score"], filters.signal_score_min)
    if filters.macd_status_whitelist:
        record("macd_status_whitelist", _filter_in(df, mask, "macd_status", filters.macd_status_whitelist))
    if filters.rsi_status_whitelist:
        record("rsi_status_whitelist", _filter_in(df, mask, "rsi_status", filters.rsi_status_whitelist))
    record_min("breakout_20d_pct_min", ["breakout_20d_pct"], filters.breakout_20d_pct_min)
    record_max("breakout_20d_pct_max", ["breakout_20d_pct"], filters.breakout_20d_pct_max)
    record_max("range_20d_pct_max", ["range_20d_pct"], filters.range_20d_pct_max)
    record_min("volume_ratio_20d_min", ["volume_ratio_20d"], filters.volume_ratio_20d_min)
    record_max("volume_ratio_20d_max", ["volume_ratio_20d"], filters.volume_ratio_20d_max)
    record_min("body_pct_min", ["body_pct"], filters.body_pct_min)
    record_max("body_pct_max", ["body_pct"], filters.body_pct_max)
    record_min("pullback_to_ma20_pct_min", ["pullback_to_ma20_pct"], filters.pullback_to_ma20_pct_min)
    record_max("pullback_to_ma20_pct_max", ["pullback_to_ma20_pct"], filters.pullback_to_ma20_pct_max)
    record_min("consolidation_days_20d_min", ["consolidation_days_20d"], filters.consolidation_days_20d_min)
    record_max("consolidation_days_20d_max", ["consolidation_days_20d"], filters.consolidation_days_20d_max)
    record_min("volatility_20d_pct_min", ["volatility_20d_pct"], filters.volatility_20d_pct_min)
    record_max("volatility_20d_pct_max", ["volatility_20d_pct"], filters.volatility_20d_pct_max)
    record_min("max_drawdown_20d_pct_min", ["max_drawdown_20d_pct"], filters.max_drawdown_20d_pct_min)
    record_max("max_drawdown_20d_pct_max", ["max_drawdown_20d_pct"], filters.max_drawdown_20d_pct_max)
    record_min("atr_20_pct_min", ["atr_20_pct"], filters.atr_20_pct_min)
    record_max("atr_20_pct_max", ["atr_20_pct"], filters.atr_20_pct_max)
    record_min("daily_data_points_min", ["daily_data_points"], filters.daily_data_points_min)
    record_min("avg_amount_20d_min", ["avg_amount_20d"], filters.avg_amount_20d_min)
    record_min("adr_20_pct_min", ["adr_20_pct"], filters.adr_20_pct_min)
    record_max("adr_20_pct_max", ["adr_20_pct"], filters.adr_20_pct_max)
    record_min("change_20d_min", ["change_20d"], filters.change_20d_min)
    record_max("change_20d_max", ["change_20d"], filters.change_20d_max)
    record_min("relative_strength_20d_pctile_min", ["relative_strength_20d_pctile"], filters.relative_strength_20d_pctile_min)
    record_min("relative_strength_60d_pctile_min", ["relative_strength_60d_pctile"], filters.relative_strength_60d_pctile_min)
    if filters.require_ema_stack:
        record("require_ema_stack", _filter_bool_true(df, mask, "ema_stack_bullish", True))
    if filters.require_ema21_rising:
        record("require_ema21_rising", _filter_bool_true(df, mask, "ema21_rising", True))
    if filters.require_ema50_rising:
        record("require_ema50_rising", _filter_bool_true(df, mask, "ema50_rising", True))
    record_min("distance_to_120d_high_pct_min", ["distance_to_120d_high_pct"], filters.distance_to_120d_high_pct_min)
    record_min("pullback_depth_10d_pct_min", ["pullback_depth_10d_pct"], filters.pullback_depth_10d_pct_min)
    record_max("pullback_depth_10d_pct_max", ["pullback_depth_10d_pct"], filters.pullback_depth_10d_pct_max)
    record_max("range_contraction_ratio_max", ["range_contraction_ratio"], filters.range_contraction_ratio_max)
    record_max("volume_contraction_ratio_max", ["volume_contraction_ratio"], filters.volume_contraction_ratio_max)
    record_min("support_confluence_count_min", ["support_confluence_count"], filters.support_confluence_count_min)
    record_min("trigger_distance_pct_min", ["trigger_distance_pct"], filters.trigger_distance_pct_min)
    record_max("trigger_distance_pct_max", ["trigger_distance_pct"], filters.trigger_distance_pct_max)
    record_min("trigger_close_location_pct_min", ["trigger_close_location_pct"], filters.trigger_close_location_pct_min)
    record_max("trigger_close_location_pct_max", ["trigger_close_location_pct"], filters.trigger_close_location_pct_max)
    record_min("trigger_volume_vs_previous_min", ["trigger_volume_vs_previous"], filters.trigger_volume_vs_previous_min)
    record_max("trigger_volume_vs_previous_max", ["trigger_volume_vs_previous"], filters.trigger_volume_vs_previous_max)
    record_min("market_median_change_20d_pct_min", ["market_median_change_20d_pct"], filters.market_median_change_20d_pct_min)
    if filters.exclude_one_word_limit_up:
        record("exclude_one_word_limit_up", _filter_bool_false(df, mask, "one_word_limit_up", True))

    return diagnostics


def hard_filter_waterfall(
    df: pd.DataFrame,
    filters: HardFilterConfig,
    *,
    sample_limit: int = 3,
) -> list[dict[str, object]]:
    """Return sequential hard-filter counts plus rejected row samples."""
    if df.empty:
        return []

    mask = pd.Series(True, index=df.index)
    steps: list[dict[str, object]] = []

    def record(label: str, next_mask: pd.Series, value_columns: list[str] | None = None) -> None:
        nonlocal mask
        before = int(mask.sum())
        after = int(next_mask.sum())
        removed = before - after
        step: dict[str, object] = {
            "filter": label,
            "before": before,
            "after": after,
            "removed": removed,
            "removed_pct": round((removed / before * 100.0), 4) if before else 0.0,
        }
        if removed > 0 and sample_limit > 0:
            step["samples"] = _rejection_samples(
                df.loc[mask & ~next_mask],
                value_columns=value_columns or [],
                limit=sample_limit,
            )
        suggestion = _waterfall_suggestion(label, before, after, removed)
        if suggestion:
            step["suggestion"] = suggestion
        steps.append(step)
        mask = next_mask

    if filters.exclude_beijing_stock_exchange:
        code_col = _find_col(df, ["code", "股票代码", "代码"])
        record(
            "exclude_beijing_stock_exchange",
            _filter_out_beijing_stocks(df, mask),
            [code_col] if code_col else [],
        )

    if filters.exclude_st:
        name_col = _find_col(df, ["name", "股票名称", "名称"]) if mask.any() else None
        if not name_col:
            raise SnapshotFieldMissingError(
                "Missing required snapshot column for exclude_st filter: name"
            )
        record("exclude_st", mask & ~df[name_col].str.contains(r"ST|退", na=False), [name_col])

    def record_min(label: str, columns: list[str], value: float | None) -> None:
        if value is not None:
            record(label, _filter_min(df, mask, columns, value), columns)

    def record_max(label: str, columns: list[str], value: float | None) -> None:
        if value is not None:
            record(label, _filter_max(df, mask, columns, value), columns)

    record_min("amount_min", ["amount", "成交额"], filters.amount_min)
    record_min("price_min", ["price", "最新价", "现价"], filters.price_min)
    record_max("price_max", ["price", "最新价", "现价"], filters.price_max)
    record_min("market_cap_min", ["total_mv", "总市值"], filters.market_cap_min)
    record_max("market_cap_max", ["total_mv", "总市值"], filters.market_cap_max)
    record_min("pe_ttm_min", ["pe_ratio", "市盈率"], filters.pe_ttm_min)
    record_max("pe_ttm_max", ["pe_ratio", "市盈率"], filters.pe_ttm_max)
    record_min("pb_min", ["pb_ratio", "市净率"], filters.pb_min)
    record_max("pb_max", ["pb_ratio", "市净率"], filters.pb_max)
    record_min("volume_ratio_min", ["volume_ratio", "量比"], filters.volume_ratio_min)
    record_min("turnover_rate_min", ["turnover_rate", "换手率"], filters.turnover_rate_min)
    record_min("change_pct_min", ["change_pct", "涨跌幅"], filters.change_pct_min)
    record_max("change_pct_max", ["change_pct", "涨跌幅"], filters.change_pct_max)
    record_min("market_positive_ratio_min", ["market_positive_ratio"], filters.market_positive_ratio_min)
    record_min("market_median_change_pct_min", ["market_median_change_pct"], filters.market_median_change_pct_min)
    record_min("change_60d_min", ["change_60d"], filters.change_60d_min)
    record_max("change_60d_max", ["change_60d"], filters.change_60d_max)

    if filters.require_ma_bullish:
        record("require_ma_bullish", _filter_bool_true(df, mask, "ma_bullish", True), ["ma_bullish"])
    if filters.require_price_above_ma20:
        record("require_price_above_ma20", _filter_bool_true(df, mask, "price_above_ma20", True), ["price_above_ma20"])

    record_min("signal_score_min", ["signal_score"], filters.signal_score_min)
    if filters.macd_status_whitelist:
        record("macd_status_whitelist", _filter_in(df, mask, "macd_status", filters.macd_status_whitelist), ["macd_status"])
    if filters.rsi_status_whitelist:
        record("rsi_status_whitelist", _filter_in(df, mask, "rsi_status", filters.rsi_status_whitelist), ["rsi_status"])
    record_min("breakout_20d_pct_min", ["breakout_20d_pct"], filters.breakout_20d_pct_min)
    record_max("breakout_20d_pct_max", ["breakout_20d_pct"], filters.breakout_20d_pct_max)
    record_max("range_20d_pct_max", ["range_20d_pct"], filters.range_20d_pct_max)
    record_min("volume_ratio_20d_min", ["volume_ratio_20d"], filters.volume_ratio_20d_min)
    record_max("volume_ratio_20d_max", ["volume_ratio_20d"], filters.volume_ratio_20d_max)
    record_min("body_pct_min", ["body_pct"], filters.body_pct_min)
    record_max("body_pct_max", ["body_pct"], filters.body_pct_max)
    record_min("pullback_to_ma20_pct_min", ["pullback_to_ma20_pct"], filters.pullback_to_ma20_pct_min)
    record_max("pullback_to_ma20_pct_max", ["pullback_to_ma20_pct"], filters.pullback_to_ma20_pct_max)
    record_min("consolidation_days_20d_min", ["consolidation_days_20d"], filters.consolidation_days_20d_min)
    record_max("consolidation_days_20d_max", ["consolidation_days_20d"], filters.consolidation_days_20d_max)
    record_min("volatility_20d_pct_min", ["volatility_20d_pct"], filters.volatility_20d_pct_min)
    record_max("volatility_20d_pct_max", ["volatility_20d_pct"], filters.volatility_20d_pct_max)
    record_min("max_drawdown_20d_pct_min", ["max_drawdown_20d_pct"], filters.max_drawdown_20d_pct_min)
    record_max("max_drawdown_20d_pct_max", ["max_drawdown_20d_pct"], filters.max_drawdown_20d_pct_max)
    record_min("atr_20_pct_min", ["atr_20_pct"], filters.atr_20_pct_min)
    record_max("atr_20_pct_max", ["atr_20_pct"], filters.atr_20_pct_max)
    record_min("daily_data_points_min", ["daily_data_points"], filters.daily_data_points_min)
    record_min("avg_amount_20d_min", ["avg_amount_20d"], filters.avg_amount_20d_min)
    record_min("adr_20_pct_min", ["adr_20_pct"], filters.adr_20_pct_min)
    record_max("adr_20_pct_max", ["adr_20_pct"], filters.adr_20_pct_max)
    record_min("change_20d_min", ["change_20d"], filters.change_20d_min)
    record_max("change_20d_max", ["change_20d"], filters.change_20d_max)
    record_min("relative_strength_20d_pctile_min", ["relative_strength_20d_pctile"], filters.relative_strength_20d_pctile_min)
    record_min("relative_strength_60d_pctile_min", ["relative_strength_60d_pctile"], filters.relative_strength_60d_pctile_min)
    if filters.require_ema_stack:
        record("require_ema_stack", _filter_bool_true(df, mask, "ema_stack_bullish", True), ["ema_stack_bullish"])
    if filters.require_ema21_rising:
        record("require_ema21_rising", _filter_bool_true(df, mask, "ema21_rising", True), ["ema21_rising"])
    if filters.require_ema50_rising:
        record("require_ema50_rising", _filter_bool_true(df, mask, "ema50_rising", True), ["ema50_rising"])
    record_min("distance_to_120d_high_pct_min", ["distance_to_120d_high_pct"], filters.distance_to_120d_high_pct_min)
    record_min("pullback_depth_10d_pct_min", ["pullback_depth_10d_pct"], filters.pullback_depth_10d_pct_min)
    record_max("pullback_depth_10d_pct_max", ["pullback_depth_10d_pct"], filters.pullback_depth_10d_pct_max)
    record_max("range_contraction_ratio_max", ["range_contraction_ratio"], filters.range_contraction_ratio_max)
    record_max("volume_contraction_ratio_max", ["volume_contraction_ratio"], filters.volume_contraction_ratio_max)
    record_min("support_confluence_count_min", ["support_confluence_count"], filters.support_confluence_count_min)
    record_min("trigger_distance_pct_min", ["trigger_distance_pct"], filters.trigger_distance_pct_min)
    record_max("trigger_distance_pct_max", ["trigger_distance_pct"], filters.trigger_distance_pct_max)
    record_min("trigger_close_location_pct_min", ["trigger_close_location_pct"], filters.trigger_close_location_pct_min)
    record_max("trigger_close_location_pct_max", ["trigger_close_location_pct"], filters.trigger_close_location_pct_max)
    record_min("trigger_volume_vs_previous_min", ["trigger_volume_vs_previous"], filters.trigger_volume_vs_previous_min)
    record_max("trigger_volume_vs_previous_max", ["trigger_volume_vs_previous"], filters.trigger_volume_vs_previous_max)
    record_min("market_median_change_20d_pct_min", ["market_median_change_20d_pct"], filters.market_median_change_20d_pct_min)
    if filters.exclude_one_word_limit_up:
        record("exclude_one_word_limit_up", _filter_bool_false(df, mask, "one_word_limit_up", True), ["one_word_limit_up"])

    return steps


def requires_daily_features(filters: HardFilterConfig) -> bool:
    """Return whether a hard-filter config needs daily K-line features."""
    return any([
        filters.daily_data_points_min is not None,
        filters.avg_amount_20d_min is not None,
        filters.adr_20_pct_min is not None,
        filters.adr_20_pct_max is not None,
        filters.change_20d_min is not None,
        filters.change_20d_max is not None,
        filters.relative_strength_20d_pctile_min is not None,
        filters.relative_strength_60d_pctile_min is not None,
        filters.require_ema_stack,
        filters.require_ema21_rising,
        filters.require_ema50_rising,
        filters.distance_to_120d_high_pct_min is not None,
        filters.pullback_depth_10d_pct_min is not None,
        filters.pullback_depth_10d_pct_max is not None,
        filters.range_contraction_ratio_max is not None,
        filters.volume_contraction_ratio_max is not None,
        filters.support_confluence_count_min is not None,
        filters.trigger_distance_pct_min is not None,
        filters.trigger_distance_pct_max is not None,
        filters.trigger_close_location_pct_min is not None,
        filters.trigger_close_location_pct_max is not None,
        filters.trigger_volume_vs_previous_min is not None,
        filters.trigger_volume_vs_previous_max is not None,
        filters.market_median_change_20d_pct_min is not None,
        filters.exclude_one_word_limit_up,
        filters.change_60d_min is not None,
        filters.change_60d_max is not None,
        filters.require_ma_bullish,
        filters.require_price_above_ma20,
        filters.signal_score_min is not None,
        bool(filters.macd_status_whitelist),
        bool(filters.rsi_status_whitelist),
        filters.breakout_20d_pct_min is not None,
        filters.breakout_20d_pct_max is not None,
        filters.range_20d_pct_max is not None,
        filters.volume_ratio_20d_min is not None,
        filters.volume_ratio_20d_max is not None,
        filters.body_pct_min is not None,
        filters.body_pct_max is not None,
        filters.pullback_to_ma20_pct_min is not None,
        filters.pullback_to_ma20_pct_max is not None,
        filters.consolidation_days_20d_min is not None,
        filters.consolidation_days_20d_max is not None,
        filters.volatility_20d_pct_min is not None,
        filters.volatility_20d_pct_max is not None,
        filters.max_drawdown_20d_pct_min is not None,
        filters.max_drawdown_20d_pct_max is not None,
        filters.atr_20_pct_min is not None,
        filters.atr_20_pct_max is not None,
    ])


def without_daily_filters(filters: HardFilterConfig) -> HardFilterConfig:
    """Return a copy with daily K-line filters disabled."""
    return replace(filters, **_DAILY_FILTER_DEFAULTS)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _filter_out_beijing_stocks(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    code_col = _find_col(df, ["code", "股票代码", "代码"])
    if not code_col:
        raise SnapshotFieldMissingError(
            "Missing required snapshot column for exclude_beijing_stock_exchange filter: code"
        )
    excluded = df[code_col].map(is_beijing_stock_code)
    return mask & ~excluded


def _filter_min(
    df: pd.DataFrame,
    mask: pd.Series,
    col_names: list[str],
    value: float | None,
) -> pd.Series:
    if value is None:
        return mask
    if not mask.any():
        return mask
    col = _find_col(df, col_names)
    if not col:
        raise SnapshotFieldMissingError(
            f"Missing required snapshot column for min filter {col_names}: "
            f"configured value={value}"
        )
    series = pd.to_numeric(df[col], errors="coerce")
    return mask & series.ge(value) & series.notna()


def _filter_max(
    df: pd.DataFrame,
    mask: pd.Series,
    col_names: list[str],
    value: float | None,
) -> pd.Series:
    if value is None:
        return mask
    if not mask.any():
        return mask
    col = _find_col(df, col_names)
    if not col:
        raise SnapshotFieldMissingError(
            f"Missing required snapshot column for max filter {col_names}: "
            f"configured value={value}"
        )
    series = pd.to_numeric(df[col], errors="coerce")
    return mask & series.le(value) & series.notna()


def _filter_bool_true(
    df: pd.DataFrame,
    mask: pd.Series,
    col_name: str,
    enabled: bool,
) -> pd.Series:
    if not enabled:
        return mask
    if not mask.any():
        return mask
    if col_name not in df.columns:
        raise SnapshotFieldMissingError(
            f"Missing required daily feature column for bool filter: {col_name}"
        )
    return mask & (df[col_name] == True)  # noqa: E712


def _filter_bool_false(
    df: pd.DataFrame,
    mask: pd.Series,
    col_name: str,
    enabled: bool,
) -> pd.Series:
    if not enabled:
        return mask
    if not mask.any():
        return mask
    if col_name not in df.columns:
        raise SnapshotFieldMissingError(
            f"Missing required daily feature column for bool filter: {col_name}"
        )
    return mask & (df[col_name] == False)  # noqa: E712


def _filter_in(
    df: pd.DataFrame,
    mask: pd.Series,
    col_name: str,
    allowed: list[str] | None,
) -> pd.Series:
    if not allowed:
        return mask
    if not mask.any():
        return mask
    if col_name not in df.columns:
        raise SnapshotFieldMissingError(
            f"Missing required daily feature column for whitelist filter: {col_name}"
        )
    allowed_set = {str(item) for item in allowed}
    return mask & df[col_name].astype(str).isin(allowed_set)


def _rejection_samples(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    limit: int,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    name_col = _find_col(df, ["name", "股票名称", "名称"])
    code_col = _find_col(df, ["code", "代码", "symbol"])
    value_col = _find_col(df, value_columns)
    for _, row in df.head(limit).iterrows():
        item: dict[str, object] = {}
        if code_col:
            item["code"] = str(row.get(code_col, ""))
        if name_col:
            item["name"] = str(row.get(name_col, ""))
        if value_col:
            value = row.get(value_col)
            item["value"] = None if pd.isna(value) else value
        samples.append(item)
    return samples


def _waterfall_suggestion(label: str, before: int, after: int, removed: int) -> str:
    if before <= 0 or removed <= 0:
        return ""
    if after == 0:
        return f"{label} eliminated all remaining candidates; inspect threshold or data availability."
    if removed / before >= 0.9:
        return f"{label} removed over 90% of remaining candidates; consider reviewing this threshold."
    return ""
