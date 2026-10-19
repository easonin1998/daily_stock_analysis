# -*- coding: utf-8 -*-
"""Point-in-time market regime scoring for the Martin momentum strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd


REGIME_STRONG_RISK_ON = "strong_risk_on"
REGIME_RISK_ON = "risk_on"
REGIME_NEUTRAL = "neutral"
REGIME_RISK_OFF = "risk_off"
REGIME_UNAVAILABLE = "unavailable"

_STATE_RANK = {
    REGIME_RISK_OFF: 0,
    REGIME_NEUTRAL: 1,
    REGIME_RISK_ON: 2,
    REGIME_STRONG_RISK_ON: 3,
}
_REGIME_HISTORY_LOCK = threading.Lock()


@dataclass(frozen=True)
class MarketRegimePolicy:
    """Stable score thresholds and position multipliers for market states."""

    min_universe_size: int = 100
    upgrade_confirmation_days: int = 2
    strong_risk_on_score: float = 70.0
    risk_on_score: float = 55.0
    neutral_score: float = 40.0
    shock_positive_ratio_max: float = 20.0
    shock_median_change_pct_max: float = -3.0
    shock_severe_decline_ratio_max: float = 8.0


_PROFILES: dict[str, dict[str, float]] = {
    REGIME_STRONG_RISK_ON: {
        "target_exposure_pct": 100.0,
        "max_positions_fraction": 1.0,
        "risk_multiplier": 1.0,
        "heat_multiplier": 1.0,
        "position_weight_multiplier": 1.0,
        "min_position_weight_multiplier": 1.0,
    },
    REGIME_RISK_ON: {
        "target_exposure_pct": 80.0,
        "max_positions_fraction": 0.8,
        "risk_multiplier": 0.8,
        "heat_multiplier": 0.8,
        "position_weight_multiplier": 1.0,
        "min_position_weight_multiplier": 0.8,
    },
    REGIME_NEUTRAL: {
        "target_exposure_pct": 40.0,
        "max_positions_fraction": 0.4,
        "risk_multiplier": 0.5,
        "heat_multiplier": 0.4,
        "position_weight_multiplier": 0.75,
        "min_position_weight_multiplier": 0.5,
    },
    REGIME_RISK_OFF: {
        "target_exposure_pct": 0.0,
        "max_positions_fraction": 0.0,
        "risk_multiplier": 0.0,
        "heat_multiplier": 0.0,
        "position_weight_multiplier": 0.0,
        "min_position_weight_multiplier": 0.0,
    },
    REGIME_UNAVAILABLE: {
        "target_exposure_pct": 100.0,
        "max_positions_fraction": 1.0,
        "risk_multiplier": 1.0,
        "heat_multiplier": 1.0,
        "position_weight_multiplier": 1.0,
        "min_position_weight_multiplier": 1.0,
    },
}


def regime_profile(state: Any) -> dict[str, float]:
    """Return a copy of the execution profile for one state."""
    normalized = str(state or REGIME_UNAVAILABLE)
    return dict(_PROFILES.get(normalized, _PROFILES[REGIME_UNAVAILABLE]))


def attach_market_regime_features(
    panel: pd.DataFrame,
    *,
    policy: MarketRegimePolicy | None = None,
) -> pd.DataFrame:
    """Attach daily regime fields to every row of a historical stock panel."""
    if panel.empty or "date" not in panel.columns:
        return panel.copy()
    observations = compute_market_regime_observations(panel)
    history = classify_market_regimes(observations, policy=policy)
    if history.empty:
        return panel.copy()
    fields = history.rename(
        columns={
            column: f"market_regime_{column}"
            for column in history.columns
            if column != "date"
        }
    )
    return panel.merge(fields, on="date", how="left", validate="many_to_one")


def compute_market_regime_observations(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one point-in-time cross-section into one observation per date."""
    if panel.empty or "date" not in panel.columns:
        return pd.DataFrame()
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work[work["date"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    if "relative_strength_60d_pctile" not in work.columns:
        change_60d = _numeric(work, "change_60d")
        work["relative_strength_60d_pctile"] = change_60d.groupby(work["date"]).rank(
            method="average", pct=True
        ) * 100.0

    rows = [_daily_observation(day, pd.Timestamp(day_date)) for day_date, day in work.groupby("date", sort=True)]
    return pd.DataFrame(rows).sort_values("date", kind="mergesort").reset_index(drop=True)


def classify_market_regimes(
    observations: pd.DataFrame,
    *,
    policy: MarketRegimePolicy | None = None,
) -> pd.DataFrame:
    """Score observations and apply immediate downgrades plus confirmed upgrades."""
    rules = policy or MarketRegimePolicy()
    if observations.empty:
        return observations.copy()
    result = observations.copy().sort_values("date", kind="mergesort").reset_index(drop=True)

    result["positive_ratio_5d"] = _rolling_mean(result, "positive_ratio", 5)
    result["above_ema21_ratio_5d"] = _rolling_mean(result, "above_ema21_ratio", 5)
    result["above_ema21_change_5d"] = (
        _numeric(result, "above_ema21_ratio")
        - _numeric(result, "above_ema21_ratio").shift(5)
    ).fillna(0.0)
    result["leader_count_baseline_20d"] = (
        _numeric(result, "leader_count").rolling(20, min_periods=5).median().shift(1)
    )
    result["leader_expansion_ratio"] = [
        _expansion_ratio(current, baseline)
        for current, baseline in zip(
            _numeric(result, "leader_count"),
            _numeric(result, "leader_count_baseline_20d"),
        )
    ]

    result["trend_score"] = (
        _scale_series(result["above_ema21_ratio"], 35.0, 65.0, 10.0)
        + _scale_series(result["above_ema50_ratio"], 30.0, 60.0, 8.0)
        + _scale_series(result["median_change_20d_pct"], -5.0, 10.0, 7.0)
    )
    result["breadth_score"] = (
        _scale_series(result["positive_ratio_5d"], 40.0, 60.0, 10.0)
        + _scale_series(result["above_ema21_change_5d"], -7.0, 7.0, 10.0)
        + _scale_series(result["ema21_rising_ratio"], 35.0, 65.0, 10.0)
    )
    result["leader_score"] = (
        _scale_series(result["leader_ratio"], 0.5, 3.0, 10.0)
        + _scale_series(result["leader_expansion_ratio"], 0.6, 1.4, 10.0)
        + _scale_series(result["leader_median_change_pct"], -1.0, 2.0, 5.0)
        + _scale_series(result["leader_median_change_20d_pct"], 10.0, 35.0, 5.0)
    )
    result["risk_score"] = (
        _scale_series(result["positive_ratio"], 25.0, 50.0, 5.0)
        + _scale_series(result["median_change_pct"], -3.0, 0.0, 5.0)
        + _inverse_scale_series(result["severe_decline_ratio"], 1.0, 8.0, 5.0)
    )
    result["score"] = result[
        ["trend_score", "breadth_score", "leader_score", "risk_score"]
    ].sum(axis=1).clip(0.0, 100.0)

    raw_states: list[str] = []
    reasons: list[str] = []
    for row in result.itertuples(index=False):
        state, reason = _raw_state(row, rules)
        raw_states.append(state)
        reasons.append(reason)
    result["raw_state"] = raw_states
    result["state"] = _apply_upgrade_confirmation(
        raw_states,
        max(int(rules.upgrade_confirmation_days), 1),
    )
    result["reason"] = reasons
    pending_upgrade = result["state"].ne(result["raw_state"])
    result.loc[pending_upgrade, "reason"] = (
        result.loc[pending_upgrade, "reason"]
        + f"；升级需连续 {max(int(rules.upgrade_confirmation_days), 1)} 日确认，"
        + "当前维持 "
        + result.loc[pending_upgrade, "state"].astype(str)
    )
    result["available"] = result["universe_size"].ge(rules.min_universe_size)
    unavailable = ~result["available"]
    result.loc[unavailable, ["raw_state", "state"]] = REGIME_UNAVAILABLE
    result.loc[unavailable, "reason"] = (
        "市场状态样本不足："
        + result.loc[unavailable, "universe_size"].astype(int).astype(str)
        + f" < {rules.min_universe_size}"
    )

    for key in (
        "target_exposure_pct",
        "max_positions_fraction",
        "risk_multiplier",
        "heat_multiplier",
        "position_weight_multiplier",
        "min_position_weight_multiplier",
    ):
        result[key] = result["state"].map(lambda state: regime_profile(state)[key])

    float_columns = [
        "positive_ratio",
        "median_change_pct",
        "above_ema21_ratio",
        "above_ema50_ratio",
        "ema21_rising_ratio",
        "ema50_rising_ratio",
        "median_change_20d_pct",
        "leader_ratio",
        "leader_median_change_pct",
        "leader_median_change_20d_pct",
        "severe_decline_ratio",
        "positive_ratio_5d",
        "above_ema21_ratio_5d",
        "above_ema21_change_5d",
        "leader_count_baseline_20d",
        "leader_expansion_ratio",
        "trend_score",
        "breadth_score",
        "leader_score",
        "risk_score",
        "score",
    ]
    for column in float_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").round(4)
    return result


def latest_market_regime(history: pd.DataFrame) -> dict[str, Any]:
    """Return a stable JSON-ready summary for the latest classified date."""
    if history.empty:
        return unavailable_market_regime("没有可用的市场状态记录")
    row = history.iloc[-1]
    state = str(row.get("state") or REGIME_UNAVAILABLE)
    return {
        "status": "available" if state != REGIME_UNAVAILABLE else "unavailable",
        "date": _date_text(row.get("date")),
        "state": state,
        "raw_state": str(row.get("raw_state") or state),
        "score": _rounded(row.get("score")),
        "components": {
            "trend": _rounded(row.get("trend_score")),
            "breadth": _rounded(row.get("breadth_score")),
            "leaders": _rounded(row.get("leader_score")),
            "risk": _rounded(row.get("risk_score")),
        },
        "measurements": {
            "universe_size": int(_number(row.get("universe_size")) or 0),
            "positive_ratio_5d": _rounded(row.get("positive_ratio_5d")),
            "above_ema21_ratio": _rounded(row.get("above_ema21_ratio")),
            "above_ema50_ratio": _rounded(row.get("above_ema50_ratio")),
            "median_change_20d_pct": _rounded(row.get("median_change_20d_pct")),
            "leader_count": int(_number(row.get("leader_count")) or 0),
            "leader_expansion_ratio": _rounded(row.get("leader_expansion_ratio")),
            "severe_decline_ratio": _rounded(row.get("severe_decline_ratio")),
        },
        "profile": regime_profile(state),
        "reason": str(row.get("reason") or ""),
        "history_days": int(len(history)),
    }


def unavailable_market_regime(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "date": "",
        "state": REGIME_UNAVAILABLE,
        "raw_state": REGIME_UNAVAILABLE,
        "score": None,
        "components": {},
        "measurements": {},
        "profile": regime_profile(REGIME_UNAVAILABLE),
        "reason": reason,
        "history_days": 0,
    }


def update_market_regime_history(
    frame: pd.DataFrame,
    *,
    history_path: str | Path,
    as_of: date,
    policy: MarketRegimePolicy | None = None,
    persist: bool = True,
    retention_days: int = 180,
) -> dict[str, Any]:
    """Append one live cross-section observation and classify stored history."""
    observation_frame = frame.copy()
    observation_frame["date"] = pd.Timestamp(as_of)
    current = compute_market_regime_observations(observation_frame)
    if current.empty:
        return unavailable_market_regime("无法从全市场快照计算市场状态")

    path = Path(history_path)
    with _REGIME_HISTORY_LOCK:
        stored = _read_history(path)
        observations = pd.concat([stored, current], ignore_index=True, sort=False)
        observations["date"] = pd.to_datetime(observations["date"], errors="coerce").dt.normalize()
        observations = (
            observations[observations["date"].notna()]
            .drop_duplicates("date", keep="last")
            .sort_values("date", kind="mergesort")
            .tail(max(int(retention_days), 20))
            .reset_index(drop=True)
        )
        if persist:
            _write_history(path, observations)
    return latest_market_regime(classify_market_regimes(observations, policy=policy))


def load_market_regime_history(
    history_path: str | Path,
    *,
    policy: MarketRegimePolicy | None = None,
) -> dict[str, Any]:
    """Read and classify the last persisted live market observation."""
    history = _read_history(Path(history_path))
    if history.empty:
        return unavailable_market_regime("尚未建立市场状态历史")
    return latest_market_regime(classify_market_regimes(history, policy=policy))


def resolve_market_regime_limits(
    regime: Mapping[str, Any] | None,
    *,
    base_max_positions: int,
    base_risk_per_trade_pct: float,
    base_max_portfolio_heat_pct: float,
    base_max_position_weight_pct: float,
    base_min_position_weight_pct: float,
) -> dict[str, Any]:
    """Resolve executable portfolio limits from one classified market state."""
    state = str((regime or {}).get("state") or REGIME_UNAVAILABLE)
    profile = regime_profile(state)
    position_fraction = profile["max_positions_fraction"]
    max_positions = (
        max(1, int(math.floor(base_max_positions * position_fraction + 1e-9)))
        if position_fraction > 0
        else 0
    )
    return {
        "state": state,
        "score": (regime or {}).get("score"),
        "target_exposure_pct": profile["target_exposure_pct"],
        "max_positions": max_positions,
        "risk_per_trade_pct": round(
            base_risk_per_trade_pct * profile["risk_multiplier"], 6
        ),
        "max_portfolio_heat_pct": round(
            base_max_portfolio_heat_pct * profile["heat_multiplier"], 6
        ),
        "max_position_weight_pct": round(
            base_max_position_weight_pct * profile["position_weight_multiplier"], 6
        ),
        "min_position_weight_pct": round(
            base_min_position_weight_pct
            * profile["min_position_weight_multiplier"],
            6,
        ),
        "new_entries_allowed": state != REGIME_RISK_OFF,
        "profile": profile,
    }


def market_regime_from_row(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Extract execution fields from a panel/signal row."""
    getter = row.get if isinstance(row, Mapping) else lambda key, default=None: getattr(row, key, default)
    state = str(getter("market_regime_state", REGIME_UNAVAILABLE) or REGIME_UNAVAILABLE)
    return {
        "status": "available" if state != REGIME_UNAVAILABLE else "unavailable",
        "date": _date_text(getter("date", None)),
        "state": state,
        "raw_state": str(getter("market_regime_raw_state", state) or state),
        "score": _rounded(getter("market_regime_score", None)),
        "profile": regime_profile(state),
        "reason": str(getter("market_regime_reason", "") or ""),
    }


def _daily_observation(day: pd.DataFrame, day_date: pd.Timestamp) -> dict[str, Any]:
    change = _numeric(day, "change_pct")
    close = _numeric(day, "close") if "close" in day.columns else _numeric(day, "price")
    ema21 = _numeric(day, "ema21")
    ema50 = _numeric(day, "ema50")
    change_20d = _numeric(day, "change_20d")
    rs60 = _numeric(day, "relative_strength_60d_pctile")
    technical_coverage = (
        close.notna()
        & ema21.notna()
        & ema50.notna()
        & change_20d.notna()
        & rs60.notna()
    )
    universe_size = int(technical_coverage.sum())

    ema_stack = _boolean(day, "ema_stack_bullish")
    ema21_rising = _boolean(day, "ema21_rising")
    ema50_rising = _boolean(day, "ema50_rising")
    distance_high = _numeric(day, "distance_to_120d_high_pct")
    leaders = (
        rs60.ge(95.0)
        & ema_stack
        & ema21_rising
        & distance_high.ge(-15.0)
        & change_20d.ge(3.0)
    )
    leader_count = int(leaders.sum())
    leader_change = change[leaders]
    leader_change_20d = change_20d[leaders]

    return {
        "date": day_date,
        "universe_size": universe_size,
        "positive_ratio": _ratio(change.gt(0), change.notna()),
        "median_change_pct": _median(change),
        "above_ema21_ratio": _ratio(close.gt(ema21), close.notna() & ema21.notna()),
        "above_ema50_ratio": _ratio(close.gt(ema50), close.notna() & ema50.notna()),
        "ema21_rising_ratio": _ratio(ema21_rising, ema21_rising.notna()),
        "ema50_rising_ratio": _ratio(ema50_rising, ema50_rising.notna()),
        "median_change_20d_pct": _median(change_20d),
        "leader_count": leader_count,
        "leader_ratio": leader_count / universe_size * 100.0 if universe_size else 0.0,
        "leader_median_change_pct": _median(leader_change),
        "leader_median_change_20d_pct": _median(leader_change_20d),
        "severe_decline_ratio": _ratio(change.le(-7.0), change.notna()),
    }


def _raw_state(row: Any, policy: MarketRegimePolicy) -> tuple[str, str]:
    universe_size = int(_number(getattr(row, "universe_size", 0)) or 0)
    if universe_size < policy.min_universe_size:
        return REGIME_UNAVAILABLE, "市场状态样本不足"
    positive_ratio = _number(getattr(row, "positive_ratio", None)) or 0.0
    median_change = _number(getattr(row, "median_change_pct", None)) or 0.0
    severe_decline = _number(getattr(row, "severe_decline_ratio", None)) or 0.0
    if (
        positive_ratio <= policy.shock_positive_ratio_max
        or median_change <= policy.shock_median_change_pct_max
        or severe_decline >= policy.shock_severe_decline_ratio_max
    ):
        return REGIME_RISK_OFF, "触发当日风险冲击否决"
    score = _number(getattr(row, "score", None)) or 0.0
    if score >= policy.strong_risk_on_score:
        return REGIME_STRONG_RISK_ON, "趋势、宽度与领涨股同步扩张"
    if score >= policy.risk_on_score:
        return REGIME_RISK_ON, "市场允许常规动量开仓"
    if score >= policy.neutral_score:
        return REGIME_NEUTRAL, "市场分歧，降低新开仓风险"
    return REGIME_RISK_OFF, "多日市场状态不足以支持新开仓"


def _apply_upgrade_confirmation(raw_states: list[str], days: int) -> list[str]:
    if not raw_states:
        return []
    confirmed: list[str] = []
    current = REGIME_NEUTRAL
    pending = ""
    streak = 0
    for raw in raw_states:
        if raw == REGIME_UNAVAILABLE:
            confirmed.append(REGIME_UNAVAILABLE)
            current = REGIME_UNAVAILABLE
            pending = ""
            streak = 0
            continue
        if current == REGIME_UNAVAILABLE:
            current = REGIME_NEUTRAL
            pending = ""
            streak = 0
        raw_rank = _STATE_RANK[raw]
        current_rank = _STATE_RANK[current]
        if raw_rank <= current_rank:
            current = raw
            pending = ""
            streak = 0
        else:
            if pending == raw:
                streak += 1
            else:
                pending = raw
                streak = 1
            if streak >= days:
                current = raw
                pending = ""
                streak = 0
        confirmed.append(current)
    return confirmed


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="boolean")
    values = frame[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype("boolean")

    def parse(value: Any) -> Any:
        if value is None or value is pd.NA:
            return pd.NA
        try:
            if pd.isna(value):
                return pd.NA
        except (TypeError, ValueError):
            pass
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        return pd.NA

    return values.map(parse).astype("boolean")


def _ratio(numerator: pd.Series, valid: pd.Series) -> float:
    count = int(valid.sum())
    return float((numerator & valid).sum() / count * 100.0) if count else 0.0


def _median(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    return float(valid.median()) if not valid.empty else 0.0


def _rolling_mean(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return _numeric(frame, column).rolling(window, min_periods=1).mean()


def _scale_series(values: pd.Series, lower: float, upper: float, points: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(lower)
    return ((numeric - lower) / (upper - lower)).clip(0.0, 1.0) * points


def _inverse_scale_series(values: pd.Series, lower: float, upper: float, points: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(upper)
    return (1.0 - (numeric - lower) / (upper - lower)).clip(0.0, 1.0) * points


def _expansion_ratio(current: Any, baseline: Any) -> float:
    current_value = _number(current) or 0.0
    baseline_value = _number(baseline)
    if baseline_value is None:
        return 1.0
    if baseline_value <= 0:
        return 1.5 if current_value > 0 else 1.0
    return current_value / baseline_value


def _read_history(path: Path) -> pd.DataFrame:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return pd.DataFrame()
    rows = payload.get("observations") if isinstance(payload, dict) else None
    return pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()


def _write_history(path: Path, observations: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in observations.to_dict(orient="records"):
        records.append({key: _json_value(value) for key, value in row.items()})
    payload = {"version": 1, "observations": records}
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _date_text(value: Any) -> str:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return ""
    return parsed.date().isoformat() if not pd.isna(parsed) else ""


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rounded(value: Any) -> float | None:
    number = _number(value)
    return round(number, 4) if number is not None else None
