# -*- coding: utf-8 -*-
# Derived from AlphaSift revision 9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf.
# Licensed under Apache-2.0 and modified for daily_stock_analysis.
"""Lightweight daily K-line enrichment for narrowed candidate pools."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable

import pandas as pd
from pandas.api.types import is_bool_dtype
import requests

from src.services.screening.source_guard import call_with_timeout, parse_source_timeout_seconds

_DAILY_FEATURE_DEFAULTS = {
    "daily_latest_date": "",
    "daily_data_points": pd.NA,
    "change_20d": pd.NA,
    "change_60d": pd.NA,
    "adr_20_pct": pd.NA,
    "ema9": pd.NA,
    "ema21": pd.NA,
    "ema50": pd.NA,
    "ema_stack_bullish": pd.NA,
    "ema21_rising": pd.NA,
    "ema50_rising": pd.NA,
    "distance_to_120d_high_pct": pd.NA,
    "pullback_depth_10d_pct": pd.NA,
    "range_contraction_ratio": pd.NA,
    "volume_contraction_ratio": pd.NA,
    "anchored_vwap_60d_high": pd.NA,
    "support_confluence_count": pd.NA,
    "trigger_distance_pct": pd.NA,
    "trigger_close_location_pct": pd.NA,
    "trigger_volume_vs_previous": pd.NA,
    "previous_5d_high": pd.NA,
    "previous_20d_high": pd.NA,
    "previous_5d_range_pct": pd.NA,
    "previous_10d_range_pct": pd.NA,
    "one_word_limit_up": pd.NA,
    "avg_amount_20d": pd.NA,
    "ma5": pd.NA,
    "ma20": pd.NA,
    "ma60": pd.NA,
    "ma_bullish": pd.NA,
    "price_above_ma20": pd.NA,
    "macd_status": "",
    "rsi_status": "",
    "rsi14": pd.NA,
    "signal_score": pd.NA,
    "prev_high_20d": pd.NA,
    "range_20d_pct": pd.NA,
    "breakout_20d_pct": pd.NA,
    "volume_ratio_20d": pd.NA,
    "body_pct": pd.NA,
    "pullback_to_ma20_pct": pd.NA,
    "consolidation_days_20d": pd.NA,
    "volatility_20d_pct": pd.NA,
    "max_drawdown_20d_pct": pd.NA,
    "atr_20_pct": pd.NA,
    "daily_quality_score": pd.NA,
    "daily_quality_flags": "",
    "daily_source": "",
    "daily_price_adjustment": "",
}
DAILY_FEATURE_COLUMNS = tuple(_DAILY_FEATURE_DEFAULTS)
_DAILY_BOOLEAN_FEATURE_COLUMNS = (
    "ema_stack_bullish",
    "ema21_rising",
    "ema50_rising",
    "one_word_limit_up",
    "ma_bullish",
    "price_above_ma20",
)
_DAILY_ENRICH_MAX_WORKERS = 1
_DAILY_HISTORY_CACHE_VERSION = 1
_DAILY_HISTORY_CACHE_TTL_SECONDS = 24 * 60 * 60
_SOURCE_HEALTH_FAILURE_THRESHOLD = 3
_SOURCE_HEALTH_COOLDOWN_SECONDS = 5 * 60
_DAILY_CALL_TIMEOUT_SECONDS = 20.0
_DEFAULT_TUSHARE_HTTP_URL = "http://api.waditu.com"
_BAOSTOCK_LOCK = threading.Lock()
_BAOSTOCK_OUTAGE_ERROR: str | None = None
_SOURCE_HEALTH: dict[str, dict[str, object]] = {}
_SOURCE_HEALTH_LOCK = threading.Lock()


def enrich_daily_features(
    df: pd.DataFrame,
    *,
    max_rows: int = 100,
    lookback_days: int = 120,
    source: str = "akshare",
    fetch_retries: int = 2,
    cache_dir: str | Path | None = None,
    cache_ttl_seconds: float | None = None,
    max_workers: int | None = None,
    history_fetcher: Callable[..., pd.DataFrame] | None = None,
    price_adjustment: str | None = None,
) -> pd.DataFrame:
    """Attach daily technical features to the first ``max_rows`` candidates.

    This intentionally runs after broad snapshot filtering; it is not a full
    market historical-data pass. ``history_fetcher`` is a request-scoped
    override; callers can reuse host data capabilities without replacing this
    module's process-global ``fetch_daily_history`` function.
    """
    if df.empty or max_rows <= 0:
        return df.copy()

    result = df.copy()
    # A live refresh can fail for one row and legitimately return pd.NA. Keep
    # precomputed boolean library columns nullable so that a partial source
    # failure is recorded instead of aborting the entire screening run.
    for column in _DAILY_BOOLEAN_FEATURE_COLUMNS:
        if column in result.columns and is_bool_dtype(result[column].dtype):
            result[column] = result[column].astype("boolean")
    daily_errors: list[str] = []
    daily_source_counts: dict[str, int] = {}
    daily_quality_flag_counts: dict[str, int] = {}
    daily_source_order_notes: list[str] = []
    daily_source_health: dict[str, object] = {}
    success_count = 0
    fetch_history = history_fetcher or fetch_daily_history
    selected_index = list(result.index[:max_rows])
    fetch_requests: list[tuple[object, str]] = []
    for idx in selected_index:
        raw_code = str(result.at[idx, "code"] if "code" in result.columns else "").strip()
        if not raw_code:
            continue
        code = raw_code.zfill(6) if raw_code.isdigit() else raw_code
        fetch_requests.append((idx, code))

    def fetch_one(request: tuple[object, str]) -> tuple[object, dict[str, object], str | None, dict[str, object]]:
        idx, code = request
        try:
            fetch_kwargs = {
                "lookback_days": lookback_days,
                "source": source,
                "retries": fetch_retries,
                "cache_dir": cache_dir,
                "cache_ttl_seconds": cache_ttl_seconds,
            }
            if price_adjustment is not None:
                fetch_kwargs["price_adjustment"] = price_adjustment
            hist = fetch_history(
                code,
                **fetch_kwargs,
            )
            features = compute_daily_features(hist)
            features["daily_source"] = str(hist.attrs.get("daily_source", ""))
            features["daily_price_adjustment"] = _history_price_adjustment(hist)
            metadata = {
                "daily_source": features["daily_source"],
                "daily_price_adjustment": features["daily_price_adjustment"],
                "daily_quality_flags": features.get("daily_quality_flags", ""),
                "daily_source_order_notes": list(hist.attrs.get("daily_source_order_notes", []) or []),
                "daily_source_health": dict(hist.attrs.get("daily_source_health", {}) or {}),
            }
            return idx, features, None, metadata
        except Exception as exc:
            features = dict(_DAILY_FEATURE_DEFAULTS)
            features["daily_quality_score"] = 0.0
            features["daily_quality_flags"] = "fetch_failed"
            return idx, features, f"{code}: {exc}", {"daily_quality_flags": "fetch_failed"}

    if len(fetch_requests) <= 1:
        fetched_rows = [fetch_one(request) for request in fetch_requests]
    else:
        worker_limit = min(_normalize_max_workers(max_workers), len(fetch_requests))
        with ThreadPoolExecutor(max_workers=worker_limit) as executor:
            fetched_rows = list(executor.map(fetch_one, fetch_requests))

    for idx, features, error, metadata in fetched_rows:
        for flag in str(metadata.get("daily_quality_flags") or "").split(";"):
            if flag:
                daily_quality_flag_counts[flag] = daily_quality_flag_counts.get(flag, 0) + 1
        if error:
            daily_errors.append(error)
        else:
            success_count += 1
            source_name = str(metadata.get("daily_source") or "unknown")
            daily_source_counts[source_name] = daily_source_counts.get(source_name, 0) + 1
            order_notes = metadata.get("daily_source_order_notes", [])
            if not isinstance(order_notes, list):
                order_notes = []
            for note in order_notes:
                note_text = str(note)
                if note_text and note_text not in daily_source_order_notes:
                    daily_source_order_notes.append(note_text)
            source_health = metadata.get("daily_source_health")
            if isinstance(source_health, dict):
                daily_source_health.update(source_health)
        for key, value in features.items():
            result.at[idx, key] = value

    result.attrs["daily_errors"] = daily_errors
    result.attrs["daily_success_count"] = success_count
    result.attrs["daily_source_counts"] = daily_source_counts
    result.attrs["daily_quality_flag_counts"] = daily_quality_flag_counts
    result.attrs["daily_source_order_notes"] = daily_source_order_notes
    result.attrs["daily_source_health"] = daily_source_health
    return result


def fetch_daily_history(
    code: str,
    *,
    lookback_days: int = 120,
    source: str = "akshare",
    retries: int = 2,
    cache_dir: str | Path | None = None,
    cache_ttl_seconds: float | None = None,
    price_adjustment: str | None = None,
) -> pd.DataFrame:
    """Fetch daily history for one stock code.

    ``source`` accepts ``tencent``, ``sina``, ``akshare``, ``baostock``, ``tushare``,
    ``yfinance`` or ``auto``. ``auto`` prefers Tushare when a token is
    configured, then Tencent's direct HTTP K-line endpoint before wrapper-based
    free sources. Without a token it starts with Tencent. Sina is a second
    direct HTTP K-line source before wrapper-based fallbacks. ``yfinance`` is
    explicit-only (never part of ``auto``) and expects a US ticker rather than
    an A-share code.
    """
    normalized_code = _normalize_daily_code(code)
    normalized_lookback_days = int(lookback_days)
    src = _normalize_daily_source(source)
    required_adjustment = _normalize_price_adjustment(price_adjustment)
    if src == "auto":
        sources: tuple[str, ...] = (
            ("tushare", "tencent", "sina", "akshare", "baostock")
            if _has_tushare_token()
            else ("tencent", "sina", "akshare", "baostock")
        )
        sources, source_order_notes = _rank_daily_sources_by_health(sources)
    elif src in ("akshare", "baostock", "tushare", "tencent", "sina", "yfinance"):
        sources = (src,)
        source_order_notes = []
    else:
        raise ValueError(f"Unsupported daily source: {source}")

    if required_adjustment is not None:
        sources = tuple(
            current
            for current in sources
            if _source_price_adjustment(current) == required_adjustment
        )
        if not sources:
            raise ValueError(
                f"Daily source {source!r} cannot provide {required_adjustment!r} prices"
            )

    cache_path = None
    if cache_dir is not None:
        cache_path = _daily_history_cache_path(
            cache_dir,
            code=normalized_code,
            source=src,
            lookback_days=normalized_lookback_days,
        )
        cached = _read_daily_history_cache(cache_path, ttl_seconds=cache_ttl_seconds)
        if cached is not None and _matches_price_adjustment(cached, required_adjustment):
            return cached

    attempts = max(int(retries), 0) + 1
    errors: list[str] = []
    for current in sources:
        disabled_reason = _source_disabled_reason(current)
        if disabled_reason:
            errors.append(f"{current}: {disabled_reason}")
            continue
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                if current == "yfinance":
                    from src.services.screening.snapshot_us import fetch_daily_history_yfinance
                    result = _call_daily_wrapper(
                        fetch_daily_history_yfinance,
                        current,
                        code,
                        lookback_days=lookback_days,
                    )
                elif current == "tencent":
                    result = _call_daily_wrapper(
                        _fetch_daily_tencent,
                        current,
                        normalized_code,
                        lookback_days=normalized_lookback_days,
                    )
                elif current == "sina":
                    result = _call_daily_wrapper(
                        _fetch_daily_sina,
                        current,
                        normalized_code,
                        lookback_days=normalized_lookback_days,
                    )
                elif current == "akshare":
                    result = _call_daily_wrapper(
                        _fetch_daily_akshare,
                        current,
                        normalized_code,
                        lookback_days=normalized_lookback_days,
                    )
                elif current == "tushare":
                    result = _call_daily_wrapper(
                        _fetch_daily_tushare,
                        current,
                        normalized_code,
                        lookback_days=normalized_lookback_days,
                    )
                else:
                    result = _call_daily_wrapper(
                        _fetch_daily_baostock,
                        current,
                        normalized_code,
                        lookback_days=normalized_lookback_days,
                    )
                _record_source_success(current, rows=len(result))
                result.attrs["daily_source"] = current
                result.attrs["daily_price_adjustment"] = _source_price_adjustment(current)
                result.attrs["daily_requested_source"] = src
                result.attrs["daily_source_order"] = list(sources)
                result.attrs["daily_source_order_notes"] = list(source_order_notes)
                result.attrs["source_errors"] = list(errors)
                result.attrs["daily_source_health"] = _daily_source_health_snapshot(sources)
                if cache_path is not None:
                    _write_daily_history_cache(
                        cache_path,
                        result,
                        code=normalized_code,
                        source=src,
                        lookback_days=normalized_lookback_days,
                    )
                return result
            except Exception as exc:  # noqa: BLE001 - aggregated below
                last_error = exc
                if attempt >= attempts - 1:
                    break
                time.sleep(min(0.5 * (attempt + 1), 2.0))
        errors.append(f"{current} after {attempts} attempts: {last_error}")
        _record_source_failure(current, last_error)

    if cache_path is not None:
        stale = _read_daily_history_cache(
            cache_path,
            ttl_seconds=cache_ttl_seconds,
            allow_stale=True,
        )
        if stale is not None and _matches_price_adjustment(stale, required_adjustment):
            stale.attrs["daily_stale"] = True
            stale.attrs["daily_source_order"] = list(sources)
            stale.attrs["daily_source_order_notes"] = list(source_order_notes)
            stale.attrs["source_errors"] = list(errors)
            stale.attrs["daily_source_health"] = _daily_source_health_snapshot(sources)
            return stale

    raise RuntimeError(
        f"daily history fetch failed for {normalized_code}: {'; '.join(errors)}"
    )


def load_cached_daily_history(
    code: str,
    *,
    source: str,
    lookback_days: int,
    cache_dir: str | Path,
    price_adjustment: str | None = None,
) -> pd.DataFrame | None:
    """Load an exact or longer compatible cache without a network refresh."""
    path = _daily_history_cache_path(
        cache_dir,
        code=_normalize_daily_code(code),
        source=_normalize_daily_source(source),
        lookback_days=int(lookback_days),
    )
    cached = _read_daily_history_cache(path, ttl_seconds=None, allow_stale=True)
    required_adjustment = _normalize_price_adjustment(price_adjustment)
    if cached is not None and _matches_price_adjustment(cached, required_adjustment):
        return cached
    compatible_path = _find_compatible_daily_history_cache(
        cache_dir,
        code=_normalize_daily_code(code),
        source=_normalize_daily_source(source),
        minimum_lookback_days=int(lookback_days),
    )
    if compatible_path is None:
        return None
    compatible = _read_daily_history_cache(
        compatible_path,
        ttl_seconds=None,
        allow_stale=True,
    )
    if compatible is not None and _matches_price_adjustment(compatible, required_adjustment):
        compatible.attrs["daily_cache_compatible_fallback"] = True
        return compatible
    return None


def load_cached_daily_history_for_period(
    code: str,
    *,
    source: str,
    requested_lookback_days: int,
    cache_dir: str | Path,
    start_date: date,
    required_end_date: date,
    minimum_pre_start_rows: int = 120,
    price_adjustment: str | None = None,
) -> pd.DataFrame | None:
    """Select the cache with the strongest actual coverage for a period."""
    normalized_code = _normalize_daily_code(code)
    normalized_source = _normalize_daily_source(source)
    required_adjustment = _normalize_price_adjustment(price_adjustment)
    candidates = _list_daily_history_cache_candidates(
        cache_dir,
        code=normalized_code,
        source=normalized_source,
    )
    covered: list[tuple[tuple[int, int, int, int], pd.DataFrame]] = []
    for stored_lookback_days, path in sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    ):
        cached = _read_daily_history_cache(
            path,
            ttl_seconds=None,
            allow_stale=True,
        )
        if cached is None or cached.empty:
            continue
        if not _matches_price_adjustment(cached, required_adjustment):
            continue
        dates = pd.to_datetime(cached.get("date"), errors="coerce").dropna()
        if dates.empty or dates.max().date() < required_end_date:
            continue
        pre_start_rows = int((dates.dt.date < start_date).sum())
        cached.attrs["daily_cache_period_coverage_fallback"] = True
        cached.attrs["daily_cache_requested_lookback_days"] = int(
            requested_lookback_days
        )
        cached.attrs["daily_cache_stored_lookback_days"] = stored_lookback_days
        cached.attrs["daily_cache_pre_start_rows"] = pre_start_rows
        score = (
            int(pre_start_rows >= int(minimum_pre_start_rows)),
            pre_start_rows,
            len(dates),
            stored_lookback_days,
        )
        covered.append((score, cached))
    if not covered:
        return None
    return max(covered, key=lambda item: item[0])[1]


def save_cached_daily_history(
    code: str,
    history: pd.DataFrame,
    *,
    source: str,
    lookback_days: int,
    cache_dir: str | Path,
) -> Path:
    """Persist a normalized cache entry used by the local daily library."""
    normalized_code = _normalize_daily_code(code)
    normalized_source = _normalize_daily_source(source)
    path = _daily_history_cache_path(
        cache_dir,
        code=normalized_code,
        source=normalized_source,
        lookback_days=int(lookback_days),
    )
    _write_daily_history_cache(
        path,
        history,
        code=normalized_code,
        source=normalized_source,
        lookback_days=int(lookback_days),
    )
    return path


def _normalize_daily_code(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)[-6:]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else text


def _normalize_daily_source(source: str | None) -> str:
    return (source or "akshare").strip().lower()


def _normalize_price_adjustment(value: str | None) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    aliases = {"forward": "qfq", "forward_adjusted": "qfq", "unadjusted": "none"}
    normalized = aliases.get(text, text)
    if normalized not in {"qfq", "hfq", "none"}:
        raise ValueError(f"Unsupported daily price adjustment: {value}")
    return normalized


def _source_price_adjustment(source: str | None) -> str:
    normalized = _normalize_daily_source(source)
    if normalized == "sina":
        return "none"
    if normalized == "tushare":
        return _normalize_tushare_adj(os.getenv("TUSHARE_DAILY_ADJ", "qfq"))
    if normalized in {"tencent", "akshare", "baostock"}:
        return "qfq"
    if normalized.startswith("dsa:") and any(
        marker in normalized for marker in ("akshare", "baostock", "tencent")
    ):
        return "qfq"
    return "unknown"


def _history_price_adjustment(history: pd.DataFrame) -> str:
    explicit = str(history.attrs.get("daily_price_adjustment") or "").strip().lower()
    if explicit:
        return explicit
    return _source_price_adjustment(history.attrs.get("daily_source"))


def _matches_price_adjustment(
    history: pd.DataFrame,
    required_adjustment: str | None,
) -> bool:
    return required_adjustment is None or _history_price_adjustment(history) == required_adjustment


def _normalize_max_workers(value: int | None) -> int:
    if value is None:
        return _DAILY_ENRICH_MAX_WORKERS
    return max(1, int(value))


def _call_daily_wrapper(fetcher, source: str, *args, **kwargs) -> pd.DataFrame:
    return call_with_timeout(
        fetcher,
        *args,
        timeout_sec=_daily_call_timeout_seconds(),
        label=f"daily source {source}",
        **kwargs,
    )


def _daily_call_timeout_seconds() -> float | None:
    return parse_source_timeout_seconds(
        "SCREENING_DAILY_CALL_TIMEOUT_SEC",
        default=_DAILY_CALL_TIMEOUT_SECONDS,
    )


def _rank_daily_sources_by_health(sources: tuple[str, ...]) -> tuple[tuple[str, ...], list[str]]:
    """Move unhealthy daily sources later while preserving default order ties."""
    now = time.monotonic()
    with _SOURCE_HEALTH_LOCK:
        health = {source: dict(_SOURCE_HEALTH.get(source, {})) for source in sources}
    default_rank = {source: idx for idx, source in enumerate(sources)}

    def rank_key(source: str) -> tuple[int, float, int]:
        state = health.get(source, {})
        disabled_until = float(state.get("disabled_until", 0.0))
        disabled = disabled_until > now
        failures = float(state.get("failures", 0.0))
        return (1 if disabled else 0, failures, default_rank[source])

    ranked = tuple(sorted(sources, key=rank_key))
    if ranked == sources:
        return sources, []
    return ranked, [f"daily source order adjusted by health: {','.join(ranked)}"]


def _source_disabled_reason(source: str) -> str | None:
    now = time.monotonic()
    with _SOURCE_HEALTH_LOCK:
        state = _SOURCE_HEALTH.get(source)
        if not state:
            return None
        disabled_until = float(state.get("disabled_until", 0.0))
        if disabled_until <= now:
            if disabled_until:
                state["disabled_until"] = 0.0
            return None
        return f"temporarily disabled for {disabled_until - now:.1f}s after repeated failures"


def _record_source_success(source: str, *, rows: int | None = None) -> None:
    with _SOURCE_HEALTH_LOCK:
        state = _SOURCE_HEALTH.setdefault(source, {"failures": 0.0, "disabled_until": 0.0})
        successes = float(state.get("successes", 0.0)) + 1.0
        state["successes"] = successes
        state["failures"] = 0.0
        state["disabled_until"] = 0.0
        state["last_success_at"] = time.time()
        if rows is not None:
            state["last_rows"] = float(rows)
            previous_avg = float(state.get("avg_rows", rows))
            state["avg_rows"] = previous_avg + (float(rows) - previous_avg) / successes


def _record_source_failure(source: str, error: object | None = None) -> None:
    now = time.monotonic()
    with _SOURCE_HEALTH_LOCK:
        state = _SOURCE_HEALTH.setdefault(source, {"failures": 0.0, "disabled_until": 0.0})
        failures = float(state.get("failures", 0.0)) + 1.0
        state["failures"] = failures
        state["total_failures"] = float(state.get("total_failures", 0.0)) + 1.0
        state["last_failure_at"] = time.time()
        if error is not None:
            state["last_error"] = " ".join(str(error).split())
        if failures >= _SOURCE_HEALTH_FAILURE_THRESHOLD:
            state["disabled_until"] = now + _SOURCE_HEALTH_COOLDOWN_SECONDS


def daily_source_health_snapshot() -> dict[str, dict[str, float | bool | str]]:
    """Return a copy of in-process daily-source health statistics."""
    return _daily_source_health_snapshot(tuple(_SOURCE_HEALTH))


def _daily_source_health_snapshot(sources: tuple[str, ...]) -> dict[str, dict[str, float | bool | str]]:
    now = time.monotonic()
    snapshot: dict[str, dict[str, float | bool | str]] = {}
    with _SOURCE_HEALTH_LOCK:
        for source in sources:
            state = dict(_SOURCE_HEALTH.get(source, {}))
            disabled_until = float(state.get("disabled_until", 0.0))
            cooldown_remaining = max(disabled_until - now, 0.0)
            snapshot[source] = {
                "successes": float(state.get("successes", 0.0)),
                "failures": float(state.get("failures", 0.0)),
                "total_failures": float(state.get("total_failures", 0.0)),
                "last_rows": float(state.get("last_rows", 0.0)),
                "avg_rows": float(state.get("avg_rows", 0.0)),
                "disabled": disabled_until > now,
                "cooldown_remaining_seconds": round(cooldown_remaining, 4),
                "last_success_at": float(state.get("last_success_at", 0.0)),
                "last_failure_at": float(state.get("last_failure_at", 0.0)),
                "last_error": str(state.get("last_error", "")),
            }
    return snapshot


def _daily_history_cache_path(
    cache_dir: str | Path,
    *,
    code: str,
    source: str,
    lookback_days: int,
) -> Path:
    key = f"{code}|{source}|{int(lookback_days)}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    safe_source = "".join(ch if ch.isalnum() else "-" for ch in source).strip("-") or "source"
    safe_code = "".join(ch if ch.isalnum() else "-" for ch in code).strip("-") or "code"
    return Path(cache_dir) / f"{safe_code}_{safe_source}_{int(lookback_days)}_{digest}.json"


def _find_compatible_daily_history_cache(
    cache_dir: str | Path,
    *,
    code: str,
    source: str,
    minimum_lookback_days: int,
) -> Path | None:
    """Return the smallest cache whose requested window covers this request."""
    candidates = [
        (candidate_lookback, path)
        for candidate_lookback, path in _list_daily_history_cache_candidates(
            cache_dir,
            code=code,
            source=source,
        )
        if candidate_lookback >= minimum_lookback_days
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _list_daily_history_cache_candidates(
    cache_dir: str | Path,
    *,
    code: str,
    source: str,
) -> list[tuple[int, Path]]:
    safe_source = "".join(ch if ch.isalnum() else "-" for ch in source).strip("-") or "source"
    safe_code = "".join(ch if ch.isalnum() else "-" for ch in code).strip("-") or "code"
    prefix = f"{safe_code}_{safe_source}_"
    candidates: list[tuple[int, Path]] = []
    for path in Path(cache_dir).glob(f"{safe_code}_{safe_source}_*.json"):
        try:
            candidate_lookback = int(path.stem.removeprefix(prefix).split("_", 1)[0])
        except (IndexError, TypeError, ValueError):
            continue
        candidates.append((candidate_lookback, path))
    return candidates


def _read_daily_history_cache(
    path: Path,
    *,
    ttl_seconds: float | None,
    allow_stale: bool = False,
) -> pd.DataFrame | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None

    ttl = _DAILY_HISTORY_CACHE_TTL_SECONDS if ttl_seconds is None else float(ttl_seconds)
    is_stale = ttl <= 0 or time.time() - stat.st_mtime > ttl
    if is_stale and not allow_stale:
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != _DAILY_HISTORY_CACHE_VERSION:
            return None
        frame = payload.get("frame")
        if not isinstance(frame, dict):
            return None
        columns = frame.get("columns")
        data = frame.get("data")
        if not isinstance(columns, list) or not isinstance(data, list):
            return None
        df = pd.DataFrame(data, columns=columns)
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in ("daily_source", "daily_price_adjustment", "daily_requested_source", "daily_source_order", "daily_source_order_notes", "source_errors", "daily_source_health"):
                if key in metadata:
                    df.attrs[key] = metadata[key]
        if is_stale:
            df.attrs["daily_stale"] = True
        return df
    except Exception:
        return None


def _write_daily_history_cache(
    path: Path,
    df: pd.DataFrame,
    *,
    code: str,
    source: str,
    lookback_days: int,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _DAILY_HISTORY_CACHE_VERSION,
            "key": {
                "code": code,
                "source": source,
                "lookback_days": int(lookback_days),
            },
            "metadata": {
                "daily_source": df.attrs.get("daily_source", source),
                "daily_price_adjustment": _history_price_adjustment(df),
                "daily_requested_source": df.attrs.get("daily_requested_source", source),
                "daily_source_order": list(df.attrs.get("daily_source_order", [])),
                "daily_source_order_notes": list(df.attrs.get("daily_source_order_notes", [])),
                "source_errors": list(df.attrs.get("source_errors", [])),
                "daily_source_health": df.attrs.get("daily_source_health", {}),
            },
            "created_at": datetime.now().isoformat(),
            "frame": json.loads(df.to_json(orient="split", date_format="iso", force_ascii=False)),
        }
        tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        return


def _fetch_daily_akshare(code: str, *, lookback_days: int) -> pd.DataFrame:
    import akshare as ak

    start_date = (datetime.now() - timedelta(days=max(lookback_days * 2, 90))).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(
        symbol=str(code).zfill(6),
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    if df is None or df.empty:
        raise RuntimeError(f"akshare daily history empty for {code}")
    return df.tail(max(lookback_days, 30)).copy()


def _fetch_daily_tencent(code: str, *, lookback_days: int) -> pd.DataFrame:
    """Fetch forward-adjusted daily history from Tencent's direct HTTP API.

    The endpoint is the same low-friction source recommended by a-stock-data for
    stable A-share market data access: no wrapper dependency, browser-like HTTP,
    and much lower IP-ban risk than Eastmoney-heavy endpoints. Tencent returns
    daily K-lines as rows shaped like ``date, open, close, high, low, volume``;
    amount is not always present. Tencent's A-share volume is reported in lots,
    so missing amount is estimated as ``volume * close * 100``. This proxy is
    used only for the liquidity gate; price/volume shape factors stay unchanged.
    """
    symbol = _to_tencent_code(code)
    count = max(int(lookback_days), 30)
    response = requests.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": f"{symbol},day,,,{count},qfq"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("code") not in (0, "0", None):
        message = payload.get("msg") if isinstance(payload, dict) else payload
        raise RuntimeError(f"tencent daily API error for {code}: {message}")
    data = payload.get("data") if isinstance(payload, dict) else None
    stock_data = data.get(symbol) if isinstance(data, dict) else None
    if not isinstance(stock_data, dict):
        raise RuntimeError(f"tencent daily history missing payload for {code}")
    rows = stock_data.get("qfqday") or stock_data.get("day") or []
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"tencent daily history empty for {code}")

    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        normalized_rows.append({
            "date": row[0],
            "open": row[1],
            "close": row[2],
            "high": row[3],
            "low": row[4],
            "volume": row[5],
            "amount": row[6] if len(row) > 6 else pd.NA,
        })
    if not normalized_rows:
        raise RuntimeError(f"tencent daily history malformed for {code}")
    df = pd.DataFrame(
        normalized_rows,
        columns=["date", "open", "close", "high", "low", "volume", "amount"],
    )
    for col in ("open", "close", "high", "low", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    _estimate_missing_amount(df, source="tencent")
    return df.tail(count).copy()


def _fetch_daily_sina(code: str, *, lookback_days: int) -> pd.DataFrame:
    """Fetch unadjusted daily history from Sina's direct K-line API.

    Sina provides a lightweight non-Eastmoney HTTP fallback for A-share daily
    bars. It does not expose forward-adjusted prices on this endpoint, so it is
    deliberately placed behind Tencent in ``auto`` but ahead of wrapper-heavy
    sources that are more prone to dependency/API drift.
    """
    symbol = _to_tencent_code(code)
    count = max(int(lookback_days), 30)
    response = requests.get(
        "https://quotes.sina.cn/cn/api/openapi.php/CN_MarketDataService.getKLineData",
        params={"symbol": symbol, "scale": 240, "ma": "no", "datalen": count},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("result", {}).get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"sina daily history empty for {code}")

    rows: list[dict[str, object]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        rows.append({
            "date": row.get("day") or row.get("date"),
            "open": row.get("open"),
            "close": row.get("close"),
            "high": row.get("high"),
            "low": row.get("low"),
            "volume": row.get("volume"),
            "amount": row.get("amount", pd.NA),
        })
    if not rows:
        raise RuntimeError(f"sina daily history malformed for {code}")

    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume", "amount"])
    if "date" in df.columns:
        df = df.sort_values("date")
    for col in ("open", "close", "high", "low", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    _estimate_missing_amount(df, source="sina")
    return df.tail(count).copy()


def _estimate_missing_amount(df: pd.DataFrame, *, source: str) -> None:
    """Fill absent A-share turnover values using source-specific volume units."""
    if not {"amount", "volume", "close"}.issubset(df.columns):
        return

    normalized_source = str(source or "").strip().lower()
    if normalized_source.startswith("tencent"):
        multiplier = 100.0  # Tencent A-share volume is expressed in lots.
        method = "tencent_volume_lots_x_close_x_100"
    elif normalized_source.startswith("sina"):
        multiplier = 1.0  # Sina's K-line endpoint expresses volume in shares.
        method = "sina_volume_shares_x_close"
    else:
        return

    missing_amount = df["amount"].isna() & df["volume"].gt(0) & df["close"].gt(0)
    if not missing_amount.any():
        return
    df.loc[missing_amount, "amount"] = (
        df.loc[missing_amount, "volume"]
        * df.loc[missing_amount, "close"]
        * multiplier
    )
    df.attrs["amount_estimation"] = method


def _fetch_daily_tushare(code: str, *, lookback_days: int) -> pd.DataFrame:
    """Fetch forward-adjusted daily history via Tushare Pro."""
    token = _tushare_token()
    if not token:
        raise RuntimeError("tushare requires TUSHARE_TOKEN")

    import tushare as ts

    pro = ts.pro_api(token)
    _configure_tushare_client(pro, token=token)

    start_date = (datetime.now() - timedelta(days=max(lookback_days * 2, 90))).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    adj = _normalize_tushare_adj(os.getenv("TUSHARE_DAILY_ADJ", "qfq"))
    ts_code = _to_tushare_code(code)
    df = pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )
    if df is None or df.empty:
        raise RuntimeError(f"tushare daily history empty for {code}")
    if adj is not None:
        df = _apply_tushare_adjustment(
            df,
            pro=pro,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            adj=adj,
        )

    normalized = _normalize_tushare_daily_frame(df)
    return normalized.tail(max(lookback_days, 30)).copy()


def _tushare_token() -> str:
    return (
        os.getenv("TUSHARE_TOKEN", "").strip()
        or os.getenv("TUSHARE_API_TOKEN", "").strip()
    )


def _has_tushare_token() -> bool:
    return bool(_tushare_token())


def _configure_tushare_client(pro: object, *, token: str) -> None:
    try:
        setattr(pro, "_DataApi__token", token)
    except Exception:
        pass

    http_url = (
        os.getenv("TUSHARE_API_URL", "").strip()
        or os.getenv("TUSHARE_HTTP_URL", "").strip()
        or _DEFAULT_TUSHARE_HTTP_URL
    )
    try:
        setattr(pro, "_DataApi__http_url", http_url)
    except Exception:
        pass


def _normalize_tushare_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "trade_date": "date",
        "vol": "volume",
    }
    normalized = df.rename(columns=rename_map).copy()
    if "date" in normalized.columns:
        normalized["date"] = normalized["date"].astype(str)
        normalized = normalized.sort_values("date")
    return normalized


def _apply_tushare_adjustment(
    df: pd.DataFrame,
    *,
    pro: object,
    ts_code: str,
    start_date: str,
    end_date: str,
    adj: str,
) -> pd.DataFrame:
    factors = pro.adj_factor(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="trade_date,adj_factor",
    )
    if factors is None or factors.empty:
        raise RuntimeError(f"tushare adj_factor empty for {ts_code}")

    merged = df.merge(factors, on="trade_date", how="left")
    merged = merged.sort_values("trade_date")
    merged["adj_factor"] = pd.to_numeric(merged["adj_factor"], errors="coerce").bfill()
    valid_factors = pd.to_numeric(factors["adj_factor"], errors="coerce").dropna()
    if valid_factors.empty:
        raise RuntimeError(f"tushare adj_factor invalid for {ts_code}")
    latest_factor = float(valid_factors.iloc[0])
    for col in ("open", "high", "low", "close"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        if adj == "hfq":
            merged[col] = merged[col] * merged["adj_factor"]
        else:
            merged[col] = merged[col] * merged["adj_factor"] / latest_factor
    return merged.drop(columns=["adj_factor"])


def _normalize_tushare_adj(value: str | None) -> str | None:
    text = (value or "").strip().lower()
    if text in {"", "none", "null", "no", "false", "0"}:
        return None
    if text not in {"qfq", "hfq"}:
        raise RuntimeError(f"unsupported TUSHARE_DAILY_ADJ: {value}")
    return text


def _fetch_daily_baostock(code: str, *, lookback_days: int) -> pd.DataFrame:
    """Fetch daily history via Baostock as a free fallback source.

    Baostock uses ``sh.600519`` / ``sz.000001`` style codes and exposes
    forward-adjusted prices via ``adjustflag='2'``.
    """
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError("baostock not installed; pip install baostock") from exc

    global _BAOSTOCK_OUTAGE_ERROR
    if _BAOSTOCK_OUTAGE_ERROR is not None:
        raise RuntimeError(_BAOSTOCK_OUTAGE_ERROR)

    bs_code = _to_baostock_code(code)
    start_date = (datetime.now() - timedelta(days=max(lookback_days * 2, 90))).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    with _BAOSTOCK_LOCK:
        if _BAOSTOCK_OUTAGE_ERROR is not None:
            raise RuntimeError(_BAOSTOCK_OUTAGE_ERROR)

        login_result = bs.login()
        try:
            login_error_code = str(getattr(login_result, "error_code", "0"))
            if login_error_code not in {"", "0"}:
                login_error_msg = getattr(login_result, "error_msg", "")
                raise RuntimeError(f"baostock login error {login_error_code}: {login_error_msg}")

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                message = f"baostock error {rs.error_code}: {rs.error_msg}"
                if _is_baostock_network_outage(rs.error_code, rs.error_msg):
                    _BAOSTOCK_OUTAGE_ERROR = message
                raise RuntimeError(message)
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    if not rows:
        raise RuntimeError(f"baostock daily history empty for {code}")

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    return df.tail(max(lookback_days, 30)).copy()


def _to_baostock_code(code: str) -> str:
    raw = str(code).strip().zfill(6)
    if raw.startswith(("6", "9", "5")):
        return f"sh.{raw}"
    return f"sz.{raw}"


def _to_tushare_code(code: str) -> str:
    raw = str(code).strip().zfill(6)
    if raw.startswith(("4", "8", "920")):
        return f"{raw}.BJ"
    if raw.startswith(("6", "9", "5")):
        return f"{raw}.SH"
    return f"{raw}.SZ"


def _to_tencent_code(code: str) -> str:
    raw = str(code).strip().zfill(6)
    if raw.startswith(("4", "8", "920")):
        return f"bj{raw}"
    if raw.startswith(("6", "9", "5")):
        return f"sh{raw}"
    return f"sz{raw}"


def _is_baostock_network_outage(error_code: object, error_msg: object) -> bool:
    code = str(error_code)
    message = str(error_msg)
    return code in {"10002007"} or "网络" in message or "接收" in message


def compute_daily_features(hist: pd.DataFrame) -> dict[str, object]:
    """Compute trend, pullback and trigger features for Martin momentum screening."""
    df = _normalize_daily_history(hist)
    if df.empty:
        raise RuntimeError("daily history is empty after normalization")

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty:
        raise RuntimeError("daily history has no valid close price")

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    last_close = float(close.iloc[-1])
    last_ma5 = _last_float(ma5)
    last_ma20 = _last_float(ma20)
    last_ma60 = _last_float(ma60)
    shape = _compute_shape_features(df, last_close=last_close, last_ma20=last_ma20)
    quality = _compute_daily_quality(hist, df)

    change_20d = _period_change_pct(close, 20)
    change_60d = _period_change_pct(close, 60)
    martin = _compute_martin_features(
        df,
        close=close,
        ema9=ema9,
        ema21=ema21,
        ema50=ema50,
    )

    macd_status = _compute_macd_status(close)
    rsi_value = _compute_rsi(close)
    rsi_status = _classify_rsi(rsi_value)
    ma_bullish = _is_true(last_ma5 is not None and last_ma20 is not None and last_ma60 is not None
                          and last_ma5 >= last_ma20 >= last_ma60)
    price_above_ma20 = _is_true(last_ma20 is not None and last_close >= last_ma20)
    signal_score = _compute_signal_score(
        change_60d=change_60d,
        ma_bullish=ma_bullish,
        price_above_ma20=price_above_ma20,
        macd_status=macd_status,
        rsi_status=rsi_status,
    )

    return {
        "daily_latest_date": _latest_daily_date(df),
        "daily_data_points": int(len(close)),
        "change_20d": _round_or_none(change_20d),
        "change_60d": None if change_60d is None else round(float(change_60d), 4),
        "ma5": last_ma5,
        "ma20": last_ma20,
        "ma60": last_ma60,
        "ma_bullish": ma_bullish,
        "price_above_ma20": price_above_ma20,
        "macd_status": macd_status,
        "rsi_status": rsi_status,
        "rsi14": None if rsi_value is None else round(float(rsi_value), 4),
        "signal_score": round(float(signal_score), 4),
        **martin,
        **shape,
        **quality,
    }


def _latest_daily_date(df: pd.DataFrame) -> str:
    if "date" not in df.columns or df.empty:
        return ""
    parsed = pd.to_datetime(
        df["date"].astype(str).str.slice(0, 10),
        format="%Y-%m-%d",
        errors="coerce",
    ).dropna()
    if parsed.empty:
        return ""
    return parsed.iloc[-1].strftime("%Y-%m-%d")


def _normalize_daily_history(hist: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = hist.rename(columns=rename_map).copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"].astype(str).str.slice(0, 10),
            format="%Y-%m-%d",
            errors="coerce",
        )
        df = df.sort_values("date")
    if "close" not in df.columns:
        raise RuntimeError("daily history has no close column")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    source = str(
        hist.attrs.get("daily_source")
        or hist.attrs.get("source")
        or ""
    ).strip().lower()
    _estimate_missing_amount(df, source=source)
    df = df.dropna(subset=["close"]).copy()
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
        else:
            df[col] = df[col].fillna(df["close"])
    return df


def _period_change_pct(close: pd.Series, sessions: int) -> float | None:
    if len(close) <= sessions:
        return None
    base = float(close.iloc[-(sessions + 1)])
    if base <= 0:
        return None
    return (float(close.iloc[-1]) / base - 1.0) * 100


def _compute_martin_features(
    df: pd.DataFrame,
    *,
    close: pd.Series,
    ema9: pd.Series,
    ema21: pd.Series,
    ema50: pd.Series,
) -> dict[str, object]:
    last_close = float(close.iloc[-1])
    last_ema9 = _last_float(ema9)
    last_ema21 = _last_float(ema21)
    last_ema50 = _last_float(ema50)
    ema_stack = bool(
        last_ema9 is not None
        and last_ema21 is not None
        and last_ema50 is not None
        and last_ema9 > last_ema21 > last_ema50
    )
    ema21_rising = _series_rising(ema21, sessions=5)
    ema50_rising = _series_rising(ema50, sessions=5)

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    adr_values = ((high / low) - 1.0).replace([float("inf"), float("-inf")], pd.NA) * 100
    adr_20_pct = _last_mean(adr_values, 20)

    high_120 = _series_max(high.tail(120))
    distance_to_120d_high_pct = (
        (last_close / high_120 - 1.0) * 100
        if high_120 is not None and high_120 > 0
        else None
    )
    high_10 = _series_max(high.tail(10))
    pullback_depth_10d_pct = (
        (last_close / high_10 - 1.0) * 100
        if high_10 is not None and high_10 > 0
        else None
    )

    daily_range_pct = ((high / low) - 1.0).replace([float("inf"), float("-inf")], pd.NA) * 100
    latest_range = _last_mean(daily_range_pct, 3)
    prior_range = _last_mean(daily_range_pct.iloc[:-3], 10)
    range_contraction_ratio = _safe_ratio(latest_range, prior_range)

    volume = pd.to_numeric(df.get("volume"), errors="coerce") if "volume" in df.columns else pd.Series(dtype=float)
    latest_volume = _last_mean(volume, 3)
    prior_volume = _last_mean(volume.iloc[:-3], 20)
    volume_contraction_ratio = _safe_ratio(latest_volume, prior_volume)

    anchored_vwap = _anchored_vwap_from_60d_high(df)
    support_levels = [last_ema9, last_ema21, anchored_vwap]
    support_confluence_count = sum(
        1
        for level in support_levels
        if level is not None and level > 0 and abs(last_close / level - 1.0) <= 0.03
    )

    previous_high = float(high.iloc[-2]) if len(high) >= 2 and pd.notna(high.iloc[-2]) else None
    prior_high = high.iloc[:-1]
    prior_low = low.iloc[:-1]
    previous_5d_high = _series_max(prior_high.tail(5))
    previous_20d_high = _series_max(prior_high.tail(20))
    previous_5d_low = _series_minimum(prior_low.tail(5))
    previous_10d_high = _series_max(prior_high.tail(10))
    previous_10d_low = _series_minimum(prior_low.tail(10))
    previous_5d_range_pct = _level_range_pct(previous_5d_high, previous_5d_low)
    previous_10d_range_pct = _level_range_pct(previous_10d_high, previous_10d_low)
    trigger_distance_pct = (
        (last_close / previous_high - 1.0) * 100
        if previous_high is not None and previous_high > 0
        else None
    )
    previous_close = float(close.iloc[-2]) if len(close) >= 2 else None
    latest_high = float(high.iloc[-1]) if pd.notna(high.iloc[-1]) else None
    latest_low = float(low.iloc[-1]) if pd.notna(low.iloc[-1]) else None
    latest_volume = float(volume.iloc[-1]) if len(volume) and pd.notna(volume.iloc[-1]) else None
    previous_volume = float(volume.iloc[-2]) if len(volume) >= 2 and pd.notna(volume.iloc[-2]) else None
    trigger_close_location_pct = (
        (last_close - latest_low) / (latest_high - latest_low) * 100.0
        if latest_high is not None
        and latest_low is not None
        and latest_high > latest_low
        else None
    )
    trigger_volume_vs_previous = (
        latest_volume / previous_volume
        if latest_volume is not None and previous_volume is not None and previous_volume > 0
        else None
    )
    one_word_limit_up = bool(
        previous_close is not None
        and previous_close > 0
        and latest_high is not None
        and latest_low is not None
        and abs(latest_high - latest_low) <= max(latest_high, 1.0) * 1e-6
        and (last_close / previous_close - 1.0) * 100 >= 9.5
    )
    avg_amount_20d = (
        _last_mean(pd.to_numeric(df["amount"], errors="coerce"), 20)
        if "amount" in df.columns
        else None
    )

    return {
        "adr_20_pct": _round_or_none(adr_20_pct),
        "ema9": last_ema9,
        "ema21": last_ema21,
        "ema50": last_ema50,
        "ema_stack_bullish": ema_stack,
        "ema21_rising": ema21_rising,
        "ema50_rising": ema50_rising,
        "distance_to_120d_high_pct": _round_or_none(distance_to_120d_high_pct),
        "pullback_depth_10d_pct": _round_or_none(pullback_depth_10d_pct),
        "range_contraction_ratio": _round_or_none(range_contraction_ratio),
        "volume_contraction_ratio": _round_or_none(volume_contraction_ratio),
        "anchored_vwap_60d_high": _round_or_none(anchored_vwap),
        "support_confluence_count": support_confluence_count,
        "trigger_distance_pct": _round_or_none(trigger_distance_pct),
        "trigger_close_location_pct": _round_or_none(trigger_close_location_pct),
        "trigger_volume_vs_previous": _round_or_none(trigger_volume_vs_previous),
        "previous_5d_high": _round_or_none(previous_5d_high),
        "previous_20d_high": _round_or_none(previous_20d_high),
        "previous_5d_range_pct": _round_or_none(previous_5d_range_pct),
        "previous_10d_range_pct": _round_or_none(previous_10d_range_pct),
        "one_word_limit_up": one_word_limit_up,
        "avg_amount_20d": _round_or_none(avg_amount_20d),
    }


def _series_rising(series: pd.Series, *, sessions: int) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return bool(len(values) > sessions and float(values.iloc[-1]) > float(values.iloc[-(sessions + 1)]))


def _last_mean(series: pd.Series, periods: int) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna().tail(periods)
    if values.empty:
        return None
    return float(values.mean())


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _series_minimum(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if not values.empty else None


def _level_range_pct(high: float | None, low: float | None) -> float | None:
    if high is None or low is None or low <= 0:
        return None
    return (high / low - 1.0) * 100.0


def _anchored_vwap_from_60d_high(df: pd.DataFrame) -> float | None:
    if not {"high", "low", "close", "volume"}.issubset(df.columns):
        return None
    recent = df.tail(60).copy()
    high = pd.to_numeric(recent["high"], errors="coerce")
    if high.dropna().empty:
        return None
    anchor_position = int(high.reset_index(drop=True).idxmax())
    anchored = recent.iloc[anchor_position:]
    volume = pd.to_numeric(anchored["volume"], errors="coerce").fillna(0)
    if volume.sum() <= 0:
        return None
    typical = (
        pd.to_numeric(anchored["high"], errors="coerce")
        + pd.to_numeric(anchored["low"], errors="coerce")
        + pd.to_numeric(anchored["close"], errors="coerce")
    ) / 3.0
    valid = typical.notna() & volume.gt(0)
    if not valid.any():
        return None
    return float((typical[valid] * volume[valid]).sum() / volume[valid].sum())


def _compute_daily_quality(raw: pd.DataFrame, normalized: pd.DataFrame) -> dict[str, object]:
    """Score daily-history quality and expose compact audit flags."""
    score = 100.0
    flags: list[str] = []
    points = len(normalized)
    if points < 30:
        score -= 35
        flags.append("short_history_lt30")
    elif points < 60:
        score -= 15
        flags.append("short_history_lt60")

    for col in ("open", "high", "low", "close"):
        if col not in normalized.columns:
            score -= 20
            flags.append(f"missing_{col}")
            continue
        missing_ratio = float(pd.to_numeric(normalized[col], errors="coerce").isna().mean())
        if missing_ratio > 0:
            score -= min(missing_ratio * 40, 20)
            flags.append(f"incomplete_{col}")

    if "volume" not in normalized.columns:
        score -= 12
        flags.append("missing_volume")
    else:
        volume = pd.to_numeric(normalized["volume"], errors="coerce")
        missing_volume_ratio = float(volume.isna().mean())
        if missing_volume_ratio > 0:
            score -= min(missing_volume_ratio * 20, 10)
            flags.append("incomplete_volume")
        if (volume.dropna() < 0).any():
            score -= 20
            flags.append("negative_volume")

    if {"open", "high", "low", "close"}.issubset(normalized.columns):
        open_ = pd.to_numeric(normalized["open"], errors="coerce")
        high = pd.to_numeric(normalized["high"], errors="coerce")
        low = pd.to_numeric(normalized["low"], errors="coerce")
        close = pd.to_numeric(normalized["close"], errors="coerce")
        invalid_ohlc = (high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)
        if invalid_ohlc.fillna(False).any():
            score -= 30
            flags.append("invalid_ohlc")
        if ((open_ <= 0) | (high <= 0) | (low <= 0) | (close <= 0)).fillna(False).any():
            score -= 35
            flags.append("non_positive_price")

    if bool(raw.attrs.get("daily_stale")):
        score -= 25
        flags.append("stale_cache")

    source_errors = list(raw.attrs.get("source_errors", []) or [])
    if source_errors:
        score -= min(len(source_errors) * 5, 20)
        flags.append("fallback_errors")

    return {
        "daily_quality_score": round(max(score, 0.0), 4),
        "daily_quality_flags": ";".join(flags),
    }


def _compute_shape_features(
    df: pd.DataFrame,
    *,
    last_close: float,
    last_ma20: float | None,
) -> dict[str, object]:
    previous = df.iloc[:-1].tail(20)
    recent = df.tail(20)
    last = df.iloc[-1]

    prev_high_20d = _series_max(previous["high"]) if "high" in previous.columns else None
    range_20d_pct = _range_pct(recent)
    breakout_20d_pct = (
        (last_close / prev_high_20d - 1.0) * 100
        if prev_high_20d is not None and prev_high_20d > 0
        else None
    )
    volume_ratio_20d = _volume_ratio_20d(df)
    body_pct = _body_pct(last)
    pullback_to_ma20_pct = (
        (last_close / last_ma20 - 1.0) * 100
        if last_ma20 is not None and last_ma20 > 0
        else None
    )
    volatility_20d_pct = _volatility_20d_pct(recent["close"])
    max_drawdown_20d_pct = _max_drawdown_pct(recent["close"])
    atr_20_pct = _atr_20_pct(df)

    return {
        "prev_high_20d": _round_or_none(prev_high_20d),
        "range_20d_pct": _round_or_none(range_20d_pct),
        "breakout_20d_pct": _round_or_none(breakout_20d_pct),
        "volume_ratio_20d": _round_or_none(volume_ratio_20d),
        "body_pct": _round_or_none(body_pct),
        "pullback_to_ma20_pct": _round_or_none(pullback_to_ma20_pct),
        "consolidation_days_20d": _consolidation_days(previous),
        "volatility_20d_pct": _round_or_none(volatility_20d_pct),
        "max_drawdown_20d_pct": _round_or_none(max_drawdown_20d_pct),
        "atr_20_pct": _round_or_none(atr_20_pct),
    }


def _series_max(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _range_pct(df: pd.DataFrame) -> float | None:
    if "high" not in df.columns or "low" not in df.columns:
        return None
    high = pd.to_numeric(df["high"], errors="coerce").dropna()
    low = pd.to_numeric(df["low"], errors="coerce").dropna()
    if high.empty or low.empty:
        return None
    low_min = float(low.min())
    if low_min <= 0:
        return None
    return (float(high.max()) / low_min - 1.0) * 100


def _volume_ratio_20d(df: pd.DataFrame) -> float | None:
    if "volume" not in df.columns:
        return None
    volume = pd.to_numeric(df["volume"], errors="coerce")
    if len(volume) < 2 or pd.isna(volume.iloc[-1]):
        return None
    previous = volume.iloc[:-1].tail(20).dropna()
    if previous.empty:
        return None
    base = float(previous.mean())
    if base <= 0:
        return None
    return float(volume.iloc[-1]) / base


def _volatility_20d_pct(close: pd.Series) -> float | None:
    values = pd.to_numeric(close, errors="coerce").dropna()
    returns = values.pct_change().dropna()
    if len(returns) < 2:
        return None
    return float(returns.std()) * (252 ** 0.5) * 100


def _max_drawdown_pct(close: pd.Series) -> float | None:
    values = pd.to_numeric(close, errors="coerce").dropna()
    if values.empty:
        return None
    running_high = values.cummax()
    drawdowns = values / running_high - 1.0
    return min(float(drawdowns.min()) * 100, 0.0)


def _atr_20_pct(df: pd.DataFrame) -> float | None:
    if not {"high", "low", "close"}.issubset(df.columns):
        return None
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.tail(20).dropna().mean()
    valid_close = close.dropna()
    if valid_close.empty:
        return None
    last_close = float(valid_close.iloc[-1])
    if pd.isna(atr) or last_close <= 0:
        return None
    return float(atr) / last_close * 100


def _consolidation_days(previous: pd.DataFrame, *, max_range_pct: float = 12.0) -> int | None:
    if previous.empty or "high" not in previous.columns or "low" not in previous.columns:
        return None
    for days in range(min(len(previous), 20), 1, -1):
        window = previous.tail(days)
        range_pct = _range_pct(window)
        if range_pct is not None and range_pct <= max_range_pct:
            return int(days)
    return 0


def _body_pct(row: pd.Series) -> float | None:
    open_price = row.get("open")
    close_price = row.get("close")
    if pd.isna(open_price) or pd.isna(close_price) or float(open_price) <= 0:
        return None
    return (float(close_price) / float(open_price) - 1.0) * 100


def _round_or_none(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _compute_macd_status(close: pd.Series) -> str:
    if len(close) < 35:
        return "neutral"
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    last_diff = float(diff.iloc[-1])
    last_dea = float(dea.iloc[-1])
    if last_diff > last_dea and last_diff > 0:
        return "bullish"
    if last_diff < last_dea and last_diff < 0:
        return "bearish"
    return "neutral"


def _compute_rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) <= period:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    value = rsi.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _classify_rsi(value: float | None) -> str:
    if value is None:
        return "neutral"
    if value <= 35:
        return "oversold"
    if value >= 70:
        return "overbought"
    return "neutral"


def _compute_signal_score(
    *,
    change_60d: float | None,
    ma_bullish: bool,
    price_above_ma20: bool,
    macd_status: str,
    rsi_status: str,
) -> float:
    score = 50.0
    if ma_bullish:
        score += 14
    if price_above_ma20:
        score += 10
    if macd_status == "bullish":
        score += 12
    elif macd_status == "bearish":
        score -= 12
    if change_60d is not None:
        if 0 <= change_60d <= 35:
            score += min(change_60d * 0.35, 12)
        elif change_60d > 60:
            score -= min((change_60d - 60) * 0.20, 12)
        elif change_60d < -25:
            score -= min(abs(change_60d + 25) * 0.25, 10)
    if rsi_status == "oversold":
        score += 4
    elif rsi_status == "overbought":
        score -= 6
    return max(0.0, min(score, 100.0))


def _last_float(series: pd.Series) -> float | None:
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    return round(float(value), 4)


def _is_true(value: bool) -> bool:
    return bool(value)
