# -*- coding: utf-8 -*-
# Derived from AlphaSift revision 9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf.
# Licensed under Apache-2.0 and modified for daily_stock_analysis.
"""Persistent local daily-feature library for broad-universe screening."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.services.screening.daily import (
    DAILY_FEATURE_COLUMNS,
    compute_daily_features,
    load_cached_daily_history,
    save_cached_daily_history,
)
from src.services.screening.normalize import normalize_code

_LIBRARY_VERSION = 1
_LIBRARY_LOCK = threading.Lock()


def merge_daily_feature_library(
    df: pd.DataFrame,
    *,
    library_path: str | Path | None,
    max_age_days: int = 7,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing daily features from the fresh portion of the local library."""
    result = df.copy()
    if result.empty or library_path is None:
        return result, _empty_diagnostics(len(result))

    library = load_daily_feature_library(
        library_path,
        max_age_days=max_age_days,
        now=now,
    )
    if library.empty or "code" not in result.columns:
        diagnostics = dict(library.attrs.get("diagnostics", {}))
        diagnostics.update({"candidate_count": len(result), "matched_count": 0})
        return result, diagnostics

    result_codes = result["code"].map(lambda value: normalize_code(value, allow_ticker=True))
    library_diagnostics = dict(library.attrs.get("diagnostics", {}))
    library = library.drop_duplicates("code", keep="last").set_index("code")
    matched = result_codes.isin(library.index)
    for column in DAILY_FEATURE_COLUMNS:
        if column not in library.columns:
            continue
        mapped = result_codes.map(library[column])
        if column in result.columns:
            result[column] = result[column].where(result[column].notna(), mapped)
        else:
            result[column] = mapped
    result["daily_library_hit"] = matched

    diagnostics = library_diagnostics
    diagnostics.update({
        "candidate_count": len(result),
        "matched_count": int(matched.sum()),
        "coverage_pct": round(float(matched.mean() * 100.0), 2) if len(matched) else 0.0,
    })
    return result, diagnostics


def load_daily_feature_library(
    path: str | Path,
    *,
    max_age_days: int = 7,
    now: datetime | None = None,
) -> pd.DataFrame:
    payload = _read_payload(Path(path))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict) or not entries:
        empty = pd.DataFrame()
        empty.attrs["diagnostics"] = _empty_diagnostics(0)
        return empty

    rows = [entry for entry in entries.values() if isinstance(entry, dict)]
    frame = pd.DataFrame(rows)
    if frame.empty or "code" not in frame.columns:
        frame.attrs["diagnostics"] = _empty_diagnostics(0)
        return frame

    frame["code"] = frame["code"].map(lambda value: normalize_code(value, allow_ticker=True))
    latest_dates = (
        frame["daily_latest_date"]
        if "daily_latest_date" in frame.columns
        else pd.Series(pd.NaT, index=frame.index)
    )
    as_of = pd.to_datetime(latest_dates, errors="coerce")
    current = now or datetime.now()
    cutoff = pd.Timestamp(current.date() - timedelta(days=max(int(max_age_days), 0)))
    fresh_mask = as_of.notna() & as_of.ge(cutoff)
    fresh = frame.loc[fresh_mask].copy()
    fresh.attrs["diagnostics"] = {
        "library_path": str(Path(path)),
        "entry_count": len(frame),
        "fresh_count": len(fresh),
        "stale_count": int((~fresh_mask).sum()),
        "max_age_days": max(int(max_age_days), 0),
        "latest_date": _latest_date_text(as_of),
        "updated_at": payload.get("updated_at", ""),
    }
    return fresh


def daily_feature_library_date_coverage(
    candidates: pd.DataFrame,
    *,
    library_path: str | Path,
    as_of: date,
) -> dict[str, Any]:
    """Measure exact-date library coverage for a concrete candidate universe."""
    if candidates.empty or "code" not in candidates.columns:
        return {
            "as_of": as_of.isoformat(),
            "candidate_count": 0,
            "matched_count": 0,
            "missing_count": 0,
            "coverage_pct": 0.0,
        }
    candidate_codes = {
        code
        for value in candidates["code"]
        if (code := normalize_code(value, allow_ticker=True))
    }
    library = load_daily_feature_library(library_path, max_age_days=36_500)
    if library.empty:
        matched_codes: set[str] = set()
    else:
        dates = pd.to_datetime(library.get("daily_latest_date"), errors="coerce").dt.date
        matched_codes = set(
            library.loc[dates.eq(as_of), "code"].dropna().astype(str)
        )
    matched_count = len(candidate_codes & matched_codes)
    candidate_count = len(candidate_codes)
    return {
        "as_of": as_of.isoformat(),
        "candidate_count": candidate_count,
        "matched_count": matched_count,
        "missing_count": max(candidate_count - matched_count, 0),
        "coverage_pct": (
            round(matched_count / candidate_count * 100.0, 4)
            if candidate_count
            else 0.0
        ),
    }


def select_daily_feature_library_date_missing(
    candidates: pd.DataFrame,
    *,
    library_path: str | Path,
    as_of: date,
) -> pd.DataFrame:
    """Return only candidate rows not represented on the exact target date."""
    if candidates.empty or "code" not in candidates.columns:
        return candidates.head(0).copy()
    library = load_daily_feature_library(library_path, max_age_days=36_500)
    if library.empty:
        return candidates.copy()
    dates = pd.to_datetime(library.get("daily_latest_date"), errors="coerce").dt.date
    current_codes = set(
        library.loc[dates.eq(as_of), "code"].dropna().astype(str)
    )
    normalized_codes = candidates["code"].map(
        lambda value: normalize_code(value, allow_ticker=True)
    )
    return candidates.loc[~normalized_codes.isin(current_codes)].copy()


def upsert_daily_feature_library(
    enriched: pd.DataFrame,
    *,
    library_path: str | Path,
) -> dict[str, Any]:
    """Persist successfully computed daily features keyed by normalized code."""
    if enriched.empty or "code" not in enriched.columns:
        return _unchanged_library_result(library_path)

    rows: list[dict[str, Any]] = []
    for _, row in enriched.iterrows():
        code = normalize_code(row.get("code"), allow_ticker=True)
        latest_date = _text(row.get("daily_latest_date"))
        points = _number(row.get("daily_data_points"))
        if not code or not latest_date or points is None or points <= 0:
            continue
        entry: dict[str, Any] = {
            "code": code,
            "daily_latest_date": latest_date,
            "daily_source": _text(row.get("daily_source")),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        for column in DAILY_FEATURE_COLUMNS:
            if column in row.index:
                entry[column] = _json_value(row.get(column))
        rows.append(entry)

    if not rows:
        return _unchanged_library_result(library_path)

    path = Path(library_path)
    with _LIBRARY_LOCK:
        payload = _read_payload(path)
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, dict):
            entries = {}
        for entry in rows:
            entries[entry["code"]] = entry
        output = {
            "version": _LIBRARY_VERSION,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "entry_count": len(entries),
            "entries": entries,
        }
        _write_payload(path, output)
    return {"updated_count": len(rows), "entry_count": len(entries), "path": str(path)}


def refresh_daily_feature_library_from_cache(
    snapshot: pd.DataFrame,
    *,
    cache_dir: str | Path,
    library_path: str | Path,
    source: str,
    lookback_days: int,
    price_adjustment: str | None = None,
    append_snapshot_bar: bool = False,
    snapshot_bar_date: date | None = None,
) -> dict[str, Any]:
    """Recompute local features, optionally appending one completed snapshot bar."""
    if snapshot.empty or "code" not in snapshot.columns:
        return {"cache_count": 0, "updated_count": 0, "appended_count": 0}

    feature_rows: list[dict[str, Any]] = []
    cache_count = 0
    appended_count = 0
    errors: list[str] = []
    for _, row in snapshot.iterrows():
        code = normalize_code(row.get("code"), allow_ticker=True)
        if not code:
            continue
        history = load_cached_daily_history(
            code,
            source=source,
            lookback_days=lookback_days,
            cache_dir=cache_dir,
            price_adjustment=price_adjustment,
        )
        if history is None or history.empty:
            continue
        cache_count += 1
        try:
            if append_snapshot_bar and snapshot_bar_date is not None:
                history, appended = _append_completed_snapshot_bar(
                    history,
                    row,
                    bar_date=snapshot_bar_date,
                )
                if appended:
                    save_cached_daily_history(
                        code,
                        history,
                        source=source,
                        lookback_days=lookback_days,
                        cache_dir=cache_dir,
                    )
                    appended_count += 1
            features = compute_daily_features(history)
            features["code"] = code
            features["daily_source"] = _text(history.attrs.get("daily_source"))
            features["daily_price_adjustment"] = _text(
                history.attrs.get("daily_price_adjustment")
            )
            feature_rows.append(features)
        except Exception as exc:  # noqa: BLE001 - one bad cache must not stop the library.
            errors.append(f"{code}: {exc}")

    update = upsert_daily_feature_library(
        pd.DataFrame(feature_rows),
        library_path=library_path,
    )
    return {
        "cache_count": cache_count,
        "updated_count": update.get("updated_count", 0),
        "entry_count": update.get("entry_count", 0),
        "appended_count": appended_count,
        "error_count": len(errors),
        "errors": errors[:10],
    }


def select_daily_library_batch(
    candidates: pd.DataFrame,
    *,
    library_path: str | Path,
    limit: int,
    period: str,
    max_age_days: int = 7,
) -> pd.DataFrame:
    """Select a deterministic batch of candidates absent from the fresh library."""
    if candidates.empty or "code" not in candidates.columns or limit <= 0:
        return candidates.head(0).copy()
    library = load_daily_feature_library(library_path, max_age_days=max_age_days)
    covered = set(library.get("code", pd.Series(dtype=str)).astype(str).tolist())
    work = candidates.copy()
    work["__library_code"] = work["code"].map(
        lambda value: normalize_code(value, allow_ticker=True)
    )
    work = work[~work["__library_code"].isin(covered)].copy()
    work["__library_order"] = work["__library_code"].map(
        lambda code: hashlib.sha256(f"{period}|{code}".encode("utf-8")).hexdigest()
    )
    return work.sort_values("__library_order", kind="mergesort").head(int(limit)).drop(
        columns=["__library_code", "__library_order"],
        errors="ignore",
    )


def latest_daily_feature_date(path: str | Path) -> date | None:
    """Return the newest completed bar date stored in the library."""
    payload = _read_payload(Path(path))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        return None
    parsed = [
        _parse_date(entry.get("daily_latest_date"))
        for entry in entries.values()
        if isinstance(entry, dict)
    ]
    valid = [value for value in parsed if value is not None]
    return max(valid) if valid else None


def completed_snapshot_bar_date(
    snapshot: pd.DataFrame,
    *,
    market: str = "cn",
    now: datetime | None = None,
) -> date | None:
    """Resolve a safe completed bar date for appending a live snapshot."""
    if snapshot.empty or bool(snapshot.attrs.get("fallback_used")):
        return None
    try:
        from src.core.trading_calendar import (
            MarketPhase,
            get_effective_trading_date,
            infer_market_phase,
        )

        phase = infer_market_phase(market, current_time=now)
        if phase not in {MarketPhase.POSTMARKET, MarketPhase.NON_TRADING}:
            return None
        return get_effective_trading_date(market, current_time=now)
    except Exception:
        return None


def _append_completed_snapshot_bar(
    history: pd.DataFrame,
    row: pd.Series,
    *,
    bar_date: date,
) -> tuple[pd.DataFrame, bool]:
    source = _text(history.attrs.get("daily_source")).lower()
    if not (source.startswith("sina") or source.startswith("tencent")):
        return history, False

    close = _number(row.get("price") if row.get("price") is not None else row.get("close"))
    open_price = _number(row.get("open")) or close
    high = _number(row.get("high")) or close
    low = _number(row.get("low")) or close
    volume = _number(row.get("volume"))
    amount = _number(row.get("amount"))
    if close is None or close <= 0 or volume is None or volume <= 0:
        return history, False
    if source.startswith("tencent"):
        volume /= 100.0

    normalized = history.copy()
    attrs = dict(history.attrs)
    if "date" not in normalized.columns:
        return history, False
    dates = pd.to_datetime(
        normalized["date"].astype(str).str.slice(0, 10),
        format="%Y-%m-%d",
        errors="coerce",
    )
    if dates.notna().any() and dates.max().date() > bar_date:
        return history, False

    bar = {
        "date": bar_date.isoformat(),
        "open": open_price,
        "close": close,
        "high": max(high or close, close, open_price or close),
        "low": min(low or close, close, open_price or close),
        "volume": volume,
        "amount": amount,
    }
    same_day = dates.dt.date.eq(bar_date)
    if same_day.any():
        return history, False
    normalized = pd.concat([normalized, pd.DataFrame([bar])], ignore_index=True)
    normalized = normalized.tail(max(len(history), 30)).reset_index(drop=True)
    normalized.attrs.update(attrs)
    return normalized, True


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"version": _LIBRARY_VERSION, "entries": {}}
    if payload.get("version") != _LIBRARY_VERSION:
        return {"version": _LIBRARY_VERSION, "entries": {}}
    return payload


def _unchanged_library_result(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    payload = _read_payload(resolved)
    entries = payload.get("entries") if isinstance(payload, dict) else None
    return {
        "updated_count": 0,
        "entry_count": len(entries) if isinstance(entries, dict) else 0,
        "path": str(resolved),
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, numbers.Integral)):
        return bool(value) if isinstance(value, bool) else int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _latest_date_text(values: pd.Series) -> str:
    valid = values.dropna()
    return valid.max().strftime("%Y-%m-%d") if not valid.empty else ""


def _parse_date(value: Any) -> date | None:
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.date()


def _empty_diagnostics(candidate_count: int) -> dict[str, Any]:
    return {
        "candidate_count": candidate_count,
        "entry_count": 0,
        "fresh_count": 0,
        "stale_count": 0,
        "matched_count": 0,
        "coverage_pct": 0.0,
    }
