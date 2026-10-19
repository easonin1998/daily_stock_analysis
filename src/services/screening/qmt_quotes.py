# -*- coding: utf-8 -*-
"""Read-only QMT quote adapter executed through a compatible sidecar Python."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

import pandas as pd

_RESULT_PREFIX = "DSA_QMT_QUOTES="
_WORKER_PATH = Path(__file__).with_name("qmt_readonly_worker.py")


def fetch_qmt_quotes(
    codes: Iterable[object],
    *,
    python_executable: str | Path | None = None,
    xtquant_path: str | Path | None = None,
    port: int | None = None,
    timeout_sec: float = 45.0,
) -> pd.DataFrame:
    """Fetch full ticks for an existing stock universe without loading trade APIs."""
    symbols = sorted(filter(None, (_to_qmt_symbol(value) for value in codes)))
    if not symbols:
        return pd.DataFrame(columns=_quote_columns())

    executable = str(python_executable or sys.executable)
    request = {
        "mode": "quotes",
        "codes": symbols,
        "xtquant_path": str(xtquant_path or ""),
        "port": port,
        "chunk_size": 300,
    }
    payload = _run_worker(
        request,
        executable=executable,
        timeout_sec=timeout_sec,
        request_label="quote",
    )
    rows = payload.get("rows")
    frame = pd.DataFrame(rows if isinstance(rows, list) else [])
    for column in _quote_columns():
        if column not in frame.columns:
            frame[column] = pd.NA
    if frame.empty:
        raise RuntimeError("QMT returned no usable full-tick rows")
    frame["code"] = frame["code"].map(_normalize_code)
    for column in (
        "price", "previous_close", "change_pct", "open", "high", "low", "amount", "volume",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["code"].ne("") & frame["price"].gt(0)].copy()
    frame = frame.drop_duplicates("code", keep="last").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("QMT returned rows, but none contained a usable last price")
    frame.attrs["snapshot_source"] = "qmt_readonly"
    frame.attrs["fallback_used"] = False
    frame.attrs["stale"] = False
    frame.attrs["requested_count"] = int(payload.get("requested_count") or len(symbols))
    quote_times = pd.to_datetime(frame["quote_time"], errors="coerce")
    if quote_times.notna().any():
        frame.attrs["quote_as_of"] = quote_times.max().isoformat()
        frame.attrs["trading_date"] = quote_times.max().strftime("%Y-%m-%d")
    return frame[_quote_columns()]


def fetch_qmt_sector_membership(
    codes: Iterable[object],
    *,
    sector_prefixes: Iterable[str] = ("TGN",),
    python_executable: str | Path | None = None,
    xtquant_path: str | Path | None = None,
    port: int | None = None,
    timeout_sec: float = 120.0,
) -> pd.DataFrame:
    """Fetch QMT concept membership for the supplied A-share universe."""
    symbols = sorted(filter(None, (_to_qmt_symbol(value) for value in codes)))
    prefixes = [str(value).strip() for value in sector_prefixes if str(value).strip()]
    if not symbols or not prefixes:
        return pd.DataFrame(columns=_sector_columns())

    executable = str(python_executable or sys.executable)
    request = {
        "mode": "sectors",
        "codes": symbols,
        "sector_prefixes": prefixes,
        "xtquant_path": str(xtquant_path or ""),
        "port": port,
    }
    payload = _run_worker(
        request,
        executable=executable,
        timeout_sec=timeout_sec,
        request_label="sector",
    )
    rows = payload.get("rows")
    frame = pd.DataFrame(rows if isinstance(rows, list) else [])
    for column in _sector_columns():
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["code"] = frame["code"].map(_normalize_code)
    frame["concepts"] = frame["concepts"].fillna("").astype(str).str.strip()
    frame["concept_count"] = pd.to_numeric(frame["concept_count"], errors="coerce")
    frame = frame[frame["code"].ne("") & frame["concepts"].ne("")].copy()
    frame = frame.drop_duplicates("code", keep="last").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("QMT returned no usable concept memberships")
    frame.attrs["membership_source"] = "qmt_sector:" + ",".join(prefixes)
    frame.attrs["requested_count"] = int(payload.get("requested_count") or len(symbols))
    frame.attrs["sector_count"] = int(payload.get("sector_count") or 0)
    frame.attrs["membership_count"] = int(payload.get("membership_count") or 0)
    return frame[_sector_columns()]


def overlay_qmt_quotes(
    snapshot: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    min_coverage_pct: float = 90.0,
    expected_trading_date: str | None = None,
) -> pd.DataFrame:
    """Use QMT as the authoritative quote layer over snapshot metadata.

    The reference snapshot continues to supply names, valuation, market-cap,
    and turnover fields that ``get_full_tick`` does not expose. QMT replaces
    only market fields for matched rows, and is accepted only when its full-
    universe coverage and trading date satisfy the configured contract.
    """
    if snapshot.empty or "code" not in snapshot.columns:
        raise RuntimeError("QMT primary overlay requires a non-empty code snapshot")
    if quotes.empty or "code" not in quotes.columns:
        raise RuntimeError("QMT primary overlay received no quote rows")

    reference_attrs = dict(snapshot.attrs)
    work = snapshot.copy()
    work["__qmt_code_key"] = work["code"].map(_normalize_code)
    work = work[work["__qmt_code_key"].ne("")].copy()
    if work.empty:
        raise RuntimeError("QMT primary overlay found no valid snapshot codes")

    incoming = quotes.copy()
    incoming["__qmt_code_key"] = incoming["code"].map(_normalize_code)
    quote_fields = [
        column
        for column in (
            "price",
            "previous_close",
            "change_pct",
            "open",
            "high",
            "low",
            "amount",
            "volume",
            "quote_time",
            "qmt_code",
        )
        if column in incoming.columns
    ]
    incoming = incoming[["__qmt_code_key", *quote_fields]].drop_duplicates(
        "__qmt_code_key", keep="last"
    )
    incoming = incoming.rename(
        columns={column: f"__qmt_{column}" for column in quote_fields}
    )
    result = work.merge(
        incoming,
        on="__qmt_code_key",
        how="left",
        validate="many_to_one",
    )
    qmt_price = pd.to_numeric(result.get("__qmt_price"), errors="coerce")
    matched = qmt_price.gt(0)
    coverage_pct = round(float(matched.mean() * 100.0), 4)
    required_coverage = max(0.0, min(float(min_coverage_pct), 100.0))
    if coverage_pct < required_coverage:
        raise RuntimeError(
            "QMT primary quote coverage below threshold: "
            f"{coverage_pct}% < {required_coverage}%"
        )

    quote_trading_date = str(quotes.attrs.get("trading_date") or "").strip()
    expected_date = str(expected_trading_date or "").strip()
    if expected_date and quote_trading_date != expected_date:
        raise RuntimeError(
            "QMT primary quote trading date mismatch: "
            f"received {quote_trading_date or '<missing>'}, expected {expected_date}"
        )

    for column in quote_fields:
        qmt_column = f"__qmt_{column}"
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = result[qmt_column].where(
            matched & result[qmt_column].notna(),
            result[column],
        )
        result = result.drop(columns=[qmt_column])
    result = result.drop(columns=["__qmt_code_key"])
    result.attrs.update(reference_attrs)
    result.attrs.update({
        "snapshot_source": "qmt_readonly",
        "snapshot_reference_source": str(
            reference_attrs.get("snapshot_source") or "unknown"
        ),
        "qmt_coverage_count": int(matched.sum()),
        "qmt_universe_count": int(len(result)),
        "qmt_coverage_pct": coverage_pct,
        "qmt_quote_as_of": str(quotes.attrs.get("quote_as_of") or ""),
        "qmt_trading_date": quote_trading_date,
        "fallback_used": False,
        "stale": False,
    })
    return result


def _run_worker(
    request: dict[str, object],
    *,
    executable: str,
    timeout_sec: float,
    request_label: str,
) -> dict[str, object]:
    if not Path(executable).is_file() and shutil.which(executable) is None:
        raise RuntimeError(f"QMT Python executable not found: {executable}")
    environment = dict(os.environ)
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        completed = subprocess.run(
            [executable, str(_WORKER_PATH)],
            input=json.dumps(request, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(float(timeout_sec), 1.0),
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"QMT read-only {request_label} request timed out after {timeout_sec:g}s"
        ) from exc

    payload = _extract_payload(completed.stdout)
    if completed.returncode != 0 or payload.get("error"):
        detail = str(
            payload.get("error") or completed.stderr or "unknown worker failure"
        ).strip()
        raise RuntimeError(f"QMT read-only {request_label} request failed: {detail}")
    return payload


def _extract_payload(output: str) -> dict[str, object]:
    for line in reversed(str(output or "").splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                parsed = json.loads(line[len(_RESULT_PREFIX):])
            except json.JSONDecodeError as exc:
                raise RuntimeError("QMT worker returned malformed JSON") from exc
            return parsed if isinstance(parsed, dict) else {}
    raise RuntimeError("QMT worker returned no structured result")


def _to_qmt_symbol(value: object) -> str:
    code = _normalize_code(value)
    if not code or not code.isdigit() or len(code) != 6:
        return ""
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{code}.SH"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{code}.SZ"
    if code.startswith(("43", "83", "87", "88", "92")):
        return f"{code}.BJ"
    return ""


def _normalize_code(value: object) -> str:
    text = str(value or "").strip().upper()
    if "." in text:
        text = text.split(".", 1)[0]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else ""


def _quote_columns() -> list[str]:
    return [
        "code", "qmt_code", "price", "previous_close", "change_pct", "open", "high",
        "low", "amount", "volume", "quote_time",
    ]


def _sector_columns() -> list[str]:
    return ["code", "qmt_code", "concepts", "concept_count"]
