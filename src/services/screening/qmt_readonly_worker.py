# -*- coding: utf-8 -*-
"""Small Python 3.6-compatible worker for read-only QMT market data."""

from __future__ import print_function

import datetime
import json
import math
import sys


_RESULT_PREFIX = "DSA_QMT_QUOTES="


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quote_time(value):
    number = _number(value)
    if number is None or number <= 0:
        return ""
    if number > 100000000000:
        number /= 1000.0
    try:
        return datetime.datetime.fromtimestamp(number).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _first(item, names):
    for name in names:
        if name in item:
            return item.get(name)
    return None


def _normalize_row(symbol, item):
    price = _number(_first(item, ("lastPrice", "last_price", "price")))
    previous_close = _number(
        _first(item, ("lastClose", "preClose", "pre_close", "previous_close"))
    )
    if price is None or price <= 0:
        return None
    change_pct = None
    if previous_close is not None and previous_close > 0:
        change_pct = (price / previous_close - 1.0) * 100.0
    return {
        "code": str(symbol).split(".", 1)[0],
        "qmt_code": str(symbol),
        "price": price,
        "previous_close": previous_close,
        "change_pct": change_pct,
        "open": _number(_first(item, ("open", "openPrice"))),
        "high": _number(_first(item, ("high", "highPrice"))),
        "low": _number(_first(item, ("low", "lowPrice"))),
        "amount": _number(_first(item, ("amount", "turnover"))),
        "volume": _number(_first(item, ("volume", "pvolume"))),
        "quote_time": _quote_time(_first(item, ("time", "timestamp"))),
    }


def _sector_rows(xtdata, symbols, prefixes):
    sectors = xtdata.get_sector_list()
    selected = [
        str(name)
        for name in sectors
        if any(str(name).startswith(prefix) for prefix in prefixes)
    ]
    if not selected:
        xtdata.download_sector_data()
        sectors = xtdata.get_sector_list()
        selected = [
            str(name)
            for name in sectors
            if any(str(name).startswith(prefix) for prefix in prefixes)
        ]

    target = set(symbols)
    concepts_by_symbol = {}
    membership_count = 0
    for sector_name in sorted(selected):
        prefix = next(
            (item for item in prefixes if sector_name.startswith(item)),
            "",
        )
        label = sector_name[len(prefix):].strip() if prefix else sector_name.strip()
        if not label:
            continue
        members = xtdata.get_stock_list_in_sector(sector_name)
        if not isinstance(members, (list, tuple)):
            continue
        for symbol in members:
            symbol = str(symbol).strip().upper()
            if symbol not in target:
                continue
            concepts_by_symbol.setdefault(symbol, []).append(label)
            membership_count += 1

    rows = []
    for symbol, concepts in sorted(concepts_by_symbol.items()):
        unique = sorted(set(concepts))
        rows.append({
            "code": symbol.split(".", 1)[0],
            "qmt_code": symbol,
            "concepts": ",".join(unique),
            "concept_count": len(unique),
        })
    return rows, len(selected), membership_count


def main():
    request = json.loads(sys.stdin.read() or "{}")
    xtquant_path = str(request.get("xtquant_path") or "").strip()
    if xtquant_path and xtquant_path not in sys.path:
        sys.path.insert(0, xtquant_path)

    from xtquant import xtdata

    port = request.get("port")
    if port not in (None, ""):
        connector = getattr(xtdata, "connect", None) or getattr(xtdata, "reconnect", None)
        if connector is None:
            raise RuntimeError("Installed xtdata has no supported connection function")
        connected = connector(port=int(port))
        if connected is False:
            raise RuntimeError("QMT market-data connection was rejected")

    symbols = [
        str(value).strip().upper()
        for value in request.get("codes", [])
        if str(value).strip()
    ]
    mode = str(request.get("mode") or "quotes").strip().lower()
    if mode == "sectors":
        prefixes = [
            str(value).strip()
            for value in request.get("sector_prefixes", [])
            if str(value).strip()
        ]
        if not prefixes:
            raise RuntimeError("QMT sector prefixes are empty")
        rows, sector_count, membership_count = _sector_rows(
            xtdata,
            symbols,
            prefixes,
        )
        print(_RESULT_PREFIX + json.dumps({
            "requested_count": len(symbols),
            "row_count": len(rows),
            "sector_count": sector_count,
            "membership_count": membership_count,
            "sector_prefixes": prefixes,
            "rows": rows,
        }, ensure_ascii=False, allow_nan=False))
        return
    if mode != "quotes":
        raise RuntimeError("Unsupported QMT worker mode: " + mode)

    chunk_size = max(1, int(request.get("chunk_size") or 300))
    rows = []
    for offset in range(0, len(symbols), chunk_size):
        batch = symbols[offset:offset + chunk_size]
        payload = xtdata.get_full_tick(batch)
        if not isinstance(payload, dict):
            continue
        for symbol in batch:
            item = payload.get(symbol)
            if not isinstance(item, dict):
                continue
            row = _normalize_row(symbol, item)
            if row is not None:
                rows.append(row)

    print(_RESULT_PREFIX + json.dumps({
        "requested_count": len(symbols),
        "row_count": len(rows),
        "rows": rows,
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(_RESULT_PREFIX + json.dumps({"error": str(exc), "rows": []}, ensure_ascii=False))
        raise
