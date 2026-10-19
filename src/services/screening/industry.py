# -*- coding: utf-8 -*-
# Derived from AlphaSift revision 9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf.
# Licensed under Apache-2.0 and modified for daily_stock_analysis.
"""Industry and concept enrichment for candidate snapshots."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.services.screening.normalize import (
    normalize_code as _normalize_code,
    safe_float as _safe_float,
    safe_text as _safe_text,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AKSHARE_BOARD_CACHE_SCHEMA = "v3"
_BAOSTOCK_INDUSTRY_CACHE_SCHEMA = "v1"
_CACHE_DIR_UNSET = object()
_NUMERIC_FIELDS = (
    "industry_rank",
    "industry_change_pct",
    "industry_heat_score",
    "concept_heat_score",
    "board_heat_score",
    "board_heat_latest_score",
    "board_heat_trend_score",
    "board_heat_persistence_score",
    "board_heat_cooling_score",
    "board_heat_observations",
    "board_fund_flow_score",
    "board_fund_flow_net_amount",
)
_TEXT_FIELDS = ("board_heat_summary", "board_heat_state")
_HEAT_FIELDS = (*_NUMERIC_FIELDS, *_TEXT_FIELDS)
_FIELD_ALIASES = {
    "industry_rank": ["industry_rank", "行业排名", "板块排名", "排名"],
    "industry_change_pct": ["industry_change_pct", "行业涨跌幅", "板块涨跌幅", "涨跌幅"],
    "industry_heat_score": ["industry_heat_score", "行业热度分"],
    "concept_heat_score": ["concept_heat_score", "概念热度分"],
    "board_heat_score": ["board_heat_score", "theme_heat_score", "板块热度分", "主题热度分"],
    "board_heat_latest_score": ["board_heat_latest_score", "板块最新热度分", "主题最新热度分"],
    "board_heat_trend_score": ["board_heat_trend_score", "板块热度趋势分", "主题热度趋势分"],
    "board_heat_persistence_score": ["board_heat_persistence_score", "板块热度持续分", "主题热度持续分"],
    "board_heat_cooling_score": ["board_heat_cooling_score", "板块降温分", "主题降温分"],
    "board_heat_observations": ["board_heat_observations", "板块热度观测数", "主题热度观测数"],
    "board_heat_summary": ["board_heat_summary", "theme_heat_summary", "板块热度", "主题热度"],
    "board_heat_state": ["board_heat_state", "板块热度状态", "主题热度状态"],
}


def enrich_industry_concepts(
    df: pd.DataFrame,
    *,
    map_files: list[str | Path] | None = None,
    provider: str = "none",
    max_boards: int = 80,
    provider_cache_dir: str | Path | None | object = _CACHE_DIR_UNSET,
    provider_cache_ttl_hours: float | None = None,
    include_fund_flow: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach industry/concepts columns from stable files and optional providers."""
    result = df.copy()
    notes: list[str] = []
    if result.empty or "code" not in result.columns:
        return result, notes

    if "industry" not in result.columns:
        result["industry"] = ""
    if "concepts" not in result.columns:
        result["concepts"] = ""
    for field in _NUMERIC_FIELDS:
        if field not in result.columns:
            result[field] = pd.NA
    for field in _TEXT_FIELDS:
        if field not in result.columns:
            result[field] = ""

    mapping: dict[str, dict[str, object]] = {}
    for path_like in map_files or []:
        file_mapping = load_industry_map(path_like)
        trend_mapping, trend_note = _load_companion_board_heat_trends(path_like)
        if trend_mapping:
            _apply_board_heat_trends(file_mapping, trend_mapping)
        _merge_mapping(mapping, file_mapping)
        notes.append(f"industry map loaded: {path_like} rows={len(file_mapping)}")
        if trend_note:
            notes.append(trend_note)

    if provider and provider.lower() not in {"", "none", "off", "false"}:
        provider_name = provider.lower()
        if provider_name == "akshare":
            provider_mapping, provider_notes = fetch_akshare_board_map(
                max_boards=max_boards,
                cache_dir=provider_cache_dir,
                cache_ttl_hours=provider_cache_ttl_hours,
                include_fund_flow=include_fund_flow,
            )
            _merge_mapping(mapping, provider_mapping)
            notes.extend(provider_notes)
        elif provider_name == "baostock":
            provider_mapping, provider_notes = fetch_baostock_industry_map(
                cache_dir=provider_cache_dir,
                cache_ttl_hours=provider_cache_ttl_hours,
            )
            _merge_mapping(mapping, provider_mapping)
            notes.extend(provider_notes)
        elif provider_name in {"free", "auto", "akshare_baostock"}:
            stable_mapping, stable_notes = fetch_baostock_industry_map(
                cache_dir=provider_cache_dir,
                cache_ttl_hours=provider_cache_ttl_hours,
            )
            _merge_mapping(mapping, stable_mapping)
            notes.extend(stable_notes)
            heat_mapping, heat_notes = fetch_akshare_board_map(
                max_boards=max_boards,
                cache_dir=provider_cache_dir,
                cache_ttl_hours=provider_cache_ttl_hours,
                include_fund_flow=include_fund_flow,
            )
            _merge_mapping(mapping, heat_mapping)
            notes.extend(heat_notes)
        elif provider_name in {"free_cached", "cached_free"}:
            stable_mapping, stable_notes = load_cached_baostock_industry_map(
                cache_dir=provider_cache_dir,
            )
            _merge_mapping(mapping, stable_mapping)
            notes.extend(stable_notes)
            heat_mapping, heat_notes = load_cached_akshare_board_map(
                max_boards=max_boards,
                cache_dir=provider_cache_dir,
                cache_ttl_hours=provider_cache_ttl_hours,
                include_fund_flow=include_fund_flow,
            )
            _merge_mapping(mapping, heat_mapping)
            notes.extend(heat_notes)
        else:
            notes.append(f"industry provider skipped: unsupported provider={provider}")

    if not mapping:
        return result, notes

    result, filled_industry, filled_concepts, filled_heat = _apply_mapping_to_snapshot(
        result,
        mapping,
    )

    notes.append(
        "industry/concepts enrichment applied: "
        f"industry={filled_industry}, concepts={filled_concepts}, heat={filled_heat}"
    )
    return result, notes


def load_industry_map(path_like: str | Path) -> dict[str, dict[str, object]]:
    """Load code -> industry/concepts mapping from CSV, JSON or JSONL."""
    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(f"Industry map file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = pd.read_csv(path, dtype=str).fillna("").to_dict(orient="records")
    elif suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            rows = []
            for code, value in data.items():
                if isinstance(value, dict):
                    rows.append({"code": code, **value})
                elif isinstance(value, str):
                    rows.append({"code": code, "industry": value})
        else:
            rows = []
    else:
        raise ValueError(f"Unsupported industry map format: {path}")

    mapping: dict[str, dict[str, object]] = {}
    for row in rows:
        code = _normalize_code(row.get("code") or row.get("代码"))
        if not code or code == "000000":
            continue
        industry = _safe_text(row.get("industry") or row.get("行业") or row.get("所属行业"))
        concepts = _safe_text(row.get("concepts") or row.get("概念") or row.get("概念题材"))
        item: dict[str, object] = {
            "industry": industry,
            "concepts": concepts,
        }
        for field in _HEAT_FIELDS:
            value = _first_row_value(row, _FIELD_ALIASES.get(field, [field]))
            if field in _NUMERIC_FIELDS:
                parsed = _safe_float(value)
                if parsed is not None:
                    item[field] = int(parsed) if field in {"industry_rank", "board_heat_observations"} else parsed
            else:
                text = _safe_text(value)
                if text:
                    item[field] = text
        mapping[code] = item
    return mapping


def fetch_akshare_board_map(
    *,
    max_boards: int = 80,
    cache_dir: str | Path | None | object = _CACHE_DIR_UNSET,
    cache_ttl_seconds: float | None = None,
    cache_ttl_hours: float | None = None,
    include_fund_flow: bool = False,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Build a code mapping from AkShare industry/concept board constituents.

    This is intentionally optional because it may require many third-party
    requests. For production, a cached CSV/JSON map is preferred.
    """
    board_limit = max(int(max_boards), 1)
    notes: list[str] = []
    mapping: dict[str, dict[str, object]] = {}
    resolved_cache_dir = _resolve_akshare_board_cache_dir(cache_dir)
    cache_path = (
        _akshare_board_cache_path(
            resolved_cache_dir,
            max_boards=board_limit,
            include_fund_flow=include_fund_flow,
        )
        if resolved_cache_dir is not None
        else None
    )
    if cache_path is not None:
        cached_mapping, cache_note = _read_akshare_board_cache(
            cache_path,
            max_boards=board_limit,
            include_fund_flow=include_fund_flow,
            ttl_seconds=_resolve_cache_ttl_seconds(
                cache_ttl_seconds=cache_ttl_seconds,
                cache_ttl_hours=cache_ttl_hours,
            ),
        )
        if cache_note:
            notes.append(cache_note)
        if cached_mapping is not None:
            if all(
                _mapping_has_board_field(cached_mapping, field)
                for field in ("industry", "concepts")
            ):
                return cached_mapping, notes
            mapping.update(cached_mapping)
            missing = [
                field
                for field in ("industry", "concepts")
                if not _mapping_has_board_field(mapping, field)
            ]
            notes.append(
                "industry provider cache is partial; refreshing missing fields: "
                + ",".join(missing)
            )

    import akshare as ak

    flow_maps = _fetch_akshare_board_fund_flow(ak, notes) if include_fund_flow else {}
    board_specs = [
        (
            "industry",
            ("eastmoney", ak.stock_board_industry_name_em, ak.stock_board_industry_cons_em),
            ("eastmoney_http", lambda: _fetch_eastmoney_http_boards("industry"), None),
        ),
        (
            "concepts",
            ("eastmoney", ak.stock_board_concept_name_em, ak.stock_board_concept_cons_em),
            ("eastmoney_http", lambda: _fetch_eastmoney_http_boards("concepts"), None),
        ),
    ]
    for field, *providers in board_specs:
        if _mapping_has_board_field(mapping, field):
            notes.append(f"akshare {field} retained from partial cache")
            continue
        provider_name = ""
        boards = pd.DataFrame()
        cons_func = None
        for candidate_name, list_func, candidate_cons_func in providers:
            try:
                boards = list_func()
            except Exception as exc:
                notes.append(f"akshare {field} {candidate_name} board list failed: {exc}")
                continue
            if isinstance(boards, pd.DataFrame) and not boards.empty:
                provider_name = candidate_name
                cons_func = candidate_cons_func
                break
        if not provider_name or boards.empty:
            continue
        board_items = _board_items(
            boards,
            infer_rank=provider_name in {"eastmoney", "eastmoney_http"},
        )[:board_limit]
        loaded = 0
        for board_item in board_items:
            board = board_item["name"]
            try:
                if provider_name == "eastmoney_http":
                    members = _fetch_eastmoney_http_constituents(
                        board_code=_safe_text(board_item.get("code")),
                    )
                else:
                    members = cons_func(symbol=board)
                if not isinstance(members, pd.DataFrame) or members.empty:
                    raise RuntimeError("empty constituent response")
            except Exception as exc:
                notes.append(f"akshare {field} board skipped {board}: {exc}")
                continue
            change_pct = _safe_float(board_item.get("change_pct"))
            rank = _safe_float(board_item.get("rank"))
            heat_score = (
                _board_heat_score(change_pct=change_pct, rank=rank)
                if change_pct is not None or rank is not None
                else None
            )
            heat_summary = _board_heat_summary(
                board,
                change_pct=change_pct,
                rank=rank,
            )
            flow_item = flow_maps.get(field, {}).get(board, {})
            for _, row in members.iterrows():
                code = _normalize_code(row.get("代码") or row.get("code"))
                if not code or code == "000000":
                    continue
                item = mapping.setdefault(code, {"industry": "", "concepts": ""})
                if field == "industry" and not item["industry"]:
                    item["industry"] = board
                    if board_item.get("rank") is not None:
                        item["industry_rank"] = int(float(board_item["rank"]))
                    if board_item.get("change_pct") is not None:
                        item["industry_change_pct"] = _safe_float(board_item.get("change_pct"))
                    item["industry_heat_score"] = heat_score
                elif field == "concepts":
                    item["concepts"] = _merge_label_text(item.get("concepts", ""), board)
                    item["concept_heat_score"] = _max_numeric(item.get("concept_heat_score"), heat_score)
                if heat_score is not None:
                    item["board_heat_score"] = _max_numeric(
                        item.get("board_heat_score"), heat_score
                    )
                flow_score = _safe_float(flow_item.get("score"))
                flow_amount = _safe_float(flow_item.get("net_amount"))
                if flow_score is not None:
                    item["board_fund_flow_score"] = _max_numeric(
                        item.get("board_fund_flow_score"),
                        flow_score,
                    )
                if flow_amount is not None:
                    current_amount = _safe_float(item.get("board_fund_flow_net_amount"))
                    if current_amount is None or abs(flow_amount) > abs(current_amount):
                        item["board_fund_flow_net_amount"] = flow_amount
                item["board_heat_summary"] = _merge_summary_text(
                    _safe_text(item.get("board_heat_summary")),
                    heat_summary,
                )
            loaded += 1
        notes.append(
            f"akshare {field} {provider_name} boards loaded: {loaded}/{len(board_items)}"
        )
    if not mapping and cache_path is not None:
        stale_mapping, stale_note = _read_akshare_board_cache(
            cache_path,
            max_boards=board_limit,
            include_fund_flow=include_fund_flow,
            ttl_seconds=float("inf"),
        )
        if stale_mapping is not None:
            notes.append(stale_note.replace("cache hit", "stale cache fallback"))
            return stale_mapping, notes
    if cache_path is not None and mapping:
        cache_note = _write_akshare_board_cache(
            cache_path,
            mapping,
            max_boards=board_limit,
            include_fund_flow=include_fund_flow,
        )
        if cache_note:
            notes.append(cache_note)
    return mapping, notes


def fetch_baostock_industry_map(
    *,
    cache_dir: str | Path | None | object = _CACHE_DIR_UNSET,
    cache_ttl_seconds: float | None = None,
    cache_ttl_hours: float | None = None,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Load the full A-share industry classification in one Baostock session."""
    notes: list[str] = []
    resolved_cache_dir = _resolve_akshare_board_cache_dir(cache_dir)
    cache_path = (
        resolved_cache_dir / f"baostock_industry_map_{_BAOSTOCK_INDUSTRY_CACHE_SCHEMA}.json"
        if resolved_cache_dir is not None
        else None
    )
    ttl_seconds = _resolve_cache_ttl_seconds(
        cache_ttl_seconds=cache_ttl_seconds,
        cache_ttl_hours=cache_ttl_hours,
    )
    if cache_path is not None:
        cached, cache_note = _read_simple_provider_cache(
            cache_path,
            provider="baostock",
            schema=_BAOSTOCK_INDUSTRY_CACHE_SCHEMA,
            ttl_seconds=ttl_seconds,
        )
        if cache_note:
            notes.append(cache_note)
        if cached is not None:
            return cached, notes

    try:
        from data_provider.baostock_fetcher import BaostockFetcher

        raw_mapping = BaostockFetcher().get_industry_map()
        mapping = {
            _normalize_code(code): {"industry": _safe_text(industry), "concepts": ""}
            for code, industry in raw_mapping.items()
            if _normalize_code(code) and _safe_text(industry)
        }
        notes.append(f"baostock industry rows loaded: {len(mapping)}")
    except Exception as exc:
        notes.append(f"baostock industry provider failed: {exc}")
        if cache_path is not None:
            stale, stale_note = _read_simple_provider_cache(
                cache_path,
                provider="baostock",
                schema=_BAOSTOCK_INDUSTRY_CACHE_SCHEMA,
                ttl_seconds=float("inf"),
            )
            if stale is not None:
                notes.append(stale_note.replace("cache hit", "stale cache fallback"))
                return stale, notes
        return {}, notes

    if cache_path is not None and mapping:
        notes.append(
            _write_simple_provider_cache(
                cache_path,
                mapping,
                provider="baostock",
                schema=_BAOSTOCK_INDUSTRY_CACHE_SCHEMA,
            )
        )
    return mapping, notes


def load_cached_akshare_board_map(
    *,
    max_boards: int = 80,
    cache_dir: str | Path | None | object = _CACHE_DIR_UNSET,
    cache_ttl_hours: float | None = None,
    include_fund_flow: bool = False,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Read the optional AkShare overlay without issuing network requests."""
    resolved_cache_dir = _resolve_akshare_board_cache_dir(cache_dir)
    if resolved_cache_dir is None:
        return {}, ["akshare cached overlay skipped: cache disabled"]
    path = _akshare_board_cache_path(
        resolved_cache_dir,
        max_boards=max(int(max_boards), 1),
        include_fund_flow=include_fund_flow,
    )
    mapping, note = _read_akshare_board_cache(
        path,
        max_boards=max(int(max_boards), 1),
        include_fund_flow=include_fund_flow,
        ttl_seconds=_resolve_cache_ttl_seconds(
            cache_ttl_seconds=None,
            cache_ttl_hours=cache_ttl_hours,
        ),
    )
    if mapping is None:
        return {}, [note or f"akshare cached overlay missing: {path}"]
    return mapping, [note]


def load_cached_baostock_industry_map(
    *,
    cache_dir: str | Path | None | object = _CACHE_DIR_UNSET,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Read the stable Baostock industry map without issuing a network request."""
    resolved_cache_dir = _resolve_akshare_board_cache_dir(cache_dir)
    if resolved_cache_dir is None:
        return {}, ["baostock cached industry skipped: cache disabled"]
    path = resolved_cache_dir / (
        f"baostock_industry_map_{_BAOSTOCK_INDUSTRY_CACHE_SCHEMA}.json"
    )
    mapping, note = _read_simple_provider_cache(
        path,
        provider="baostock",
        schema=_BAOSTOCK_INDUSTRY_CACHE_SCHEMA,
        ttl_seconds=float("inf"),
    )
    if mapping is None:
        return {}, [note or f"baostock cached industry missing: {path}"]
    return mapping, [note.replace("cache hit", "cached industry loaded")]


def save_industry_map(mapping: dict[str, dict[str, object]], path_like: str | Path) -> Path:
    """Persist a code->industry/concepts mapping as CSV or JSON."""
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "code": code,
            "industry": item.get("industry", ""),
            "concepts": item.get("concepts", ""),
            "industry_rank": item.get("industry_rank", ""),
            "industry_change_pct": item.get("industry_change_pct", ""),
            "industry_heat_score": item.get("industry_heat_score", ""),
            "concept_heat_score": item.get("concept_heat_score", ""),
            "board_heat_score": item.get("board_heat_score", ""),
            "board_heat_latest_score": item.get("board_heat_latest_score", ""),
            "board_heat_trend_score": item.get("board_heat_trend_score", ""),
            "board_heat_persistence_score": item.get("board_heat_persistence_score", ""),
            "board_heat_cooling_score": item.get("board_heat_cooling_score", ""),
            "board_heat_observations": item.get("board_heat_observations", ""),
            "board_fund_flow_score": item.get("board_fund_flow_score", ""),
            "board_fund_flow_net_amount": item.get("board_fund_flow_net_amount", ""),
            "board_heat_summary": item.get("board_heat_summary", ""),
            "board_heat_state": item.get("board_heat_state", ""),
        }
        for code, item in sorted(mapping.items())
    ]
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def _apply_mapping_to_snapshot(
    result: pd.DataFrame,
    mapping: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, int, int, int]:
    map_df = _mapping_dataframe(mapping)
    if map_df.empty:
        return result, 0, 0, 0

    output = result.copy()
    work = output.copy()
    work["__industry_row"] = range(len(work))
    work["__industry_code"] = work["code"].map(_normalize_code)
    merged = work.merge(map_df, on="__industry_code", how="left", sort=False)
    merged = merged.sort_values("__industry_row", kind="stable")
    merged.index = output.index

    filled_industry = _apply_industry_column(output, merged)
    filled_concepts = _apply_concepts_column(output, merged)
    filled_heat = 0
    for field in _NUMERIC_FIELDS:
        filled_heat += _apply_numeric_column(output, merged, field)
    for field in _TEXT_FIELDS:
        filled_heat += _apply_text_column(output, merged, field)
    return output, filled_industry, filled_concepts, filled_heat


def _mapping_dataframe(mapping: dict[str, dict[str, object]]) -> pd.DataFrame:
    fields = ("industry", "concepts", *_HEAT_FIELDS)
    rows: list[dict[str, object]] = []
    for code, item in mapping.items():
        normalized = _normalize_code(code)
        if not normalized or normalized == "000000" or not isinstance(item, dict):
            continue
        row = {"__industry_code": normalized}
        for field in fields:
            row[f"__map_{field}"] = item.get(field, pd.NA)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["__industry_code", *(f"__map_{field}" for field in fields)])
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(subset=["__industry_code"], keep="last")


def _apply_industry_column(output: pd.DataFrame, merged: pd.DataFrame) -> int:
    current = output["industry"].map(_safe_text)
    incoming = merged["__map_industry"].map(_safe_text)
    mask = current.eq("") & incoming.ne("")
    if mask.any():
        output.loc[mask, "industry"] = incoming[mask].to_numpy()
    return int(mask.sum())


def _apply_concepts_column(output: pd.DataFrame, merged: pd.DataFrame) -> int:
    current = output["concepts"].map(_safe_text)
    incoming = merged["__map_concepts"].map(_safe_text)
    candidate_mask = incoming.ne("")
    if not candidate_mask.any():
        return 0
    merged_values = pd.Series(
        [
            _merge_label_text(left, right) if right else left
            for left, right in zip(current.tolist(), incoming.tolist(), strict=False)
        ],
        index=output.index,
    )
    mask = candidate_mask & merged_values.ne(current)
    if mask.any():
        output.loc[mask, "concepts"] = merged_values[mask].to_numpy()
    return int(mask.sum())


def _apply_numeric_column(output: pd.DataFrame, merged: pd.DataFrame, field: str) -> int:
    incoming = merged[f"__map_{field}"].map(_safe_float)
    current = output[field].map(_safe_float)
    mask = _numeric_replacement_mask(field, incoming, current)
    if not mask.any():
        return 0
    values = incoming[mask]
    if field in {"industry_rank", "board_heat_observations"}:
        values = values.map(int)
    output.loc[mask, field] = values.to_numpy()
    return int(mask.sum())


def _apply_text_column(output: pd.DataFrame, merged: pd.DataFrame, field: str) -> int:
    current = output[field].map(_safe_text)
    incoming = merged[f"__map_{field}"].map(_safe_text)
    candidate_mask = incoming.ne("")
    if not candidate_mask.any():
        return 0
    if field == "board_heat_summary":
        merged_values = pd.Series(
            [
                _merge_summary_text(left, right) if right else left
                for left, right in zip(current.tolist(), incoming.tolist(), strict=False)
            ],
            index=output.index,
        )
    else:
        merged_values = pd.Series(
            [left or right for left, right in zip(current.tolist(), incoming.tolist(), strict=False)],
            index=output.index,
        )
    mask = candidate_mask & merged_values.ne(current)
    if mask.any():
        output.loc[mask, field] = merged_values[mask].to_numpy()
    return int(mask.sum())


def _numeric_replacement_mask(field: str, incoming: pd.Series, current: pd.Series) -> pd.Series:
    candidate_mask = incoming.notna()
    missing_mask = current.isna()
    comparable_mask = candidate_mask & ~missing_mask
    wins = pd.Series(False, index=incoming.index)
    if comparable_mask.any():
        new_values = incoming[comparable_mask].astype(float)
        current_values = current[comparable_mask].astype(float)
        if field == "industry_rank":
            wins.loc[comparable_mask] = new_values < current_values
        elif field == "board_heat_observations":
            wins.loc[comparable_mask] = new_values > current_values
        elif field in {"board_heat_latest_score", "board_heat_persistence_score", "board_heat_cooling_score"}:
            wins.loc[comparable_mask] = new_values > current_values
        elif field == "board_heat_trend_score":
            wins.loc[comparable_mask] = new_values.abs() > current_values.abs()
        elif field.endswith("heat_score"):
            wins.loc[comparable_mask] = new_values > current_values
    return candidate_mask & (missing_mask | wins)


def _resolve_akshare_board_cache_dir(cache_dir: str | Path | None | object) -> Path | None:
    if cache_dir is _CACHE_DIR_UNSET:
        return _default_akshare_board_cache_dir()
    if cache_dir is None:
        return None
    return Path(cache_dir)


def _default_akshare_board_cache_dir() -> Path:
    explicit = (
        os.getenv("SCREENING_INDUSTRY_PROVIDER_CACHE_DIR", "").strip()
        or os.getenv("INDUSTRY_PROVIDER_CACHE_DIR", "").strip()
    )
    if explicit:
        return Path(explicit)
    data_dir = Path(os.getenv("SCREENING_DATA_DIR", str(_PROJECT_ROOT / "data")))
    return data_dir / "industry_provider_cache"


def _resolve_cache_ttl_seconds(
    *,
    cache_ttl_seconds: float | None,
    cache_ttl_hours: float | None,
) -> float:
    if cache_ttl_seconds is not None:
        return float(cache_ttl_seconds)
    if cache_ttl_hours is not None:
        return float(cache_ttl_hours) * 3600
    raw_hours = (
        os.getenv("SCREENING_INDUSTRY_PROVIDER_CACHE_TTL_HOURS", "").strip()
        or os.getenv("INDUSTRY_PROVIDER_CACHE_TTL_HOURS", "").strip()
        or "24"
    )
    return max(0.0, float(raw_hours)) * 3600


def _akshare_board_cache_path(
    cache_dir: Path,
    *,
    max_boards: int,
    include_fund_flow: bool,
) -> Path:
    flow = "with_flow" if include_fund_flow else "no_flow"
    return cache_dir / (
        f"akshare_board_map_{_AKSHARE_BOARD_CACHE_SCHEMA}_"
        f"max_boards_{int(max_boards)}_{flow}.json"
    )


def _read_akshare_board_cache(
    path: Path,
    *,
    max_boards: int,
    include_fund_flow: bool,
    ttl_seconds: float,
) -> tuple[dict[str, dict[str, object]] | None, str]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None, ""
    if ttl_seconds <= 0:
        return None, f"industry provider cache expired: {path}"
    age_seconds = time.time() - stat.st_mtime
    if age_seconds > ttl_seconds:
        return None, f"industry provider cache expired: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"industry provider cache skipped: {path} error={exc}"
    if not isinstance(payload, dict):
        return None, f"industry provider cache skipped: {path} invalid payload"
    if (
        payload.get("schema") != _AKSHARE_BOARD_CACHE_SCHEMA
        or payload.get("provider") != "akshare"
        or int(payload.get("max_boards", 0) or 0) != int(max_boards)
        or bool(payload.get("include_fund_flow", False)) != bool(include_fund_flow)
    ):
        return None, f"industry provider cache skipped: {path} schema mismatch"
    mapping = _normalize_cached_mapping(payload.get("mapping"))
    if mapping is None:
        return None, f"industry provider cache skipped: {path} invalid mapping"
    return mapping, f"industry provider cache hit: {path} rows={len(mapping)}"


def _write_akshare_board_cache(
    path: Path,
    mapping: dict[str, dict[str, object]],
    *,
    max_boards: int,
    include_fund_flow: bool,
) -> str:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _AKSHARE_BOARD_CACHE_SCHEMA,
            "provider": "akshare",
            "max_boards": int(max_boards),
            "include_fund_flow": bool(include_fund_flow),
            "created_at": datetime.now().isoformat(),
            "mapping": _json_safe_mapping(mapping),
        }
        tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        return f"industry provider cache saved: {path} rows={len(mapping)}"
    except Exception as exc:
        return f"industry provider cache skipped: {path} error={exc}"


def _read_simple_provider_cache(
    path: Path,
    *,
    provider: str,
    schema: str,
    ttl_seconds: float,
) -> tuple[dict[str, dict[str, object]] | None, str]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None, ""
    if ttl_seconds <= 0 or time.time() - stat.st_mtime > ttl_seconds:
        return None, f"industry provider cache expired: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"industry provider cache skipped: {path} error={exc}"
    if not isinstance(payload, dict):
        return None, f"industry provider cache skipped: {path} invalid payload"
    if payload.get("schema") != schema or payload.get("provider") != provider:
        return None, f"industry provider cache skipped: {path} schema mismatch"
    mapping = _normalize_cached_mapping(payload.get("mapping"))
    if mapping is None:
        return None, f"industry provider cache skipped: {path} invalid mapping"
    return mapping, f"industry provider cache hit: {path} rows={len(mapping)}"


def _write_simple_provider_cache(
    path: Path,
    mapping: dict[str, dict[str, object]],
    *,
    provider: str,
    schema: str,
) -> str:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": schema,
            "provider": provider,
            "created_at": datetime.now().isoformat(),
            "mapping": _json_safe_mapping(mapping),
        }
        tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        return f"industry provider cache saved: {path} rows={len(mapping)}"
    except Exception as exc:
        return f"industry provider cache skipped: {path} error={exc}"


def _normalize_cached_mapping(value: object) -> dict[str, dict[str, object]] | None:
    if not isinstance(value, dict):
        return None
    mapping: dict[str, dict[str, object]] = {}
    for code, raw_item in value.items():
        normalized = _normalize_code(code)
        if not normalized or normalized == "000000" or not isinstance(raw_item, dict):
            continue
        item: dict[str, object] = {
            "industry": _safe_text(raw_item.get("industry")),
            "concepts": _safe_text(raw_item.get("concepts")),
        }
        for field in _NUMERIC_FIELDS:
            parsed = _safe_float(raw_item.get(field))
            if parsed is not None:
                item[field] = int(parsed) if field in {"industry_rank", "board_heat_observations"} else parsed
        for field in _TEXT_FIELDS:
            text = _safe_text(raw_item.get(field))
            if text:
                item[field] = text
        mapping[normalized] = item
    return mapping


def _json_safe_mapping(mapping: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        code: _json_safe_item(item)
        for code, item in sorted(mapping.items())
        if isinstance(item, dict)
    }


def _json_safe_item(item: dict[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {
        "industry": _safe_text(item.get("industry")),
        "concepts": _safe_text(item.get("concepts")),
    }
    for field in _NUMERIC_FIELDS:
        value = _safe_float(item.get(field))
        if value is not None:
            cleaned[field] = int(value) if field in {"industry_rank", "board_heat_observations"} else value
    for field in _TEXT_FIELDS:
        text = _safe_text(item.get(field))
        if text:
            cleaned[field] = text
    return cleaned


def _board_names(df: pd.DataFrame) -> list[str]:
    for column in ("板块名称", "名称", "name"):
        if column in df.columns:
            return [_safe_text(item) for item in df[column].tolist() if _safe_text(item)]
    return []


def _board_items(
    df: pd.DataFrame,
    *,
    infer_rank: bool = True,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        name = _first_row_value(row, ["板块名称", "名称", "name"])
        if not _safe_text(name):
            continue
        rank = _safe_float(_first_row_value(row, ["排名", "序号", "rank"]))
        if rank is None and infer_rank:
            rank = float(idx + 1)
        change_pct = _safe_float(_first_row_value(row, ["涨跌幅", "涨幅", "change_pct"]))
        items.append({
            "name": _safe_text(name),
            "code": _safe_text(_first_row_value(row, ["板块代码", "代码", "code"])),
            "rank": rank,
            "change_pct": change_pct,
        })
    return items


def _fetch_eastmoney_http_boards(board_type: str) -> pd.DataFrame:
    """Fetch the live board ranking from the HTTP endpoint used as EM fallback."""
    market_type = "2" if board_type == "industry" else "3"
    rows = _fetch_eastmoney_http_pages({
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": f"m:90 t:{market_type} f:!50",
        "fields": "f3,f12,f14",
    })
    return pd.DataFrame([
        {
            "name": _safe_text(item.get("f14")),
            "code": _safe_text(item.get("f12")),
            "change_pct": _safe_float(item.get("f3")),
            "rank": index + 1,
        }
        for index, item in enumerate(rows)
        if _safe_text(item.get("f12")) and _safe_text(item.get("f14"))
    ])


def _fetch_eastmoney_http_constituents(*, board_code: str) -> pd.DataFrame:
    """Fetch complete constituents for one EastMoney board code."""
    if not board_code:
        raise RuntimeError("EastMoney board code is missing")
    rows = _fetch_eastmoney_http_pages({
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": f"b:{board_code} f:!50",
        "fields": "f3,f6,f8,f10,f12,f14",
    })
    records = [
        {
            "code": _normalize_code(item.get("f12")),
            "name": _safe_text(item.get("f14")),
            "change_pct": _safe_float(item.get("f3")),
            "amount": _safe_float(item.get("f6")),
            "turnover_rate": _safe_float(item.get("f8")),
            "volume_ratio": _safe_float(item.get("f10")),
        }
        for item in rows
        if _normalize_code(item.get("f12"))
    ]
    return pd.DataFrame(
        records,
        columns=("code", "name", "change_pct", "amount", "turnover_rate", "volume_ratio"),
    ).drop_duplicates("code", keep="last")


def _fetch_eastmoney_http_pages(params: dict[str, str]) -> list[dict[str, object]]:
    import requests

    hosts = (
        "http://79.push2.eastmoney.com/api/qt/clist/get",
        "http://17.push2.eastmoney.com/api/qt/clist/get",
        "http://push2.eastmoney.com/api/qt/clist/get",
    )
    page_size = 100
    rows: list[dict[str, object]] = []
    page = 1
    while True:
        request_params = {**params, "pn": str(page), "pz": str(page_size)}
        payload = None
        errors: list[str] = []
        for url in hosts:
            try:
                response = requests.get(
                    url,
                    params=request_params,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Referer": "http://quote.eastmoney.com/",
                    },
                    timeout=15,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        data = payload.get("data") if isinstance(payload, dict) else None
        batch = data.get("diff") if isinstance(data, dict) else None
        if not isinstance(batch, list):
            raise RuntimeError("EastMoney HTTP board request failed: " + "; ".join(errors))
        rows.extend(item for item in batch if isinstance(item, dict))
        total = int(data.get("total") or len(rows))
        if page * page_size >= total or not batch:
            break
        page += 1
        time.sleep(0.05)
    return rows


def _fetch_akshare_board_fund_flow(
    ak: object,
    notes: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Fetch optional current board-flow overlays without making them mandatory."""
    output: dict[str, dict[str, dict[str, float]]] = {}
    specs = (
        ("industry", "行业资金流", "stock_fund_flow_industry"),
        ("concepts", "概念资金流", "stock_fund_flow_concept"),
    )
    for field, sector_type, fallback_name in specs:
        frame, error = _fetch_akshare_board_flow_frame(
            ak,
            sector_type=sector_type,
            fallback_name=fallback_name,
        )
        if error:
            notes.append(f"akshare {field} fund flow skipped: {error}")
            continue
        parsed = _parse_board_fund_flow(frame)
        if parsed:
            output[field] = parsed
            notes.append(f"akshare {field} fund flow loaded: {len(parsed)}")
        else:
            notes.append(f"akshare {field} fund flow skipped: unsupported columns")
    return output


def _fetch_akshare_board_flow_frame(
    ak: object,
    *,
    sector_type: str,
    fallback_name: str,
) -> tuple[pd.DataFrame, str]:
    """Fetch one board-flow table with the domestic free source fallback."""
    try:
        rank_func = getattr(ak, "stock_sector_fund_flow_rank")
        frame = rank_func(indicator="今日", sector_type=sector_type)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame, ""
        primary_error = "rank returned no rows"
    except Exception as exc:
        primary_error = str(exc)
    try:
        fallback = getattr(ak, fallback_name)
        frame = fallback()
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame, ""
        fallback_error = "fallback returned no rows"
    except Exception as exc:
        fallback_error = str(exc)
    return pd.DataFrame(), f"rank={primary_error}; fallback={fallback_error}"


def fetch_akshare_board_fund_flow_snapshot() -> tuple[pd.DataFrame, list[str]]:
    """Return current industry/concept fund flow in the board-heat schema."""
    notes: list[str] = []
    try:
        import akshare as ak
    except Exception as exc:
        return pd.DataFrame(), [f"akshare board fund flow unavailable: {exc}"]

    frames: list[pd.DataFrame] = []
    specs = (
        ("industry", "行业资金流", "stock_fund_flow_industry"),
        ("concept", "概念资金流", "stock_fund_flow_concept"),
    )
    for board_type, sector_type, fallback_name in specs:
        frame, error = _fetch_akshare_board_flow_frame(
            ak,
            sector_type=sector_type,
            fallback_name=fallback_name,
        )
        if error:
            notes.append(f"akshare {board_type} fund flow skipped: {error}")
            continue
        normalized = _normalize_board_fund_flow_snapshot(frame, board_type=board_type)
        if normalized.empty:
            notes.append(
                f"akshare {board_type} fund flow skipped: unsupported columns"
            )
            continue
        notes.append(f"akshare {board_type} fund flow loaded: {len(normalized)}")
        frames.append(normalized)
    if not frames:
        return pd.DataFrame(), notes
    return pd.concat(frames, ignore_index=True), notes


def _normalize_board_fund_flow_snapshot(
    df: pd.DataFrame,
    *,
    board_type: str,
) -> pd.DataFrame:
    """Normalize board-level price and flow fields from AkShare/THS tables."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    def find_column(
        candidates: tuple[str, ...],
        *,
        excluded: tuple[str, ...] = (),
    ) -> object | None:
        for column in df.columns:
            text = str(column).strip()
            if any(token == text or token in text for token in candidates) and not any(
                token in text for token in excluded
            ):
                return column
        return None

    name_column = find_column(
        ("板块名称", "行业名称", "概念名称", "名称", "行业"),
        excluded=("指数", "涨跌幅", "领涨股"),
    )
    amount_column = find_column(
        ("主力净流入-净额", "主力净流入", "净流入", "净额"),
        excluded=("占比", "比例", "%"),
    )
    if name_column is None or amount_column is None:
        return pd.DataFrame()
    change_column = find_column(
        ("板块-涨跌幅", "行业-涨跌幅", "涨跌幅"),
        excluded=("领涨股",),
    )
    member_column = find_column(("成分股数量", "公司家数", "成分股数"))

    result = pd.DataFrame({
        "board_name": df[name_column].map(_safe_text),
        "main_net_inflow": df[amount_column].map(_safe_float),
    })
    result = result[
        result["board_name"].ne("") & result["main_net_inflow"].notna()
    ].copy()
    if result.empty:
        return result
    result["board_type"] = board_type
    result["fund_flow_score"] = (
        result["main_net_inflow"].rank(method="average", pct=True).mul(100.0).round(4)
    )
    result["weighted_change_pct"] = (
        df.loc[result.index, change_column].map(_safe_float).to_numpy()
        if change_column is not None
        else pd.NA
    )
    result["member_count"] = (
        df.loc[result.index, member_column].map(_safe_float).to_numpy()
        if member_column is not None
        else pd.NA
    )
    result["board_heat_score"] = (
        pd.to_numeric(result["weighted_change_pct"], errors="coerce")
        .rank(method="average", pct=True)
        .mul(100.0)
        .round(4)
    )
    result["data_source"] = "akshare_ths_fund_flow"
    return result[
        [
            "board_type",
            "board_name",
            "board_heat_score",
            "fund_flow_score",
            "main_net_inflow",
            "weighted_change_pct",
            "member_count",
            "data_source",
        ]
    ].drop_duplicates(["board_type", "board_name"], keep="last")


def attach_qmt_industry_overlays(
    universe: pd.DataFrame,
    membership: pd.DataFrame,
    board_overlays: pd.DataFrame,
    *,
    level_prefix: str = "2",
    include_fund_flow: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Bridge THS industry metrics to the stable Baostock industry universe.

    QMT ``THY`` membership uses numbered hierarchy labels.  AkShare's free
    industry flow table uses the level-2 labels without that prefix.  Matching
    those exact names at stock level avoids guessing between the incompatible
    Baostock and THS industry taxonomies.
    """
    result = universe.copy()
    diagnostics = {
        "membership_rows": 0,
        "matched_stock_count": 0,
        "external_heat_stock_count": 0,
        "fund_flow_stock_count": 0,
        "matched_board_count": 0,
    }
    if (
        result.empty
        or membership.empty
        or board_overlays.empty
        or "code" not in result.columns
        or not {"code", "concepts"}.issubset(membership.columns)
        or not {"board_type", "board_name"}.issubset(board_overlays.columns)
    ):
        return result, diagnostics

    overlays = board_overlays[board_overlays["board_type"].eq("industry")].copy()
    if overlays.empty:
        return result, diagnostics
    overlays["board_name"] = overlays["board_name"].map(_safe_text)
    overlays = overlays[overlays["board_name"].ne("")].drop_duplicates(
        "board_name", keep="last"
    )
    if overlays.empty:
        return result, diagnostics

    metrics = overlays.set_index("board_name")
    available_names = set(metrics.index)

    def median_metric(frame: pd.DataFrame, column: str) -> float:
        if column not in frame.columns:
            return float("nan")
        return float(pd.to_numeric(frame[column], errors="coerce").median())

    bridge_rows: list[dict[str, object]] = []
    matched_boards: set[str] = set()
    diagnostics["membership_rows"] = int(len(membership))
    for row in membership[["code", "concepts"]].itertuples(index=False):
        code = _normalize_code(row.code)
        labels = {
            item.strip()[len(level_prefix):].strip()
            for item in str(row.concepts or "").split(",")
            if item.strip().startswith(level_prefix)
            and item.strip()[len(level_prefix):].strip() in available_names
        }
        if not code or not labels:
            continue
        matched_boards.update(labels)
        selected = metrics.loc[sorted(labels)]
        if isinstance(selected, pd.Series):
            selected = selected.to_frame().T
        bridge_rows.append({
            "code": code,
            "board_heat_score": median_metric(selected, "board_heat_score"),
            "board_fund_flow_score": median_metric(selected, "fund_flow_score"),
            "board_fund_flow_net_amount": median_metric(selected, "main_net_inflow"),
        })
    if not bridge_rows:
        return result, diagnostics

    bridge = pd.DataFrame(bridge_rows).drop_duplicates("code", keep="last").set_index("code")
    keys = result["code"].map(_normalize_code)
    heat_values = keys.map(bridge["board_heat_score"])
    if "board_heat_score" not in result.columns:
        result["board_heat_score"] = pd.NA
    result["board_heat_score"] = heat_values.where(
        heat_values.notna(), result["board_heat_score"]
    )

    if include_fund_flow:
        for field in ("board_fund_flow_score", "board_fund_flow_net_amount"):
            values = keys.map(bridge[field])
            if field not in result.columns:
                result[field] = pd.NA
            result[field] = values.where(values.notna(), result[field])

    diagnostics.update({
        "matched_stock_count": int(keys.isin(bridge.index).sum()),
        "external_heat_stock_count": int(heat_values.notna().sum()),
        "fund_flow_stock_count": int(
            keys.map(bridge["board_fund_flow_score"]).notna().sum()
        ) if include_fund_flow else 0,
        "matched_board_count": len(matched_boards),
    })
    return result, diagnostics


def _parse_board_fund_flow(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    work = _normalize_board_fund_flow_snapshot(df, board_type="industry")
    if work.empty:
        return {}
    return {
        name: {
            "net_amount": round(float(amount), 4),
            "score": round(float(score), 4),
        }
        for name, amount, score in work[
            ["board_name", "main_net_inflow", "fund_flow_score"]
        ].itertuples(
            index=False,
            name=None,
        )
    }


def _mapping_has_board_field(
    mapping: dict[str, dict[str, object]],
    field: str,
) -> bool:
    return any(_safe_text(item.get(field)) for item in mapping.values())


def _merge_mapping(target: dict[str, dict[str, object]], source: dict[str, dict[str, object]]) -> None:
    for code, item in source.items():
        existing = target.setdefault(code, {"industry": "", "concepts": ""})
        if item.get("industry") and not existing.get("industry"):
            existing["industry"] = item["industry"]
        if item.get("concepts"):
            existing["concepts"] = _merge_label_text(existing.get("concepts", ""), item["concepts"])
        if item.get("board_heat_summary"):
            existing["board_heat_summary"] = _merge_summary_text(
                _safe_text(existing.get("board_heat_summary")),
                item.get("board_heat_summary", ""),
            )
        if item.get("board_heat_state") and not existing.get("board_heat_state"):
            existing["board_heat_state"] = item["board_heat_state"]
        for field in _NUMERIC_FIELDS:
            value = _safe_float(item.get(field))
            if value is None:
                continue
            current = _safe_float(existing.get(field))
            if current is None or _should_replace_numeric(field, value, current):
                existing[field] = int(value) if field in {"industry_rank", "board_heat_observations"} else value


def _load_companion_board_heat_trends(path_like: str | Path) -> tuple[dict[str, dict[str, object]], str]:
    path = Path(path_like)
    history_path = path.with_suffix(path.suffix + ".history.jsonl")
    if not history_path.is_file():
        return {}, ""
    try:
        trends = load_board_heat_trends(history_path)
    except Exception as exc:
        return {}, f"board heat trends skipped: {history_path} error={exc}"
    return trends, f"board heat trends loaded: {history_path} boards={len(trends)}"


def load_board_heat_trends(
    path_like: str | Path,
    *,
    window_size: int = 5,
    hot_score: float = 60.0,
    cooling_threshold: float = 5.0,
) -> dict[str, dict[str, object]]:
    """Load board heat trend stats from an industry-cache history JSONL file."""
    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(f"Board heat history file not found: {path}")
    grouped: dict[str, list[dict[str, object]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        board = _safe_text(item.get("board"))
        heat = _safe_float(item.get("max_board_heat_score"))
        if not board or heat is None:
            continue
        if heat < 0 or heat > 100:
            continue
        grouped.setdefault(board, []).append({
            "generated_at": _safe_text(item.get("generated_at")),
            "heat": heat,
        })

    trends: dict[str, dict[str, object]] = {}
    for board, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: str(item.get("generated_at", "")))
        recent = ordered[-max(int(window_size), 1):]
        heat_values = [
            heat
            for heat in (_safe_float(item.get("heat")) for item in recent)
            if heat is not None
        ]
        if not heat_values:
            continue
        first = heat_values[0]
        last = heat_values[-1]
        if first is None or last is None:
            continue
        previous = heat_values[-2] if len(heat_values) >= 2 else last
        trend_score = last - first
        cooling_score = max(previous - last, 0.0)
        persistence_score = sum(1 for heat in heat_values if heat >= hot_score) / len(heat_values) * 100
        trends[board] = {
            "board_heat_latest_score": round(last, 4),
            "board_heat_trend_score": round(trend_score, 4),
            "board_heat_persistence_score": round(persistence_score, 4),
            "board_heat_cooling_score": round(cooling_score, 4),
            "board_heat_observations": len(heat_values),
            "board_heat_state": _board_heat_state(
                trend_score=trend_score,
                cooling_score=cooling_score,
                persistence_score=persistence_score,
                hot_score=hot_score,
                cooling_threshold=cooling_threshold,
            ),
        }
    return trends


def _apply_board_heat_trends(
    mapping: dict[str, dict[str, object]],
    trends: dict[str, dict[str, object]],
) -> None:
    for item in mapping.values():
        boards = _summary_boards(item.get("board_heat_summary", ""))
        matches = [trends[board] for board in boards if board in trends]
        if not matches:
            continue
        best = max(
            matches,
            key=lambda trend: (
                int(trend.get("board_heat_observations", 0) or 0),
                _safe_float(trend.get("board_heat_latest_score")) or 0.0,
                abs(_safe_float(trend.get("board_heat_trend_score")) or 0.0),
            ),
        )
        for field in (
            "board_heat_latest_score",
            "board_heat_trend_score",
            "board_heat_persistence_score",
            "board_heat_cooling_score",
            "board_heat_observations",
            "board_heat_state",
        ):
            if field in best:
                item[field] = best.get(field)


def _summary_boards(value: object) -> list[str]:
    boards = []
    for summary in _merge_summary_text("", value).split("|"):
        board = summary.strip().split(":", 1)[0].strip()
        if board:
            boards.append(board)
    return boards


def _merge_label_text(left: str, right: str) -> str:
    labels: list[str] = []
    seen = set()
    for raw in (left, right):
        for item in str(raw or "").replace("，", ",").replace("、", ",").split(","):
            label = item.strip()
            if label and label.lower() not in {"nan", "none", "<na>"} and label not in seen:
                seen.add(label)
                labels.append(label)
    return ",".join(labels)


def _merge_summary_text(left: object, right: object, *, limit: int = 8) -> str:
    labels: list[str] = []
    seen = set()
    for raw in (left, right):
        for item in str(raw or "").replace("\n", " | ").split("|"):
            label = item.strip()
            if label and label.lower() not in {"nan", "none", "<na>"} and label not in seen:
                seen.add(label)
                labels.append(label)
    return " | ".join(labels[:limit])


def _first_row_value(row: dict | pd.Series, columns: list[str]) -> object:
    for column in columns:
        if column in row:
            return row.get(column)
    return None


def _max_numeric(left: object, right: object) -> float | None:
    left_num = _safe_float(left)
    right_num = _safe_float(right)
    if left_num is None:
        return right_num
    if right_num is None:
        return left_num
    return max(left_num, right_num)


def _should_replace_numeric(field: str, new_value: float, current_value: float) -> bool:
    if field == "industry_rank":
        return new_value < current_value
    if field == "board_heat_observations":
        return new_value > current_value
    if field in {"board_heat_latest_score", "board_heat_persistence_score", "board_heat_cooling_score"}:
        return new_value > current_value
    if field == "board_heat_trend_score":
        return abs(new_value) > abs(current_value)
    if field.endswith("heat_score"):
        return new_value > current_value
    return False


def _board_heat_state(
    *,
    trend_score: float,
    cooling_score: float,
    persistence_score: float,
    hot_score: float,
    cooling_threshold: float,
) -> str:
    if cooling_score >= cooling_threshold:
        return "cooling"
    if trend_score >= cooling_threshold:
        return "warming"
    if persistence_score >= 66.6667 and hot_score > 0:
        return "persistent_hot"
    if trend_score <= -cooling_threshold:
        return "weakening"
    return "flat"


def _board_heat_score(*, change_pct: float | None, rank: float | None) -> float:
    score = 50.0
    if change_pct is not None:
        score += change_pct * 6.0
    if rank is not None and rank > 0:
        score += max(0.0, 12.0 - min(rank, 12.0))
    return round(max(0.0, min(score, 100.0)), 4)


def _board_heat_summary(board: str, *, change_pct: float | None, rank: float | None) -> str:
    parts = [board]
    if change_pct is not None:
        parts.append(f"{change_pct:+.2f}%")
    if rank is not None:
        parts.append(f"rank={int(rank)}")
    return ":".join(parts)
