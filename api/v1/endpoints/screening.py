# -*- coding: utf-8 -*-
"""Stock screening routes."""

from __future__ import annotations

import ipaddress
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.deps import get_config_dep, get_database_manager
from api.v1.errors import api_error
from src.config import Config
from src.notification_sender.feishu_sender import FeishuSender
from src.services.screening_service import ScreeningService
from src.services.task_queue import TaskStatus as QueueTaskStatus
from src.services.task_queue import get_task_queue
from src.storage import DatabaseManager

router = APIRouter()
logger = logging.getLogger(__name__)

_CN_MARKET_TIMEZONE = timezone(timedelta(hours=8))


class ScreeningScreenRequest(BaseModel):
    market: str = Field("cn", min_length=1, max_length=16)
    strategy: str = Field("martin_momentum_pullback", min_length=1, max_length=64)
    max_results: int = Field(20, ge=1, le=100)
    variant_seed: str = Field("", max_length=128)
    notify: bool = False
    account_id: Optional[int] = Field(None, gt=0)
    web_base_url: str = Field("", max_length=2048)


class ScreeningStrategyResponse(BaseModel):
    id: str
    name: str = ""
    title: str = ""
    description: str = ""
    category: str = ""
    tag: str = ""
    tags: List[str] = Field(default_factory=list)
    market_scope: List[str] = Field(default_factory=list)
    market: str = ""
    analysis_skills: List[str] = Field(default_factory=list)


class ScreeningScreenAccepted(BaseModel):
    task_id: str
    trace_id: str
    status: str = "pending"
    message: str
    strategy: str
    market: str
    max_results: int
    account_id: Optional[int] = None


class ScreeningScreenTaskStatus(BaseModel):
    task_id: str
    trace_id: Optional[str] = None
    status: str
    progress: int = 0
    message: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


def _service(config: Config, db_manager: Any = None) -> ScreeningService:
    usable_db = db_manager if callable(getattr(db_manager, "save_screening_run", None)) else None
    return ScreeningService(config=config, db_manager=usable_db)


def _screening_task_not_found(task_id: str) -> HTTPException:
    return api_error(
        404,
        "screening_screen_task_not_found",
        f"选股任务 {task_id} 不存在或已过期",
    )


def _normalize_web_base_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        return ""
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _same_web_origin(left: str, right: str) -> bool:
    left_url = urlsplit(_normalize_web_base_url(left))
    right_url = urlsplit(_normalize_web_base_url(right))
    try:
        left_port = left_url.port
        right_port = right_url.port
    except ValueError:
        return False
    return bool(
        left_url.scheme
        and right_url.scheme
        and left_url.scheme == right_url.scheme
        and left_url.hostname == right_url.hostname
        and left_port == right_port
    )


def _trusted_request_web_base_url(http_request: Request, submitted_url: str) -> str:
    """Use a client origin only when it matches the request that submitted it."""
    request_url = _normalize_web_base_url(getattr(http_request, "base_url", ""))
    candidate = _normalize_web_base_url(submitted_url)
    if candidate and request_url and not _same_web_origin(candidate, request_url):
        logger.warning(
            "忽略与当前请求不同源的选股详情地址: submitted=%s request=%s",
            urlsplit(candidate).netloc,
            urlsplit(request_url).netloc,
        )
        candidate = ""
    return candidate or request_url


def _screening_detail_url(web_base_url: str, run_id: Any) -> str:
    base_url = _normalize_web_base_url(web_base_url)
    normalized_run_id = str(run_id or "").strip()
    if not base_url or not normalized_run_id:
        return ""
    parsed = urlsplit(base_url)
    hostname = (parsed.hostname or "").strip().lower()
    if parsed.scheme != "https" or hostname == "localhost" or hostname.endswith(".local"):
        return ""
    try:
        if not ipaddress.ip_address(hostname).is_global:
            return ""
    except ValueError:
        pass
    return f"{base_url}/screening?runId={quote(normalized_run_id, safe='')}"


def _screening_notification_content(result: Dict[str, Any]) -> str:
    candidates = result.get("candidates") if isinstance(result.get("candidates"), list) else []
    candidate_names = [
        str(item.get("name") or item.get("code") or "").strip()
        for item in candidates[:5]
        if isinstance(item, dict)
    ]
    lines = [
        "## 选股任务已完成",
        f"策略：{result.get('strategy') or 'martin_momentum_pullback'}",
        f"市场：{'A股' if str(result.get('market') or 'cn').lower() == 'cn' else result.get('market')}",
        f"数据日期：{result.get('analysis_date') or '--'}",
        f"候选：{result.get('candidate_count', len(candidates))} 只",
    ]
    if candidate_names:
        lines.append(f"前列候选：{'、'.join(candidate_names)}")
    else:
        lines.append("本次没有股票通过全部硬条件，请保持观察。")
    market_regime = result.get("market_regime")
    if isinstance(market_regime, dict):
        lines.append(
            "市场状态："
            f"{market_regime.get('state') or 'unavailable'}"
            f"（评分 {market_regime.get('score') if market_regime.get('score') is not None else '--'}）"
        )
    return "\n\n".join(lines)


def _screening_theme_cache_freshness(
    payload: Dict[str, Any],
    *,
    analysis_date: str,
) -> Dict[str, Any]:
    """Classify fallback topic data against the completed market session."""
    cached_at = str(payload.get("cached_at") or "").strip()
    cache_used = bool(payload.get("cache_used") or payload.get("fallback_used"))
    stale = bool(payload.get("stale"))
    stale_age_hours = payload.get("stale_age_hours")
    data_as_of = ""
    reason = ""

    cached_dt: Optional[datetime] = None
    if cached_at:
        try:
            cached_dt = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
        except ValueError:
            cached_dt = None
        if cached_dt is not None:
            if cached_dt.tzinfo is None:
                cached_dt = cached_dt.replace(tzinfo=timezone.utc)
            cached_dt = cached_dt.astimezone(timezone.utc)
            data_as_of = cached_dt.astimezone(_CN_MARKET_TIMEZONE).strftime("%Y-%m-%d %H:%M")
            if stale_age_hours is None:
                stale_age_hours = round(
                    max(0.0, (datetime.now(timezone.utc) - cached_dt).total_seconds()) / 3600,
                    2,
                )

    if cache_used:
        if cached_dt is None:
            stale = True
            reason = "回退缓存缺少可验证的数据时间"
        elif analysis_date:
            try:
                session_date = datetime.strptime(analysis_date, "%Y-%m-%d").date()
            except ValueError:
                session_date = None
            if session_date is not None:
                session_close = datetime.combine(
                    session_date,
                    datetime.min.time().replace(hour=15),
                    tzinfo=_CN_MARKET_TIMEZONE,
                ).astimezone(timezone.utc)
                if cached_dt < session_close:
                    stale = True
                    reason = "回退缓存早于分析日收盘"

    if stale and not reason:
        reason = "上游已将该题材快照标记为过期"
    return {
        "stale": stale,
        "stale_age_hours": stale_age_hours,
        "data_as_of": data_as_of,
        "freshness_reason": reason,
    }


def _screening_theme_overview(
    config: Config,
    *,
    top: int = 5,
    analysis_date: str = "",
) -> Dict[str, Any]:
    """Fetch a concise theme snapshot without making notification delivery fragile."""
    try:
        payload = ScreeningService(config=config).hotspots(
            top=max(1, min(int(top), 10)),
            refresh=True,
            include_details=False,
            lightweight=True,
        )
    except Exception as exc:  # Topic heat is optional enrichment.
        logger.warning("选股飞书题材热度刷新失败: %s", exc)
        return {
            "status": "unavailable",
            "provider": "",
            "stale": False,
            "themes": [],
            "source_errors": [str(exc)],
        }

    raw_rows = payload.get("hotspots") if isinstance(payload, dict) else []
    rows = raw_rows if isinstance(raw_rows, list) else []
    themes: List[Dict[str, Any]] = []
    for row in rows[:top]:
        if not isinstance(row, dict):
            continue
        raw_leaders = row.get("leaders")
        leaders: List[str] = []
        if isinstance(raw_leaders, list):
            for item in raw_leaders[:3]:
                if isinstance(item, dict):
                    label = str(item.get("name") or item.get("code") or "").strip()
                else:
                    label = str(item or "").strip()
                if label:
                    leaders.append(label)
        themes.append({
            "name": str(row.get("name") or row.get("topic") or "").strip(),
            "topic": str(row.get("topic") or "").strip(),
            "rank": row.get("rank"),
            "heat_score": row.get("heat_score"),
            "change_pct": row.get("change_pct"),
            "state": str(row.get("state") or "").strip(),
            "stage": str(row.get("stage") or "").strip(),
            "leaders": leaders,
            "quality_status": str(row.get("quality_status") or "").strip(),
        })
    freshness = _screening_theme_cache_freshness(
        payload,
        analysis_date=analysis_date,
    )
    stale = bool(freshness["stale"])
    return {
        "status": "stale" if themes and stale else ("available" if themes else "unavailable"),
        "provider": str(payload.get("provider_used") or payload.get("provider") or "").strip(),
        "cached_at": payload.get("cached_at"),
        "stale": stale,
        "stale_age_hours": freshness["stale_age_hours"],
        "data_as_of": freshness["data_as_of"],
        "freshness_reason": freshness["freshness_reason"],
        "cache_used": bool(payload.get("cache_used")),
        "fallback_used": bool(payload.get("fallback_used")),
        "themes": themes,
        "source_errors": [
            str(item)
            for item in (payload.get("source_errors") or [])[:3]
            if str(item).strip()
        ],
    }


def _notify_screening_result(
    *,
    config: Config,
    result: Dict[str, Any],
    web_base_url: str,
) -> Dict[str, Any]:
    detail_url = _screening_detail_url(web_base_url, result.get("run_id"))
    theme_overview = (
        _screening_theme_overview(
            config,
            analysis_date=str(result.get("analysis_date") or "").strip(),
        )
        if str(result.get("market") or "cn").strip().lower() == "cn"
        else {"status": "unavailable", "themes": []}
    )
    warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
    payload = {
        "kind": "screening_result",
        "strategy": result.get("strategy"),
        "strategy_version": result.get("strategy_version"),
        "market": result.get("market"),
        "run_id": result.get("run_id"),
        "analysis_date": result.get("analysis_date"),
        "candidate_count": result.get("candidate_count"),
        "snapshot_count": result.get("snapshot_count"),
        "snapshot_source": result.get("snapshot_source"),
        "snapshot_filtered_count": result.get("snapshot_filtered_count"),
        "seed_pool_count": result.get("seed_pool_count"),
        "daily_enrich_count": result.get("daily_enrich_count"),
        "after_filter_count": result.get("after_filter_count"),
        "ranking_mode": result.get("ranking_mode"),
        "candidates": result.get("candidates") or [],
        "market_regime": result.get("market_regime") or {},
        "sector_overview": result.get("sector_overview") or {},
        "board_heat_overview": result.get("board_heat_overview") or {},
        "portfolio_plan": result.get("portfolio_plan") or {},
        "theme_overview": theme_overview,
        "data_diagnostics": {
            "analysis_date": result.get("analysis_date"),
            "snapshot_source": result.get("snapshot_source"),
            "snapshot_count": result.get("snapshot_count"),
            "snapshot_filtered_count": result.get("snapshot_filtered_count"),
            "seed_pool_count": result.get("seed_pool_count"),
            "daily_enrich_count": result.get("daily_enrich_count"),
            "warning_count": len(warnings),
            "warnings": [str(item) for item in warnings[:3]],
        },
        "detail_url": detail_url,
    }
    try:
        sent = FeishuSender(config).send_to_feishu(
            _screening_notification_content(result),
            structured_payload=payload,
        )
    except Exception as exc:  # Notification must never fail completed screening work.
        logger.warning("选股完成后的飞书推送异常: %s", exc)
        sent = False
    return {
        "requested": True,
        "channel": "feishu",
        "sent": bool(sent),
        "detail_url": detail_url or None,
    }


@router.get("/status")
def screening_status(config: Config = Depends(get_config_dep)) -> Dict[str, Any]:
    return _service(config).status()


@router.get("/strategies")
def screening_strategies(
    request: Request,
    config: Config = Depends(get_config_dep),
) -> Dict[str, Any]:
    return _service(config).strategies()


@router.get("/hotspots")
def screening_hotspots(
    provider: str = Query("", max_length=32),
    top: int = Query(12, ge=1, le=50),
    refresh: bool = Query(False),
    include_details: bool = Query(False),
    config: Config = Depends(get_config_dep),
) -> Dict[str, Any]:
    refresh_value = refresh if isinstance(refresh, bool) else bool(getattr(refresh, "default", False))
    include_details_value = (
        include_details
        if isinstance(include_details, bool)
        else bool(getattr(include_details, "default", False))
    )
    return _service(config).hotspots(
        provider=provider,
        top=top,
        refresh=refresh_value,
        include_details=include_details_value,
    )


@router.get("/hotspots/{topic:path}")
def screening_hotspot_detail(
    topic: str,
    provider: str = Query("", max_length=32),
    refresh: bool = Query(False),
    include_search: bool = Query(False),
    config: Config = Depends(get_config_dep),
) -> Dict[str, Any]:
    refresh_value = refresh if isinstance(refresh, bool) else bool(getattr(refresh, "default", False))
    include_search_value = (
        include_search
        if isinstance(include_search, bool)
        else bool(getattr(include_search, "default", False))
    )
    return _service(config).hotspot_detail(
        topic=topic,
        provider=provider,
        refresh=refresh_value,
        include_search=include_search_value,
    )


@router.post("/screen/tasks", status_code=202, response_model=ScreeningScreenAccepted)
def screening_start_screen_task(
    request: ScreeningScreenRequest,
    http_request: Request,
    config: Config = Depends(get_config_dep),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> ScreeningScreenAccepted:
    task_id = uuid.uuid4().hex
    task_queue = get_task_queue()
    request_base_url = _trusted_request_web_base_url(http_request, request.web_base_url)

    def run_screen() -> Dict[str, Any]:
        task_queue.update_task_progress(
            task_id,
            20,
            "正在执行选股，外部数据源较慢时会持续后台运行",
        )

        def report_progress(progress: int, message: str) -> None:
            task_queue.update_task_progress(task_id, progress, message)

        result = _service(config, db_manager).screen(
            strategy=request.strategy,
            market=request.market,
            max_results=request.max_results,
            account_id=request.account_id,
            selection_seed=request.variant_seed,
            progress_callback=report_progress,
        )
        task_queue.update_task_progress(
            task_id,
            98,
            f"选股已完成，正在整理 {result.get('candidate_count', 0)} 条候选",
        )
        if request.notify:
            task_queue.update_task_progress(task_id, 99, "正在推送飞书结果卡片")
            result["notification"] = _notify_screening_result(
                config=config,
                result=result,
                web_base_url=request_base_url,
            )
        return result

    task = task_queue.submit_background_task(
        run_screen,
        stock_code="screening_screen",
        stock_name=f"{request.strategy} / {request.market}",
        report_type="screening_screen",
        message="选股任务已提交",
        task_id=task_id,
        trace_id=task_id,
    )
    return ScreeningScreenAccepted(
        task_id=task.task_id,
        trace_id=task.trace_id or task.task_id,
        status=task.status.value if isinstance(task.status, QueueTaskStatus) else str(task.status),
        message=task.message or "选股任务已提交",
        strategy=request.strategy,
        market=request.market,
        max_results=request.max_results,
        account_id=request.account_id,
    )


@router.get("/screen/tasks/{task_id}", response_model=ScreeningScreenTaskStatus)
def screening_screen_task_status(task_id: str) -> ScreeningScreenTaskStatus:
    task = get_task_queue().get_task(task_id)
    if task is None or task.report_type != "screening_screen":
        raise _screening_task_not_found(task_id)

    result = task.result if task.status == QueueTaskStatus.COMPLETED and isinstance(task.result, dict) else None
    return ScreeningScreenTaskStatus(
        task_id=task.task_id,
        trace_id=task.trace_id or task.task_id,
        status=task.status.value if isinstance(task.status, QueueTaskStatus) else str(task.status),
        progress=task.progress,
        message=task.message,
        error=task.error,
        result=result,
    )


@router.post("/screen")
def screening_screen(
    request: ScreeningScreenRequest,
    http_request: Request,
    config: Config = Depends(get_config_dep),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> Dict[str, Any]:
    result = _service(config, db_manager).screen(
        strategy=request.strategy,
        market=request.market,
        max_results=request.max_results,
        account_id=request.account_id,
        selection_seed=request.variant_seed,
    )
    if request.notify:
        result["notification"] = _notify_screening_result(
            config=config,
            result=result,
            web_base_url=_trusted_request_web_base_url(http_request, request.web_base_url),
        )
    return result


@router.get("/history")
def screening_history(
    limit: int = Query(20, ge=1, le=100),
    strategy: str = Query("", max_length=64),
    market: str = Query("", max_length=16),
    config: Config = Depends(get_config_dep),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> Dict[str, Any]:
    return _service(config, db_manager).history(
        limit=limit,
        strategy=strategy,
        market=market,
    )


@router.get("/history/{run_id}")
def screening_history_detail(
    run_id: str,
    config: Config = Depends(get_config_dep),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> Dict[str, Any]:
    return _service(config, db_manager).history_detail(run_id)


@router.get("/source-history")
def screening_source_history(
    limit: int = Query(100, ge=1, le=100),
    config: Config = Depends(get_config_dep),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> Dict[str, Any]:
    return _service(config, db_manager).source_history(limit=limit)
