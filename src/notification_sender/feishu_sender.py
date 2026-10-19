# -*- coding: utf-8 -*-
"""
飞书 发送提醒服务

职责：
1. 通过 webhook 发送飞书消息
2. 通过飞书应用机器人（App Bot）发送消息（lark-oapi SDK）
"""
import base64
import hashlib
import hmac
import importlib.util
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid as uuid_mod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.config import Config
from src.formatters import (
    MIN_MAX_BYTES,
    PAGE_MARKER_SAFE_BYTES,
    chunk_content_by_max_bytes,
    format_feishu_markdown,
    strip_hidden_markdown_metadata,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# lark-oapi SDK availability
# ---------------------------------------------------------------------------

FEISHU_SDK_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None
_lark: Any = None  # type: ignore[assignment]
FEISHU_DOMAIN = "feishu"
LARK_DOMAIN = "lark"
_SDK_IMPORT_LOCK = threading.Lock()
_SDK_IMPORT_ATTEMPTED = False


class _LazyCreateMessageRequest:
    """Keep request construction patchable without importing the SDK at startup."""

    @staticmethod
    def builder() -> Any:
        if not _load_feishu_sdk():
            raise RuntimeError("lark-oapi SDK is unavailable")
        return CreateMessageRequest.builder()


class _LazyCreateMessageRequestBody:
    @staticmethod
    def builder() -> Any:
        if not _load_feishu_sdk():
            raise RuntimeError("lark-oapi SDK is unavailable")
        return CreateMessageRequestBody.builder()


CreateMessageRequest: Any = _LazyCreateMessageRequest
CreateMessageRequestBody: Any = _LazyCreateMessageRequestBody
FEISHU_FILE_SDK_AVAILABLE = FEISHU_SDK_AVAILABLE
_CreateFileRequest: Any = None
_CreateFileRequestBody: Any = None


def _load_feishu_sdk() -> bool:
    """Import the large lark-oapi package only when App Bot mode is used."""
    global _SDK_IMPORT_ATTEMPTED
    global _lark
    global CreateMessageRequest
    global CreateMessageRequestBody
    global FEISHU_DOMAIN
    global LARK_DOMAIN
    global FEISHU_SDK_AVAILABLE
    global FEISHU_FILE_SDK_AVAILABLE
    global _CreateFileRequest
    global _CreateFileRequestBody

    if _lark is not None:
        return True
    if _SDK_IMPORT_ATTEMPTED or not FEISHU_SDK_AVAILABLE:
        return False

    with _SDK_IMPORT_LOCK:
        if _lark is not None:
            return True
        if _SDK_IMPORT_ATTEMPTED:
            return False
        _SDK_IMPORT_ATTEMPTED = True
        try:
            import lark_oapi as sdk
            from lark_oapi.api.im.v1 import (
                CreateMessageRequest as sdk_create_message_request,
                CreateMessageRequestBody as sdk_create_message_request_body,
            )
            from lark_oapi.core.const import FEISHU_DOMAIN as sdk_feishu_domain
            from lark_oapi.core.const import LARK_DOMAIN as sdk_lark_domain
        except (ImportError, AttributeError) as exc:
            logger.warning("加载飞书 App Bot SDK 失败: %s", exc)
            FEISHU_SDK_AVAILABLE = False
            FEISHU_FILE_SDK_AVAILABLE = False
            return False

        _lark = sdk
        CreateMessageRequest = sdk_create_message_request
        CreateMessageRequestBody = sdk_create_message_request_body
        FEISHU_DOMAIN = sdk_feishu_domain
        LARK_DOMAIN = sdk_lark_domain
        try:
            from lark_oapi.api.im.v1 import (
                CreateFileRequest as sdk_create_file_request,
                CreateFileRequestBody as sdk_create_file_request_body,
            )
        except (ImportError, AttributeError):
            FEISHU_FILE_SDK_AVAILABLE = False
        else:
            _CreateFileRequest = sdk_create_file_request
            _CreateFileRequestBody = sdk_create_file_request_body
            FEISHU_FILE_SDK_AVAILABLE = True
        return True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APP_SEND_RETRIES = 3
_APP_SEND_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
_WEBHOOK_SEND_TIMEOUT_SECONDS = 30

# Sentinel for "client not yet initialised".
_NO_CLIENT = object()


class FeishuSender:

    def __init__(self, config: Config):
        """
        Initialise Feishu sender.

        Two mutually exclusive routing modes are supported:
          1. **Webhook** – configured via ``feishu_webhook_url`` (legacy).
          2. **App Bot** – configured via ``feishu_app_id`` + ``feishu_app_secret``
             + ``feishu_chat_id``, sends through the ``lark-oapi`` SDK.

        Webhook mode takes precedence when both are configured.
        """
        # -- Webhook mode --
        self._feishu_url = getattr(config, "feishu_webhook_url", None)
        self._feishu_secret = (getattr(config, "feishu_webhook_secret", None) or "").strip()
        self._feishu_keyword = (getattr(config, "feishu_webhook_keyword", None) or "").strip()
        self._feishu_max_bytes = getattr(config, "feishu_max_bytes", 20000)
        self._feishu_send_as_file = getattr(config, "feishu_send_as_file", False)
        self._webhook_verify_ssl = getattr(config, "webhook_verify_ssl", True)

        # -- App Bot mode --
        self._feishu_app_id = (getattr(config, "feishu_app_id", None) or "").strip()
        self._feishu_app_secret = (getattr(config, "feishu_app_secret", None) or "").strip()
        self._feishu_chat_id = (getattr(config, "feishu_chat_id", None) or "").strip()
        self._feishu_receive_id_type = (
            getattr(config, "feishu_receive_id_type", None) or "chat_id"
        ).strip().lower()
        if self._feishu_receive_id_type not in ("chat_id", "open_id"):
            logger.warning(
                "无效的 FEISHU_RECEIVE_ID_TYPE=%s，回退为 chat_id",
                self._feishu_receive_id_type,
            )
            self._feishu_receive_id_type = "chat_id"
        # domain_name must be "feishu" or "lark"; anything else defaulted to feishu.
        raw_domain = (
            getattr(config, "feishu_domain", None) or os.getenv("FEISHU_DOMAIN", "feishu")
        ).strip().lower()
        if raw_domain not in ("feishu", "lark"):
            logger.warning(
                "无效的 FEISHU_DOMAIN=%s，回退为 feishu", raw_domain
            )
            raw_domain = "feishu"
        self._feishu_domain = FEISHU_DOMAIN if raw_domain == "feishu" else LARK_DOMAIN

        self._app_client: Any = _NO_CLIENT
        self._app_client_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_card_body(content: str) -> dict:
        """Build a Feishu interactive-card body (without the ``msg_type`` wrapper)."""
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "股票智能分析报告"},
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": content},
                }
            ],
        }

    @staticmethod
    def _format_change(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "--"
        return f"{number:+.2f}%"

    @staticmethod
    def _format_decimal(value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "--"

    @staticmethod
    def _format_quantity(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "--"
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.2f}"

    @staticmethod
    def _strip_markdown_tables(markdown: str) -> str:
        """Keep narrative in markdown while structured tables render natively."""
        lines = (markdown or "").splitlines()
        kept: List[str] = []
        index = 0
        while index < len(lines):
            current = lines[index].strip()
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            next_next_line = lines[index + 2].strip() if index + 2 < len(lines) else ""
            is_table_start = current.startswith("|") and current.endswith("|") and re.match(r"^\|?\s*:?-{3,}", next_line)
            heading_before_table = current.startswith("####") and next_line.startswith("|") and next_next_line and re.match(r"^\|?\s*:?-{3,}", next_next_line)
            if heading_before_table:
                index += 1
                continue
            if is_table_start:
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    index += 1
                continue
            kept.append(lines[index])
            index += 1
        return "\n".join(kept).strip()

    @staticmethod
    def _build_table(
        element_id: str,
        columns: List[Dict[str, Any]],
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "tag": "table",
            "element_id": element_id,
            "margin": "4px 0",
            "page_size": min(max(len(rows), 1), 10),
            "row_height": "auto",
            "freeze_first_column": True,
            "header_style": {
                "text_align": "left",
                "text_size": "normal",
                "background_style": "grey",
                "text_color": "default",
                "bold": True,
                "lines": 1,
            },
            "columns": columns,
            "rows": rows,
        }

    @classmethod
    def _build_index_table(cls, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rows = []
        for item in payload.get("indices", [])[:6]:
            if not isinstance(item, dict):
                continue
            current = item.get("current")
            amount = item.get("amount")
            rows.append({
                "name": str(item.get("name") or "--"),
                "last": f"{float(current):.2f}" if isinstance(current, (int, float)) else "--",
                "change": cls._format_change(item.get("change_pct")),
                "amount": f"{float(amount):.0f}亿" if isinstance(amount, (int, float)) and amount else "--",
            })
        if not rows:
            return None
        return cls._build_table("index_table", [
            {"name": "name", "display_name": "指数", "data_type": "text", "width": "30%"},
            {"name": "last", "display_name": "最新", "data_type": "text", "width": "20%", "horizontal_align": "right"},
            {"name": "change", "display_name": "涨跌幅", "data_type": "text", "width": "20%", "horizontal_align": "right"},
            {"name": "amount", "display_name": "成交额", "data_type": "text", "width": "20%", "horizontal_align": "right"},
        ], rows)

    @classmethod
    def _build_sector_table(cls, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rows = []
        for direction, items in (("领涨", payload.get("sectors", {}).get("top", [])), ("领跌", payload.get("sectors", {}).get("bottom", []))):
            for item in (items or [])[:3]:
                if not isinstance(item, dict):
                    continue
                rows.append({
                    "direction": direction,
                    "name": str(item.get("name") or "--"),
                    "change": cls._format_change(item.get("change_pct")),
                })
        if not rows:
            return None
        return cls._build_table("sector_table", [
            {"name": "direction", "display_name": "方向", "data_type": "text", "width": "20%"},
            {"name": "name", "display_name": "行业板块", "data_type": "text", "width": "55%"},
            {"name": "change", "display_name": "涨跌幅", "data_type": "text", "width": "20%", "horizontal_align": "right"},
        ], rows)

    @classmethod
    def _build_us_market_table(cls, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        overseas = payload.get("overseas") if isinstance(payload.get("overseas"), dict) else {}
        rows = []
        for item in overseas.get("us_indices", [])[:4]:
            if not isinstance(item, dict):
                continue
            current = item.get("current")
            rows.append({
                "name": str(item.get("name") or "--"),
                "last": f"{float(current):.2f}" if isinstance(current, (int, float)) else "--",
                "change": cls._format_change(item.get("change_pct")),
            })
        if not rows:
            return None
        return cls._build_table("us_market_table", [
            {"name": "name", "display_name": "美股指数", "data_type": "text", "width": "45%"},
            {"name": "last", "display_name": "最新", "data_type": "text", "width": "25%", "horizontal_align": "right"},
            {"name": "change", "display_name": "涨跌幅", "data_type": "text", "width": "25%", "horizontal_align": "right"},
        ], rows)

    @classmethod
    def _build_recommendation_table(
        cls,
        element_id: str,
        items: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        rows = []
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            rows.append({
                "name": f"{item.get('name') or '--'}\n{item.get('code') or '--'}",
                "theme": str(item.get("theme") or "--"),
                "change": cls._format_change(item.get("change_pct")),
                "signal": str(item.get("technical") or "数据不足"),
                "strength": str(item.get("strength") or "观察"),
            })
        if not rows:
            return None
        return cls._build_table(element_id, [
            {"name": "name", "display_name": "标的", "data_type": "text", "width": "22%", "vertical_align": "top"},
            {"name": "theme", "display_name": "方向", "data_type": "text", "width": "15%", "vertical_align": "top"},
            {"name": "change", "display_name": "涨跌", "data_type": "text", "width": "13%", "horizontal_align": "right", "vertical_align": "top"},
            {"name": "signal", "display_name": "技术状态", "data_type": "text", "width": "34%", "vertical_align": "top"},
            {"name": "strength", "display_name": "强度", "data_type": "text", "width": "14%", "vertical_align": "top"},
        ], rows)

    @classmethod
    def _build_market_review_card_body(cls, payload: Dict[str, Any], content: str) -> dict:
        """Build a full A-share recap card from the persisted market payload."""
        date_text = str(payload.get("date") or "").strip()
        generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
        model = str(generation.get("model") or "").strip()
        model_label = model.split("/", 1)[-1] if model.startswith("deepseek/") else model
        analysis_source = f"DeepSeek / {model_label}" if model.startswith("deepseek/") else "行情数据规则整理"
        if generation.get("mode") == "llm" and not model:
            analysis_source = "大模型分析"

        sections = payload.get("sections") if isinstance(payload.get("sections"), list) else []
        elements: List[Dict[str, Any]] = [{
            "tag": "markdown",
            "content": f"<font color='grey'>交易日 {date_text or '--'} | 分析来源：{analysis_source}</font>",
        }]
        index_table = cls._build_index_table(payload)
        if index_table:
            elements.extend([
                {"tag": "markdown", "content": "### 📈 指数表现"},
                index_table,
            ])
        sector_table = cls._build_sector_table(payload)
        if sector_table:
            elements.extend([
                {"tag": "markdown", "content": "### 🧭 板块强弱"},
                sector_table,
            ])
        us_market_table = cls._build_us_market_table(payload)
        if us_market_table:
            elements.extend([
                {"tag": "markdown", "content": "### 🌎 最新美股动态"},
                us_market_table,
            ])
        has_section = False
        for item in sections:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "")
            title = str(item.get("title") or "").strip()
            markdown = str(item.get("markdown") or "").strip()
            if not markdown:
                continue
            display_title = "核心结论" if key == "overview" else title
            section_content = format_feishu_markdown(cls._strip_markdown_tables(markdown))
            section_prefix = f"**{display_title}**\n" if display_title else ""
            elements.append(
                {
                    "tag": "markdown",
                    "content": section_prefix + section_content,
                }
            )
            if key in {"六_etf_个股关注", "etf_个股关注", "ETF/个股关注"} or "ETF/个股" in title:
                recommendations = payload.get("recommendations") if isinstance(payload.get("recommendations"), dict) else {}
                etf_table = cls._build_recommendation_table("etf_watch_table", recommendations.get("etf", []))
                stock_table = cls._build_recommendation_table("stock_watch_table", recommendations.get("stocks", []))
                if etf_table:
                    elements.extend([{"tag": "markdown", "content": "#### 🎯 ETF 关注"}, etf_table])
                if stock_table:
                    elements.extend([{"tag": "markdown", "content": "#### 🏷️ 个股关注"}, stock_table])
            has_section = True

        if not has_section:
            elements.append(
                {"tag": "markdown", "content": content}
            )
        elements.append({"tag": "markdown", "content": "<font color='grey'>数据仅用于研究复盘，不构成投资建议。</font>"})
        return {
            "schema": "2.0",
            "config": {"update_multi": True},
            "header": {
                "template": "blue",
                "title": {"tag": "plain_text", "content": "A股盘后智能复盘"},
            },
            "body": {"direction": "vertical", "padding": "12px 12px 12px 12px", "elements": elements},
        }

    @staticmethod
    def _compact_card_text(value: Any, limit: int = 120) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= limit:
            return text
        return text[: max(1, limit - 1)].rstrip() + "…"

    @classmethod
    def _build_screening_card_body(cls, payload: Dict[str, Any], content: str) -> dict:
        """Build a compact completion card for one persisted screening run."""
        candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
        candidate_count = payload.get("candidate_count")
        if candidate_count is None:
            candidate_count = len(candidates)
        market = str(payload.get("market") or "cn").strip().lower()
        market_label = "A股" if market == "cn" else (market.upper() or "--")
        ranking_mode = str(payload.get("ranking_mode") or "").strip().lower()
        ranking_label = "智能重排" if ranking_mode == "llm" else "确定性因子"
        run_id = cls._compact_card_text(payload.get("run_id"), 32) or "--"
        strategy_version = cls._compact_card_text(payload.get("strategy_version"), 16)
        strategy_label = cls._compact_card_text(payload.get("strategy"), 64) or "选股策略"
        if strategy_version:
            strategy_label = f"{strategy_label} v{strategy_version}"
        analysis_date = cls._compact_card_text(payload.get("analysis_date"), 16) or "--"

        elements: List[Dict[str, Any]] = [{
            "tag": "markdown",
            "content": (
                f"**{market_label} · {strategy_label}**\n"
                f"候选 **{candidate_count}** 只 ｜ {ranking_label}\n"
                f"<font color='grey'>数据日 {analysis_date} · 快照 {payload.get('snapshot_count') or '--'} · "
                f"初筛后 {payload.get('after_filter_count') if payload.get('after_filter_count') is not None else '--'} · Run ID {run_id}</font>"
            ),
        }]
        market_regime = (
            payload.get("market_regime")
            if isinstance(payload.get("market_regime"), dict)
            else {}
        )
        state = str(market_regime.get("state") or "unavailable")
        state_label = {
            "strong_risk_on": "强势进攻",
            "risk_on": "正常进攻",
            "neutral": "中性防守",
            "risk_off": "风险关闭",
            "unavailable": "状态待建立",
        }.get(state, state)
        profile = (
            market_regime.get("profile")
            if isinstance(market_regime.get("profile"), dict)
            else {}
        )
        score = market_regime.get("score")
        score_text = f"{float(score):.1f}" if isinstance(score, (int, float)) else "--"
        target_exposure = profile.get("target_exposure_pct")
        exposure_text = f"{target_exposure}%" if target_exposure is not None else "--"
        elements.append({
            "tag": "markdown",
            "content": (
                "### 📊 市场状态\n"
                f"**{state_label}** ｜ 评分 **{score_text}** ｜ "
                f"目标总仓位 **{exposure_text}**\n"
                f"<font color='grey'>{cls._compact_card_text(market_regime.get('reason'), 120) or '尚未建立完整市场状态'}</font>"
            ),
        })

        board_heat_overview = (
            payload.get("board_heat_overview")
            if isinstance(payload.get("board_heat_overview"), dict)
            else {}
        )
        board_heat_rows: List[Dict[str, Any]] = []
        raw_board_heat = board_heat_overview.get("industry")
        if isinstance(raw_board_heat, list):
            for index, item in enumerate(raw_board_heat[:5], start=1):
                if not isinstance(item, dict):
                    continue
                board_heat_rows.append({
                    "sector": f"{index}. {cls._compact_card_text(item.get('name'), 24) or '--'}",
                    "heat": cls._format_decimal(item.get("score")),
                    "change": cls._format_change(item.get("change_pct")),
                    "advance": cls._format_change(item.get("advance_ratio")),
                    "amount": cls._format_change(item.get("amount_share_pct")),
                })
        board_heat_status = str(board_heat_overview.get("status") or "unavailable")
        board_heat_status_text = {
            "available": "可用",
            "unavailable": "不可用",
            "stale": "过期缓存",
        }.get(board_heat_status, board_heat_status)
        board_heat_date = cls._compact_card_text(
            board_heat_overview.get("trading_date"), 16
        ) or analysis_date
        board_heat_observed_at = cls._compact_card_text(
            board_heat_overview.get("observed_at"), 32
        )
        elements.append({
            "tag": "markdown",
            "content": (
                "### ⚡ 当日行业热度\n"
                f"状态 **{board_heat_status_text}** ｜ 行情源 "
                f"**{cls._compact_card_text(board_heat_overview.get('quote_source'), 20) or '--'}**\n"
                f"<font color='grey'>Baostock 行业归属 · 交易日 {board_heat_date}"
                f"{(' · 观测时间 ' + board_heat_observed_at) if board_heat_observed_at else ''}</font>"
            ),
        })
        if board_heat_rows and board_heat_status == "available":
            elements.append(cls._build_table("screening_daily_sectors", [
                {"name": "sector", "display_name": "行业", "data_type": "text", "width": "32%", "vertical_align": "top"},
                {"name": "heat", "display_name": "当日热度", "data_type": "text", "width": "17%", "horizontal_align": "right"},
                {"name": "change", "display_name": "涨跌", "data_type": "text", "width": "15%", "horizontal_align": "right"},
                {"name": "advance", "display_name": "上涨占比", "data_type": "text", "width": "18%", "horizontal_align": "right"},
                {"name": "amount", "display_name": "成交占比", "data_type": "text", "width": "18%", "horizontal_align": "right"},
            ], board_heat_rows))
        else:
            elements.append({
                "tag": "markdown",
                "content": "<font color='grey'>暂无与分析交易日一致的收盘行业热度快照。</font>",
            })

        concept_heat_rows: List[Dict[str, Any]] = []
        raw_concept_heat = board_heat_overview.get("concept")
        if isinstance(raw_concept_heat, list):
            for index, item in enumerate(raw_concept_heat[:5], start=1):
                if not isinstance(item, dict):
                    continue
                concept_heat_rows.append({
                    "theme": f"{index}. {cls._compact_card_text(item.get('name'), 24) or '--'}",
                    "heat": cls._format_decimal(item.get("score")),
                    "change": cls._format_change(item.get("change_pct")),
                    "flow": cls._format_decimal(item.get("fund_flow_score")),
                    "members": str(item.get("member_count") or "--"),
                })
        concept_source = cls._compact_card_text(
            board_heat_overview.get("concept_source"), 40
        ) or "--"
        elements.append({
            "tag": "markdown",
            "content": (
                "### 🎯 当日题材热度\n"
                f"状态 **{'可用' if concept_heat_rows else '不可用'}** ｜ 来源 **{concept_source}**\n"
                "<font color='grey'>题材与行业分榜；资金分位为当日概念板块横截面排名。</font>"
            ),
        })
        if concept_heat_rows:
            elements.append(cls._build_table("screening_daily_concepts", [
                {"name": "theme", "display_name": "题材", "data_type": "text", "width": "34%", "vertical_align": "top"},
                {"name": "heat", "display_name": "综合热度", "data_type": "text", "width": "18%", "horizontal_align": "right"},
                {"name": "change", "display_name": "涨跌", "data_type": "text", "width": "16%", "horizontal_align": "right"},
                {"name": "flow", "display_name": "资金分位", "data_type": "text", "width": "18%", "horizontal_align": "right"},
                {"name": "members", "display_name": "成分数", "data_type": "text", "width": "14%", "horizontal_align": "right"},
            ], concept_heat_rows))
        else:
            elements.append({
                "tag": "markdown",
                "content": "<font color='grey'>当日概念资金流源不可用，未用行业榜代替题材榜。</font>",
            })

        sector_overview = (
            payload.get("sector_overview")
            if isinstance(payload.get("sector_overview"), dict)
            else {}
        )
        status_label = {"available": "可用", "unavailable": "不可用", "stale": "缓存"}
        sector_rows: List[Dict[str, Any]] = []
        raw_sectors = sector_overview.get("top_sectors")
        if isinstance(raw_sectors, list):
            for index, item in enumerate(raw_sectors[:5], start=1):
                if not isinstance(item, dict):
                    continue
                sector_rows.append({
                    "sector": f"{index}. {cls._compact_card_text(item.get('industry'), 24) or '--'}",
                    "heat": cls._format_decimal(item.get("local_heat_score")),
                    "breadth": cls._format_change(item.get("breadth_ratio")),
                    "leaders": str(item.get("high_adr_leader_count") if item.get("high_adr_leader_count") is not None else "--"),
                    "trend": cls._format_change(item.get("change_20d_pct")),
                })
        local_status = status_label.get(
            str(sector_overview.get("local_heat_status") or "unavailable"), "不可用"
        )
        external_status = status_label.get(
            str(sector_overview.get("external_heat_status") or "unavailable"), "不可用"
        )
        flow_status = status_label.get(
            str(sector_overview.get("fund_flow_status") or "unavailable"), "不可用"
        )
        sector_count = int(sector_overview.get("sector_count") or 0)
        external_count = int(sector_overview.get("external_heat_coverage_count") or 0)
        flow_count = int(sector_overview.get("fund_flow_coverage_count") or 0)
        sector_as_of = cls._compact_card_text(sector_overview.get("as_of"), 16) or analysis_date
        feature_coverage = cls._format_decimal(
            sector_overview.get("daily_feature_coverage_pct")
        )
        industry_coverage = cls._format_decimal(
            sector_overview.get("industry_coverage_pct")
        )
        elements.append({
            "tag": "markdown",
            "content": (
                "### 🔥 行业板块热度（中期趋势）\n"
                f"本地量价 **{local_status}** ｜ 外部热度 **{external_status} "
                f"({external_count}/{sector_count})** ｜ 资金流 **{flow_status} "
                f"({flow_count}/{sector_count})**\n"
                f"<font color='grey'>数据日 {sector_as_of} · 覆盖 {sector_count} 个行业"
                f" · 日线覆盖 {feature_coverage}% · 行业归属覆盖 {industry_coverage}%"
                f"{' · 使用最近缓存' if sector_overview.get('stale') else ''}</font>"
            ),
        })
        if sector_rows:
            elements.append(cls._build_table("screening_sectors", [
                {"name": "sector", "display_name": "行业", "data_type": "text", "width": "34%", "vertical_align": "top"},
                {"name": "heat", "display_name": "量价热度", "data_type": "text", "width": "18%", "horizontal_align": "right"},
                {"name": "breadth", "display_name": "趋势宽度", "data_type": "text", "width": "18%", "horizontal_align": "right"},
                {"name": "leaders", "display_name": "高ADR领涨", "data_type": "text", "width": "14%", "horizontal_align": "right"},
                {"name": "trend", "display_name": "20日", "data_type": "text", "width": "16%", "horizontal_align": "right"},
            ], sector_rows))
        else:
            elements.append({
                "tag": "markdown",
                "content": "<font color='grey'>暂无可用行业强度排行；该缺失不会被填成中性分。</font>",
            })

        theme_overview = (
            payload.get("theme_overview")
            if isinstance(payload.get("theme_overview"), dict)
            else {}
        )
        theme_rows: List[Dict[str, Any]] = []
        raw_themes = theme_overview.get("themes")
        if isinstance(raw_themes, list):
            for index, item in enumerate(raw_themes[:5], start=1):
                if not isinstance(item, dict):
                    continue
                leaders = item.get("leaders") if isinstance(item.get("leaders"), list) else []
                theme_rows.append({
                    "theme": f"{index}. {cls._compact_card_text(item.get('name') or item.get('topic'), 24) or '--'}",
                    "heat": cls._format_decimal(item.get("heat_score")),
                    "change": cls._format_change(item.get("change_pct")),
                    "stage": cls._compact_card_text(item.get("stage") or item.get("state"), 18) or "--",
                    "leaders": cls._compact_card_text("、".join(str(value) for value in leaders), 30) or "--",
                })
        theme_status = str(theme_overview.get("status") or "unavailable")
        theme_status_text = {
            "available": "可用",
            "unavailable": "不可用",
            "stale": "过期缓存",
        }.get(theme_status, theme_status)
        provider_text = cls._compact_card_text(theme_overview.get("provider"), 36) or "免费源"
        theme_data_as_of = cls._compact_card_text(
            theme_overview.get("data_as_of") or theme_overview.get("cached_at"),
            32,
        )
        theme_freshness_reason = cls._compact_card_text(
            theme_overview.get("freshness_reason"),
            80,
        )
        freshness_line = ""
        if theme_status == "stale":
            freshness_line = (
                "\n<font color='red'>"
                f"缓存时间 {theme_data_as_of or '--'}"
                f" · {theme_freshness_reason or '未通过分析日时效校验'}"
                "；旧榜单已隐藏，不参与当日题材判断。</font>"
            )
        elif theme_data_as_of:
            freshness_line = f"\n<font color='grey'>数据时间 {theme_data_as_of}</font>"
        elements.append({
            "tag": "markdown",
            "content": (
                "### 🧭 题材异动热度\n"
                f"状态 **{theme_status_text}** ｜ 来源 **{provider_text}**"
                f"{' ｜ 已回退缓存' if theme_overview.get('cache_used') or theme_overview.get('fallback_used') else ''}"
                f"{freshness_line}"
            ),
        })
        if theme_rows and theme_status != "stale":
            elements.append(cls._build_table("screening_themes", [
                {"name": "theme", "display_name": "题材", "data_type": "text", "width": "30%", "vertical_align": "top"},
                {"name": "heat", "display_name": "热度", "data_type": "text", "width": "13%", "horizontal_align": "right"},
                {"name": "change", "display_name": "涨跌", "data_type": "text", "width": "13%", "horizontal_align": "right"},
                {"name": "stage", "display_name": "阶段", "data_type": "text", "width": "18%"},
                {"name": "leaders", "display_name": "代表股", "data_type": "text", "width": "26%", "vertical_align": "top"},
            ], theme_rows))
        elif theme_status != "stale":
            source_errors = theme_overview.get("source_errors")
            reason = ""
            if isinstance(source_errors, list) and source_errors:
                reason = cls._compact_card_text(source_errors[0], 100)
            elements.append({
                "tag": "markdown",
                "content": (
                    "<font color='grey'>外部题材榜当前不可用；行业量价强度仍可独立使用。"
                    f"{(' 原因：' + reason) if reason else ''}</font>"
                ),
            })

        portfolio_plan = (
            payload.get("portfolio_plan")
            if isinstance(payload.get("portfolio_plan"), dict)
            else {}
        )
        portfolio_account = (
            portfolio_plan.get("account")
            if isinstance(portfolio_plan.get("account"), dict)
            else {}
        )
        holding_rows: List[Dict[str, Any]] = []
        raw_holdings = portfolio_plan.get("holdings")
        if isinstance(raw_holdings, list):
            action_labels = {
                "sell_next_open_call_auction": "次日竞价卖出",
                "partial_sell_next_open_call_auction": "次日竞价减仓",
                "keep_and_review_after_close": "继续持有",
                "wait_for_complete_close": "等待当日收盘",
                "hold_without_rule_change": "数据不足，保持条件",
                "manual_review": "人工确认",
            }
            for item in raw_holdings[:8]:
                if not isinstance(item, dict):
                    continue
                action = action_labels.get(
                    str(item.get("action") or ""),
                    cls._compact_card_text(item.get("status"), 16) or "--",
                )
                if item.get("execution_status") == "blocked_t1":
                    action = "T+1 暂不可卖"
                elif item.get("recommended_partial_sell_quantity"):
                    action = (
                        f"减仓 {cls._format_quantity(item.get('recommended_partial_sell_quantity'))}股"
                    )
                add_on = item.get("add_on_plan")
                if (
                    action == "继续持有"
                    and isinstance(add_on, dict)
                    and add_on.get("status") == "ready"
                ):
                    action = (
                        f"可加仓 {cls._format_quantity(add_on.get('recommended_quantity'))}股\n"
                        f"竞价≤{cls._format_decimal(add_on.get('max_price'))}"
                    )
                holding_rows.append({
                    "symbol": cls._compact_card_text(item.get("symbol"), 14) or "--",
                    "quantity": (
                        f"{cls._format_quantity(item.get('quantity'))} / "
                        f"{cls._format_quantity(item.get('sellable_quantity'))}"
                    ),
                    "cost_close": (
                        f"{cls._format_decimal(item.get('avg_cost'))} / "
                        f"{cls._format_decimal(item.get('current_close'))}"
                    ),
                    "risk": (
                        f"{cls._format_decimal(item.get('current_r_multiple'))}R / "
                        f"{cls._format_decimal(item.get('active_exit_price'))}"
                        if item.get("managed")
                        else "未绑定策略"
                    ),
                    "action": action,
                })
        if portfolio_account:
            regime_labels = {
                "strong_risk_on": "强势进攻",
                "risk_on": "正常进攻",
                "neutral": "中性防守",
                "risk_off": "风险关闭",
                "unavailable": "暂不可用",
            }
            regime_state = str(portfolio_account.get("market_regime_state") or "unavailable")
            elements.append({
                "tag": "markdown",
                "content": (
                    "### 💼 账户执行摘要\n"
                    f"账户 **{cls._compact_card_text(portfolio_account.get('account_name'), 28) or portfolio_account.get('account_id')}** ｜ "
                    f"权益 **{cls._format_decimal(portfolio_account.get('total_equity'))}** ｜ "
                    f"现金 **{cls._format_decimal(portfolio_account.get('total_cash'))}**\n"
                    f"持仓 **{portfolio_account.get('position_count', 0)}** 只 ｜ "
                    f"策略管理 **{portfolio_account.get('managed_position_count', 0)}** 只 ｜ "
                    f"风险热度 **{cls._format_decimal(portfolio_account.get('strategy_risk_heat_pct'))}%**\n"
                    f"市场状态 **{regime_labels.get(regime_state, regime_state)}**"
                    f"（{portfolio_account.get('market_regime_date') or '日期不可用'}）\n"
                    f"<font color='grey'>计划日期 {portfolio_account.get('as_of') or analysis_date}；"
                    "未纳入策略的持仓不计入策略风险热度，也不会自动生成卖出条件。</font>"
                ),
            })
            if holding_rows:
                elements.append(cls._build_table("screening_portfolio_holdings", [
                    {"name": "symbol", "display_name": "持仓", "data_type": "text", "width": "16%"},
                    {"name": "quantity", "display_name": "持有/可卖", "data_type": "text", "width": "20%", "horizontal_align": "right"},
                    {"name": "cost_close", "display_name": "成本/收盘", "data_type": "text", "width": "22%", "horizontal_align": "right"},
                    {"name": "risk", "display_name": "当前R/防线", "data_type": "text", "width": "22%", "horizontal_align": "right"},
                    {"name": "action", "display_name": "动作", "data_type": "text", "width": "20%"},
                ], holding_rows))

        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(candidates[:5], start=1):
            if not isinstance(item, dict):
                continue
            score = item.get("score")
            try:
                score_text = f"{float(score):.1f}"
            except (TypeError, ValueError):
                score_text = "--"
            candidate_plan = (
                item.get("trade_plan")
                if isinstance(item.get("trade_plan"), dict)
                else {}
            )
            candidate_sizing = (
                candidate_plan.get("position_sizing")
                if isinstance(candidate_plan.get("position_sizing"), dict)
                else {}
            )
            rows.append({
                "name": f"{index}. {cls._compact_card_text(item.get('name'), 24) or '--'}\n{cls._compact_card_text(item.get('code'), 16) or '--'}",
                "score": score_text,
                "order": (
                    f"{int(candidate_sizing.get('recommended_quantity') or 0)}股\n"
                    f"≈{cls._format_decimal(candidate_sizing.get('estimated_notional'))}"
                    if candidate_sizing.get("recommended_quantity")
                    else "观察/受限"
                ),
                "reason": cls._compact_card_text(item.get("reason") or item.get("recommendation_reason"), 88) or "通过策略硬条件",
            })
        if rows:
            elements.extend([
                {"tag": "markdown", "content": "### 🎯 前列候选"},
                cls._build_table("screening_candidates", [
                    {"name": "name", "display_name": "标的", "data_type": "text", "width": "28%", "vertical_align": "top"},
                    {"name": "score", "display_name": "评分", "data_type": "text", "width": "12%", "horizontal_align": "right", "vertical_align": "top"},
                    {"name": "order", "display_name": "账户计划", "data_type": "text", "width": "22%", "horizontal_align": "right", "vertical_align": "top"},
                    {"name": "reason", "display_name": "入选依据", "data_type": "text", "width": "38%", "vertical_align": "top"},
                ], rows),
            ])
        else:
            elements.append({
                "tag": "markdown",
                "content": "**本次无候选**\n没有股票通过全部硬条件，保留现金与观察清单也是结果。",
            })

        plan_sections: List[str] = []
        for index, item in enumerate(candidates[:5], start=1):
            if not isinstance(item, dict):
                continue
            plan = item.get("trade_plan") if isinstance(item.get("trade_plan"), dict) else {}
            if not plan:
                continue
            status = str(plan.get("signal_status") or "")
            name = cls._compact_card_text(item.get("name"), 20) or "--"
            code = cls._compact_card_text(item.get("code"), 12) or "--"
            if status == "watch_only_setup":
                setup_permission = (
                    plan.get("setup_permission")
                    if isinstance(plan.get("setup_permission"), dict)
                    else {}
                )
                setup_type = cls._compact_card_text(
                    setup_permission.get("setup_type"), 24
                ) or "当前形态"
                plan_sections.append(
                    f"**{index}. {name} {code}｜形态观察**\n"
                    f"{setup_type} 已确认，但当前版本未开放该形态交易；"
                    "不设置买入条件单。"
                )
                continue
            if status == "market_blocked":
                plan_sections.append(
                    f"**{index}. {name} {code}｜市场状态暂停**\n"
                    f"{plan.get('market_block_reason') or '当前市场状态不支持新开仓，不设置买入条件单。'}"
                )
                continue
            if status == "risk_rejected":
                plan_sections.append(
                    f"**{index}. {name} {code}｜风险过滤**\n"
                    f"{plan.get('risk_rejection_reason') or '技术止损距离超过策略上限，不设置买入条件单。'}"
                )
                continue
            if status != "buy_signal":
                plan_sections.append(
                    f"**{index}. {name} {code}｜仅观察**\n"
                    "尚未完成收盘突破确认，不设置买入条件单。"
                )
                continue
            entry = plan.get("entry") if isinstance(plan.get("entry"), dict) else {}
            exit_plan = plan.get("exit") if isinstance(plan.get("exit"), dict) else {}
            sizing = plan.get("position_sizing") if isinstance(plan.get("position_sizing"), dict) else {}
            weight_cap = sizing.get("max_position_weight_pct")
            minimum_weight = sizing.get("min_position_weight_pct")
            risk_budget = sizing.get("max_portfolio_risk_per_trade_pct")
            heat_cap = sizing.get("max_portfolio_heat_pct")
            risk_limits = []
            if risk_budget is not None:
                risk_limits.append(f"单笔风险预算 {risk_budget}%")
            if heat_cap is not None:
                risk_limits.append(f"组合风险热度上限 {heat_cap}%")
            risk_limits_text = f"（{'｜'.join(risk_limits)}）" if risk_limits else ""
            account_quantity = sizing.get("recommended_quantity")
            account_notional = sizing.get("estimated_notional")
            execution_status = str(sizing.get("execution_status") or "")
            account_sizing_line = ""
            if account_quantity:
                account_sizing_line = (
                    f"账户计划：**{int(account_quantity)} 股**，按最高允许价估算 **{account_notional}**；"
                    f"预计仓位 {sizing.get('estimated_weight_pct', '--')}%，风险热度 "
                    f"{sizing.get('estimated_heat_pct', '--')}%\n"
                )
            elif portfolio_account:
                account_sizing_line = (
                    f"账户计划：**不下单**（{execution_status or '受限'}）"
                    f"；{sizing.get('constraint_reason') or '未满足账户执行约束'}\n"
                )
            sizing_line = (
                f"仓位区间：**{minimum_weight if minimum_weight is not None else '--'}% - {weight_cap}%**"
                f"{risk_limits_text}\n"
                if weight_cap is not None
                else ""
            )
            risk_line = ""
            technical_risk = exit_plan.get(
                "sizing_risk_pct",
                exit_plan.get("effective_risk_pct"),
            )
            if technical_risk is not None:
                risk_line = (
                    f"技术止损/计仓风险 **{technical_risk}%**；"
                    f"1 ADR 压力 {exit_plan.get('adr_stress_pct', '--')}%（不参与计仓）\n"
                )
            max_holding_days = exit_plan.get("max_holding_days")
            holding_period_text = (
                f"研究上限 {max_holding_days} 日"
                if max_holding_days is not None
                else "不设固定期限，趋势未破坏则持有"
            )
            plan_sections.append(
                f"**{index}. {name} {code}｜买入信号**\n"
                f"开盘竞价区间：**{entry.get('min_price', '--')} - {entry.get('max_price', '--')}**；"
                f"确认价 {entry.get('trigger_price', '--')}\n"
                f"初始防守：**{exit_plan.get('initial_stop_price', '--')}**；"
                f"1R 保本观察 {exit_plan.get('breakeven_activation_price', '--')}；"
                f"2R 观察价 {exit_plan.get('two_r_reference_price', exit_plan.get('first_profit_price', '--'))}（非固定止盈）\n"
                f"{risk_line}"
                f"{account_sizing_line}"
                f"{sizing_line}"
                f"持仓纪律：{exit_plan.get('follow_through_days', 10)} 日至少 "
                f"{exit_plan.get('follow_through_min_return_pct', 3)}%；{holding_period_text}；"
                "每日收盘更新 EMA21 与移动防线。\n"
                "<font color='grey'>1R/2R 为信号日预估，实际成交后按成交价重算；防守触发后于下一开盘竞价卖出。</font>"
            )
        if plan_sections:
            elements.append({
                "tag": "markdown",
                "content": "### 🧾 条件单计划\n" + "\n\n".join(plan_sections),
            })

        diagnostics = (
            payload.get("data_diagnostics")
            if isinstance(payload.get("data_diagnostics"), dict)
            else {}
        )
        warning_count = diagnostics.get("warning_count") or 0
        elements.append({
            "tag": "markdown",
            "content": (
                "### 🧪 数据质量\n"
                f"快照源 **{cls._compact_card_text(diagnostics.get('snapshot_source'), 30) or '--'}** ｜ "
                f"全市场 **{diagnostics.get('snapshot_count') or '--'}** ｜ "
                f"种子池 **{diagnostics.get('seed_pool_count') if diagnostics.get('seed_pool_count') is not None else '--'}** ｜ "
                f"日线成功 **{diagnostics.get('daily_enrich_count') if diagnostics.get('daily_enrich_count') is not None else '--'}**\n"
                f"<font color='grey'>诊断提醒 {warning_count} 条；缺失项会明确降级，不以估计值伪装为实时数据。</font>"
            ),
        })

        detail_url = str(payload.get("detail_url") or "").strip()
        if detail_url:
            elements.append({
                "tag": "markdown",
                "content": f"[🔎 在网页中查看完整结果]({detail_url})",
            })
        elements.append({
            "tag": "markdown",
            "content": "<font color='grey'>结果用于量化研究与复盘，不构成投资建议。</font>",
        })
        return {
            "schema": "2.0",
            "config": {"update_multi": True},
            "header": {
                "template": "orange",
                "title": {"tag": "plain_text", "content": "选股任务完成"},
            },
            "body": {"direction": "vertical", "padding": "12px 12px 12px 12px", "elements": elements},
        }

    @staticmethod
    def _metric_number(mapping: Any, key: str) -> Optional[float]:
        if not isinstance(mapping, dict):
            return None
        try:
            return float(mapping.get(key))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _build_strategy_backtest_card_body(cls, payload: Dict[str, Any], content: str) -> dict:
        """Build a concise, mobile-readable card from a Martin backtest JSON result."""
        period = payload.get("period") if isinstance(payload.get("period"), dict) else {}
        results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
        benchmark = payload.get("benchmark") if isinstance(payload.get("benchmark"), dict) else {}
        benchmark_metrics = benchmark.get("metrics") if isinstance(benchmark.get("metrics"), dict) else {}
        signal_diagnostics = (
            payload.get("signal_diagnostics")
            if isinstance(payload.get("signal_diagnostics"), dict)
            else {}
        )
        data_diagnostics = (
            payload.get("data_diagnostics")
            if isinstance(payload.get("data_diagnostics"), dict)
            else {}
        )
        strategy = payload.get("strategy") if isinstance(payload.get("strategy"), dict) else {}

        unified = results.get("strategy") if isinstance(results.get("strategy"), dict) else None
        variants = (
            (("strategy", "Martin策略"),)
            if unified is not None
            else (("left", "左侧承接"), ("right", "右侧突破"))
        )
        mode_rows: List[Dict[str, Any]] = []
        for mode_key, mode_label in variants:
            result = results.get(mode_key) if isinstance(results.get(mode_key), dict) else {}
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            mode_rows.append({
                "mode": mode_label,
                "return": cls._format_change(metrics.get("total_return_pct")),
                "excess": cls._format_change(metrics.get("excess_return_pct")),
                "drawdown": cls._format_change(metrics.get("max_drawdown_pct")),
                "trades": str(metrics.get("trade_count") if metrics.get("trade_count") is not None else "--"),
                "win_rate": cls._format_change(metrics.get("win_rate_pct")),
            })

        primary_result = unified or (
            results.get("right") if isinstance(results.get("right"), dict) else {}
        )
        primary_metrics = (
            primary_result.get("metrics") if isinstance(primary_result.get("metrics"), dict) else {}
        )
        primary_return = cls._metric_number(primary_metrics, "total_return_pct")
        benchmark_return = cls._metric_number(benchmark_metrics, "total_return_pct")

        conclusions: List[str] = []
        if unified is not None:
            if primary_return is not None:
                conclusions.append("策略取得正收益" if primary_return > 0 else "策略本区间录得亏损")
            if primary_return is not None and benchmark_return is not None:
                conclusions.append(
                    "跑赢同池等权基准"
                    if primary_return >= benchmark_return
                    else "仍跑输同池等权基准"
                )
        else:
            left_metrics = (
                results.get("left", {}).get("metrics", {})
                if isinstance(results.get("left"), dict)
                else {}
            )
            left_return = cls._metric_number(left_metrics, "total_return_pct")
            if primary_return is not None and left_return is not None and primary_return > left_return:
                conclusions.append("右侧突破明显优于左侧承接")
            if left_return is not None and left_return < 0:
                conclusions.append("本区间不支持采用左侧版本")
            if primary_return is not None and benchmark_return is not None and primary_return < benchmark_return:
                conclusions.append("右侧虽为正收益，但仍跑输同池等权基准")
        verdict = "；".join(conclusions) or "样本尚不足以形成稳健结论"

        elements: List[Dict[str, Any]] = [
            {
                "tag": "markdown",
                "content": (
                    f"<font color='grey'>{period.get('start') or '--'} 至 {period.get('end') or '--'} | "
                    "收盘确认，次日开盘集合竞价成交</font>\n"
                    f"**结论：{verdict}。**"
                ),
            },
            {"tag": "markdown", "content": "### 📊 策略表现" if unified is not None else "### 📊 历史版本对比"},
            cls._build_table(
                "strategy_backtest_summary",
                [
                    {"name": "mode", "display_name": "模式", "data_type": "text", "width": "21%"},
                    {"name": "return", "display_name": "净收益", "data_type": "text", "width": "17%", "horizontal_align": "right"},
                    {"name": "excess", "display_name": "超额", "data_type": "text", "width": "17%", "horizontal_align": "right"},
                    {"name": "drawdown", "display_name": "回撤", "data_type": "text", "width": "17%", "horizontal_align": "right"},
                    {"name": "trades", "display_name": "交易", "data_type": "text", "width": "12%", "horizontal_align": "right"},
                    {"name": "win_rate", "display_name": "胜率", "data_type": "text", "width": "16%", "horizontal_align": "right"},
                ],
                mode_rows,
            ),
            {
                "tag": "markdown",
                "content": (
                    f"**同池等权基准：{cls._format_change(benchmark_return)}**（不计费）\n"
                    f"策略估算成本约 **{float(primary_metrics.get('estimated_total_cost') or 0):,.0f} 元**。\n"
                    f"<font color='grey'>策略 v{strategy.get('version') or '--'}：技术止损计仓；"
                    "1 ADR 仅作跳空压力，不压缩股数。</font>"
                ),
            },
        ]
        regime_summary = (
            primary_result.get("market_regime")
            if isinstance(primary_result.get("market_regime"), dict)
            else {}
        )
        if regime_summary:
            elements.append({
                "tag": "markdown",
                "content": (
                    "### 🛡️ 多日市场状态\n"
                    f"区间天数分布：`{regime_summary.get('state_day_counts') or {}}`\n"
                    "<font color='grey'>降级即时生效，升级连续两日确认；仅调节新开仓，不强制卖出已有持仓。</font>"
                ),
            })

        impactful_trades: List[Dict[str, Any]] = []
        for mode_key, mode_label in variants:
            result = results.get(mode_key) if isinstance(results.get(mode_key), dict) else {}
            trades = result.get("trades") if isinstance(result.get("trades"), list) else []
            for trade in trades:
                if not isinstance(trade, dict):
                    continue
                try:
                    pnl = float(trade.get("pnl") or 0)
                except (TypeError, ValueError):
                    pnl = 0.0
                impactful_trades.append({"mode": mode_label, "pnl_value": pnl, **trade})
        impactful_trades.sort(key=lambda item: abs(float(item.get("pnl_value") or 0)), reverse=True)
        trade_rows = [
            {
                "name": f"{cls._compact_card_text(item.get('name'), 16) or '--'}\n{item.get('code') or '--'}",
                "return": cls._format_change(item.get("return_pct")),
                "pnl": f"{float(item.get('pnl_value') or 0):+,.0f}元",
            }
            for item in impactful_trades[:6]
        ]
        if trade_rows:
            elements.extend([
                {"tag": "markdown", "content": "### 🔎 影响最大的交易"},
                cls._build_table(
                    "strategy_backtest_trades",
                    [
                        {"name": "name", "display_name": "标的", "data_type": "text", "width": "48%", "vertical_align": "top"},
                        {"name": "return", "display_name": "收益", "data_type": "text", "width": "24%", "horizontal_align": "right"},
                        {"name": "pnl", "display_name": "盈亏", "data_type": "text", "width": "28%", "horizontal_align": "right"},
                    ],
                    trade_rows,
                ),
            ])

        validation = (
            payload.get("source_validation")
            if isinstance(payload.get("source_validation"), dict)
            else {}
        )
        validation_summary = (
            validation.get("summary") if isinstance(validation.get("summary"), dict) else {}
        )
        full_market = payload.get("universe_mode") == "full-a-share"
        universe_note = (
            "• 覆盖完整当前 A 股快照，但历史已退市股票仍缺失；当前 ST 名称按全区间保守排除。\n"
            if full_market
            else "• 当前可交易池存在幸存者偏差和当前流动性偏差。\n"
        )
        elements.extend([
            {
                "tag": "markdown",
                "content": (
                    "### 🧪 数据与信号\n"
                    f"• 日线覆盖 **{data_diagnostics.get('library_codes') or '--'}** 只，"
                    f"区间 **{data_diagnostics.get('panel_dates') or '--'}** 个交易日\n"
                    f"• 起点预热覆盖 **{data_diagnostics.get('codes_with_120_bars_at_period_start') or 0}/"
                    f"{data_diagnostics.get('codes_available_at_period_start') or 0}** 只"
                    f"（{cls._format_change(data_diagnostics.get('start_history_coverage_pct'))}）\n"
                    f"• 硬过滤合格 **{signal_diagnostics.get('setup_count', signal_diagnostics.get('total_setups', 0))}** 条；"
                    f"形态确认 {signal_diagnostics.get('confirmed_setup_count', signal_diagnostics.get('entry_signal_count', 0))}，"
                    f"可交易 {signal_diagnostics.get('entry_signal_count', signal_diagnostics.get('right_setups', 0))}，"
                    f"仅观察 {signal_diagnostics.get('watch_only_signal_count', 0)}，"
                    f"未确认 {signal_diagnostics.get('unconfirmed_setup_count', signal_diagnostics.get('left_setups', 0))}\n"
                    f"• 首个严格信号日：**{signal_diagnostics.get('first_signal_date') or '--'}**\n"
                    f"• AkShare + Baostock 抽样复核：**{validation_summary.get('ok') or 0}/"
                    f"{validation_summary.get('requested') or 0} 通过**"
                ),
            },
            {
                "tag": "markdown",
                "content": (
                    "### ⚠️ 使用边界\n"
                    f"{universe_note}"
                    "• 历史 ST / 市值 / 停复牌状态未按日重建。\n"
                    "• 缺少历史板块成分与热度，本次板块因子中性、未执行板块配额。\n"
                    "• 日线开盘价仅近似开盘集合竞价成交价，无法还原委托、排队和撤单。\n"
                    "• 收盘只用于确认信号，成交统一顺延至下一交易日，避免未来函数。\n"
                    "• 区间较短，暂不能把结果视为稳定超额。"
                ),
            },
            {
                "tag": "markdown",
                "content": "<font color='grey'>研究回测，不构成投资建议。完整逐笔明细保留在本地回测产物中。</font>",
            },
        ])
        return {
            "schema": "2.0",
            "config": {"update_multi": True},
            "header": {
                "template": "turquoise",
                "title": {"tag": "plain_text", "content": "Martin A股策略回测"},
            },
            "body": {"direction": "vertical", "padding": "12px 12px 12px 12px", "elements": elements},
        }

    @classmethod
    def _build_strategy_backtest_analysis_card_body(
        cls,
        payload: Dict[str, Any],
        content: str,
    ) -> dict:
        """Build capital, monthly-return, and exit diagnostics for a backtest."""
        period = payload.get("period") if isinstance(payload.get("period"), dict) else {}
        results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
        result = results.get("strategy") if isinstance(results.get("strategy"), dict) else {}
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        config = result.get("config") if isinstance(result.get("config"), dict) else {}
        diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
        initial_cash = float(config.get("initial_cash") or 0)
        final_equity = float(metrics.get("final_equity") or 0)
        net_profit = final_equity - initial_cash

        risk_rows = [{
            "capital": f"{initial_cash:,.0f} → {final_equity:,.0f}元",
            "return": cls._format_change(metrics.get("total_return_pct")),
            "drawdown": cls._format_change(metrics.get("max_drawdown_pct")),
            "sharpe": cls._format_decimal(metrics.get("sharpe_ratio")),
        }]
        quality_rows = [{
            "profit": f"{net_profit:+,.0f}元",
            "factor": cls._format_decimal(metrics.get("profit_factor")),
            "average": cls._format_change(metrics.get("average_trade_pct")),
            "exposure": cls._format_change(metrics.get("average_exposure_pct")),
        }]
        sizing_rows = [{
            "average": cls._format_change(metrics.get("average_entry_position_weight_pct")),
            "median": cls._format_change(metrics.get("median_entry_position_weight_pct")),
            "one_lot": str(metrics.get("one_lot_trade_count") or 0),
            "adr_stress": cls._format_change(metrics.get("max_adr_stress_heat_pct")),
        }]
        month_rows = cls._strategy_backtest_month_rows(result)
        exit_rows = cls._strategy_backtest_exit_rows(result)
        elements: List[Dict[str, Any]] = [
            {
                "tag": "markdown",
                "content": (
                    f"<font color='grey'>{period.get('start') or '--'} 至 {period.get('end') or '--'} | "
                    "完整资金曲线与退出归因</font>"
                ),
            },
            {"tag": "markdown", "content": "### 💰 资金与风险"},
            cls._build_table(
                "strategy_backtest_risk",
                [
                    {"name": "capital", "display_name": "本金→期末", "data_type": "text", "width": "38%"},
                    {"name": "return", "display_name": "收益", "data_type": "text", "width": "20%", "horizontal_align": "right"},
                    {"name": "drawdown", "display_name": "最大回撤", "data_type": "text", "width": "22%", "horizontal_align": "right"},
                    {"name": "sharpe", "display_name": "夏普", "data_type": "text", "width": "20%", "horizontal_align": "right"},
                ],
                risk_rows,
            ),
            cls._build_table(
                "strategy_backtest_quality",
                [
                    {"name": "profit", "display_name": "净利润", "data_type": "text", "width": "27%"},
                    {"name": "factor", "display_name": "利润因子", "data_type": "text", "width": "24%", "horizontal_align": "right"},
                    {"name": "average", "display_name": "单笔均值", "data_type": "text", "width": "24%", "horizontal_align": "right"},
                    {"name": "exposure", "display_name": "平均仓位", "data_type": "text", "width": "25%", "horizontal_align": "right"},
                ],
                quality_rows,
            ),
            cls._build_table(
                "strategy_backtest_sizing",
                [
                    {"name": "average", "display_name": "平均入场仓位", "data_type": "text", "width": "27%", "horizontal_align": "right"},
                    {"name": "median", "display_name": "中位入场仓位", "data_type": "text", "width": "27%", "horizontal_align": "right"},
                    {"name": "one_lot", "display_name": "100股交易", "data_type": "text", "width": "20%", "horizontal_align": "right"},
                    {"name": "adr_stress", "display_name": "最高ADR压力", "data_type": "text", "width": "26%", "horizontal_align": "right"},
                ],
                sizing_rows,
            ),
        ]
        if month_rows:
            elements.extend([
                {"tag": "markdown", "content": "### 📅 月度资金曲线"},
                cls._build_table(
                    "strategy_backtest_months",
                    [
                        {"name": "month", "display_name": "月份", "data_type": "text", "width": "34%"},
                        {"name": "return", "display_name": "月收益", "data_type": "text", "width": "28%", "horizontal_align": "right"},
                        {"name": "equity", "display_name": "月末权益", "data_type": "text", "width": "38%", "horizontal_align": "right"},
                    ],
                    month_rows,
                ),
            ])
        if exit_rows:
            elements.extend([
                {"tag": "markdown", "content": "### 🧭 退出归因"},
                cls._build_table(
                    "strategy_backtest_exits",
                    [
                        {"name": "reason", "display_name": "退出原因", "data_type": "text", "width": "44%"},
                        {"name": "count", "display_name": "次数", "data_type": "text", "width": "20%", "horizontal_align": "right"},
                        {"name": "pnl", "display_name": "合计盈亏", "data_type": "text", "width": "36%", "horizontal_align": "right"},
                    ],
                    exit_rows,
                ),
            ])
        elements.extend([
            {
                "tag": "markdown",
                "content": (
                    "### 🚦 执行过滤\n"
                    f"• 高开超区间跳过 **{diagnostics.get('skipped_gap_up') or 0}** 次；"
                    f"低开超区间跳过 **{diagnostics.get('skipped_gap_down') or 0}** 次\n"
                    f"• 开盘破坏支撑跳过 **{diagnostics.get('skipped_broken_support') or 0}** 次；"
                    f"涨停无法买入 **{diagnostics.get('skipped_limit_up') or 0}** 次\n"
                    f"• 技术止损超限跳过 **{diagnostics.get('skipped_technical_stop_too_wide') or 0}** 次；"
                    f"仓位低于最低有效值跳过 **{diagnostics.get('skipped_min_position_weight') or 0}** 次\n"
                    f"• 市场状态禁止新开仓 **{diagnostics.get('skipped_market_regime') or 0}** 条信号\n"
                    f"• 期末未平仓 **{diagnostics.get('open_positions_at_end') or 0}** 只；"
                    f"估算总成本 **{float(metrics.get('estimated_total_cost') or 0):,.0f} 元**"
                ),
            },
            {
                "tag": "markdown",
                "content": (
                    f"胜率仅 **{cls._format_change(metrics.get('win_rate_pct'))}**，但平均盈利 "
                    f"**{cls._format_change(metrics.get('average_win_pct'))}**、平均亏损 "
                    f"**{cls._format_change(metrics.get('average_loss_pct'))}**。"
                    "收益依赖少数大幅盈利交易，策略具有明显右偏分布，必须严格执行小亏退出。"
                ),
            },
        ])
        return {
            "schema": "2.0",
            "config": {"update_multi": True},
            "header": {
                "template": "blue",
                "title": {"tag": "plain_text", "content": "Martin回测详细诊断"},
            },
            "body": {"direction": "vertical", "padding": "12px 12px 12px 12px", "elements": elements},
        }

    @classmethod
    def _build_strategy_backtest_trades_card_body(
        cls,
        payload: Dict[str, Any],
        content: str,
    ) -> dict:
        """Build one paged trade-ledger card."""
        period = payload.get("period") if isinstance(payload.get("period"), dict) else {}
        trades = payload.get("trades") if isinstance(payload.get("trades"), list) else []
        page = int(payload.get("page") or 1)
        pages = int(payload.get("pages") or 1)
        total = int(payload.get("total_trades") or len(trades))
        rows: List[Dict[str, Any]] = []
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            entry_date = str(trade.get("entry_date") or "--")
            exit_date = str(trade.get("exit_date") or "--")
            shares = float(trade.get("shares") or 0)
            entry_price = float(trade.get("entry_price") or 0)
            exit_price = float(trade.get("exit_price") or 0)
            entry_total = float(
                trade.get("entry_total")
                or shares * entry_price + float(trade.get("entry_fee") or 0)
            )
            exit_proceeds = float(
                trade.get("exit_proceeds")
                or shares * exit_price - float(trade.get("exit_fee") or 0)
            )
            rows.append({
                "stock": f"{cls._compact_card_text(trade.get('name'), 12) or '--'}\n{trade.get('code') or '--'}",
                "dates": f"{entry_date[5:]}\n→ {exit_date[5:]}",
                "shares": f"{int(shares):,}",
                "prices": f"买 {entry_price:,.3f}\n卖 {exit_price:,.3f}",
                "cashflow": f"支 {entry_total:,.2f}\n收 {exit_proceeds:,.2f}",
                "return": (
                    f"{cls._format_change(trade.get('return_pct'))}\n"
                    f"{float(trade.get('pnl') or 0):+,.0f}元"
                ),
                "exit": (
                    f"{int(trade.get('holding_days') or 0)}日\n"
                    f"{cls._strategy_exit_reason_label(trade.get('exit_reason'))}"
                ),
            })
        elements: List[Dict[str, Any]] = [
            {
                "tag": "markdown",
                "content": (
                    f"<font color='grey'>{period.get('start') or '--'} 至 {period.get('end') or '--'} | "
                    f"第 {page}/{pages} 页，共 {total} 笔</font>\n"
                    "信号在收盘确认，买卖统一于下一交易日开盘集合竞价近似成交。"
                ),
            },
            cls._build_table(
                f"strategy_backtest_trade_page_{page}",
                [
                    {"name": "stock", "display_name": "标的", "data_type": "text", "width": "20%", "vertical_align": "top"},
                    {"name": "dates", "display_name": "买→卖日", "data_type": "text", "width": "15%", "vertical_align": "top"},
                    {"name": "shares", "display_name": "股数", "data_type": "text", "width": "9%", "horizontal_align": "right", "vertical_align": "top"},
                    {"name": "prices", "display_name": "买/卖价", "data_type": "text", "width": "15%", "horizontal_align": "right", "vertical_align": "top"},
                    {"name": "cashflow", "display_name": "支出/回款", "data_type": "text", "width": "19%", "horizontal_align": "right", "vertical_align": "top"},
                    {"name": "return", "display_name": "净收益", "data_type": "text", "width": "12%", "horizontal_align": "right", "vertical_align": "top"},
                    {"name": "exit", "display_name": "持有/退出", "data_type": "text", "width": "10%", "vertical_align": "top"},
                ],
                rows,
            ),
            {
                "tag": "markdown",
                "content": (
                    "<font color='grey'>支出=买入成交额+买入佣金；回款=卖出成交额-卖出佣金-印花税。"
                    "逐笔净收益已计滑点；研究回测，不构成投资建议。</font>"
                ),
            },
        ]
        return {
            "schema": "2.0",
            "config": {"update_multi": True},
            "header": {
                "template": "wathet",
                "title": {"tag": "plain_text", "content": f"Martin逐笔交易（含金额） {page}/{pages}"},
            },
            "body": {"direction": "vertical", "padding": "12px 12px 12px 12px", "elements": elements},
        }

    @classmethod
    def _strategy_backtest_month_rows(cls, result: Dict[str, Any]) -> List[Dict[str, str]]:
        curve = result.get("equity_curve") if isinstance(result.get("equity_curve"), list) else []
        config = result.get("config") if isinstance(result.get("config"), dict) else {}
        previous = float(config.get("initial_cash") or 0)
        month_ends: Dict[str, float] = {}
        for row in curve:
            if not isinstance(row, dict):
                continue
            date_text = str(row.get("date") or "")
            try:
                month_ends[date_text[:7]] = float(row.get("equity"))
            except (TypeError, ValueError):
                continue
        rows: List[Dict[str, str]] = []
        for month, equity in month_ends.items():
            month_return = (equity / previous - 1.0) * 100.0 if previous > 0 else None
            rows.append({
                "month": month,
                "return": cls._format_change(month_return),
                "equity": f"{equity:,.0f}元",
            })
            previous = equity
        return rows

    @classmethod
    def _strategy_backtest_exit_rows(cls, result: Dict[str, Any]) -> List[Dict[str, str]]:
        trades = result.get("trades") if isinstance(result.get("trades"), list) else []
        grouped: Dict[str, Dict[str, float]] = {}
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            reason = str(trade.get("exit_reason") or "unknown")
            item = grouped.setdefault(reason, {"count": 0.0, "pnl": 0.0})
            item["count"] += 1
            try:
                item["pnl"] += float(trade.get("pnl") or 0)
            except (TypeError, ValueError):
                pass
        return [
            {
                "reason": cls._strategy_exit_reason_label(reason),
                "count": str(int(values["count"])),
                "pnl": f"{values['pnl']:+,.0f}元",
            }
            for reason, values in grouped.items()
        ]

    @staticmethod
    def _strategy_exit_reason_label(reason: Any) -> str:
        labels = {
            "close_below_initial_stop": "初始防守",
            "close_below_ema21": "跌破EMA21",
            "close_below_breakeven": "保本防线",
            "trailing_close_drawdown": "移动回撤",
            "no_follow_through": "10日未跟随",
            "max_holding_days": "研究持仓上限",
            "period_end_mark": "期末估值",
        }
        value = str(reason or "unknown")
        return labels.get(value, value)

    @staticmethod
    def _is_cn_market_review_payload(payload: Optional[Dict[str, Any]]) -> bool:
        return bool(
            isinstance(payload, dict)
            and payload.get("kind") == "market_review"
            and str(payload.get("region") or "").strip().lower() == "cn"
        )

    @staticmethod
    def _is_screening_result_payload(payload: Optional[Dict[str, Any]]) -> bool:
        return bool(isinstance(payload, dict) and payload.get("kind") == "screening_result")

    @staticmethod
    def _is_strategy_backtest_payload(payload: Optional[Dict[str, Any]]) -> bool:
        return bool(isinstance(payload, dict) and payload.get("kind") == "strategy_backtest")

    @staticmethod
    def _is_strategy_backtest_analysis_payload(payload: Optional[Dict[str, Any]]) -> bool:
        return bool(
            isinstance(payload, dict)
            and payload.get("kind") == "strategy_backtest_analysis"
        )

    @staticmethod
    def _is_strategy_backtest_trades_payload(payload: Optional[Dict[str, Any]]) -> bool:
        return bool(
            isinstance(payload, dict)
            and payload.get("kind") == "strategy_backtest_trades"
        )

    @classmethod
    def _build_structured_card_body(
        cls,
        payload: Optional[Dict[str, Any]],
        content: str,
    ) -> dict:
        if cls._is_cn_market_review_payload(payload):
            return cls._build_market_review_card_body(payload or {}, content)
        if cls._is_screening_result_payload(payload):
            return cls._build_screening_card_body(payload or {}, content)
        if cls._is_strategy_backtest_payload(payload):
            return cls._build_strategy_backtest_card_body(payload or {}, content)
        if cls._is_strategy_backtest_analysis_payload(payload):
            return cls._build_strategy_backtest_analysis_card_body(payload or {}, content)
        if cls._is_strategy_backtest_trades_payload(payload):
            return cls._build_strategy_backtest_trades_card_body(payload or {}, content)
        return cls._build_card_body(content)

    # ------------------------------------------------------------------
    # Webhook helpers (unchanged legacy path)
    # ------------------------------------------------------------------

    def _get_keyword_prefix(self) -> str:
        if not self._feishu_keyword:
            return ""
        return f"{self._feishu_keyword}\n"

    def _apply_keyword_prefix(self, content: str) -> str:
        prefix = self._get_keyword_prefix()
        if not prefix:
            return content
        return f"{prefix}{content}" if content else self._feishu_keyword

    def _build_security_fields(self) -> Dict[str, str]:
        if not self._feishu_secret:
            return {}
        timestamp = str(int(time.time()))
        string_to_sign = f"{timestamp}\n{self._feishu_secret}"
        sign = base64.b64encode(
            hmac.new(
                string_to_sign.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        return {"timestamp": timestamp, "sign": sign}

    # ------------------------------------------------------------------
    # App Bot client (lazy, thread-safe)
    # ------------------------------------------------------------------

    def _ensure_app_client(self) -> Any:
        """Lazily initialise the ``lark-oapi`` client for App Bot mode."""
        if self._app_client is not _NO_CLIENT:
            return self._app_client
        with self._app_client_lock:
            if self._app_client is not _NO_CLIENT:
                return self._app_client
            if not FEISHU_SDK_AVAILABLE or not _load_feishu_sdk():
                logger.warning(
                    "飞书 App Bot 需要 lark-oapi 库；标准安装请运行: pip install -r requirements.txt"
                )
                self._app_client = None
                return None
            if not self._feishu_app_id or not self._feishu_app_secret:
                missing = []
                if not self._feishu_app_id:
                    missing.append("FEISHU_APP_ID")
                if not self._feishu_app_secret:
                    missing.append("FEISHU_APP_SECRET")
                logger.warning("飞书 App Bot 凭据不全，缺少: %s", ", ".join(missing))
                self._app_client = None
                return None
            try:
                self._app_client = (
                    _lark.Client.builder()
                    .app_id(self._feishu_app_id)
                    .app_secret(self._feishu_app_secret)
                    .domain(self._feishu_domain)
                    .log_level(_lark.LogLevel.WARNING)
                    .build()
                )
                logger.info("飞书 App Bot 客户端初始化成功 (domain=%s)", self._feishu_domain)
            except Exception as e:
                logger.error("飞书 App Bot 客户端初始化失败: %s", e)
                self._app_client = None
            return self._app_client

    # ------------------------------------------------------------------
    # App Bot send helpers
    # ------------------------------------------------------------------

    def _send_via_app_bot(
        self,
        content: str,
        *,
        structured_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send message through the Feishu App Bot, chunking if necessary."""
        if not self._feishu_chat_id:
            logger.warning("FEISHU_CHAT_ID 未配置，跳过 App Bot 推送")
            return False

        client = self._ensure_app_client()
        if client is None:
            return False

        formatted = format_feishu_markdown(content)
        content_bytes = len(formatted.encode("utf-8"))

        if content_bytes > self._feishu_max_bytes:
            logger.info(
                "App Bot 消息超长 (%d 字节)，将分批发送", content_bytes
            )
            return self._app_send_chunked(client, formatted)

        return self._app_send_once(client, formatted, structured_payload=structured_payload)

    def _app_send_chunked(self, client: Any, content: str) -> bool:
        """Chunk and send long content through App Bot."""
        try:
            chunks = chunk_content_by_max_bytes(
                content, self._feishu_max_bytes, add_page_marker=True
            )
        except (ValueError, TypeError, Exception) as e:
            logger.error("App Bot 分片失败: %s", e)
            return False

        success = True
        for i, chunk in enumerate(chunks):
            ok = self._app_send_once(client, chunk)
            if not ok:
                logger.error("App Bot 第 %d/%d 批发送失败", i + 1, len(chunks))
                success = False
            if i < len(chunks) - 1:
                time.sleep(1)
        return success

    def _app_send_once(
        self,
        client: Any,
        content: str,
        *,
        structured_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Single-shot send via App Bot with card-first / text-fallback.

        Content received here has already been through ``format_feishu_markdown``
        which converts all Markdown constructs to ``lark_md``-compatible format.
        The interactive card uses ``tag: lark_md`` for rendering.
        """
        card_body = self._build_structured_card_body(structured_payload, content)
        card_payload = json.dumps(card_body, ensure_ascii=False)

        if self._app_send_raw(client, "interactive", card_payload):
            return True

        # Fallback to plain text.
        text_payload = json.dumps({"text": content}, ensure_ascii=False)
        return self._app_send_raw(client, "text", text_payload)

    def _app_send_raw(self, client: Any, msg_type: str, content_json: str) -> bool:
        """Low-level send via lark-oapi SDK with retry and idempotency UUID.

        Request construction is done once outside the retry loop; it is
        deterministic and a construction error is a programming error, not
        a transient failure.
        """
        if client is None:
            return False

        send_uuid = str(uuid_mod.uuid4())
        try:
            req = (
                CreateMessageRequest.builder()
                .receive_id_type(self._feishu_receive_id_type)
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(self._feishu_chat_id)
                    .content(content_json)
                    .msg_type(msg_type)
                    .uuid(send_uuid)
                    .build()
                )
                .build()
            )
        except Exception as e:
            logger.error("App Bot 请求构建失败: %s: %s", type(e).__name__, e)
            return False

        last_status: Optional[str] = None

        for attempt in range(_APP_SEND_RETRIES):
            try:
                resp = client.im.v1.message.create(req)
            except Exception as e:
                logger.warning(
                    "App Bot 发送异常 (attempt=%d/%d): %s: %s",
                    attempt + 1, _APP_SEND_RETRIES, type(e).__name__, e,
                )
                if attempt < _APP_SEND_RETRIES - 1:
                    time.sleep(
                        _APP_SEND_BACKOFF_SECONDS[
                            min(attempt, len(_APP_SEND_BACKOFF_SECONDS) - 1)
                        ]
                    )
                continue

            if resp.success():
                logger.info("App Bot 消息发送成功 (type=%s)", msg_type)
                return True

            try:
                log_id = resp.get_log_id()
            except (AttributeError, Exception):
                log_id = "N/A"
            status = "code=%s, msg=%s, log_id=%s" % (
                resp.code, resp.msg, log_id,
            )
            last_status = status
            logger.warning(
                "App Bot 发送失败 (attempt=%d/%d): %s",
                attempt + 1, _APP_SEND_RETRIES, status,
            )

            if attempt < _APP_SEND_RETRIES - 1:
                time.sleep(
                    _APP_SEND_BACKOFF_SECONDS[
                        min(attempt, len(_APP_SEND_BACKOFF_SECONDS) - 1)
                    ]
                )

        if last_status:
            logger.error("App Bot 发送最终失败: %s", last_status)
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_to_feishu(
        self,
        content: str,
        *,
        timeout_seconds: Optional[float] = None,
        structured_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Push a message to Feishu.

        Routing priority:
          1. **Webhook** – when ``feishu_webhook_url`` is configured.
          2. **App Bot** – when ``feishu_app_id`` + ``feishu_app_secret``
             + ``feishu_chat_id`` are all configured and webhook is absent.

        Returns:
            Whether the send succeeded.
        """
        if content is None:
            logger.error("send_to_feishu: content 不能为 None")
            return False
        sanitized_content = strip_hidden_markdown_metadata(content).strip()
        if self._feishu_url:
            if structured_payload is None:
                return self._send_via_webhook(
                    sanitized_content,
                    timeout_seconds=timeout_seconds,
                )
            return self._send_via_webhook(
                sanitized_content,
                timeout_seconds=timeout_seconds,
                structured_payload=structured_payload,
            )
        if structured_payload is None:
            return self._send_via_app_bot(sanitized_content)
        return self._send_via_app_bot(sanitized_content, structured_payload=structured_payload)

    def send_feishu_file(self, file_path: str) -> bool:
        """
        Upload and send a file to the Feishu chat.

        .. note::

           * **App Bot mode** – uploads the file via the lark-oapi SDK and
             sends it as a file message.  This is the recommended path.
           * **Webhook mode** – reads the file content and sends it as a
             regular text/card message (webhooks do not support file upload).

        Args:
            file_path: Absolute or relative path to the local file.

        Returns:
            Whether the send succeeded.
        """
        path = Path(file_path)
        if not path.is_file():
            logger.error("send_feishu_file: 文件不存在: %s", file_path)
            return False

        upload_path, display_name = self._prepare_sanitized_file(path)
        try:
            if self._feishu_url:
                # Webhook mode: send file content as a message (best-effort).
                return self._send_file_via_webhook(upload_path, display_name=display_name)

            # App Bot mode: upload file via SDK.
            return self._send_file_via_app_bot(upload_path, display_name=display_name)
        finally:
            if upload_path != path:
                try:
                    upload_path.unlink()
                except OSError:
                    logger.warning("清理飞书临时文件失败: %s", upload_path)

    @staticmethod
    def _prepare_sanitized_file(path: Path) -> Tuple[Path, str]:
        display_name = path.name
        if path.suffix.lower() not in {".md", ".markdown", ".txt"}:
            return path, display_name

        try:
            raw_text = path.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            return path, display_name
        if not isinstance(raw_text, str):
            return path, display_name

        sanitized_text = strip_hidden_markdown_metadata(raw_text).strip()
        if sanitized_text == raw_text:
            return path, display_name

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=path.suffix,
            prefix="feishu_sanitized_",
            delete=False,
        )
        with temp_file:
            temp_file.write(sanitized_text)
        return Path(temp_file.name), display_name

    def _send_file_via_app_bot(self, path: Path, *, display_name: Optional[str] = None) -> bool:
        """Upload *path* to Feishu via App Bot SDK and send as file message."""
        if not self._feishu_chat_id:
            logger.warning("FEISHU_CHAT_ID 未配置，跳过 App Bot 文件推送")
            return False

        client = self._ensure_app_client()
        if client is None:
            return False
        if not FEISHU_FILE_SDK_AVAILABLE:
            logger.warning("lark-oapi SDK does not support file upload; upgrade lark-oapi")
            return False

        file_name = display_name or path.name
        # Determine file_type from extension; fall back to "stream" for unknown types.
        feishu_file_types = {
            ".opus": "opus", ".aac": "aac", ".amr": "amr", ".mp3": "mp3",
            ".wma": "wma", ".pcm": "pcm", ".wav": "wav",
            ".mp4": "mp4", ".gif": "gif",
            ".pdf": "pdf",
            ".doc": "doc", ".docx": "docx",
            ".xls": "xls", ".xlsx": "xlsx",
            ".ppt": "ppt", ".pptx": "pptx",
        }
        file_type = feishu_file_types.get(path.suffix.lower(), "stream")

        try:
            with path.open("rb") as f:
                body = (
                    _CreateFileRequestBody.builder()
                    .file_type(file_type)
                    .file_name(file_name)
                    .file(f)  # type: ignore[arg-type]
                    .build()
                )
                req = (
                    _CreateFileRequest.builder()
                    .request_body(body)
                    .build()
                )
                resp = client.im.v1.file.create(req)
        except Exception as e:
            logger.error("App Bot 文件上传异常: %s: %s", type(e).__name__, e)
            return False

        if not resp.success():
            try:
                log_id = resp.get_log_id()
            except (AttributeError, Exception):
                log_id = "N/A"
            logger.error(
                "App Bot 文件上传失败: code=%s, msg=%s, log_id=%s",
                resp.code, resp.msg, log_id,
            )
            return False

        file_key = resp.data.file_key if resp.data else None
        if not file_key:
            logger.error("App Bot 文件上传成功但未返回 file_key")
            return False

        logger.info("App Bot 文件上传成功: file_key=%s, file_name=%s", file_key, file_name)

        # Send a file message with the uploaded file_key.
        content_json = json.dumps({"file_key": file_key})
        return self._app_send_raw(client, "file", content_json)

    @staticmethod
    def _guess_mime_for_webhook(path: Path) -> str:
        """Determine a human-readable label for webhook fallback."""
        suffix = path.suffix.lower()
        labels = {".md": "Markdown", ".txt": "文本", ".pdf": "PDF", ".csv": "CSV"}
        return labels.get(suffix, suffix.lstrip(".").upper() or "文件")

    def _send_file_via_webhook(self, path: Path, *, display_name: Optional[str] = None) -> bool:
        """Send file *content* as a Feishu message (webhook fallback)."""
        try:
            text = path.read_text("utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.error("读取文件内容失败 (webhook fallback): %s: %s", type(e).__name__, e)
            return False

        file_label = self._guess_mime_for_webhook(path)
        header = f"**📄 {file_label} 文件内容: {display_name or path.name}**\n\n"
        content = header + strip_hidden_markdown_metadata(text).strip()
        # Delegate to the existing webhook send path.
        return self._send_via_webhook(content)

    # ------------------------------------------------------------------
    # Webhook path (legacy, unchanged)
    # ------------------------------------------------------------------

    def _send_via_webhook(
        self,
        content: str,
        *,
        timeout_seconds: Optional[float] = None,
        structured_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Legacy webhook send path."""
        formatted_content = format_feishu_markdown(content)

        max_bytes = self._feishu_max_bytes
        keyword_overhead = len(self._get_keyword_prefix().encode("utf-8"))
        effective_max_bytes = max_bytes - keyword_overhead

        if effective_max_bytes <= 0:
            logger.error("飞书关键词过长，超过单条消息允许的最大字节数，无法发送")
            return False

        content_bytes = len(formatted_content.encode("utf-8")) + keyword_overhead
        if content_bytes > max_bytes:
            min_chunk_bytes = MIN_MAX_BYTES + PAGE_MARKER_SAFE_BYTES
            if effective_max_bytes < min_chunk_bytes:
                logger.error(
                    "飞书关键词过长，剩余分片预算(%s字节)不足以安全分页发送，至少需要 %s 字节",
                    effective_max_bytes,
                    min_chunk_bytes,
                )
                return False
            logger.info("飞书消息内容超长(%d字节/%d字符)，将分批发送", content_bytes, len(content))
            return self._send_feishu_chunked(formatted_content, effective_max_bytes)

        try:
            if structured_payload is None:
                return self._send_feishu_message(formatted_content, timeout_seconds=timeout_seconds)
            return self._send_feishu_message(
                formatted_content,
                timeout_seconds=timeout_seconds,
                structured_payload=structured_payload,
            )
        except Exception as e:
            logger.error("发送飞书消息失败: %s", e)
            return False

    def _send_feishu_chunked(self, content: str, max_bytes: int) -> bool:
        try:
            chunks = chunk_content_by_max_bytes(content, max_bytes, add_page_marker=True)
        except ValueError as e:
            logger.error("飞书消息分片失败，单片预算不足以安全分页（关键词过长或 max_bytes 过小）: %s", e)
            return False

        total_chunks = len(chunks)
        success_count = 0
        logger.info("飞书分批发送：共 %d 批", total_chunks)
        for i, chunk in enumerate(chunks):
            try:
                if self._send_feishu_message(chunk):
                    success_count += 1
                    logger.info("飞书第 %d/%d 批发送成功", i + 1, total_chunks)
                else:
                    logger.error("飞书第 %d/%d 批发送失败", i + 1, total_chunks)
            except Exception as e:
                logger.error("飞书第 %d/%d 批发送异常: %s", i + 1, total_chunks, e)
            if i < total_chunks - 1:
                time.sleep(1)
        return success_count == total_chunks

    def _send_feishu_message(
        self,
        content: str,
        *,
        timeout_seconds: Optional[float] = None,
        structured_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a single Feishu webhook message (interactive card, fallback text)."""
        prepared_content = self._apply_keyword_prefix(content)
        security_fields = self._build_security_fields()

        def _post_payload(payload: Dict[str, Any]) -> bool:
            request_payload = dict(payload)
            request_payload.update(security_fields)
            try:
                response = requests.post(
                    self._feishu_url,
                    json=request_payload,
                    timeout=timeout_seconds or _WEBHOOK_SEND_TIMEOUT_SECONDS,
                    verify=self._webhook_verify_ssl,
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                logger.error("飞书 Webhook 网络请求异常: %s", e)
                return False
            if response.status_code == 200:
                try:
                    result = response.json()
                except (ValueError, AttributeError):
                    logger.error("飞书 Webhook 返回非 JSON 响应: %s", response.text[:200])
                    return False
                if not isinstance(result, dict):
                    logger.error("飞书 Webhook 返回非预期格式: %s", type(result).__name__)
                    return False
                code = result.get("code") if "code" in result else result.get("StatusCode")
                if code == 0:
                    logger.info("飞书 Webhook 消息发送成功")
                    return True
                logger.error(
                    "飞书 Webhook 返回错误 [code=%s]: %s",
                    code,
                    result.get("msg") or result.get("StatusMessage", "未知错误"),
                )
                return False
            logger.error("飞书 Webhook 请求失败: HTTP %d", response.status_code)
            return False

        card_body = self._build_structured_card_body(structured_payload, prepared_content)
        card_payload = {"msg_type": "interactive", "card": card_body}

        if _post_payload(card_payload):
            return True

        text_payload = {
            "msg_type": "text",
            "content": {"text": prepared_content},
        }
        return _post_payload(text_payload)
