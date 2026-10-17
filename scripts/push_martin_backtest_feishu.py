#!/usr/bin/env python3
"""Push a persisted Martin strategy backtest as a Feishu card."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.notification_sender.feishu_sender import FeishuSender  # noqa: E402


DEFAULT_BACKTEST_DIR = ROOT / "data" / "screening" / "backtests"


def _latest_report_json() -> Path:
    candidates = sorted(
        DEFAULT_BACKTEST_DIR.glob("martin_backtest_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No Martin backtest JSON found under {DEFAULT_BACKTEST_DIR}")
    return candidates[0]


def _load_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError(f"Invalid Martin backtest JSON: {path}")
    payload["kind"] = "strategy_backtest"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Backtest JSON path; defaults to the newest martin_backtest_*.json artifact.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Send only the overview card instead of the detailed card set.",
    )
    parser.add_argument(
        "--trades-per-card",
        type=int,
        default=14,
        help="Number of trade-ledger rows per detail card.",
    )
    args = parser.parse_args()

    report_path = (args.report_json or _latest_report_json()).resolve()
    payload = _load_payload(report_path)
    sender = FeishuSender(Config.get_instance())
    if not sender.send_to_feishu("Martin A股策略回测", structured_payload=payload):
        print("Feishu backtest card send failed.", file=sys.stderr)
        return 1
    sent_cards = 1
    if not args.summary_only:
        time.sleep(0.8)
        analysis_payload = dict(payload)
        analysis_payload["kind"] = "strategy_backtest_analysis"
        if not sender.send_to_feishu("Martin回测详细诊断", structured_payload=analysis_payload):
            print("Feishu backtest analysis card send failed.", file=sys.stderr)
            return 1
        sent_cards += 1

        strategy_result = payload.get("results", {}).get("strategy", {})
        trades = strategy_result.get("trades") if isinstance(strategy_result, dict) else []
        trades = trades if isinstance(trades, list) else []
        per_card = max(int(args.trades_per_card), 1)
        pages = max((len(trades) + per_card - 1) // per_card, 1)
        for page_index in range(pages):
            page_trades = trades[page_index * per_card : (page_index + 1) * per_card]
            if not page_trades:
                break
            time.sleep(0.8)
            trade_payload = {
                "kind": "strategy_backtest_trades",
                "period": payload.get("period", {}),
                "page": page_index + 1,
                "pages": pages,
                "total_trades": len(trades),
                "trades": page_trades,
            }
            if not sender.send_to_feishu(
                f"Martin逐笔交易 {page_index + 1}/{pages}",
                structured_payload=trade_payload,
            ):
                print(
                    f"Feishu trade card {page_index + 1}/{pages} send failed.",
                    file=sys.stderr,
                )
                return 1
            sent_cards += 1
    print(f"Feishu backtest cards sent ({sent_cards}): {report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
