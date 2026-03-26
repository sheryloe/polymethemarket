from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from polymethemoney.config import Settings

logger = logging.getLogger(__name__)


class TelegramBotService:
    def __init__(
        self,
        settings: Settings,
        runtime_state,
        store,
        execution_engine,
        risk_engine,
        gatekeeper,
        reporter,
    ) -> None:
        self.settings = settings
        self.runtime_state = runtime_state
        self.store = store
        self.execution_engine = execution_engine
        self.risk_engine = risk_engine
        self.gatekeeper = gatekeeper
        self.reporter = reporter
        self._token = (settings.telegram_bot_token or "").strip()
        self._chat_id = (settings.telegram_chat_id or "").strip()
        self._api_base = f"https://api.telegram.org/bot{self._token}"
        self._offset = 0

    async def run(self) -> None:
        if not self._token or not self._chat_id:
            logger.warning("telegram disabled: missing token/chat id")
            return

        async with httpx.AsyncClient(timeout=35) as client:
            while True:
                try:
                    updates = await self._get_updates(client)
                    for update in updates:
                        await self._handle_update(client, update)
                except Exception:
                    logger.exception("telegram polling failed")
                    await asyncio.sleep(3)

    async def notify(self, text: str) -> None:
        if not self._token or not self._chat_id:
            return
        if self._should_rewrite_report(text):
            text = await self._build_report_card(self.settings.report_interval_minutes)
        async with httpx.AsyncClient(timeout=15) as client:
            await self._send_message(client, text)

    async def _get_updates(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        payload = {
            "offset": self._offset,
            "timeout": 25,
            "allowed_updates": ["message"],
        }
        resp = await client.post(f"{self._api_base}/getUpdates", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            return []
        return data.get("result", [])

    async def _handle_update(self, client: httpx.AsyncClient, update: dict[str, Any]) -> None:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            self._offset = max(self._offset, update_id + 1)

        message = update.get("message") or {}
        text = (message.get("text") or "").strip()
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id") or "")
        if not text or chat_id != self._chat_id:
            return

        cmd, *rest = text.split(maxsplit=1)
        arg = rest[0] if rest else ""

        if cmd in ("/help", "/start"):
            await self._send_message(client, self._help_text())
            return
        if cmd == "/status":
            await self._send_message(client, await self._build_status_card())
            return
        if cmd in ("/report60", "/report"):
            await self._send_message(client, await self._build_report_card(60))
            return
        if cmd == "/report6h":
            await self._send_message(client, await self._build_report_card(360))
            return
        if cmd == "/pnl":
            await self._send_message(client, await self._build_pnl_card())
            return
        if cmd == "/positions":
            await self._send_message(client, await self._build_positions_card(arg))
            return

        await self._send_message(client, "지원하지 않는 명령입니다. /help")

    async def _send_message(self, client: httpx.AsyncClient, text: str) -> None:
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        await client.post(f"{self._api_base}/sendMessage", json=payload)

    def _help_text(self) -> str:
        return (
            "[명령어]\n"
            "/status - 요약 카드\n"
            "/report60 - 60분 요약\n"
            "/report6h - 6시간 요약\n"
            "/pnl - 손익 요약\n"
            "/positions [N] - 포지션 상세\n"
            "/help - 도움말"
        )

    async def _build_status_card(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        positions = await self._collect_positions()
        pending_count = await self._collect_pending_count()
        risk_line = await self._collect_risk_line()
        stats = await self._collect_signal_stats(60)
        fill_line = "체결률 N/A"
        reject_line = "거절: 없음"
        if stats is not None:
            signals = stats["signals"]
            fills = stats["fills"]
            fill_rate = fills / max(1, signals)
            fill_line = f"체결률 {fill_rate:.0%} ({fills}/{signals})"
            reject_line = f"거절: {self._format_top_rejects(stats.get('rejections', {}), 3)}"

        pos_total = len(positions)
        pos_pl = self._summarize_position_pnl(positions)
        mode = getattr(self.runtime_state, "trading_mode", "unknown")
        status = "운영중"
        if getattr(self.runtime_state, "paused", False) or getattr(self.runtime_state, "trading_paused", False):
            status = "일시정지"

        return (
            "[상태]\n"
            f"시각 {now}\n"
            f"모드 {mode} | 상태 {status}\n"
            f"포지션 {pos_total}개 (+{pos_pl['pos']}/0{pos_pl['flat']}/-{pos_pl['neg']}) | 승인대기 {pending_count}건\n"
            f"{risk_line}\n"
            f"{fill_line}\n"
            f"{reject_line}"
        )

    async def _build_report_card(self, window_minutes: int) -> str:
        stats = await self._collect_signal_stats(window_minutes)
        if stats is None:
            return "[요약]\n데이터 없음"

        signals = stats["signals"]
        fills = stats["fills"]
        fill_rate = fills / max(1, signals)
        pnl = stats.get("pnl_usd")
        rejections = stats.get("rejections", {})

        top_rejects = self._format_top_rejects(rejections, 3)
        pnl_line = "손익 N/A" if pnl is None else f"손익 {pnl:+.2f} USD"

        return (
            f"[요약 {window_minutes}m]\n"
            f"신호 {signals} | 체결 {fills} ({fill_rate:.0%})\n"
            f"{pnl_line}\n"
            f"거절: {top_rejects}"
        )

    async def _build_pnl_card(self) -> str:
        stats = await self._collect_signal_stats(self.settings.report_interval_minutes)
        pnl = None if stats is None else stats.get("pnl_usd")
        if pnl is None:
            return "[손익]\n데이터 없음"
        return f"[손익]\n누적 {pnl:+.2f} USD"

    async def _build_positions_card(self, arg: str) -> str:
        positions = await self._collect_positions()
        count = len(positions)
        if count == 0:
            return "[포지션]\n없음"

        limit = self._safe_int(arg, 10)
        limit = max(1, min(limit, 20))
        lines = [f"[포지션] 총 {count}개 (상위 {limit})"]
        for idx, pos in enumerate(positions[:limit], 1):
            lines.append(self._format_position_line(idx, pos))
        return "\n".join(lines)

    async def _collect_signal_stats(self, window_minutes: int) -> dict[str, Any] | None:
        candidates = [
            ("get_report_window", {"window_minutes": window_minutes, "include_rejections": True}),
            ("get_recent_signal_stats", {"window_minutes": window_minutes}),
            ("get_signal_stats", {"window_minutes": window_minutes}),
        ]
        for name, kwargs in candidates:
            method = getattr(self.store, name, None)
            if method is None:
                continue
            result = method(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            parsed = self._parse_signal_stats_payload(result)
            if parsed is not None:
                return parsed
        return None

    async def _collect_positions(self) -> list[dict[str, Any]]:
        candidates = [
            ("get_open_positions", {}),
            ("list_open_positions", {}),
            ("get_positions", {"status": "open"}),
        ]
        for name, kwargs in candidates:
            method = getattr(self.store, name, None)
            if method is None:
                continue
            result = method(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, list):
                return result
        return []

    async def _collect_pending_count(self) -> int:
        candidates = [
            ("get_pending_signals", {}),
            ("list_pending_signals", {}),
            ("get_pending_approvals", {}),
        ]
        for name, kwargs in candidates:
            method = getattr(self.store, name, None)
            if method is None:
                continue
            result = method(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, list):
                return len(result)
        return 0

    async def _collect_risk_line(self) -> str:
        daily_dd = None
        weekly_dd = None
        kill = None

        for attr in ("risk_state", "state"):
            state = getattr(self.risk_engine, attr, None)
            if isinstance(state, dict):
                daily_dd = state.get("daily_dd")
                weekly_dd = state.get("weekly_dd")
                kill = state.get("kill_switch")
                break

        if daily_dd is None or weekly_dd is None:
            method = getattr(self.risk_engine, "get_state", None) or getattr(self.risk_engine, "get_risk_state", None)
            if method is not None:
                result = method()
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, dict):
                    daily_dd = result.get("daily_dd")
                    weekly_dd = result.get("weekly_dd")
                    kill = result.get("kill_switch")

        dd_line = "DD N/A"
        if isinstance(daily_dd, (int, float)) and isinstance(weekly_dd, (int, float)):
            dd_line = f"DD {daily_dd:.2%}/{weekly_dd:.2%}"
        kill_line = "킬스위치 N/A"
        if isinstance(kill, bool):
            kill_line = "킬스위치 ON" if kill else "킬스위치 OFF"
        return f"{dd_line} | {kill_line}"

    def _parse_signal_stats_payload(self, payload: Any) -> dict[str, Any] | None:
        if payload is None:
            return None
        if isinstance(payload, dict):
            signals = int(payload.get("signals") or payload.get("signal_count") or 0)
            fills = int(payload.get("fills") or payload.get("fill_count") or 0)
            pnl = payload.get("pnl_usd") or payload.get("net_pnl_usd") or payload.get("realized_pnl_usd")
            rejections = payload.get("rejections") or payload.get("rejects") or {}
            if not isinstance(rejections, dict):
                rejections = {}
            return {"signals": signals, "fills": fills, "pnl_usd": pnl, "rejections": rejections}
        return None

    def _format_top_rejects(self, rejections: dict[str, Any], top_n: int) -> str:
        if not rejections:
            return "없음"
        items = []
        for key, val in rejections.items():
            try:
                count = int(val)
            except (TypeError, ValueError):
                count = 0
            items.append((key, count))
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:top_n]
        return ", ".join([f"{k} {v}" for k, v in items])

    def _format_position_line(self, idx: int, pos: dict[str, Any]) -> str:
        side = pos.get("side") or pos.get("outcome") or "N/A"
        size = pos.get("size_usd") or pos.get("notional_usd") or pos.get("size") or "N/A"
        entry = pos.get("entry_price") or pos.get("avg_entry_price") or pos.get("price")
        mark = pos.get("mark_price") or pos.get("current_price") or pos.get("last_price")
        pnl = pos.get("pnl_unrealized") or pos.get("pnl") or pos.get("unrealized_pnl")
        market_id = pos.get("market_id") or pos.get("market") or "-"
        title = pos.get("title") or pos.get("question") or ""

        entry_str = f"{entry:.4f}" if isinstance(entry, (int, float)) else "N/A"
        mark_str = f"{mark:.4f}" if isinstance(mark, (int, float)) else "N/A"
        pnl_str = f"{pnl:+.2f}" if isinstance(pnl, (int, float)) else "N/A"
        size_str = f"{size:.2f}" if isinstance(size, (int, float)) else str(size)

        suffix = f" | {market_id}"
        if title:
            suffix += f" | {title[:40]}"
        return f"{idx}. {side} {size_str} USD @ {entry_str} -> {mark_str} ({pnl_str}){suffix}"

    def _summarize_position_pnl(self, positions: list[dict[str, Any]]) -> dict[str, int]:
        pos = neg = flat = 0
        for item in positions:
            pnl = item.get("pnl_unrealized") or item.get("pnl") or item.get("unrealized_pnl")
            if not isinstance(pnl, (int, float)):
                flat += 1
                continue
            if pnl > 0:
                pos += 1
            elif pnl < 0:
                neg += 1
            else:
                flat += 1
        return {"pos": pos, "neg": neg, "flat": flat}

    def _safe_int(self, value: str, default: int) -> int:
        try:
            return int(value.strip())
        except (TypeError, ValueError, AttributeError):
            return default

    def _should_rewrite_report(self, text: str) -> bool:
        if not text:
            return False
        return "정기 리포트" in text or "상태 요약" in text or "리포트" in text
