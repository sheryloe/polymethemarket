from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx

from polymethemoney.config import Settings
from polymethemoney.domain import STRATEGY_EXPIRY_ANCHOR, STRATEGY_TAPE_RIDER, get_strategy_spec, iter_active_strategy_specs
from polymethemoney.services.execution_engine import ExecutionEngine
from polymethemoney.services.gatekeeper import Gatekeeper
from polymethemoney.services.reporter import ReporterService
from polymethemoney.services.risk_engine import RiskEngine
from polymethemoney.state import RuntimeState
from polymethemoney.storage import Store

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PositionView:
    index: int
    position_id: int
    market_id: str
    side: str
    size_usd: float
    entry_price: float
    mark_price: float
    unrealized_pnl_usd: float
    question: str
    strategy_id: str


class TelegramBotService:
    def __init__(
        self,
        settings: Settings,
        runtime_state: RuntimeState,
        store: Store,
        execution_engine: ExecutionEngine,
        risk_engine: RiskEngine,
        gatekeeper: Gatekeeper,
        reporter: ReporterService,
    ) -> None:
        self.settings = settings
        self.runtime_state = runtime_state
        self.store = store
        self.execution_engine = execution_engine
        self.risk_engine = risk_engine
        self.gatekeeper = gatekeeper
        self.reporter = reporter
        self._token = (self.settings.telegram_bot_token or "").strip()
        self._chat_id = int(self.settings.telegram_chat_id) if self.settings.telegram_chat_id else None
        self._base_url = f"https://api.telegram.org/bot{self._token}" if self._token else ""
        self._last_update_id = 0
        self._client: httpx.AsyncClient | None = None
        self._poll_timeout = 20
        self.enabled = bool(self._token and self._chat_id)
        self.instant_alerts_enabled = os.getenv("TELEGRAM_INSTANT_ALERTS_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._handlers: dict[str, Callable[[int, str], Awaitable[None]]] = {
            "help": self._cmd_help,
            "start": self._cmd_help,
            "status": self._cmd_status,
            "report30": self._cmd_report30,
            "report60": self._cmd_report60,
            "report6h": self._cmd_report6h,
            "positions": self._cmd_positions,
            "reset_paper": self._cmd_reset_paper,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "close": self._cmd_close,
            "close_market": self._cmd_close_market,
            "close_side": self._cmd_close_side,
            "closeall": self._cmd_closeall,
            "panic": self._cmd_panic,
        }

    async def run(self) -> None:
        if not self.enabled:
            logger.warning("Telegram disabled. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
            while True:
                await asyncio.sleep(3600)
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        while True:
            try:
                await self._poll_once()
            except Exception:
                logger.exception("telegram poll failed")
                await asyncio.sleep(2)

    async def notify(self, text: str) -> None:
        if not self.enabled or self._chat_id is None:
            logger.info("notify skipped: %s", text)
            return
        if not self.instant_alerts_enabled and not self._is_periodic_report(text):
            return
        await self._send_text(self._chat_id, text)

    async def _poll_once(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        params = {"timeout": self._poll_timeout, "allowed_updates": "message"}
        if self._last_update_id:
            params["offset"] = self._last_update_id + 1
        response = await self._client.get(f"{self._base_url}/getUpdates", params=params)
        payload = response.json()
        if not payload.get("ok"):
            return
        for update in payload.get("result", []):
            update_id = int(update.get("update_id", 0))
            if update_id > self._last_update_id:
                self._last_update_id = update_id
            message = update.get("message")
            if isinstance(message, dict):
                await self._handle_message(message)

    async def _handle_message(self, message: dict) -> None:
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if self._chat_id is None or chat_id != self._chat_id:
            return
        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return
        command, args = self._parse_command(text)
        handler = self._handlers.get(command)
        if handler is None:
            await self._send_text(chat_id, "[안내]\n지원하지 않는 명령입니다. /help 를 확인하세요.")
            return
        await handler(chat_id, args)

    async def _send_text(self, chat_id: int, text: str) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        await self._client.post(f"{self._base_url}/sendMessage", json={"chat_id": chat_id, "text": text})

    @staticmethod
    def _parse_command(text: str) -> tuple[str, str]:
        parts = text.split()
        command = parts[0].split("@", 1)[0].lstrip("/")
        args = " ".join(parts[1:]) if len(parts) > 1 else ""
        return command, args

    @staticmethod
    def _is_periodic_report(text: str) -> bool:
        return text.startswith(("[60분 누적]", "[30분 전략 비교]", "[6시간 누적]", "[상태 요약]"))

    async def _cmd_help(self, chat_id: int, args: str) -> None:
        await self._send_text(
            chat_id,
            "[도움말]\n"
            "/status : 현재 전략 상태 요약\n"
            "/report60 : 최근 60분 비교 리포트\n"
            "/report30 : 최근 30분 비교 리포트\n"
            "/report6h : 최근 6시간 비교 리포트\n"
            "/positions expiry_anchor_active [개수] : ExpiryAnchor 포지션 조회\n"
            "/positions tape_rider_shadow [개수] : TapeRider 포지션 조회\n"
            "/positions funded [개수] / /positions shadow [개수] 도 가능\n"
            "/reset_paper : funded/shadow 포지션과 런타임 상태 초기화\n"
            "/pause /resume : 신규 진입 일시중지 / 재개\n"
            "/close <position_id>\n"
            "/close_market <market_id>\n"
            "/close_side <YES|NO> [limit]\n"
            "/closeall\n"
            "/panic",
        )

    async def _cmd_status(self, chat_id: int, args: str) -> None:
        await self._send_text(chat_id, await self.reporter.build_status_text())

    async def _cmd_report30(self, chat_id: int, args: str) -> None:
        await self._send_text(chat_id, await self.reporter.build_report30_text())

    async def _cmd_report60(self, chat_id: int, args: str) -> None:
        await self._send_text(chat_id, await self.reporter.build_report60_text())

    async def _cmd_report6h(self, chat_id: int, args: str) -> None:
        await self._send_text(chat_id, await self.reporter.build_report6h_text())

    async def _cmd_positions(self, chat_id: int, args: str) -> None:
        strategy_id, limit = self._parse_positions_args(args)
        if strategy_id is None:
            await self._send_text(
                chat_id,
                "[사용법]\n/positions expiry_anchor_active [개수]\n/positions tape_rider_shadow [개수]\n예: /positions funded 10",
            )
            return
        views, unrealized_total = await self._build_position_views(strategy_id=strategy_id, limit=limit)
        spec = get_strategy_spec(strategy_id)
        if not views:
            await self._send_text(chat_id, f"[포지션]\n{spec.display_name} 현재 오픈 포지션이 없습니다.")
            return
        lines = [
            f"[포지션] {spec.display_name} | {self._mode_label(spec.execution_mode)} | 총 {len(views)}개 | 미실현 {unrealized_total:+.2f} USD",
        ]
        for view in views:
            question = view.question if len(view.question) <= 54 else f"{view.question[:51]}..."
            lines.append(
                f"{view.index}. id={view.position_id} {view.side} {view.size_usd:.2f} USD | {view.entry_price:.4f} -> {view.mark_price:.4f} | {view.unrealized_pnl_usd:+.2f}\n{view.market_id} | {question}"
            )
        await self._send_text(chat_id, "\n".join(lines))

    async def _cmd_reset_paper(self, chat_id: int, args: str) -> None:
        strategy_ids = [spec.strategy_id for spec in iter_active_strategy_specs()]
        reset = await self.store.reset_paper_open_positions(strategy_ids=strategy_ids)
        self.runtime_state.pending_approvals.clear()
        self.runtime_state.recent_signals.clear()
        self.runtime_state.recent_signal_candidate_sides.clear()
        self.runtime_state.recent_signal_candidate_rejects.clear()
        self.runtime_state.recent_approved_signal_sides.clear()
        self.runtime_state.recent_no_guard_rejects.clear()
        self.runtime_state.min_position_usd_override = None
        self.runtime_state.auto_tune_zero_fill_applied = False
        self.runtime_state.auto_tune_zero_fill_applied_at = None
        self.runtime_state.auto_tune_zero_fill_last_outcome = None
        self.runtime_state.auto_tune_zero_fill_last_checked_at = None
        self.runtime_state.paused = False
        self.runtime_state.manual_live_approved = False
        if hasattr(self.gatekeeper, "_local_trades"):
            self.gatekeeper._local_trades.clear()
        if hasattr(self.gatekeeper, "_local_violations"):
            self.gatekeeper._local_violations.clear()
        await self.risk_engine.refresh_state()
        await self._send_text(
            chat_id,
            "[초기화 완료]\n"
            f"포지션 종료 {reset['positions_closed']}건 | 구조 포지션 종료 {reset['structure_legs_closed']}건\n"
            "런타임 승인대기/최근 신호/튜닝 상태까지 초기화했습니다.",
        )

    async def _cmd_pause(self, chat_id: int, args: str) -> None:
        self.runtime_state.paused = True
        await self._send_text(chat_id, "[운영 제어]\n신규 진입을 일시중지했습니다.")

    async def _cmd_resume(self, chat_id: int, args: str) -> None:
        self.runtime_state.paused = False
        await self._send_text(chat_id, "[운영 제어]\n신규 진입을 재개했습니다.")

    async def _cmd_close(self, chat_id: int, args: str) -> None:
        position_id = self._parse_positive_int(args)
        if position_id is None:
            await self._send_text(chat_id, "[사용법]\n/close <position_id>")
            return
        result = await self.execution_engine.force_close_position(position_id)
        await self._send_text(chat_id, self._render_close_result("개별 포지션 강제 청산", result))

    async def _cmd_close_market(self, chat_id: int, args: str) -> None:
        market_id = (args or "").strip()
        if not market_id:
            await self._send_text(chat_id, "[사용법]\n/close_market <market_id>")
            return
        result = await self.execution_engine.force_close_market(market_id)
        await self._send_text(chat_id, self._render_close_result(f"시장 강제 청산 {market_id}", result))

    async def _cmd_close_side(self, chat_id: int, args: str) -> None:
        parts = (args or "").split()
        if not parts:
            await self._send_text(chat_id, "[사용법]\n/close_side <YES|NO> [limit]")
            return
        side = parts[0].upper()
        limit = self._parse_positive_int(parts[1]) if len(parts) > 1 else None
        if side not in {"YES", "NO"}:
            await self._send_text(chat_id, "[실패]\n방향은 YES 또는 NO 만 가능합니다.")
            return
        result = await self.execution_engine.force_close_side(side, limit=limit)
        await self._send_text(chat_id, self._render_close_result(f"방향 강제 청산 {side}", result))

    async def _cmd_closeall(self, chat_id: int, args: str) -> None:
        closed = await self.execution_engine.close_all_positions()
        await self._send_text(chat_id, f"[전체 청산 완료]\n오픈 포지션 {closed}건을 정리했습니다.")

    async def _cmd_panic(self, chat_id: int, args: str) -> None:
        result = await self.execution_engine.emergency_stop(flatten=True)
        flatten = result.get("flatten", {})
        await self._send_text(
            chat_id,
            "[긴급 정지]\n"
            f"paused={result.get('paused')} | 승인대기 정리 {result.get('rejected_pending', 0)}건\n"
            f"강제 청산 {flatten.get('closed', 0)}건 | 실현 {float(flatten.get('realized_pnl_usd', 0.0)):+.2f} USD",
        )

    async def _build_position_views(self, strategy_id: str, limit: int) -> tuple[list[PositionView], float]:
        rows = await self.store.get_open_positions(self.runtime_state.trading_mode, strategy_id=strategy_id)
        rows = sorted(rows, key=lambda row: getattr(row, "opened_at", None) or 0, reverse=True)[:limit]
        market_ids = [str(row.market_id) for row in rows]
        prices = await self.store.latest_market_prices(market_ids)
        questions = await self.store.market_questions(market_ids)
        views: list[PositionView] = []
        unrealized_total = 0.0
        for idx, row in enumerate(rows, start=1):
            side = str(row.side).upper()
            market_id = str(row.market_id)
            yes_price = prices.get(market_id, float(row.entry_price))
            mark_price = (1.0 - float(yes_price)) if side == "NO" else float(yes_price)
            mtm = Store.estimate_position_unrealized(
                size_usd=float(row.size_usd),
                entry_price=float(row.entry_price),
                mark_price=max(0.001, min(0.999, mark_price)),
                entry_fee_usd=float(getattr(row, "entry_fee_usd", 0.0) or 0.0),
                exit_fee_bps=float(self.settings.taker_fee_bps),
            )
            unrealized = float(mtm["net_unrealized_pnl_usd"])
            unrealized_total += unrealized
            views.append(
                PositionView(
                    index=idx,
                    position_id=int(row.id),
                    market_id=market_id,
                    side=side,
                    size_usd=float(row.size_usd),
                    entry_price=float(row.entry_price),
                    mark_price=float(mtm["mark_price"]),
                    unrealized_pnl_usd=unrealized,
                    question=questions.get(market_id, ""),
                    strategy_id=strategy_id,
                )
            )
        return views, unrealized_total

    def _parse_positions_args(self, args: str) -> tuple[str | None, int]:
        parts = (args or "").split()
        if not parts:
            return None, 10
        strategy_id = self._resolve_strategy_alias(parts[0])
        limit = self._parse_positive_int(parts[1]) if len(parts) > 1 else 10
        return strategy_id, limit or 10

    @staticmethod
    def _parse_positive_int(value: str | None, default: int | None = None) -> int | None:
        if value is None or str(value).strip() == "":
            return default
        try:
            parsed = int(str(value).strip())
        except ValueError:
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def _resolve_strategy_alias(value: str) -> str | None:
        token = (value or "").strip().lower()
        alias_map = {
            "expiry_anchor_active": STRATEGY_EXPIRY_ANCHOR,
            "expiryanchor": STRATEGY_EXPIRY_ANCHOR,
            "expiry": STRATEGY_EXPIRY_ANCHOR,
            "anchor": STRATEGY_EXPIRY_ANCHOR,
            "funded": STRATEGY_EXPIRY_ANCHOR,
            "tape_rider_shadow": STRATEGY_TAPE_RIDER,
            "taperider": STRATEGY_TAPE_RIDER,
            "tape": STRATEGY_TAPE_RIDER,
            "shadow": STRATEGY_TAPE_RIDER,
        }
        if token in alias_map:
            return alias_map[token]
        if token in {spec.strategy_id for spec in iter_active_strategy_specs()}:
            return token
        return None

    @staticmethod
    def _mode_label(execution_mode: str) -> str:
        return "funded" if execution_mode == "funded_paper" else "shadow"

    @staticmethod
    def _render_close_result(title: str, result: dict) -> str:
        return (
            f"[{title}]\n"
            f"요청 {int(result.get('requested', 0))}건 | 청산 {int(result.get('closed', 0))}건 | 누락 {int(result.get('missing', 0))}건 | 가격없음 {int(result.get('no_price', 0))}건\n"
            f"실현 {float(result.get('realized_pnl_usd', 0.0)):+.2f} USD"
        )
