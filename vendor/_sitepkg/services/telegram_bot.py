
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import timezone

import httpx

from polymethemoney.config import Settings
from polymethemoney.domain import TradingMode
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
        self.instant_alerts_enabled = (
            os.getenv("TELEGRAM_INSTANT_ALERTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        )

        self._handlers: dict[str, callable] = {
            "help": self._cmd_help,
            "start": self._cmd_help,
            "status": self._cmd_status,
            "report60": self._cmd_report60,
            "report6h": self._cmd_report6h,
            "positions": self._cmd_positions,
            "pending": self._cmd_pending,
            "pnl": self._cmd_pnl,
            "risk": self._cmd_risk,
            "gate": self._cmd_gate,
            "data": self._cmd_data,
            "config": self._cmd_config,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "go_live": self._cmd_go_live,
            "go_paper": self._cmd_go_paper,
            "set_minpos": self._cmd_set_minpos,
            "clear_minpos": self._cmd_clear_minpos,
            "reset_paper": self._cmd_reset_paper,
            "approve": self._cmd_approve,
            "reject": self._cmd_reject,
            "reject_all": self._cmd_reject_all,
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
        params = {
            "timeout": self._poll_timeout,
            "allowed_updates": "message",
        }
        if self._last_update_id:
            params["offset"] = self._last_update_id + 1
        resp = await self._client.get(f"{self._base_url}/getUpdates", params=params)
        payload = resp.json()
        if not payload.get("ok"):
            return
        for update in payload.get("result", []):
            update_id = int(update.get("update_id", 0))
            if update_id > self._last_update_id:
                self._last_update_id = update_id
            message = update.get("message")
            if not isinstance(message, dict):
                continue
            await self._handle_message(message)

    async def _handle_message(self, message: dict) -> None:
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if self._chat_id is None or chat_id != self._chat_id:
            return
        text = (message.get("text") or "").strip()
        if not text or not text.startswith("/"):
            return
        command, args = self._parse_command(text)
        handler = self._handlers.get(command)
        if handler is None:
            await self._send_text(chat_id, "[알림]\n알 수 없는 명령입니다. /help 를 확인해 주세요.")
            return
        await handler(chat_id, args)

    async def _send_text(self, chat_id: int, text: str) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        try:
            await self._client.post(
                f"{self._base_url}/sendMessage",
                json={"chat_id": chat_id, "text": text},
            )
        except Exception:
            logger.exception("telegram send failed")

    @staticmethod
    def _parse_command(text: str) -> tuple[str, str]:
        parts = text.split()
        cmd = parts[0].split("@", 1)[0].lstrip("/")
        args = " ".join(parts[1:]) if len(parts) > 1 else ""
        return cmd, args

    @staticmethod
    def _is_periodic_report(text: str) -> bool:
        if not text:
            return False
        prefixes = (
            "[10분 누적]",
            "[60분 누적]",
            "[6시간 누적]",
            "[상태 요약]",
        )
        return text.startswith(prefixes)

    async def _cmd_help(self, chat_id: int, args: str) -> None:
        await self._send_text(
            chat_id,
            "[봇 도움말]\n"
            "이 봇은 BTC 5분 Up/Down 페이퍼 운용을 텔레그램에서 제어합니다.\n\n"
            "모니터링\n"
            "/status : 현재 상태 요약(모드/오픈 포지션/손익/리스크)\n"
            "/report60 : 최근 60분 누적(신호/체결/거절 사유)\n"
            "/report6h : 최근 6시간 누적\n"
            "/positions [개수] : 오픈 포지션 상세(기본 10, 최대 40)\n"
            "/pending : 승인 대기 신호 목록\n"
            "/pnl : 오늘/이번주 실현 손익 요약\n"
            "/risk : 리스크 상태(DD)\n"
            "/gate : 게이트 통과 현황\n"
            "/data : 데이터 신선도(최근 tick/feature/signal)\n"
            "/config : 운영 설정 요약\n\n"
            "운영 제어\n"
            "/pause : 신규 진입 일시 중지\n"
            "/resume : 재개\n"
            "/go_paper : PAPER 모드로 강제 전환\n"
            "/go_live : 게이트 통과 시 LIVE 전환(하드락이면 차단)\n"
            "/set_minpos <usd> : 최소 진입 금액 임시 override\n"
            "/clear_minpos : 최소 진입 override 해제\n"
            "/reset_paper : PAPER 포지션 모두 닫고 상태 초기화\n\n"
            "승인/거절\n"
            "/approve <signal_id> : 승인 대기 신호 승인\n"
            "/reject <signal_id> : 승인 대기 신호 거절\n"
            "/reject_all : 승인 대기 전체 거절\n\n"
            "강제 청산\n"
            "/close <position_id> : 특정 포지션 강제 청산\n"
            "/close_market <market_id> : 특정 시장 전체 청산\n"
            "/close_side <YES|NO> [limit] : 방향 기준 일부/전체 청산\n"
            "/closeall : 오픈 포지션 전체 청산\n"
            "/panic : 긴급 중지(일시중지+전량 청산)\n\n"
            "예시\n"
            "/positions 20\n"
            "/close 123\n"
            "/close_market 0x1234abcd\n"
            "/close_side YES 5",
        )

    async def _cmd_status(self, chat_id: int, args: str) -> None:
        await self._send_text(chat_id, await self.reporter.build_status_text())

    async def _cmd_report60(self, chat_id: int, args: str) -> None:
        await self._send_text(chat_id, await self.reporter.build_report60_text())

    async def _cmd_report6h(self, chat_id: int, args: str) -> None:
        await self._send_text(chat_id, await self.reporter.build_report6h_text())

    async def _cmd_positions(self, chat_id: int, args: str) -> None:
        limit = self._parse_positive_int(args, default=10, max_value=40)
        if limit is None:
            await self._send_text(chat_id, "[사용법]\n/positions [개수]\n예: /positions 20")
            return

        views, unrealized_total = await self._build_position_views(limit=limit)
        if not views:
            await self._send_text(chat_id, "[포지션]\n현재 오픈된 포지션이 없습니다.")
            return

        pos_plus = sum(1 for v in views if v.unrealized_pnl_usd > 1e-6)
        pos_minus = sum(1 for v in views if v.unrealized_pnl_usd < -1e-6)
        pos_flat = len(views) - pos_plus - pos_minus
        lines = [
            f"[포지션 요약] 총 {len(views)}개",
            f"미실현 합계 {unrealized_total:+.2f} USD | +{pos_plus}/0:{pos_flat}/-{pos_minus}",
        ]
        for view in views:
            short_q = view.question if len(view.question) <= 44 else f"{view.question[:41]}..."
            lines.append(
                f"{view.index}. id={view.position_id} {view.side} {view.size_usd:.2f} USD "
                f"({view.entry_price:.4f} -> {view.mark_price:.4f}, {view.unrealized_pnl_usd:+.2f})\n"
                f"   {view.market_id} | {short_q}"
            )
        await self._send_text(chat_id, "\n".join(lines))

    async def _cmd_pending(self, chat_id: int, args: str) -> None:
        items = list(self.runtime_state.pending_approvals.items())
        if not items:
            await self._send_text(chat_id, "[승인 대기]\n현재 대기 중인 신호가 없습니다.")
            return
        lines = [f"[승인 대기] 총 {len(items)}건"]
        for idx, (signal_id, intent) in enumerate(items[:20], start=1):
            lines.append(
                f"{idx}. {signal_id}\n"
                f"   시장 {intent.market_id} | 방향 {intent.side.value} | 가격 {intent.price:.4f} | 금액 {intent.size_usd:.2f} USD"
            )
        if len(items) > 20:
            lines.append(f"... 외 {len(items) - 20}건")
        await self._send_text(chat_id, "\n".join(lines))

    async def _cmd_pnl(self, chat_id: int, args: str) -> None:
        snapshot = await self.store.status_snapshot()
        _, unrealized_total = await self._build_position_views(limit=10_000)
        est_total = float(snapshot["week_pnl"]) + unrealized_total
        await self._send_text(
            chat_id,
            "[손익 요약]\n"
            f"일간 실현 손익 {snapshot['day_pnl']:+.2f} USD\n"
            f"주간 실현 손익 {snapshot['week_pnl']:+.2f} USD\n"
            f"현재 미실현 손익 {unrealized_total:+.2f} USD\n"
            f"주간 실현+미실현 {est_total:+.2f} USD\n"
            f"오픈 포지션 {snapshot['open_positions']}개",
        )

    async def _cmd_risk(self, chat_id: int, args: str) -> None:
        state = await self.risk_engine.refresh_state()
        pause_text = "ON" if self.runtime_state.paused else "OFF"
        await self._send_text(
            chat_id,
            "[리스크 상태]\n"
            f"일시중지 {pause_text}\n"
            f"일간 DD {state.daily_drawdown_pct:.2%}\n"
            f"주간 DD {state.weekly_drawdown_pct:.2%}\n"
            f"한도(일/주) {self.settings.daily_loss_limit_pct:.2%} / {self.settings.weekly_loss_limit_pct:.2%}",
        )

    async def _cmd_gate(self, chat_id: int, args: str) -> None:
        await self._send_text(chat_id, f"[게이트]\n{await self.gatekeeper.summary_text()}")

    async def _cmd_data(self, chat_id: int, args: str) -> None:
        snapshot = await self.store.data_freshness_snapshot()
        latest_tick = self._fmt_ts(snapshot.get("latest_tick_ts"))
        latest_feature = self._fmt_ts(snapshot.get("latest_feature_ts"))
        latest_signal = self._fmt_ts(snapshot.get("latest_signal_ts"))
        ticks_5m = int(snapshot.get("ticks_5m") or 0)
        await self._send_text(
            chat_id,
            "[데이터 신선도]\n"
            f"최근 tick {latest_tick}\n"
            f"최근 feature {latest_feature}\n"
            f"최근 signal {latest_signal}\n"
            f"최근 5분 tick 수 {ticks_5m}건",
        )

    async def _cmd_config(self, chat_id: int, args: str) -> None:
        mode = self.runtime_state.trading_mode.value.upper()
        await self._send_text(
            chat_id,
            "[운영 설정]\n"
            f"모드 {mode} | DEMO_PAPER_HARDLOCK {self.settings.demo_paper_hardlock}\n"
            f"슬러그 프리픽스 {self.settings.contrarian_market_slug_prefix}\n"
            f"진입 간격 {self.settings.contrarian_entry_interval_seconds}s | 금액 {self.settings.contrarian_position_usd:.2f} USD\n"
            f"최대 포지션 {self.settings.contrarian_max_positions} | 만기 그레이스 {self.settings.contrarian_expiry_grace_seconds}s\n"
            f"리포트 주기 {self.settings.report_interval_minutes}분 | 시드 {self.settings.starting_capital_usd:.2f} USD",
        )

    async def _cmd_pause(self, chat_id: int, args: str) -> None:
        self.runtime_state.paused = True
        await self._send_text(chat_id, "[운영 제어]\n신규 진입을 일시 중지했습니다.")

    async def _cmd_resume(self, chat_id: int, args: str) -> None:
        self.runtime_state.paused = False
        await self._send_text(chat_id, "[운영 제어]\n거래를 재개했습니다.")

    async def _cmd_go_live(self, chat_id: int, args: str) -> None:
        if self.settings.demo_paper_hardlock:
            self.runtime_state.trading_mode = TradingMode.PAPER
            self.runtime_state.manual_live_approved = False
            await self._send_text(chat_id, "[모드 전환 불가]\nDEMO_PAPER_HARDLOCK이 켜져 있어 LIVE로 전환할 수 없습니다.")
            return

        hist_ok = await self.gatekeeper.historical_gate_passed()
        paper_ok = await self.gatekeeper.paper_gate_passed()
        if not (hist_ok and paper_ok):
            reasons: list[str] = []
            if not hist_ok:
                reasons.append("히스토리 게이트 미통과")
            if not paper_ok:
                reasons.append("페이퍼 게이트 미통과")
            reason_text = "\n".join(f"- {reason}" for reason in reasons)
            await self._send_text(
                chat_id,
                "[모드 전환 불가]\n"
                f"{reason_text}\n"
                f"{await self.gatekeeper.summary_text()}",
            )
            return

        self.runtime_state.trading_mode = TradingMode.LIVE
        self.runtime_state.manual_live_approved = True
        await self._send_text(chat_id, "[모드 전환]\nLIVE 모드로 전환했습니다.")

    async def _cmd_go_paper(self, chat_id: int, args: str) -> None:
        self.runtime_state.trading_mode = TradingMode.PAPER
        self.runtime_state.manual_live_approved = False
        await self._send_text(chat_id, "[모드 전환]\nPAPER 모드로 전환했습니다.")

    async def _cmd_set_minpos(self, chat_id: int, args: str) -> None:
        if not args:
            await self._send_text(chat_id, "[사용법]\n/set_minpos <usd>\n예: /set_minpos 1.5")
            return
        value = self._parse_positive_float(args)
        if value is None:
            await self._send_text(chat_id, "[실패]\n숫자를 입력해 주세요. 예: /set_minpos 1.5")
            return
        if value > float(self.settings.max_position_usd):
            await self._send_text(
                chat_id,
                f"[실패]\n입력 값이 MAX_POSITION_USD({self.settings.max_position_usd:.2f})를 초과합니다.",
            )
            return
        self.runtime_state.min_position_usd_override = float(value)
        await self._send_text(
            chat_id,
            "[설정 변경]\n"
            f"최소 진입 금액 override 적용: {self.runtime_state.min_position_usd_override:.2f} USD",
        )

    async def _cmd_clear_minpos(self, chat_id: int, args: str) -> None:
        self.runtime_state.min_position_usd_override = None
        await self._send_text(chat_id, "[설정 변경]\n최소 진입 금액 override를 해제했습니다.")

    async def _cmd_reset_paper(self, chat_id: int, args: str) -> None:
        reset = await self.store.reset_paper_open_positions()
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
            "PAPER 오픈 포지션을 모두 닫고 상태를 초기화했습니다.\n"
            f"닫은 포지션 {reset['positions_closed']}건, 구조물 레그 {reset['structure_legs_closed']}건.\n"
            "주의: DB 히스토리는 유지됩니다. 완전 초기화는 컨테이너 DB 리셋이 필요합니다.",
        )

    async def _cmd_approve(self, chat_id: int, args: str) -> None:
        signal_id = (args or "").strip()
        if not signal_id:
            await self._send_text(chat_id, "[사용법]\n/approve <signal_id>")
            return
        await self._send_text(chat_id, await self.execution_engine.approve_signal(signal_id))

    async def _cmd_reject(self, chat_id: int, args: str) -> None:
        signal_id = (args or "").strip()
        if not signal_id:
            await self._send_text(chat_id, "[사용법]\n/reject <signal_id>")
            return
        await self._send_text(chat_id, await self.execution_engine.reject_signal(signal_id))

    async def _cmd_reject_all(self, chat_id: int, args: str) -> None:
        count = await self.execution_engine.reject_all_pending()
        await self._send_text(chat_id, f"[승인 대기 정리]\n대기 신호 {count}건을 모두 거절했습니다.")

    async def _cmd_close(self, chat_id: int, args: str) -> None:
        position_id = self._parse_positive_int(args, default=None, max_value=10_000_000)
        if position_id is None:
            await self._send_text(chat_id, "[사용법]\n/close <position_id>\n예: /close 123")
            return
        result = await self.execution_engine.force_close_position(position_id)
        await self._send_text(chat_id, self._render_close_result("포지션 강제 청산", result))

    async def _cmd_close_market(self, chat_id: int, args: str) -> None:
        market_id = (args or "").strip()
        if not market_id:
            await self._send_text(chat_id, "[사용법]\n/close_market <market_id>")
            return
        result = await self.execution_engine.force_close_market(market_id)
        await self._send_text(chat_id, self._render_close_result(f"시장 강제 청산 ({market_id})", result))

    async def _cmd_close_side(self, chat_id: int, args: str) -> None:
        raw = (args or "").strip()
        if not raw:
            await self._send_text(chat_id, "[사용법]\n/close_side <YES|NO> [limit]\n예: /close_side YES 5")
            return
        parts = raw.split()
        side = parts[0].upper()
        if side not in {"YES", "NO"}:
            await self._send_text(chat_id, "[실패]\nside는 YES 또는 NO만 가능합니다.")
            return
        limit = None
        if len(parts) >= 2:
            parsed = self._parse_positive_int(parts[1], default=None, max_value=10_000)
            if parsed is None:
                await self._send_text(chat_id, "[실패]\nlimit은 양의 정수여야 합니다.")
                return
            limit = parsed
        result = await self.execution_engine.force_close_side(side, limit=limit)
        label = f"방향 강제 청산 ({side}" + ("" if limit is None else f", limit={limit}") + ")"
        await self._send_text(chat_id, self._render_close_result(label, result))

    async def _cmd_closeall(self, chat_id: int, args: str) -> None:
        mode = TradingMode.PAPER if self.settings.demo_paper_hardlock else self.runtime_state.trading_mode
        rows = await self.store.get_open_positions(mode)
        result = await self.execution_engine.force_close_positions(
            [int(row.id) for row in rows],
            reason="manual_close_all",
        )
        await self._send_text(chat_id, self._render_close_result("전체 강제 청산", result))

    async def _cmd_panic(self, chat_id: int, args: str) -> None:
        result = await self.execution_engine.emergency_stop(flatten=True)
        flat = result.get("flatten", {})
        await self._send_text(
            chat_id,
            "[긴급 중지]\n"
            f"paused={result.get('paused')}\n"
            f"승인 대기 거절 {result.get('rejected_pending')}건\n"
            f"청산 요청 {flat.get('requested', 0)} | 청산 {flat.get('closed', 0)} | "
            f"미존재 {flat.get('missing', 0)} | 가격없음 {flat.get('no_price', 0)}\n"
            f"실현 손익 {float(flat.get('realized_pnl_usd', 0.0)):+.2f} USD",
        )

    async def _build_position_views(self, limit: int) -> tuple[list[PositionView], float]:
        rows = await self.store.get_open_positions(self.runtime_state.trading_mode)
        market_ids = [str(row.market_id) for row in rows]
        prices = await self.store.latest_market_prices(market_ids)
        questions = await self.store.market_questions(market_ids)

        views: list[PositionView] = []
        unrealized_total = 0.0
        for idx, row in enumerate(rows[:limit], start=1):
            side = str(row.side).upper()
            market_id = str(row.market_id)
            yes_price = prices.get(market_id, float(row.entry_price))
            mark_price = max(0.001, min(0.999, 1.0 - yes_price)) if side == "NO" else max(0.001, min(0.999, yes_price))
            shares = float(row.size_usd) / max(float(row.entry_price), 0.001)
            unrealized = (shares * mark_price) - float(row.size_usd)
            unrealized_total += unrealized
            views.append(
                PositionView(
                    index=idx,
                    position_id=int(row.id),
                    market_id=market_id,
                    side=side,
                    size_usd=float(row.size_usd),
                    entry_price=float(row.entry_price),
                    mark_price=float(mark_price),
                    unrealized_pnl_usd=float(unrealized),
                    question=questions.get(market_id, "").strip(),
                )
            )
        return views, unrealized_total

    @staticmethod
    def _parse_positive_int(raw: str | None, default: int | None, max_value: int) -> int | None:
        if raw is None or not raw.strip():
            return default
        text = raw.strip()
        if not text.isdigit():
            return None
        value = int(text)
        if value <= 0:
            return None
        return min(value, max_value)

    @staticmethod
    def _parse_positive_float(raw: str | None) -> float | None:
        if raw is None:
            return None
        text = raw.strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        if value <= 0:
            return None
        return value

    @staticmethod
    def _fmt_ts(value: object) -> str:
        if value is None:
            return "-"
        if hasattr(value, "astimezone"):
            return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return str(value)

    @staticmethod
    def _render_close_result(title: str, result: dict) -> str:
        error = result.get("error")
        if error:
            return (
                f"[{title}]\n"
                f"모드 {str(result.get('mode')).upper()} | 실패 사유 {error}\n"
                f"요청 {result.get('requested', 0)} | 청산 {result.get('closed', 0)}"
            )
        return (
            f"[{title}]\n"
            f"모드 {str(result.get('mode')).upper()}\n"
            f"요청 {result.get('requested', 0)} | 청산 {result.get('closed', 0)} | "
            f"미존재 {result.get('missing', 0)} | 가격없음 {result.get('no_price', 0)}\n"
            f"실현 손익 {float(result.get('realized_pnl_usd', 0.0)):+.2f} USD"
        )
