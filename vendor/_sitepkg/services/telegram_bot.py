from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import timezone

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.filters.command import CommandObject
from aiogram.types import Message

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

        self.enabled = bool(self.settings.telegram_bot_token and self.settings.telegram_chat_id)
        self.chat_id = int(self.settings.telegram_chat_id) if self.settings.telegram_chat_id else None
        self.bot: Bot | None = None
        self.dp: Dispatcher | None = None
        self.instant_alerts_enabled = os.getenv("TELEGRAM_INSTANT_ALERTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        if self.enabled:
            self.bot = Bot(
                token=self.settings.telegram_bot_token,
                default=DefaultBotProperties(parse_mode=ParseMode.HTML),
            )
            self.dp = Dispatcher()
            self._register_handlers()

    async def run(self) -> None:
        if not self.enabled or self.bot is None or self.dp is None:
            logger.warning("Telegram disabled. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
            while True:
                await asyncio.sleep(3600)
        await self.dp.start_polling(self.bot)

    async def notify(self, text: str) -> None:
        if not self.enabled or self.bot is None or self.chat_id is None:
            logger.info("notify skipped: %s", text)
            return
        if not self.instant_alerts_enabled and not self._is_periodic_report(text):
            return
        await self.bot.send_message(chat_id=self.chat_id, text=text)

    @staticmethod
    def _is_periodic_report(text: str) -> bool:
        if not text:
            return False
        prefixes = (
            "[정기 리포트]",
            "[60분 실측 리포트]",
            "[6시간 리포트]",
            "[상태 요약]",
        )
        return text.startswith(prefixes)

    def _register_handlers(self) -> None:
        assert self.dp is not None
        self.dp.message.register(self._cmd_help, Command("help"))

        self.dp.message.register(self._cmd_status, Command("status"))
        self.dp.message.register(self._cmd_report60, Command("report60"))
        self.dp.message.register(self._cmd_report6h, Command("report6h"))
        self.dp.message.register(self._cmd_positions, Command("positions"))
        self.dp.message.register(self._cmd_pending, Command("pending"))
        self.dp.message.register(self._cmd_pnl, Command("pnl"))
        self.dp.message.register(self._cmd_risk, Command("risk"))
        self.dp.message.register(self._cmd_gate, Command("gate"))
        self.dp.message.register(self._cmd_data, Command("data"))

        self.dp.message.register(self._cmd_pause, Command("pause"))
        self.dp.message.register(self._cmd_resume, Command("resume"))
        self.dp.message.register(self._cmd_kill_on, Command("kill_on"))
        self.dp.message.register(self._cmd_kill_off, Command("kill_off"))
        self.dp.message.register(self._cmd_go_live, Command("go_live"))
        self.dp.message.register(self._cmd_go_paper, Command("go_paper"))
        self.dp.message.register(self._cmd_set_minpos, Command("set_minpos"))
        self.dp.message.register(self._cmd_clear_minpos, Command("clear_minpos"))

        self.dp.message.register(self._cmd_approve, Command("approve"))
        self.dp.message.register(self._cmd_reject, Command("reject"))
        self.dp.message.register(self._cmd_reject_all, Command("reject_all"))
        self.dp.message.register(self._cmd_close, Command("close"))
        self.dp.message.register(self._cmd_close_market, Command("close_market"))
        self.dp.message.register(self._cmd_close_side, Command("close_side"))
        self.dp.message.register(self._cmd_closeall, Command("closeall"))
        self.dp.message.register(self._cmd_panic, Command("panic"))

    def _authorized(self, message: Message) -> bool:
        if self.chat_id is None:
            return False
        return message.chat.id == self.chat_id

    async def _cmd_help(self, message: Message) -> None:
        if not self._authorized(message):
            return
        await message.answer(
            "[명령어 가이드]\n"
            "모니터링\n"
            "/status : 운영 핵심 요약(모드/포지션/손익/게이트)\n"
            "/report60 : 최근 60분 실측(신호/체결/거절 사유)\n"
            "/report6h : 최근 6시간 실측\n"
            "/positions [개수] : 오픈 포지션 상세 조회(기본 10, 최대 40)\n"
            "/pending : 승인 대기 신호 목록\n"
            "/pnl : 실현/미실현 손익 요약\n"
            "/risk : 드로다운/킬스위치/손실한도\n"
            "/gate : 히스토리+페이퍼 게이트 상세\n"
            "/data : tick/feature/signal 최신 시각\n"
            "제어\n"
            "/pause : 신규 진입 일시정지\n"
            "/resume : 일시정지 해제 + 킬스위치 해제\n"
            "/kill_on : 킬스위치 즉시 ON\n"
            "/kill_off : 킬스위치 OFF\n"
            "/go_paper : PAPER 모드 강제 전환\n"
            "/go_live : 게이트 통과 시 LIVE 전환(하드락이면 차단)\n"
            "/set_minpos <usd> : 런타임 최소 진입금액 override\n"
            "/clear_minpos : 런타임 최소 진입금액 override 해제\n"
            "승인/거절\n"
            "/approve <signal_id> : 반자동 신호 승인\n"
            "/reject <signal_id> : 반자동 신호 거절\n"
            "/reject_all : 승인 대기 전체 거절\n"
            "강제 청산\n"
            "/close <position_id> : 특정 포지션 강제 청산\n"
            "/close_market <market_id> : 해당 마켓 오픈 포지션 전부 청산\n"
            "/close_side <YES|NO> [limit] : 방향 기준 부분/전체 청산\n"
            "/closeall : 오픈 포지션 전체 강제 청산\n"
            "/panic : 긴급정지(일시정지+킬스위치+대기거절+전체청산)"
        )

    async def _cmd_status(self, message: Message) -> None:
        if not self._authorized(message):
            return
        await message.answer(await self.reporter.build_status_text())

    async def _cmd_report60(self, message: Message) -> None:
        if not self._authorized(message):
            return
        await message.answer(await self.reporter.build_report60_text())

    async def _cmd_report6h(self, message: Message) -> None:
        if not self._authorized(message):
            return
        await message.answer(await self.reporter.build_report6h_text())

    async def _cmd_positions(self, message: Message, command: CommandObject) -> None:
        if not self._authorized(message):
            return
        limit = self._parse_positive_int(command.args, default=10, max_value=40)
        if limit is None:
            await message.answer("[사용법]\n/positions [개수]\n예: /positions 20")
            return

        views, unrealized_total = await self._build_position_views(limit=limit)
        if not views:
            await message.answer("[포지션]\n현재 오픈 포지션이 없습니다.")
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
        await message.answer("\n".join(lines))

    async def _cmd_pending(self, message: Message) -> None:
        if not self._authorized(message):
            return
        items = list(self.runtime_state.pending_approvals.items())
        if not items:
            await message.answer("[승인 대기]\n현재 대기 신호가 없습니다.")
            return
        lines = [f"[승인 대기] 총 {len(items)}건"]
        for idx, (signal_id, intent) in enumerate(items[:20], start=1):
            lines.append(
                f"{idx}. {signal_id}\n"
                f"   시장 {intent.market_id} | 방향 {intent.side.value} | 가격 {intent.price:.4f} | 금액 {intent.size_usd:.2f} USD"
            )
        if len(items) > 20:
            lines.append(f"... 외 {len(items) - 20}건")
        await message.answer("\n".join(lines))

    async def _cmd_pnl(self, message: Message) -> None:
        if not self._authorized(message):
            return
        snapshot = await self.store.status_snapshot()
        _, unrealized_total = await self._build_position_views(limit=10_000)
        est_total = float(snapshot["week_pnl"]) + unrealized_total
        await message.answer(
            "[손익 요약]\n"
            f"일간 실현손익 {snapshot['day_pnl']:+.2f} USD\n"
            f"주간 실현손익 {snapshot['week_pnl']:+.2f} USD\n"
            f"현재 미실현손익 {unrealized_total:+.2f} USD\n"
            f"주간실현+미실현 {est_total:+.2f} USD\n"
            f"오픈 포지션 {snapshot['open_positions']}개"
        )

    async def _cmd_risk(self, message: Message) -> None:
        if not self._authorized(message):
            return
        state = await self.risk_engine.refresh_state()
        pause_text = "ON" if self.runtime_state.paused else "OFF"
        kill_text = "ON" if self.runtime_state.kill_switch else "OFF"
        await message.answer(
            "[리스크 상태]\n"
            f"일시정지 {pause_text} | 킬스위치 {kill_text}\n"
            f"일간 DD {state.daily_drawdown_pct:.2%}\n"
            f"주간 DD {state.weekly_drawdown_pct:.2%}\n"
            f"손실한도(일/주) {self.settings.daily_loss_limit_pct:.2%} / {self.settings.weekly_loss_limit_pct:.2%}"
        )

    async def _cmd_gate(self, message: Message) -> None:
        if not self._authorized(message):
            return
        await message.answer(f"[게이트]\n{await self.gatekeeper.summary_text()}")

    async def _cmd_data(self, message: Message) -> None:
        if not self._authorized(message):
            return
        snapshot = await self.store.data_freshness_snapshot()
        latest_tick = self._fmt_ts(snapshot.get("latest_tick_ts"))
        latest_feature = self._fmt_ts(snapshot.get("latest_feature_ts"))
        latest_signal = self._fmt_ts(snapshot.get("latest_signal_ts"))
        ticks_5m = int(snapshot.get("ticks_5m") or 0)
        await message.answer(
            "[데이터 신선도]\n"
            f"최근 tick {latest_tick}\n"
            f"최근 feature {latest_feature}\n"
            f"최근 signal {latest_signal}\n"
            f"최근 5분 tick 수 {ticks_5m}건"
        )

    async def _cmd_pause(self, message: Message) -> None:
        if not self._authorized(message):
            return
        self.runtime_state.paused = True
        await message.answer("[제어]\n트레이딩을 일시정지했습니다.")

    async def _cmd_resume(self, message: Message) -> None:
        if not self._authorized(message):
            return
        self.runtime_state.paused = False
        self.runtime_state.kill_switch = False
        await message.answer("[제어]\n트레이딩을 재개했습니다. 킬스위치도 해제했습니다.")

    async def _cmd_kill_on(self, message: Message) -> None:
        if not self._authorized(message):
            return
        await self.risk_engine.activate_kill_switch("manual_command:/kill_on")
        await message.answer("[긴급 제어]\n킬스위치 ON, 신규 진입 차단.")

    async def _cmd_kill_off(self, message: Message) -> None:
        if not self._authorized(message):
            return
        self.runtime_state.kill_switch = False
        await message.answer("[긴급 제어]\n킬스위치 OFF.")

    async def _cmd_go_live(self, message: Message) -> None:
        if not self._authorized(message):
            return
        if self.settings.demo_paper_hardlock:
            self.runtime_state.trading_mode = TradingMode.PAPER
            self.runtime_state.manual_live_approved = False
            await message.answer("[실거래 전환 불가]\n데모 모드에서는 실거래 전환 불가")
            return

        hist_ok = await self.gatekeeper.historical_gate_passed()
        paper_ok = await self.gatekeeper.paper_gate_passed()
        if not (hist_ok and paper_ok):
            reasons: list[str] = []
            if not hist_ok:
                reasons.append("히스토리 60일 검증 미통과")
            if not paper_ok:
                reasons.append("Paper 3일 실검증 미통과")
            reason_text = "\n".join(f"- {reason}" for reason in reasons)
            await message.answer(
                "[실거래 전환 불가]\n"
                f"{reason_text}\n"
                f"{await self.gatekeeper.summary_text()}"
            )
            return

        self.runtime_state.trading_mode = TradingMode.LIVE
        self.runtime_state.manual_live_approved = True
        await message.answer("[모드 전환]\nLIVE 모드로 전환했습니다.")

    async def _cmd_go_paper(self, message: Message) -> None:
        if not self._authorized(message):
            return
        self.runtime_state.trading_mode = TradingMode.PAPER
        self.runtime_state.manual_live_approved = False
        await message.answer("[모드 전환]\nPAPER 모드로 전환했습니다.")

    async def _cmd_set_minpos(self, message: Message, command: CommandObject) -> None:
        if not self._authorized(message):
            return
        if not command.args:
            await message.answer("[사용법]\n/set_minpos <usd>\n예: /set_minpos 1.5")
            return
        value = self._parse_positive_float(command.args)
        if value is None:
            await message.answer("[실패]\n숫자를 입력하세요. 예: /set_minpos 1.5")
            return
        if value > float(self.settings.max_position_usd):
            await message.answer(
                f"[실패]\n최소진입은 MAX_POSITION_USD({self.settings.max_position_usd:.2f})를 넘을 수 없습니다."
            )
            return
        self.runtime_state.min_position_usd_override = float(value)
        await message.answer(
            "[사이징]\n"
            f"런타임 최소진입금액 override 적용: {self.runtime_state.min_position_usd_override:.2f} USD"
        )

    async def _cmd_clear_minpos(self, message: Message) -> None:
        if not self._authorized(message):
            return
        self.runtime_state.min_position_usd_override = None
        await message.answer("[사이징]\n런타임 최소진입금액 override 해제.")

    async def _cmd_approve(self, message: Message, command: CommandObject) -> None:
        if not self._authorized(message):
            return
        signal_id = (command.args or "").strip()
        if not signal_id:
            await message.answer("[사용법]\n/approve <signal_id>")
            return
        await message.answer(await self.execution_engine.approve_signal(signal_id))

    async def _cmd_reject(self, message: Message, command: CommandObject) -> None:
        if not self._authorized(message):
            return
        signal_id = (command.args or "").strip()
        if not signal_id:
            await message.answer("[사용법]\n/reject <signal_id>")
            return
        await message.answer(await self.execution_engine.reject_signal(signal_id))

    async def _cmd_reject_all(self, message: Message) -> None:
        if not self._authorized(message):
            return
        count = await self.execution_engine.reject_all_pending()
        await message.answer(f"[승인 대기 정리]\n일괄 거절 {count}건 완료.")

    async def _cmd_close(self, message: Message, command: CommandObject) -> None:
        if not self._authorized(message):
            return
        position_id = self._parse_positive_int(command.args, default=None, max_value=10_000_000)
        if position_id is None:
            await message.answer("[사용법]\n/close <position_id>\n예: /close 123")
            return
        result = await self.execution_engine.force_close_position(position_id)
        await message.answer(self._render_close_result("포지션 강제청산", result))

    async def _cmd_close_market(self, message: Message, command: CommandObject) -> None:
        if not self._authorized(message):
            return
        market_id = (command.args or "").strip()
        if not market_id:
            await message.answer("[사용법]\n/close_market <market_id>")
            return
        result = await self.execution_engine.force_close_market(market_id)
        await message.answer(self._render_close_result(f"마켓 강제청산 ({market_id})", result))

    async def _cmd_close_side(self, message: Message, command: CommandObject) -> None:
        if not self._authorized(message):
            return
        args = (command.args or "").strip().split()
        if not args:
            await message.answer("[사용법]\n/close_side <YES|NO> [limit]\n예: /close_side YES 5")
            return
        side = args[0].upper()
        if side not in {"YES", "NO"}:
            await message.answer("[실패]\nside는 YES 또는 NO만 가능합니다.")
            return
        limit = None
        if len(args) >= 2:
            parsed = self._parse_positive_int(args[1], default=None, max_value=10_000)
            if parsed is None:
                await message.answer("[실패]\nlimit는 양의 정수여야 합니다.")
                return
            limit = parsed
        result = await self.execution_engine.force_close_side(side, limit=limit)
        label = f"사이드 강제청산 ({side}" + ("" if limit is None else f", limit={limit}") + ")"
        await message.answer(self._render_close_result(label, result))

    async def _cmd_closeall(self, message: Message) -> None:
        if not self._authorized(message):
            return
        mode = TradingMode.PAPER if self.settings.demo_paper_hardlock else self.runtime_state.trading_mode
        rows = await self.store.get_open_positions(mode)
        result = await self.execution_engine.force_close_positions(
            [int(row.id) for row in rows],
            reason="manual_close_all",
        )
        await message.answer(self._render_close_result("전체 강제청산", result))

    async def _cmd_panic(self, message: Message) -> None:
        if not self._authorized(message):
            return
        result = await self.execution_engine.emergency_stop(flatten=True)
        flat = result.get("flatten", {})
        await message.answer(
            "[긴급정지]\n"
            f"paused={result.get('paused')} | kill_switch={result.get('kill_switch')}\n"
            f"대기신호 거절 {result.get('rejected_pending')}건\n"
            f"청산 요청 {flat.get('requested', 0)} | 청산 {flat.get('closed', 0)} | "
            f"미존재 {flat.get('missing', 0)} | 가격없음 {flat.get('no_price', 0)}\n"
            f"실현손익 {float(flat.get('realized_pnl_usd', 0.0)):+.2f} USD"
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
                f"모드 {str(result.get('mode')).upper()} | 실패사유 {error}\n"
                f"요청 {result.get('requested', 0)} | 청산 {result.get('closed', 0)}"
            )
        return (
            f"[{title}]\n"
            f"모드 {str(result.get('mode')).upper()}\n"
            f"요청 {result.get('requested', 0)} | 청산 {result.get('closed', 0)} | "
            f"미존재 {result.get('missing', 0)} | 가격없음 {result.get('no_price', 0)}\n"
            f"실현손익 {float(result.get('realized_pnl_usd', 0.0)):+.2f} USD"
        )
