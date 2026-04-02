from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from polymethemoney.config import Settings
from polymethemoney.domain import TradingMode
from polymethemoney.services.llm_client import LLMClient

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
        self.llm = LLMClient(settings)
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
        cmd = cmd.lower()

        if cmd in ("/help", "/start"):
            await self._send_message(client, self._help_text())
            return
        if cmd == "/status":
            await self._send_message(client, await self.reporter.build_status_text())
            return
        if cmd in ("/report60", "/report"):
            await self._send_message(client, await self.reporter.build_report60_text())
            return
        if cmd == "/report6h":
            await self._send_message(client, await self.reporter.build_report6h_text())
            return
        if cmd == "/positions":
            await self._send_message(client, await self._build_positions_card(arg))
            return
        if cmd == "/pending":
            await self._send_message(client, await self._build_pending_card())
            return
        if cmd == "/pnl":
            await self._send_message(client, await self._build_pnl_card())
            return
        if cmd == "/risk":
            await self._send_message(client, await self._build_risk_card())
            return
        if cmd == "/gate":
            await self._send_message(client, await self._build_gate_card())
            return
        if cmd == "/data":
            await self._send_message(client, await self._build_data_card())
            return
        if cmd == "/pause":
            await self._send_message(client, await self._cmd_pause())
            return
        if cmd == "/resume":
            await self._send_message(client, await self._cmd_resume())
            return
        if cmd == "/kill_on":
            await self._send_message(client, await self._cmd_kill_on())
            return
        if cmd == "/kill_off":
            await self._send_message(client, await self._cmd_kill_off())
            return
        if cmd == "/go_live":
            await self._send_message(client, await self._cmd_go_live())
            return
        if cmd == "/go_paper":
            await self._send_message(client, await self._cmd_go_paper())
            return
        if cmd == "/set_minpos":
            await self._send_message(client, await self._cmd_set_minpos(arg))
            return
        if cmd == "/clear_minpos":
            await self._send_message(client, await self._cmd_clear_minpos())
            return
        if cmd == "/tuning":
            await self._send_message(client, await self._cmd_tuning(arg))
            return
        if cmd == "/tuning_reset":
            await self._send_message(client, await self._cmd_tuning_reset())
            return
        if cmd == "/approve":
            await self._send_message(client, await self._cmd_approve(arg))
            return
        if cmd == "/reject":
            await self._send_message(client, await self._cmd_reject(arg))
            return
        if cmd == "/reject_all":
            await self._send_message(client, await self._cmd_reject_all())
            return
        if cmd == "/close":
            await self._send_message(client, await self._cmd_close(arg))
            return
        if cmd == "/close_market":
            await self._send_message(client, await self._cmd_close_market(arg))
            return
        if cmd == "/close_side":
            await self._send_message(client, await self._cmd_close_side(arg))
            return
        if cmd == "/closeall":
            await self._send_message(client, await self._cmd_closeall())
            return
        if cmd == "/panic":
            await self._send_message(client, await self._cmd_panic())
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
            "모니터링\n"
            "/status - 상태 요약\n"
            "/report60 - 최근 60분 요약\n"
            "/report6h - 최근 6시간 요약\n"
            "/positions [N] - 포지션 상세(기본 10, 최대 40)\n"
            "/pending - 승인 대기 신호\n"
            "/pnl - 손익 요약\n"
            "/risk - 리스크 상태\n"
            "/gate - 게이트 요약\n"
            "/data - 데이터 신선도\n"
            "제어\n"
            "/pause - 신규 진입 일시정지\n"
            "/resume - 일시정지 해제\n"
            "/kill_on - 킬스위치 ON\n"
            "/kill_off - 킬스위치 OFF\n"
            "/go_paper - PAPER 모드 전환\n"
            "/go_live - 게이트 통과 시 LIVE 전환(하드락이면 차단)\n"
            "/set_minpos <usd> - 최소 진입 금액 override\n"
            "/clear_minpos - 최소 진입 override 해제\n"
            "/tuning <goal> - LLM 튜닝 제안 런타임 적용\n"
            "/tuning_reset - LLM 튜닝 오버라이드 해제\n"
            "승인/거절\n"
            "/approve <signal_id> - 승인\n"
            "/reject <signal_id> - 거절\n"
            "/reject_all - 승인대기 일괄 거절\n"
            "강제 청산\n"
            "/close <position_id>\n"
            "/close_market <market_id>\n"
            "/close_side <YES|NO> [limit]\n"
            "/closeall - 전체 강제 청산\n"
            "/panic - 일시정지+킬스위치+전체청산"
        )

    async def _build_positions_card(self, arg: str) -> str:
        limit = self._parse_positive_int(arg, default=10, max_value=40)
        rows = await self.store.get_open_positions(self.runtime_state.trading_mode)
        if not rows:
            return "[포지션]\n없음"
        if limit is None:
            return "[사용법]\n/positions [개수]\n예) /positions 20"
        limit = max(1, min(limit, 40))

        views, unrealized_total = await self._build_position_views(rows, limit=limit)
        pos_plus = sum(1 for view in views if view["unrealized_pnl_usd"] > 1e-6)
        pos_minus = sum(1 for view in views if view["unrealized_pnl_usd"] < -1e-6)
        pos_flat = len(views) - pos_plus - pos_minus

        lines = [
            f"[포지션] 총 {len(rows)}개 (상위 {limit})",
            f"미실현 합계 {unrealized_total:+.2f} USD | +{pos_plus}/0:{pos_flat}/-{pos_minus}",
        ]
        for view in views:
            short_q = view["question"] if len(view["question"]) <= 44 else f"{view['question'][:41]}..."
            lines.append(
                f"{view['index']}. id={view['position_id']} {view['side']} {view['size_usd']:.2f} USD "
                f"({view['entry_price']:.4f} -> {view['mark_price']:.4f}, {view['unrealized_pnl_usd']:+.2f})\n"
                f"   {view['market_id']} | {short_q}"
            )
        return "\n".join(lines)

    async def _build_pending_card(self) -> str:
        items = list(self.runtime_state.pending_approvals.items())
        if not items:
            return "[승인 대기]\n없음"
        lines = [f"[승인 대기] 총 {len(items)}건"]
        for idx, (signal_id, intent) in enumerate(items[:20], start=1):
            lines.append(
                f"{idx}. {signal_id}\n"
                f"   시장 {intent.market_id} | 방향 {intent.side.value} | 가격 {intent.price:.4f} | 금액 {intent.size_usd:.2f} USD"
            )
        if len(items) > 20:
            lines.append(f"... 외 {len(items) - 20}건")
        return "\n".join(lines)

    async def _build_pnl_card(self) -> str:
        snapshot = await self.store.status_snapshot()
        _, unrealized_total = await self._build_position_views(
            await self.store.get_open_positions(self.runtime_state.trading_mode),
            limit=10_000,
        )
        est_total = float(snapshot["week_pnl"]) + unrealized_total
        return (
            "[손익 요약]\n"
            f"일간 실현 {snapshot['day_pnl']:+.2f} USD\n"
            f"주간 실현 {snapshot['week_pnl']:+.2f} USD\n"
            f"미실현 {unrealized_total:+.2f} USD\n"
            f"주간+미실현 {est_total:+.2f} USD\n"
            f"오픈 포지션 {snapshot['open_positions']}개"
        )

    async def _build_risk_card(self) -> str:
        state = await self.risk_engine.refresh_state()
        pause_text = "ON" if self.runtime_state.paused else "OFF"
        kill_text = "ON" if self.runtime_state.kill_switch else "OFF"
        return (
            "[리스크 상태]\n"
            f"일시정지 {pause_text} | 킬스위치 {kill_text}\n"
            f"일간 DD {state.daily_drawdown_pct:.2%}\n"
            f"주간 DD {state.weekly_drawdown_pct:.2%}\n"
            f"손실한도(일/주) {self.settings.daily_loss_limit_pct:.2%} / {self.settings.weekly_loss_limit_pct:.2%}"
        )

    async def _build_gate_card(self) -> str:
        return f"[게이트]\n{await self.gatekeeper.summary_text()}"

    async def _build_data_card(self) -> str:
        snapshot = await self.store.data_freshness_snapshot()
        latest_tick = self._fmt_ts(snapshot.get("latest_tick_ts"))
        latest_feature = self._fmt_ts(snapshot.get("latest_feature_ts"))
        latest_signal = self._fmt_ts(snapshot.get("latest_signal_ts"))
        ticks_5m = int(snapshot.get("ticks_5m") or 0)
        return (
            "[데이터 신선도]\n"
            f"최근 tick {latest_tick}\n"
            f"최근 feature {latest_feature}\n"
            f"최근 signal {latest_signal}\n"
            f"최근 5분 tick 수 {ticks_5m}건"
        )

    async def _cmd_pause(self) -> str:
        self.runtime_state.paused = True
        rejected = await self.execution_engine.reject_all_pending()
        return f"[제어]\n일시정지 적용. 승인대기 {rejected}건 거절."

    async def _cmd_resume(self) -> str:
        self.runtime_state.paused = False
        self.runtime_state.kill_switch = False
        return "[제어]\n일시정지 해제. 킬스위치 OFF."

    async def _cmd_kill_on(self) -> str:
        await self.risk_engine.activate_kill_switch("manual:/kill_on")
        return "[긴급 제어]\n킬스위치 ON, 신규 진입 차단."

    async def _cmd_kill_off(self) -> str:
        self.runtime_state.kill_switch = False
        return "[긴급 제어]\n킬스위치 OFF."

    async def _cmd_go_live(self) -> str:
        if self.settings.demo_paper_hardlock:
            self.runtime_state.trading_mode = TradingMode.PAPER
            self.runtime_state.manual_live_approved = False
            return "[실거래 전환 불가]\n데모 모드에서 실거래 전환은 차단됩니다."

        hist_ok = await self.gatekeeper.historical_gate_passed()
        paper_ok = await self.gatekeeper.paper_gate_passed()
        if not (hist_ok and paper_ok):
            reasons = []
            if not hist_ok:
                reasons.append("- 히스토리 60일 검증 미통과")
            if not paper_ok:
                reasons.append("- Paper 3일 검증 미통과")
            reason_text = "\n".join(reasons)
            return f"[실거래 전환 불가]\n{reason_text}\n{await self.gatekeeper.summary_text()}"

        self.runtime_state.trading_mode = TradingMode.LIVE
        self.runtime_state.manual_live_approved = True
        return "[모드 전환]\nLIVE 모드로 전환 완료."

    async def _cmd_go_paper(self) -> str:
        self.runtime_state.trading_mode = TradingMode.PAPER
        self.runtime_state.manual_live_approved = False
        return "[모드 전환]\nPAPER 모드로 전환 완료."

    async def _cmd_set_minpos(self, arg: str) -> str:
        value = self._parse_positive_float(arg)
        if value is None:
            return "[사용법]\n/set_minpos <usd>\n예) /set_minpos 1.5"
        if value > float(self.settings.max_position_usd):
            return f"[실패]\nMAX_POSITION_USD({self.settings.max_position_usd:.2f}) 초과 불가."
        self.runtime_state.min_position_usd_override = float(value)
        return (
            "[사이징]\n"
            f"최소 진입금액 override 적용: {self.runtime_state.min_position_usd_override:.2f} USD"
        )

    async def _cmd_clear_minpos(self) -> str:
        self.runtime_state.min_position_usd_override = None
        return "[사이징]\n최소 진입금액 override 해제."

    async def _cmd_tuning(self, arg: str) -> str:
        if not self.settings.llm_enabled or not self.settings.llm_tuning_enabled:
            return "[튜닝]\nLLM 튜닝 비활성 상태입니다."
        if self.runtime_state.trading_mode != TradingMode.PAPER:
            return "[튜닝]\nPAPER 모드에서만 사용 가능합니다."
        healthy = await self.llm.health()
        if not healthy:
            return "[튜닝]\nLLM 브리지 상태 불가(health 실패)."

        goal = arg.strip() or "수익 개선"
        summary = await self.store.signal_window_summary(window_minutes=self.settings.llm_tuning_window_minutes)
        snapshot = await self.store.status_snapshot()
        risk_state = await self.risk_engine.refresh_state()

        allowlist = self._tuning_allowlist()
        context = {
            "window_minutes": int(self.settings.llm_tuning_window_minutes),
            "signals": int(summary["total_signals"]),
            "fills": int(summary["filled_signals"]),
            "fill_rate": float(summary["fill_rate"]),
            "yes_no": {"yes": int(summary["yes_signals"]), "no": int(summary["no_signals"])},
            "reject_top": summary.get("reject_top", []),
            "pnl_day": float(snapshot["day_pnl"]),
            "pnl_week": float(snapshot["week_pnl"]),
            "drawdown_day": float(risk_state.daily_drawdown_pct),
            "drawdown_week": float(risk_state.weekly_drawdown_pct),
            "params": {key: value["current"] for key, value in allowlist.items()},
            "bounds": {key: {"min": value["min"], "max": value["max"]} for key, value in allowlist.items()},
        }

        response = await self.llm.suggest_tuning(goal, context)
        if not response.ok or response.payload is None:
            return f"[튜닝]\nLLM 응답 실패: {response.error}"

        changes = response.payload.get("changes") or []
        if not isinstance(changes, list) or not changes:
            return "[튜닝]\n변경 제안 없음."

        max_changes = max(1, int(self.settings.llm_tuning_max_changes))
        applied: list[str] = []
        skipped: list[str] = []
        for item in changes[:max_changes]:
            if not isinstance(item, dict):
                skipped.append("invalid_item")
                continue
            key = str(item.get("key") or "").strip().upper()
            if key not in allowlist:
                skipped.append(f"{key}:not_allowed")
                continue
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                skipped.append(f"{key}:invalid_value")
                continue
            applied_line = self._apply_tuning_override(key, value, allowlist[key])
            if applied_line is None:
                skipped.append(f"{key}:no_change")
                continue
            applied.append(applied_line)

        if not applied:
            reason = ", ".join(skipped[:3]) if skipped else "no_change"
            return f"[튜닝]\n적용 없음. 사유: {reason}"

        self.runtime_state.tuning_last_applied_at = datetime.now(timezone.utc)
        applied_text = "\n".join(applied[:5])
        skipped_text = ""
        if skipped:
            skipped_text = "\n스킵 " + ", ".join(skipped[:5])
        return f"[튜닝 적용]\n{applied_text}{skipped_text}"

    async def _cmd_tuning_reset(self) -> str:
        if not self.runtime_state.tuning_originals:
            return "[튜닝 리셋]\n해제할 오버라이드가 없습니다."
        for key, original in list(self.runtime_state.tuning_originals.items()):
            attr = self._tuning_attr_from_key(key)
            if attr:
                setattr(self.settings, attr, original)
        self.runtime_state.tuning_overrides.clear()
        self.runtime_state.tuning_originals.clear()
        self.runtime_state.tuning_last_applied_at = None
        return "[튜닝 리셋]\n모든 런타임 오버라이드 해제."

    async def _cmd_approve(self, arg: str) -> str:
        signal_id = arg.strip()
        if not signal_id:
            return "[사용법]\n/approve <signal_id>"
        return await self.execution_engine.approve_signal(signal_id)

    async def _cmd_reject(self, arg: str) -> str:
        signal_id = arg.strip()
        if not signal_id:
            return "[사용법]\n/reject <signal_id>"
        return await self.execution_engine.reject_signal(signal_id)

    async def _cmd_reject_all(self) -> str:
        count = await self.execution_engine.reject_all_pending()
        return f"[승인 대기 정리]\n일괄 거절 {count}건 완료."

    async def _cmd_close(self, arg: str) -> str:
        position_id = self._parse_positive_int(arg, default=None, max_value=10_000_000)
        if position_id is None:
            return "[사용법]\n/close <position_id>\n예) /close 123"
        result = await self.execution_engine.force_close_position(position_id)
        return self._render_close_result("포지션 강제 청산", result)

    async def _cmd_close_market(self, arg: str) -> str:
        market_id = arg.strip()
        if not market_id:
            return "[사용법]\n/close_market <market_id>"
        result = await self.execution_engine.force_close_market(market_id)
        return self._render_close_result(f"마켓 강제 청산 ({market_id})", result)

    async def _cmd_close_side(self, arg: str) -> str:
        args = arg.strip().split()
        if not args:
            return "[사용법]\n/close_side <YES|NO> [limit]\n예) /close_side YES 5"
        side = args[0].upper()
        if side not in {"YES", "NO"}:
            return "[실패]\nside는 YES 또는 NO만 가능합니다."
        limit = None
        if len(args) >= 2:
            parsed = self._parse_positive_int(args[1], default=None, max_value=10_000)
            if parsed is None:
                return "[실패]\nlimit은 양의 정수만 가능합니다."
            limit = parsed
        result = await self.execution_engine.force_close_side(side, limit=limit)
        label = f"사이드 강제 청산 ({side}" + ("" if limit is None else f", limit={limit}") + ")"
        return self._render_close_result(label, result)

    async def _cmd_closeall(self) -> str:
        mode = TradingMode.PAPER if self.settings.demo_paper_hardlock else self.runtime_state.trading_mode
        rows = await self.store.get_open_positions(mode)
        result = await self.execution_engine.force_close_positions(
            [int(row.id) for row in rows],
            reason="manual_close_all",
        )
        return self._render_close_result("전체 강제 청산", result)

    async def _cmd_panic(self) -> str:
        result = await self.execution_engine.emergency_stop(flatten=True)
        flat = result.get("flatten", {})
        return (
            "[긴급정지]\n"
            f"paused={result.get('paused')} | kill_switch={result.get('kill_switch')}\n"
            f"승인대기 거절 {result.get('rejected_pending')}건\n"
            f"청산 요청 {flat.get('requested', 0)} | 청산 {flat.get('closed', 0)} | "
            f"미존재 {flat.get('missing', 0)} | 가격없음 {flat.get('no_price', 0)}\n"
            f"실현손익 {float(flat.get('realized_pnl_usd', 0.0)):+.2f} USD"
        )

    async def _build_position_views(self, rows: list, limit: int) -> tuple[list[dict[str, Any]], float]:
        market_ids = [str(row.market_id) for row in rows]
        prices = await self.store.latest_market_prices(market_ids)
        questions = await self.store.market_questions(market_ids)

        views: list[dict[str, Any]] = []
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
                {
                    "index": idx,
                    "position_id": int(row.id),
                    "market_id": market_id,
                    "side": side,
                    "size_usd": float(row.size_usd),
                    "entry_price": float(row.entry_price),
                    "mark_price": float(mark_price),
                    "unrealized_pnl_usd": float(unrealized),
                    "question": questions.get(market_id, "").strip(),
                }
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

    def _tuning_allowlist(self) -> dict[str, dict[str, float]]:
        return {
            "MIN_POSITION_USD": {
                "attr": "min_position_usd",
                "min": float(self.settings.auto_threshold_tune_min_position_usd_min),
                "max": float(self.settings.auto_threshold_tune_min_position_usd_max),
                "current": float(self.settings.min_position_usd),
            },
            "SIGNAL_MIN_NET_EV": {
                "attr": "signal_min_net_ev",
                "min": float(self.settings.auto_threshold_tune_min_net_ev),
                "max": float(self.settings.auto_threshold_tune_max_net_ev),
                "current": float(self.settings.signal_min_net_ev),
            },
            "SIGNAL_MIN_CONTRACT_PRICE": {
                "attr": "signal_min_contract_price",
                "min": float(self.settings.auto_threshold_tune_min_contract_price_min),
                "max": float(self.settings.auto_threshold_tune_min_contract_price_max),
                "current": float(self.settings.signal_min_contract_price),
            },
            "AUTO_THRESHOLD_TUNE_WINDOW_MINUTES": {
                "attr": "auto_threshold_tune_window_minutes",
                "min": 30.0,
                "max": 240.0,
                "current": float(self.settings.auto_threshold_tune_window_minutes),
            },
            "AUTO_THRESHOLD_TUNE_MIN_SIGNALS": {
                "attr": "auto_threshold_tune_min_signals",
                "min": 5.0,
                "max": 200.0,
                "current": float(self.settings.auto_threshold_tune_min_signals),
            },
            "AUTO_THRESHOLD_TUNE_FILLRATE_LOW": {
                "attr": "auto_threshold_tune_fillrate_low",
                "min": 0.05,
                "max": 0.6,
                "current": float(self.settings.auto_threshold_tune_fillrate_low),
            },
            "AUTO_THRESHOLD_TUNE_FILLRATE_HIGH": {
                "attr": "auto_threshold_tune_fillrate_high",
                "min": 0.2,
                "max": 0.95,
                "current": float(self.settings.auto_threshold_tune_fillrate_high),
            },
            "AUTO_THRESHOLD_TUNE_STEP_NET_EV": {
                "attr": "auto_threshold_tune_step_net_ev",
                "min": 0.000001,
                "max": 0.001,
                "current": float(self.settings.auto_threshold_tune_step_net_ev),
            },
            "AUTO_THRESHOLD_TUNE_STEP_MIN_PRICE": {
                "attr": "auto_threshold_tune_step_min_price",
                "min": 0.0001,
                "max": 0.01,
                "current": float(self.settings.auto_threshold_tune_step_min_price),
            },
            "AUTO_THRESHOLD_TUNE_STEP_MIN_USD": {
                "attr": "auto_threshold_tune_step_min_usd",
                "min": 0.1,
                "max": 5.0,
                "current": float(self.settings.auto_threshold_tune_step_min_usd),
            },
            "AUTO_THRESHOLD_TUNE_COOLDOWN_MINUTES": {
                "attr": "auto_threshold_tune_cooldown_minutes",
                "min": 5.0,
                "max": 240.0,
                "current": float(self.settings.auto_threshold_tune_cooldown_minutes),
            },
        }

    @staticmethod
    def _tuning_attr_from_key(key: str) -> str | None:
        mapping = {
            "MIN_POSITION_USD": "min_position_usd",
            "SIGNAL_MIN_NET_EV": "signal_min_net_ev",
            "SIGNAL_MIN_CONTRACT_PRICE": "signal_min_contract_price",
            "AUTO_THRESHOLD_TUNE_WINDOW_MINUTES": "auto_threshold_tune_window_minutes",
            "AUTO_THRESHOLD_TUNE_MIN_SIGNALS": "auto_threshold_tune_min_signals",
            "AUTO_THRESHOLD_TUNE_FILLRATE_LOW": "auto_threshold_tune_fillrate_low",
            "AUTO_THRESHOLD_TUNE_FILLRATE_HIGH": "auto_threshold_tune_fillrate_high",
            "AUTO_THRESHOLD_TUNE_STEP_NET_EV": "auto_threshold_tune_step_net_ev",
            "AUTO_THRESHOLD_TUNE_STEP_MIN_PRICE": "auto_threshold_tune_step_min_price",
            "AUTO_THRESHOLD_TUNE_STEP_MIN_USD": "auto_threshold_tune_step_min_usd",
            "AUTO_THRESHOLD_TUNE_COOLDOWN_MINUTES": "auto_threshold_tune_cooldown_minutes",
        }
        return mapping.get(key)

    def _apply_tuning_override(self, key: str, value: float, spec: dict[str, float]) -> str | None:
        attr = str(spec["attr"])
        minimum = float(spec["min"])
        maximum = float(spec["max"])
        new_value = max(minimum, min(maximum, float(value)))
        if attr.endswith("_minutes") or attr.endswith("_signals"):
            new_value = float(int(new_value))
        if attr == "auto_threshold_tune_fillrate_high":
            low = float(self.settings.auto_threshold_tune_fillrate_low)
            if new_value <= low:
                new_value = min(0.95, max(low + 0.05, new_value))
        if attr == "auto_threshold_tune_fillrate_low":
            high = float(self.settings.auto_threshold_tune_fillrate_high)
            if new_value >= high:
                new_value = max(0.05, min(high - 0.05, new_value))
        current = float(getattr(self.settings, attr))
        if abs(current - new_value) < 1e-12:
            return None
        if key not in self.runtime_state.tuning_originals:
            self.runtime_state.tuning_originals[key] = current
        setattr(self.settings, attr, new_value)
        self.runtime_state.tuning_overrides[key] = new_value
        return f"{key} {current:.6f}->{new_value:.6f}"
