from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from redis.asyncio import Redis

from polymethemoney.config import Settings
from polymethemoney.domain import PaperGateMetrics

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HistoricalMetrics:
    pf: float
    mdd_pct: float
    ece: float
    window_days: int
    source: str
    generated_at: datetime | None
    loaded_at: datetime


class Gatekeeper:
    REDIS_PAPER_TRADES_KEY = "gate:paper:trades"
    REDIS_VIOLATIONS_KEY = "gate:paper:violations"

    def __init__(self, settings: Settings, redis_client: Redis | None = None) -> None:
        self.settings = settings
        self.redis_client = redis_client
        self.historical_metrics: HistoricalMetrics | None = None
        self._local_trades: list[tuple[datetime, float]] = []
        self._local_violations: list[datetime] = []

    async def load_historical_from_file(self, file_path: Path) -> None:
        if not file_path.exists():
            logger.warning("Historical metrics file missing: %s", file_path)
            return
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        pf = float(payload.get("pf", 0.0))
        mdd = float(payload.get("mdd_pct", 1.0))
        ece = float(payload.get("ece", 1.0))
        window_days = int(payload.get("window_days", 0))
        source = str(payload.get("source", "unknown"))
        generated_at = self._parse_datetime(payload.get("generated_at"))
        self.historical_metrics = HistoricalMetrics(
            pf=pf,
            mdd_pct=mdd,
            ece=ece,
            window_days=window_days,
            source=source,
            generated_at=generated_at,
            loaded_at=datetime.now(timezone.utc),
        )
        logger.info(
            "Historical metrics loaded window=%s pf=%.4f mdd=%.4f ece=%.4f source=%s",
            window_days,
            pf,
            mdd,
            ece,
            source,
        )

    async def register_paper_trade(self, pnl_usd: float, when: datetime | None = None) -> None:
        ts = when or datetime.now(timezone.utc)
        self._local_trades.append((ts, pnl_usd))
        if self.redis_client is not None:
            value = json.dumps({"timestamp": ts.isoformat(), "pnl_usd": pnl_usd})
            await self.redis_client.rpush(self.REDIS_PAPER_TRADES_KEY, value)

    async def register_violation(self, when: datetime | None = None) -> None:
        ts = when or datetime.now(timezone.utc)
        self._local_violations.append(ts)
        if self.redis_client is not None:
            await self.redis_client.rpush(self.REDIS_VIOLATIONS_KEY, ts.isoformat())

    async def paper_metrics(self) -> PaperGateMetrics:
        since = datetime.now(timezone.utc) - timedelta(days=self.settings.gate_paper_days)
        trades = await self._paper_trade_values(since)
        if not trades:
            return PaperGateMetrics(0.0, 1.0, 0, await self._violations_count(since), 0.0)
        gross_profit = sum(x for x in trades if x > 0)
        gross_loss = abs(sum(x for x in trades if x < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 99.0
        curve = [0.0]
        for pnl in trades:
            curve.append(curve[-1] + pnl)
        peak = curve[0]
        max_dd_usd = 0.0
        for value in curve:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd_usd:
                max_dd_usd = dd
        mdd_pct = max_dd_usd / self.settings.starting_capital_usd
        covered_days = self._covered_days(since)
        violations = await self._violations_count(since)
        return PaperGateMetrics(
            profit_factor=pf,
            max_drawdown_pct=mdd_pct,
            trades=len(trades),
            violations=violations,
            covered_days=covered_days,
        )

    async def historical_gate_passed(self) -> bool:
        if self.historical_metrics is None:
            return False
        return (
            self.historical_metrics.window_days >= self.settings.gate_hist_window_days
            and self.historical_metrics.pf >= self.settings.gate_hist_min_pf
            and self.historical_metrics.mdd_pct <= self.settings.gate_hist_max_dd_pct
            and self.historical_metrics.ece <= self.settings.gate_hist_max_ece
        )

    async def paper_gate_passed(self) -> bool:
        metrics = await self.paper_metrics()
        return (
            metrics.covered_days >= self.settings.gate_paper_days
            and metrics.profit_factor >= self.settings.gate_paper_min_pf
            and metrics.max_drawdown_pct <= self.settings.gate_paper_max_dd_pct
            and metrics.trades >= self.settings.gate_paper_min_trades
            and metrics.violations <= self.settings.gate_paper_max_violations
        )

    async def is_live_gate_passed(self) -> bool:
        if self.settings.allow_live_without_gate:
            return True
        return await self.historical_gate_passed() and await self.paper_gate_passed()

    async def summary_text(self) -> str:
        h_ok = await self.historical_gate_passed()
        p_ok = await self.paper_gate_passed()
        paper = await self.paper_metrics()
        hist = self.historical_metrics
        hist_status = "\ud1b5\uacfc" if h_ok else "\uc2e4\ud328"
        paper_status = "\ud1b5\uacfc" if p_ok else "\uc2e4\ud328"
        hist_text = "\uc9c0\ud45c \ud30c\uc77c \uc5c6\uc74c"
        if hist is not None:
            generated = hist.generated_at.isoformat() if hist.generated_at is not None else "\ubbf8\uae30\ub85d"
            hist_text = (
                f"\uac80\uc99d\ucc3d {hist.window_days}\uc77c (\uae30\uc900 {self.settings.gate_hist_window_days}\uc77c)\n"
                f"PF {hist.pf:.2f} | MDD {hist.mdd_pct:.2%} | ECE {hist.ece:.4f}\n"
                f"\uc0dd\uc131\uc2dc\uac01 {generated}"
            )
        return (
            "\uac8c\uc774\ud2b8 \uc0c1\ud0dc\n"
            f"\ud788\uc2a4\ud1a0\ub9ac: {hist_status}\n"
            f"{hist_text}\n"
            f"\ud398\uc774\ud37c: {paper_status}\n"
            f"PF {paper.profit_factor:.2f} | MDD {paper.max_drawdown_pct:.2%} | "
            f"\uc2e4\ud604\uac70\ub798(\uccad\uc0b0\uc644\ub8cc) {paper.trades}\uac74 | \uc704\ubc18 {paper.violations}\uac74 | "
            f"\ucee4\ubc84 {paper.covered_days:.1f}\uc77c"
        )

    async def _paper_trade_values(self, since: datetime) -> list[float]:
        values: list[float] = [pnl for ts, pnl in self._local_trades if ts >= since]
        if self.redis_client is None:
            return values
        raw_rows = await self.redis_client.lrange(self.REDIS_PAPER_TRADES_KEY, 0, -1)
        for raw in raw_rows:
            try:
                item = json.loads(raw)
                ts = datetime.fromisoformat(item["timestamp"])
                if ts >= since:
                    values.append(float(item["pnl_usd"]))
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return values

    async def _violations_count(self, since: datetime) -> int:
        count = sum(1 for ts in self._local_violations if ts >= since)
        if self.redis_client is None:
            return count
        raw_rows = await self.redis_client.lrange(self.REDIS_VIOLATIONS_KEY, 0, -1)
        for raw in raw_rows:
            try:
                raw_value = raw if isinstance(raw, str) else raw.decode("utf-8")
                ts = datetime.fromisoformat(raw_value)
            except (AttributeError, ValueError):
                continue
            if ts >= since:
                count += 1
        return count

    def _covered_days(self, since: datetime) -> float:
        ts_values = [ts for ts, _ in self._local_trades if ts >= since]
        if not ts_values:
            return 0.0
        earliest = min(ts_values)
        return max(0.0, (datetime.now(timezone.utc) - earliest).total_seconds() / 86400.0)

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).astimezone(timezone.utc)
        except ValueError:
            return None
