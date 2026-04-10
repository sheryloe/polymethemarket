from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict, deque
from datetime import datetime, timezone

from polymethemoney.adapters.paper_exchange import PaperExchange
from polymethemoney.domain import FeatureVector, MarketTick
from polymethemoney.storage import Store

logger = logging.getLogger(__name__)


class FeatureEngine:
    def __init__(
        self,
        store: Store,
        tick_queue: asyncio.Queue[MarketTick],
        feature_queues: list[asyncio.Queue[FeatureVector]],
        paper_exchange: PaperExchange,
    ) -> None:
        self.store = store
        self.tick_queue = tick_queue
        self.feature_queues = feature_queues
        self.paper_exchange = paper_exchange
        self._price_windows: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=120))

    async def run(self) -> None:
        while True:
            tick = await self.tick_queue.get()
            await self.paper_exchange.on_tick(tick)
            feature = self._build_feature(tick)
            await self.store.add_feature(feature)
            for queue in self.feature_queues:
                await queue.put(feature)

    def _build_feature(self, tick: MarketTick) -> FeatureVector:
        mid = max(0.001, min(0.999, (tick.bid + tick.ask) / 2.0))
        spread = max(0.0, tick.ask - tick.bid)
        spread_pct = spread / max(mid, 1e-6)
        now = datetime.now(timezone.utc)
        if tick.expiry_ts is not None:
            diff = tick.expiry_ts - now
            tte_hours = max(0.0, diff.total_seconds() / 3600.0)
        else:
            tte_hours = 168.0

        window = self._price_windows[tick.market_id]
        window.append(mid)
        volatility_long = self._realized_volatility(window)
        volatility_short = self._realized_volatility(window, size=20)
        trend_score = self._trend_score(window, size=20)
        momentum_20 = self._momentum(window, size=20)
        zscore_20 = self._zscore(window, size=20)
        imbalance = self._orderbook_imbalance(tick.bid, tick.ask, tick.last_price)
        blended_imbalance = self._blend_imbalance(imbalance, trend_score)
        volume_oi_ratio = float(tick.volume_1h) / max(float(tick.open_interest), 1.0)

        return FeatureVector(
            market_id=tick.market_id,
            implied_prob=mid,
            spread=spread,
            spread_pct=spread_pct,
            volume_1h=max(0.0, tick.volume_1h),
            open_interest=max(0.0, tick.open_interest),
            volume_oi_ratio=volume_oi_ratio,
            time_to_expiry_hours=tte_hours,
            orderbook_imbalance=blended_imbalance,
            momentum_20=momentum_20,
            zscore_20=zscore_20,
            volatility_20=volatility_short,
            volatility_30=volatility_long,
            timestamp=tick.timestamp,
        )

    @staticmethod
    def _realized_volatility(window: deque[float], size: int | None = None) -> float:
        series = list(window)
        if size is not None and size > 0:
            series = series[-size:]
        if len(series) < 5:
            return 0.0
        returns: list[float] = []
        for i in range(1, len(series)):
            prev = max(series[i - 1], 1e-6)
            curr = max(series[i], 1e-6)
            returns.append(math.log(curr / prev))
        if not returns:
            return 0.0
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / max(1, len(returns) - 1)
        return math.sqrt(max(0.0, var))

    @staticmethod
    def _trend_score(window: deque[float], size: int | None = None) -> float:
        series = list(window)
        if size is not None and size > 0:
            series = series[-size:]
        if len(series) < 10:
            return 0.0
        first = max(series[0], 1e-6)
        last = max(series[-1], 1e-6)
        momentum = (last - first) / first
        # Normalize around a 2% move range.
        scaled = momentum / 0.02
        return max(-1.0, min(1.0, scaled))

    @staticmethod
    def _momentum(window: deque[float], size: int) -> float:
        series = list(window)
        if size > 0:
            series = series[-size:]
        if len(series) < 2:
            return 0.0
        first = max(series[0], 1e-6)
        last = max(series[-1], 1e-6)
        return (last - first) / first

    @staticmethod
    def _zscore(window: deque[float], size: int) -> float:
        series = list(window)
        if size > 0:
            series = series[-size:]
        if len(series) < 5:
            return 0.0
        mean = sum(series) / len(series)
        var = sum((value - mean) ** 2 for value in series) / max(1, len(series) - 1)
        stdev = math.sqrt(max(0.0, var))
        if stdev <= 1e-9:
            return 0.0
        return (series[-1] - mean) / stdev

    @staticmethod
    def _orderbook_imbalance(bid: float, ask: float, last: float) -> float:
        spread = max(ask - bid, 1e-6)
        mid = (ask + bid) / 2.0
        imbalance = (mid - last) / spread
        return max(-1.0, min(1.0, imbalance))

    @staticmethod
    def _blend_imbalance(imbalance: float, trend_score: float) -> float:
        blended = (0.7 * imbalance) + (0.3 * trend_score)
        return max(-1.0, min(1.0, blended))
