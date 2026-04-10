from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx
import websockets

from polymethemoney.config import Settings
from polymethemoney.domain import FillResult, MarketTick, OrderIntent, TradingMode


class PolymarketClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._headers = {
            "Content-Type": "application/json",
            "X-API-KEY": settings.polymarket_api_key,
            "X-API-SECRET": settings.polymarket_api_secret,
            "X-API-PASSPHRASE": settings.polymarket_api_passphrase,
        }

    async def fetch_markets(self, limit: int = 250) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                self.settings.polymarket_gamma_url,
                params={
                    "limit": limit,
                    "active": True,
                    "closed": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and "markets" in payload and isinstance(payload["markets"], list):
                return payload["markets"]
            return []

    async def fetch_order_book(self, token_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{self.settings.polymarket_clob_rest_url}/book",
                params={"token_id": token_id},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            return {"token_id": token_id, "bids": [], "asks": []}
        payload["token_id"] = token_id
        return payload

    async def fetch_order_books(self, token_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not token_ids:
            return {}
        unique_ids = list(dict.fromkeys(token_ids))
        async with httpx.AsyncClient(timeout=20.0) as client:
            if not self.settings.polymarket_use_bulk_books:
                return await self._fetch_order_books_fallback(client, unique_ids)
            payload: Any
            try:
                response = await client.post(
                    f"{self.settings.polymarket_clob_rest_url}/books",
                    json={"token_ids": unique_ids},
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError:
                # Fallback for environments where /books rejects payload variants.
                return await self._fetch_order_books_fallback(client, unique_ids)
        books: dict[str, dict[str, Any]] = {}
        rows: list[Any]
        if isinstance(payload, dict) and isinstance(payload.get("books"), list):
            rows = payload["books"]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            token_id = str(row.get("asset_id") or row.get("token_id") or "")
            if not token_id:
                continue
            books[token_id] = row
        for token_id in unique_ids:
            books.setdefault(token_id, {"token_id": token_id, "bids": [], "asks": []})
        return books

    async def _fetch_order_books_fallback(
        self,
        client: httpx.AsyncClient,
        token_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        books: dict[str, dict[str, Any]] = {}
        for token_id in token_ids:
            try:
                response = await client.get(
                    f"{self.settings.polymarket_clob_rest_url}/book",
                    params={"token_id": token_id},
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    payload["token_id"] = str(payload.get("asset_id") or payload.get("token_id") or token_id)
                    books[token_id] = payload
                    continue
            except httpx.HTTPError:
                pass
            books[token_id] = {"token_id": token_id, "bids": [], "asks": []}
        return books

    async def stream_market_ticks(self, market_ids: list[str]) -> AsyncIterator[MarketTick]:
        if not market_ids:
            return
        sub_payload = {
            "type": "subscribe",
            "channel": "market",
            "market_ids": market_ids,
        }
        async with websockets.connect(self.settings.polymarket_ws_url, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps(sub_payload))
            async for raw in ws:
                tick = self._parse_ws_message(raw)
                if tick is not None:
                    yield tick

    def _parse_ws_message(self, raw: str) -> MarketTick | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        market_id = str(
            data.get("market")
            or data.get("market_id")
            or data.get("asset_id")
            or data.get("condition_id")
            or ""
        )
        if not market_id:
            return None
        bid = self._to_float(data, ["best_bid", "bid", "b"])
        ask = self._to_float(data, ["best_ask", "ask", "a"])
        last_price = self._to_float(data, ["last_price", "price", "p"], fallback=(bid + ask) / 2 if bid and ask else 0.5)
        volume_1h = self._to_float(data, ["volume_1h", "volume"], fallback=0.0)
        open_interest = self._to_float(data, ["open_interest", "oi"], fallback=0.0)
        if bid <= 0 or ask <= 0:
            return None
        expiry = self._parse_datetime(data.get("end_date") or data.get("expiry") or data.get("end_time"))
        return MarketTick(
            market_id=market_id,
            bid=bid,
            ask=ask,
            last_price=max(0.001, min(0.999, last_price)),
            volume_1h=volume_1h,
            open_interest=open_interest,
            expiry_ts=expiry,
            timestamp=datetime.now(timezone.utc),
            raw=data,
        )

    @staticmethod
    def _to_float(data: dict[str, Any], keys: list[str], fallback: float = 0.0) -> float:
        for key in keys:
            if key in data:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    continue
        return fallback

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            try:
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                return datetime.fromisoformat(value).astimezone(timezone.utc)
            except ValueError:
                return None
        return None

    async def place_limit_order(self, intent: OrderIntent, trading_mode: TradingMode) -> FillResult:
        if trading_mode != TradingMode.LIVE:
            raise ValueError("Live client received non-live trading mode.")
        if not self.settings.polymarket_api_key:
            raise ValueError("POLYMARKET_API_KEY is required for live trading mode.")
        endpoint = f"{self.settings.polymarket_clob_rest_url}/order"
        order_id = str(uuid4())
        body = {
            "order_id": order_id,
            "market_id": intent.market_id,
            "side": intent.side.value.lower(),
            "price": intent.price,
            "size": intent.size_usd,
            "order_type": "limit",
            "ttl": 0,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(endpoint, json=body, headers=self._headers)
            response.raise_for_status()
            payload = response.json() if response.content else {}
        fill_price = float(payload.get("fill_price", intent.price))
        size_usd = float(payload.get("fill_size", intent.size_usd))
        fee_usd = float(payload.get("fee_usd", 0.0))
        status = str(payload.get("status", "filled"))
        return FillResult(
            order_id=str(payload.get("order_id", order_id)),
            strategy_id=intent.strategy_id,
            market_id=intent.market_id,
            side=intent.side,
            fill_price=fill_price,
            size_usd=size_usd,
            status=status,
            fee_usd=fee_usd,
            mode=TradingMode.LIVE,
        )
