from __future__ import annotations

import asyncio
import logging

from polymethemoney.config import get_settings
from polymethemoney.orchestrator import TradingApp


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


async def _run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = TradingApp(settings)
    await app.run()


def run() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()

