# Polymethemoney v1.5

Polymarket paper/live trading engine with:
- dual-model probabilistic signals (settlement + intraday)
- structure alpha (pair/basket opportunities)
- Telegram control and reporting
- gate policy (`historical 60d + paper 3d`) before live

## Core Policy

- Default mode: `paper`
- Demo unlimited: `DEMO_UNLIMITED=false` (실거래 유사 검증 권장)
- Demo hardlock: `DEMO_PAPER_HARDLOCK=true` (live transition and live execution are blocked)
- Startup reset: `DEMO_RESET_PAPER_ON_STARTUP=false` (기본값, 재시작 시 포지션 자동 초기화 금지)
- Live transition: only after gate pass and manual `/go_live`
- Risk limits: `MAX_POSITIONS`, `MAX_POSITION_USD`, `DAILY_LOSS_LIMIT_PCT`, `WEEKLY_LOSS_LIMIT_PCT`
- Set `DAILY_LOSS_LIMIT_PCT=0`, `WEEKLY_LOSS_LIMIT_PCT=0` to disable drawdown kill switch
- Daily profit target: `DAILY_PROFIT_TARGET_PCT` (default `0.10` for 10%)
- Per-position stop loss: `POSITION_STOP_LOSS_PCT`
- Per-position take profit: `POSITION_TAKE_PROFIT_PCT`
- Paper fill source for demo: `PAPER_EXCHANGE_VENUES=binance,bybit`
- Paper symbol mapping: `PAPER_MARKET_SYMBOL_MAP` (JSON map: `{"market_id":"SYMBOL"}`)
- Paper ticker list: `PAPER_TICKER_SYMBOLS` (default BTC/ETH/BNB/SOL/ADA/..)
- Paper post-only mode: `PAPER_POST_ONLY=true`
- 3/12시간 성능 튜닝: `PAPER_PERF_TUNE_ENABLED`, `PAPER_PERF_TUNE_INTERVAL_HOURS`, `PAPER_PERF_TUNE_INTERVAL_HOURS_SECONDARY`, `PAPER_PERF_TUNE_MIN_PNL_USD`, `PAPER_PERF_TUNE_MIN_TRADES`, `PAPER_PERF_TUNE_TARGET`

## Telegram Commands

- `/help`
- `/status`
- `/report60`
- `/report6h`
- `/positions [개수]`
- `/pending`
- `/pnl`
- `/risk`
- `/gate`
- `/data`
- `/pause`
- `/resume`
- `/kill_on`
- `/kill_off`
- `/go_live`
- `/go_paper`
- `/set_minpos <usd>`
- `/clear_minpos`
- `/approve <signal_id>`
- `/reject <signal_id>`
- `/reject_all`
- `/close <position_id>`
- `/close_market <market_id>`
- `/close_side <YES|NO> [limit]`
- `/closeall`
- `/panic`

## Setup (WSL)

```bash
cd /mnt/d/Donggri_Platform/Polymethemoney
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Fill `.env` with Polymarket and Telegram credentials.

## Run

```bash
docker compose up -d --build
docker compose logs -f app
```

## Reset Paper Round

```bash
python scripts/reset_paper_round.py --confirm
docker compose up -d --build app
```

## Backfill + Retrain

```bash
./backfill_markets --since 2y --sources api,subgraph
./build_training_dataset --target both --lookback-days 730
./retrain_models --target both --kind full
```

## Tuning Controls

- `TREND_RECENCY_HALF_LIFE_DAYS`
- `TREND_INTRADAY_RETRAIN_MINUTES`
- `TREND_ADAPTIVE_ENABLED`
- `TREND_WINDOW_SIZE`
- `TREND_ALIGN_BOOST`
- `TREND_REVERT_PENALTY`
- `MICRO_REVERT_ENABLED`
- `MICRO_REVERT_IMB_THRESHOLD`
- `MICRO_REVERT_MAX_VOL`
- `MICRO_REVERT_BOOST`
- `POLYMARKET_USE_BULK_BOOKS` (`false` uses per-token `/book` fallback)
- `MAX_OPEN_PER_MARKET_SIDE` (prevent repeated stacking on same market+side)
- `SIGNAL_REENTRY_COOLDOWN_SECONDS`
- `SIGNAL_MIN_CONTRACT_PRICE` (hard floor for entry contract price)
- `SIGNAL_MIN_NET_EV`
- `SIGNAL_TAIL_PROB_FLOOR`
- `SIGNAL_TAIL_PROB_CEILING`
- `SIGNAL_TAIL_EXTRA_NET_EV`
- `SIGNAL_SIDE_BALANCE_ENABLED`
- `SIGNAL_SIDE_BALANCE_WINDOW`
- `SIGNAL_MAX_SIDE_RATIO`
- `SIGNAL_DOMINANT_SIDE_PENALTY`
- `SIGNAL_DOMINANT_SIDE_HARD_BLOCK` (`실거래 유사 paper에서는 false 권장`)
- `SIGNAL_SIDE_POLICY` (`YES_PRIORITY` 권장)
- `SIGNAL_NO_MIN_NET_EV`
- `SIGNAL_NO_MIN_CONFIDENCE`
- `SIGNAL_NO_REGIME_TREND_BIAS`
- `SIGNAL_NO_REGIME_IMBALANCE`
- `SIGNAL_NO_MAX_RATIO`
- `AUTO_TUNE_ZERO_FILL_ENABLED`
  - `AUTO_TUNE_WINDOW_MINUTES`
  - `AUTO_TUNE_MIN_SIGNALS`
  - `AUTO_TUNE_MIN_POSITION_USD`
  - `AUTO_TUNE_APPLY_ONCE`
  - `PAPER_PERF_TUNE_ENABLED`
  - `PAPER_PERF_TUNE_INTERVAL_HOURS`
  - `PAPER_PERF_TUNE_MIN_PNL_USD`
  - `PAPER_PERF_TUNE_MIN_TRADES`
  - `PAPER_PERF_TUNE_TARGET`
  - `PAPER_PERF_TUNE_ONLY_IN_PAPER`

## Historical Metrics Contract

`data/historical_metrics.json`

```json
{
  "pf": 1.08,
  "mdd_pct": 0.06,
  "ece": 0.03,
  "window_days": 60,
  "generated_at": "2026-03-21T00:00:00+00:00",
  "source": "clob_api"
}
```

Pass conditions:
- historical: `window_days >= 60`, `pf >= 1.05`, `mdd_pct <= 0.08`, `ece <= 0.05`
- paper: `covered_days >= 3`, `pf >= 1.10`, `max_dd_pct <= 0.04`, `trades >= 10`, `violations <= 0`
