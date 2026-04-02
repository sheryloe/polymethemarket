# Polymethemoney

High‑frequency, risk‑aware Polymarket paper trading engine with Telegram control, auto‑tuning, and dual‑model signals.

Keywords: prediction market, Polymarket, trading bot, paper trading, risk engine, Telegram bot, algorithmic trading

## What This Is

- Paper‑only by default (live execution is blocked unless gate + manual approval).
- Dual model: intraday + settlement signals with calibration.
- Gate policy: historical 60d + paper 3d before live.
- Telegram commands for control, status, and reports.

## Quick Start (WSL)

```bash
cd /mnt/d/Donggri_Platform/Polymethemoney
python3 -m venv .venv
source .venv/bin/activate
pip install -e "[dev]"
cp .env.example .env
```

### 1) Create Telegram Bot (BotFather)

1. Open Telegram and search `@BotFather`.
2. Send `/newbot` and follow prompts.
3. Save the bot token (format: `123456:ABC...`).
4. Search `@userinfobot`, send `/start` to get your numeric chat ID.

Put into `.env`:
```
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID
```

### 2) Polymarket API Keys

You need API key/secret/passphrase derived from your wallet private key.

```bash
pip install py-clob-client python-dotenv
python - <<'PY'
import os
from dotenv import load_dotenv
from py_clob_client.client import ClobClient

load_dotenv('.env')

pk = os.getenv('POLYMARKET_PRIVATE_KEY', '').strip()
if not pk:
    raise SystemExit('POLYMARKET_PRIVATE_KEY missing')

client = ClobClient(
    host='https://clob.polymarket.com',
    chain_id=int(os.getenv('POLYMARKET_CHAIN_ID', '137')),
    key=pk,
    signature_type=int(os.getenv('POLYMARKET_SIGNATURE_TYPE', '0')),
    funder=(os.getenv('POLYMARKET_FUNDER') or None),
)
creds = client.create_or_derive_api_creds()

print('POLYMARKET_API_KEY=' + creds.api_key)
print('POLYMARKET_API_SECRET=' + creds.api_secret)
print('POLYMARKET_API_PASSPHRASE=' + creds.api_passphrase)
PY
```

Put outputs into `.env`:
```
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...
```

### 3) Run (Docker)

```bash
docker compose up -d --build
docker compose logs -f polymethemoney_app
```

## Core Commands (Telegram)

- `/status`
- `/report60`
- `/report6h`
- `/positions [N]`
- `/pnl`
- `/risk`
- `/pause` / `/resume`
- `/closeall`
- `/go_live` (blocked unless gate passes + manual approval)

## Security

- `.env` contains secrets and must never be committed.
- Use `.env.example` for safe defaults.

## GitHub Pages

This repo includes a landing page in `docs/`.

Steps:
1. GitHub → Settings → Pages
2. Source: `Deploy from a branch`
3. Branch: `main` (or your default) + Folder: `/docs`
4. Save

Your site will be published at:
`https://<user>.github.io/<repo>/`

## Troubleshooting

- If no trades occur: check `report60` and rejection reasons.
- If paper fills are 0: set `PAPER_POST_ONLY=false` and rebuild.
- If training rows are 0: remove `data/training_*.csv` then restart.
