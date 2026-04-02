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

## Gemini OAuth Bridge (LLM 튜닝)

이 프로젝트는 `mirofish-ko-oauthbridge`의 `codex-bridge`를 벤더링해 LLM 튜닝에 사용합니다.
로컬 Gemini CLI 로그인 상태가 필요합니다.

### 브리지 실행 (WSL)

```bash
cd /mnt/d/Donggri_Platform/Polymethemoney
bash scripts/run-bridge.sh
```

### 브리지 실행 (PowerShell)

```powershell
cd D:\Donggri_Platform\Polymethemoney
.\scripts\run-bridge.ps1
```

### LLM 튜닝 활성화 (.env)

```
LLM_ENABLED=true
LLM_API_BASE=http://127.0.0.1:8787/v1
LLM_MODEL=gemini:gemini-2.5-flash
LLM_TUNING_ENABLED=true
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
- `/tuning <goal>` (LLM 튜닝 제안 런타임 적용)
- `/tuning_reset` (튜닝 오버라이드 해제)

## Security

- `.env` contains secrets and must never be committed.
- Use `.env.example` for safe defaults.

## License Notice (AGPL)

`vendor/mirofish-oauthbridge`에는 AGPL 라이선스가 포함됩니다.
해당 디렉터리의 `LICENSE` 및 `NOTICE.md`를 준수해야 합니다.

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
