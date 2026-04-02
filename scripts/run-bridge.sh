#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BRIDGE_DIR="${ROOT_DIR}/vendor/mirofish-oauthbridge/codex-bridge"

if [[ ! -d "${BRIDGE_DIR}" ]]; then
  echo "codex-bridge not found: ${BRIDGE_DIR}"
  exit 1
fi

cd "${BRIDGE_DIR}"

if [[ ! -d "node_modules" ]]; then
  echo "Installing bridge dependencies..."
  npm install
fi

export PORT=8787
export BRIDGE_PROVIDER=gemini
export GEMINI_MODEL=gemini-2.5-flash
export CODEX_BRIDGE_WORKDIR="${ROOT_DIR}"

echo "Starting codex-bridge on http://127.0.0.1:8787 ..."
npm start
