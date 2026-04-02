from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from polymethemoney.config import Settings


@dataclass(slots=True)
class LLMResponse:
    ok: bool
    error: str | None
    payload: dict[str, Any] | None


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = bool(settings.llm_enabled and settings.llm_tuning_enabled)
        self.base = (settings.llm_api_base or "").rstrip("/")
        self.model = settings.llm_model or "gemini:gemini-2.5-flash"
        self.timeout = max(5, int(settings.llm_tuning_timeout_seconds))

    def _root_base(self) -> str:
        if self.base.endswith("/v1"):
            return self.base[:-3]
        return self.base

    async def health(self) -> bool:
        if not self.enabled or not self.base:
            return False
        url = f"{self._root_base()}/health"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

    async def suggest_tuning(self, goal: str, payload: dict[str, Any]) -> LLMResponse:
        if not self.enabled or not self.base:
            return LLMResponse(ok=False, error="LLM_DISABLED", payload=None)

        system = (
            "You are a trading-system tuning assistant. "
            "Return ONLY JSON. No markdown. No extra text. "
            "Schema: {\"changes\":[{\"key\":\"STRING\",\"value\":NUMBER,\"reason\":\"STRING\"}]}"
        )
        user = {
            "goal": goal.strip() or "tuning",
            "context": payload,
            "rules": {
                "max_changes": int(self.settings.llm_tuning_max_changes),
                "apply_runtime_only": True,
                "keys_must_be_in_allowlist": True,
            },
        }

        req = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base}/chat/completions", json=req)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return LLMResponse(ok=False, error=str(exc), payload=None)

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = self._parse_json(content)
        if parsed is None:
            return LLMResponse(ok=False, error="INVALID_JSON", payload=None)
        if not isinstance(parsed, dict) or "changes" not in parsed:
            return LLMResponse(ok=False, error="INVALID_SCHEMA", payload=None)
        return LLMResponse(ok=True, error=None, payload=parsed)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
