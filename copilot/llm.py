"""OpenRouter chat client.

Small on purpose: one blocking call, JSON-object mode, and usage accounting so
the UI can show a running cost. OpenRouter reports the real credit cost per
request when asked, which beats guessing from a price table.
"""

import json
import logging
import re
import threading

import requests

import config

log = logging.getLogger(__name__)

TIMEOUT = (10, 90)  # connect, read


class LLMError(RuntimeError):
    pass


class Usage:
    """Thread-safe running total of LLM spend for one meeting."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

    def add(self, usage: dict | None) -> None:
        with self._lock:
            self.calls += 1
            if not usage:
                return
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            # OpenRouter returns `cost` in USD when usage accounting is enabled.
            try:
                self.cost_usd += float(usage.get("cost") or 0.0)
            except (TypeError, ValueError):
                pass

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 5),
            }


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, usage: Usage | None = None):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.usage = usage or Usage()
        self._session = requests.Session()

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 700,
        json_mode: bool = False,
    ) -> str:
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")

        body = {
            "model": model or config.OPENROUTER_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Ask OpenRouter to report what the call actually cost.
            "usage": {"include": True},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        try:
            resp = self._session.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "Cantonese Meeting Copilot",
                },
                json=body,
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            raise LLMError(f"OpenRouter request failed: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(
                f"OpenRouter returned {resp.status_code}: {resp.text[:400]}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise LLMError("OpenRouter returned a non-JSON body") from exc

        if "error" in payload and not payload.get("choices"):
            raise LLMError(f"OpenRouter error: {payload['error']}")

        self.usage.add(payload.get("usage"))

        choices = payload.get("choices") or []
        if not choices:
            raise LLMError("OpenRouter returned no choices")
        return (choices[0].get("message") or {}).get("content") or ""

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        """Chat and parse a JSON object, tolerating fences and stray prose."""
        kwargs.setdefault("json_mode", True)
        raw = self.chat(messages, **kwargs)
        parsed = extract_json(raw)
        if parsed is None:
            raise LLMError(f"model did not return JSON: {raw[:300]}")
        return parsed


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Best-effort JSON object extraction from a model response."""
    if not text:
        return None
    candidates = [text.strip()]
    match = _FENCE.search(text)
    if match:
        candidates.insert(0, match.group(1).strip())

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            value = None
        if isinstance(value, dict):
            return value

    # Fall back to the first balanced {...} span.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : i + 1])
                    except ValueError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    return None
