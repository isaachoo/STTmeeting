"""OpenRouter chat client.

Small on purpose: one blocking call, JSON-object mode, and usage accounting so
the UI can show a running cost. OpenRouter reports the real credit cost per
request when asked, which beats guessing from a price table.
"""

import json
import logging
import re
import threading
import time

import requests

import config

log = logging.getLogger(__name__)

TIMEOUT = (10, 90)  # connect, read -- the live meeting; a slow answer there is useless anyway
REVIEW_TIMEOUT = (10, 300)  # after the meeting a long section may take minutes to write


class LLMError(RuntimeError):
    def __init__(self, message: str, *, empty: bool = False, finish_reason: str = ""):
        super().__init__(message)
        self.empty = empty  # the model answered with nothing at all
        self.finish_reason = finish_reason


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

    def seed(self, snapshot: dict | None) -> None:
        """Start from a previous total -- a resumed meeting's earlier spend."""
        if not snapshot:
            return
        with self._lock:
            self.calls += int(snapshot.get("calls") or 0)
            self.prompt_tokens += int(snapshot.get("prompt_tokens") or 0)
            self.completion_tokens += int(snapshot.get("completion_tokens") or 0)
            try:
                self.cost_usd += float(snapshot.get("cost_usd") or 0.0)
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
        timeout: tuple | None = None,
        label: str = "",
    ) -> str:
        """One completion. `label` names the call in the log ("digest 3/14"), so
        a long job is visibly alive in the console instead of silent until it
        fails; unlabelled calls (the live meeting's steady stream) log at DEBUG."""
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        started = time.monotonic()
        prompt_chars = sum(len(m.get("content") or "") for m in messages)

        body = {
            "model": model or config.OPENROUTER_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Ask OpenRouter to report what the call actually cost.
            "usage": {"include": True},
            # "Thinking" models (DeepSeek V3.2 among them) reason in private
            # before answering, and that reasoning is charged against
            # max_tokens. On a long prompt it can use the whole budget and the
            # visible answer comes back empty. Nothing in this app wants
            # deliberation over speed and cost, so reasoning is off unless the
            # user turns it on. OpenRouter ignores this for models without it.
            "reasoning": {"enabled": config.OPENROUTER_REASONING},
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
                timeout=timeout or TIMEOUT,
            )
        except requests.RequestException as exc:
            log.warning("llm %s: request failed after %.1fs: %s",
                        label or "call", time.monotonic() - started, exc)
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
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        usage = payload.get("usage") or {}
        log.log(
            logging.INFO if label else logging.DEBUG,
            "llm %s: %s, %d chars in, %s tokens out, %.1fs, finish=%s",
            label or "call", body["model"], prompt_chars,
            usage.get("completion_tokens", "?"), time.monotonic() - started,
            choice.get("finish_reason") or "?",
        )
        if not content.strip():
            # Say what actually happened; "did not return JSON" hid this for a
            # whole afternoon. The usual cause is the output budget being spent
            # on reasoning, which the finish_reason and token counts reveal.
            finish = choice.get("finish_reason") or choice.get("native_finish_reason") or "?"
            details = payload.get("usage") or {}
            reasoning_tokens = (details.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            )
            hint = ""
            if finish == "length":
                hint = (
                    " -- the model ran out of output budget"
                    + (f" after {reasoning_tokens} reasoning tokens" if reasoning_tokens else "")
                    + "; raise max_tokens or use a model that does not reason first"
                )
            elif message.get("reasoning") or message.get("reasoning_content"):
                hint = " -- it returned reasoning but no answer"
            raise LLMError(
                f"model {body['model']} returned an empty answer (finish_reason={finish}){hint}",
                empty=True,
                finish_reason=finish,
            )
        return content

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        """Chat and parse a JSON object, tolerating fences and stray prose.

        An empty answer that ran out of budget is retried once with twice the
        room -- long transcript sections on a verbose model need it -- before
        giving up with the real reason.
        """
        kwargs.setdefault("json_mode", True)
        try:
            raw = self.chat(messages, **kwargs)
        except LLMError as exc:
            if not (exc.empty and exc.finish_reason == "length"):
                raise
            kwargs["max_tokens"] = int(kwargs.get("max_tokens", 700)) * 2
            log.warning("empty answer at the token limit; retrying with max_tokens=%s",
                        kwargs["max_tokens"])
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
