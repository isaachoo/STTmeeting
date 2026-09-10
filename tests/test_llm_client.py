"""The OpenRouter client against a fake HTTP session.

Written after a real failure: DeepSeek V3.2 spent its whole output budget on
private reasoning, returned an empty answer, and the app reported "model did not
return JSON:" with nothing after the colon -- for every section of a meeting.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from copilot.llm import LLMError, OpenRouterClient  # noqa: E402


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Answers in order from a queue and remembers every request body."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.bodies.append(json)
        return self.responses.pop(0)


def reply(content, finish="stop", reasoning_tokens=None, reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0001}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return FakeResponse({"choices": [{"message": message, "finish_reason": finish}], "usage": usage})


def client(*responses):
    c = OpenRouterClient(api_key="test-key")
    c._session = FakeSession(*responses)
    return c


MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


class TestReasoningIsOff(unittest.TestCase):
    def test_every_request_turns_reasoning_off_by_default(self):
        c = client(reply("hi"))
        c.chat(MESSAGES)
        self.assertEqual(c._session.bodies[0]["reasoning"], {"enabled": False})

    def test_the_user_can_turn_it_on(self):
        original = config.OPENROUTER_REASONING
        self.addCleanup(setattr, config, "OPENROUTER_REASONING", original)
        config.OPENROUTER_REASONING = True
        c = client(reply("hi"))
        c.chat(MESSAGES)
        self.assertEqual(c._session.bodies[0]["reasoning"], {"enabled": True})


class TestEmptyAnswers(unittest.TestCase):
    def test_an_empty_answer_says_why_not_did_not_return_json(self):
        c = client(reply("", finish="length", reasoning_tokens=1600))
        with self.assertRaises(LLMError) as caught:
            c.chat(MESSAGES, model="deepseek/deepseek-v3.2")
        text = str(caught.exception)
        self.assertIn("empty answer", text)
        self.assertIn("deepseek/deepseek-v3.2", text)
        self.assertIn("1600 reasoning tokens", text)
        self.assertIn("finish_reason=length", text)
        self.assertTrue(caught.exception.empty)

    def test_reasoning_without_an_answer_is_named(self):
        c = client(reply("", finish="stop", reasoning="I was thinking about it"))
        with self.assertRaises(LLMError) as caught:
            c.chat(MESSAGES)
        self.assertIn("reasoning but no answer", str(caught.exception))

    def test_whitespace_counts_as_empty(self):
        c = client(reply("   \n"))
        with self.assertRaises(LLMError):
            c.chat(MESSAGES)

    def test_chat_json_retries_once_with_double_the_budget(self):
        c = client(
            reply("", finish="length", reasoning_tokens=900),
            reply('{"ok": true}'),
        )
        result = c.chat_json(MESSAGES, max_tokens=800)
        self.assertEqual(result, {"ok": True})
        self.assertEqual([b["max_tokens"] for b in c._session.bodies], [800, 1600])

    def test_chat_json_does_not_retry_a_non_length_failure(self):
        c = client(reply("", finish="stop"))
        with self.assertRaises(LLMError):
            c.chat_json(MESSAGES)
        self.assertEqual(len(c._session.bodies), 1)

    def test_a_second_empty_answer_gives_up_with_the_reason(self):
        c = client(
            reply("", finish="length", reasoning_tokens=800),
            reply("", finish="length", reasoning_tokens=1600),
        )
        with self.assertRaises(LLMError) as caught:
            c.chat_json(MESSAGES, max_tokens=800)
        self.assertIn("ran out of output budget", str(caught.exception))
        self.assertEqual(len(c._session.bodies), 2)

    def test_a_good_answer_still_counts_usage(self):
        c = client(reply('{"a": 1}'))
        self.assertEqual(c.chat_json(MESSAGES), {"a": 1})
        self.assertEqual(c.usage.snapshot()["calls"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
