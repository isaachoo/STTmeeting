"""End-to-end test of the WebSocket wiring, with no external network.

Runs the real Flask app on a real port and talks to it with a real WebSocket
client, exactly as the browser does -- start_meeting, binary audio frames,
Deepgram result frames, stop_meeting -- with Deepgram's socket and OpenRouter
replaced by fakes. This is what catches breakage in app.py, which the unit tests
never touch.

Run with:  python -m unittest discover -s tests -v
"""

import json
import socket as socketlib
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

_TMP = tempfile.TemporaryDirectory()
config.DATA_DIR = Path(_TMP.name)
config.AUDIO_DIR = config.DATA_DIR / "audio"
config.DB_PATH = config.DATA_DIR / "socketflow.sqlite3"

import simple_websocket  # noqa: E402

import stt.deepgram_live as dg  # noqa: E402


# ------------------------------------------------------------------ fake Deepgram


class FakeWebSocketApp:
    """Stands in for websocket.WebSocketApp: opens, accepts sends, replays frames."""

    latest: "FakeWebSocketApp | None" = None

    def __init__(self, url, header=None, on_open=None, on_message=None,
                 on_error=None, on_close=None):
        self.url = url
        self.header = header or {}
        self.on_open = on_open
        self.on_message = on_message
        self.on_close = on_close
        self.audio_bytes = 0
        self.text_frames: list[str] = []
        self.opened = threading.Event()
        self._closed = threading.Event()
        FakeWebSocketApp.latest = self

    def run_forever(self, **_kwargs):
        self.on_open(self)
        self.opened.set()
        self._closed.wait(20)
        self.on_close(self, 1000, "closed by test")

    def send(self, data, opcode=None):
        if isinstance(data, (bytes, bytearray)):
            self.audio_bytes += len(data)
        else:
            self.text_frames.append(data)
            if '"CloseStream"' in data:
                self._closed.set()

    def close(self):
        self._closed.set()

    def push(self, frame: dict):
        self.on_message(self, json.dumps(frame))

    def push_utterance(self, text: str, speaker: int = 0):
        """Interim, then final with an endpoint -- the real Deepgram sequence."""
        self.push(_results(text[: max(1, len(text) // 2)], False, speaker=speaker))
        self.push(_results(text, True, speech_final=True, speaker=speaker))


def _results(text, is_final, speech_final=False, speaker=0):
    return {
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "start": 0.0,
        "duration": 1.0,
        "channel": {
            "alternatives": [
                {
                    "transcript": text,
                    "words": [
                        {"word": w, "speaker": speaker} for w in (text.split() or [text])
                    ],
                }
            ]
        },
    }


# ----------------------------------------------------------------- fake OpenRouter

ADVICE_JSON = {
    "key_point": "Alan 想加兩個 headcount",
    "suggested_questions": ["每個 head 的 fully loaded cost 係幾多?"],
    "watch_out": "budget model 未有人負責",
    "question_to_answer": None,
}

NOTES_JSON = {
    "summary": "討論 Q3 headcount。",
    "decisions": ["Q3 加一個 head"],
    "action_items": [{"who": "Wing", "what": "update budget model", "due": "Friday"}],
    "open_questions": ["財務部幾時批?"],
    "topics": ["headcount", "budget"],
}


class FakeLLM:
    def __init__(self):
        from copilot.llm import Usage

        self.usage = Usage()
        self.prompts: list[str] = []
        self.json_reply = None  # set by a test to override
        self._lock = threading.Lock()

    def chat_json(self, messages, **_kwargs):
        with self._lock:
            self.prompts.append(messages[1]["content"])
        self.usage.add({"prompt_tokens": 900, "completion_tokens": 120, "cost": 0.0004})
        if "note taker" in messages[0]["content"]:
            return dict(NOTES_JSON)
        return dict(self.json_reply or ADVICE_JSON)

    def chat(self, messages, **_kwargs):
        with self._lock:
            self.prompts.append(messages[1]["content"])
        self.usage.add({"prompt_tokens": 500, "completion_tokens": 90, "cost": 0.0002})
        return "A short answer."


# --------------------------------------------------------------------- harness


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def free_port() -> int:
    with socketlib.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Browser:
    """A real WebSocket client that collects server events in the background."""

    def __init__(self, port: int):
        self.ws = simple_websocket.Client(f"ws://127.0.0.1:{port}/ws")
        self.events: dict[str, list[dict]] = {}
        self.order: list[str] = []
        self._lock = threading.Lock()
        self._closed = False
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        while not self._closed:
            try:
                raw = self.ws.receive(timeout=0.1)
            except Exception:
                return
            if raw is None:
                continue
            message = json.loads(raw)
            with self._lock:
                self.events.setdefault(message["event"], []).append(
                    message.get("data") or {}
                )
                self.order.append(message["event"])

    def emit(self, event: str, data: dict | None = None):
        self.ws.send(json.dumps({"event": event, "data": data or {}}))

    def send_audio(self, chunk: bytes):
        self.ws.send(chunk)

    def got(self, event: str) -> list[dict]:
        with self._lock:
            return list(self.events.get(event, []))

    def wait(self, event: str, count: int = 1, timeout: float = 10.0) -> list[dict]:
        ok = wait_for(lambda: len(self.got(event)) >= count, timeout)
        if not ok:
            with self._lock:
                seen = {k: len(v) for k, v in self.events.items()}
            raise AssertionError(f"never received {count}x {event!r}; saw {seen}")
        return self.got(event)

    def clear(self):
        with self._lock:
            self.events.clear()
            self.order.clear()

    def close(self):
        self._closed = True
        try:
            self.ws.close()
        except Exception:
            pass


_SERVER: dict = {}


def setUpModule():
    """One real server for the whole module -- a Flask app can only run once."""
    config.DEEPGRAM_API_KEY = "dg-test"
    config.OPENROUTER_API_KEY = "or-test"
    config.ADVICE_MIN_INTERVAL = 0
    config.ADVICE_MIN_NEW_CHARS = 1
    config.NOTES_INTERVAL = 0

    import app as app_module
    from storage import db

    db.init()
    port = free_port()
    threading.Thread(
        target=app_module.app.run,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "threaded": True,
            "debug": False,
            "use_reloader": False,
        },
        daemon=True,
    ).start()

    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socketlib.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("test server never came up")

    _SERVER.update(port=port, app_module=app_module, db=db)


class LiveServerCase(unittest.TestCase):
    """Talks to the module-wide server over a real WebSocket."""

    @property
    def port(self):
        return _SERVER["port"]

    @property
    def app_module(self):
        return _SERVER["app_module"]

    @property
    def db(self):
        return _SERVER["db"]

    def setUp(self):
        dg.websocket.WebSocketApp = FakeWebSocketApp
        FakeWebSocketApp.latest = None

        self.llm = FakeLLM()
        import session as session_module

        real_client = session_module.OpenRouterClient
        session_module.OpenRouterClient = lambda **_kw: self.llm
        self.addCleanup(setattr, session_module, "OpenRouterClient", real_client)

        self.app_module._session = None
        self.browser = Browser(self.port)
        self.addCleanup(self.browser.close)
        self.browser.wait("snapshot")
        self.browser.clear()

    def tearDown(self):
        session = self.app_module._session
        if session is not None and not session.stopped:
            session.stopped = True
            session.stt.stop()
            session.engine.close(wait=False)
        self.app_module._session = None

    def start_meeting(self, brief="Q3 budget 會議，同 Alan 同 Wing 開。", rate=16000):
        self.browser.emit(
            "start_meeting", {"title": "Q3 planning", "brief": brief, "sample_rate": rate}
        )
        self.browser.wait("meeting_started")
        self.assertTrue(wait_for(lambda: FakeWebSocketApp.latest is not None))
        ws = FakeWebSocketApp.latest
        self.assertTrue(ws.opened.wait(5))
        return ws


# ----------------------------------------------------------------------- tests


class TestMeetingLifecycle(LiveServerCase):
    def test_full_meeting_lifecycle(self):
        ws = self.start_meeting()

        # The connection Deepgram was asked for reflects our configuration.
        self.assertIn("language=zh-HK", ws.url)
        self.assertIn("sample_rate=16000", ws.url)
        self.assertIn("model=nova-3", ws.url)
        self.assertEqual(ws.header["Authorization"], "Token dg-test")

        statuses = self.browser.wait("status")
        self.assertTrue(any(s.get("state") == "listening" for s in statuses), statuses)

        # Binary audio frames from the browser reach Deepgram.
        chunk = b"\x11\x22" * 2048
        for _ in range(5):
            self.browser.send_audio(chunk)
        self.assertTrue(
            wait_for(lambda: ws.audio_bytes >= len(chunk) * 5),
            f"only {ws.audio_bytes} bytes arrived",
        )

        # Deepgram speaks; transcript, advice and notes follow.
        ws.push_utterance("我想爭取多兩個 headcount", speaker=0)

        segments = self.browser.wait("segment")
        self.assertEqual(segments[0]["text"], "我想爭取多兩個 headcount")
        self.assertEqual(segments[0]["speaker_label"], "S1")

        advice = self.browser.wait("advice")
        self.assertEqual(advice[0]["key_point"], "Alan 想加兩個 headcount")
        self.assertEqual(advice[0]["questions"], ["每個 head 的 fully loaded cost 係幾多?"])
        self.assertEqual(advice[0]["watch_out"], "budget model 未有人負責")

        notes = self.browser.wait("notes")
        self.assertEqual(notes[-1]["notes"]["decisions"], ["Q3 加一個 head"])
        self.assertEqual(notes[-1]["notes"]["action_items"][0]["who"], "Wing")

        # Cost comes from real reported usage and covers both halves of the bill.
        def priced():
            costs = self.browser.got("cost")
            return costs and costs[-1]["llm_usd"] > 0

        self.assertTrue(wait_for(priced), "LLM cost never appeared")
        latest = self.browser.got("cost")[-1]
        self.assertGreater(latest["audio_minutes"], 0)
        self.assertAlmostEqual(
            latest["total_usd"], latest["stt_usd"] + latest["llm_usd"], places=4
        )
        self.assertEqual(latest["stt_model"], "nova-3")

        # Stopping closes the Deepgram stream cleanly and persists everything.
        meeting_id = self.app_module._session.meeting_id
        self.browser.emit("stop_meeting")
        self.browser.wait("meeting_stopped", timeout=20)
        self.assertTrue(any('"CloseStream"' in f for f in ws.text_frames))

        stored = self.db.get_meeting(meeting_id)
        self.assertIsNotNone(stored["ended_at"])
        self.assertEqual([s["text"] for s in stored["segments"]], ["我想爭取多兩個 headcount"])
        self.assertEqual(stored["notes_json"]["decisions"], ["Q3 加一個 head"])
        self.assertGreater(stored["audio_seconds"], 0)
        self.assertIn("advice", [e["kind"] for e in stored["events"]])

    def test_interim_results_reach_the_browser(self):
        ws = self.start_meeting()
        ws.push(_results("我想爭", is_final=False))
        self.assertEqual(self.browser.wait("interim")[-1]["text"], "我想爭")

    def test_keepalive_is_sent_while_nobody_is_talking(self):
        ws = self.start_meeting()
        self.assertTrue(
            wait_for(lambda: any('"KeepAlive"' in f for f in ws.text_frames), timeout=15),
            "an idle Deepgram socket must be kept alive or it closes",
        )

    def test_audio_after_stop_is_dropped(self):
        ws = self.start_meeting()
        self.browser.emit("stop_meeting")
        self.browser.wait("meeting_stopped", timeout=20)
        before = ws.audio_bytes
        self.browser.send_audio(b"\x00\x01" * 1024)
        time.sleep(0.3)
        self.assertEqual(ws.audio_bytes, before)


class TestCopilotOverTheWire(LiveServerCase):
    def test_brief_is_given_to_the_advisor(self):
        ws = self.start_meeting(brief="留意 project 代號 Falcon")
        ws.push_utterance("Falcon 幾時 launch?")
        self.browser.wait("advice")
        self.assertTrue(
            any("Falcon" in prompt for prompt in self.llm.prompts),
            "the pre-meeting brief must be in the advisor prompt",
        )

    def test_a_question_in_the_meeting_gets_answered(self):
        self.llm.json_reply = {
            **ADVICE_JSON,
            "question_to_answer": {
                "question": "最低工資係幾多?",
                "needs_web": True,
                "search_query": "Hong Kong minimum wage",
            },
        }
        ws = self.start_meeting()
        ws.push_utterance("最低工資係幾多?")

        answers = self.browser.wait("answer")
        self.assertEqual(answers[0]["answer"], "A short answer.")
        self.assertFalse(answers[0]["from_user"])
        self.assertFalse(answers[0]["web_enabled"], "no TAVILY_API_KEY in tests")

    def test_user_can_ask_mid_meeting(self):
        self.start_meeting()
        self.browser.emit("ask", {"question": "點樣講服財務部?"})
        answers = self.browser.wait("answer")
        self.assertTrue(answers[0]["from_user"])
        self.assertEqual(answers[0]["question"], "點樣講服財務部?")

    def test_user_notes_persist(self):
        self.start_meeting()
        meeting_id = self.app_module._session.meeting_id
        self.browser.emit("user_notes", {"text": "我自己嘅筆記"})
        self.assertTrue(
            wait_for(lambda: self.db.get_meeting(meeting_id)["user_notes"] == "我自己嘅筆記")
        )


class TestReconnectAndGuards(LiveServerCase):
    def test_a_second_tab_gets_a_snapshot_and_shares_the_meeting(self):
        ws = self.start_meeting()
        ws.push_utterance("第一句話")
        self.browser.wait("segment")

        second = Browser(self.port)
        self.addCleanup(second.close)
        snapshot = second.wait("snapshot")[-1]
        self.assertTrue(snapshot["running"])
        self.assertEqual(snapshot["segments"][0]["text"], "第一句話")
        self.assertEqual(snapshot["brief"], "Q3 budget 會議，同 Alan 同 Wing 開。")
        self.assertGreaterEqual(len(snapshot["cards"]), 1)

        # Both tabs see live events from here on.
        ws.push_utterance("第二句話")
        self.assertEqual(second.wait("segment")[-1]["text"], "第二句話")
        self.assertEqual(self.browser.wait("segment", 2)[-1]["text"], "第二句話")

    def test_resync_replays_the_meeting(self):
        ws = self.start_meeting()
        ws.push_utterance("一句話")
        self.browser.wait("segment")
        self.browser.emit("resync")
        snapshot = self.browser.wait("snapshot")[-1]
        self.assertEqual(snapshot["segments"][0]["text"], "一句話")

    def test_a_second_meeting_is_refused_while_one_runs(self):
        self.start_meeting()
        self.browser.emit("start_meeting", {"title": "another", "sample_rate": 16000})
        errors = self.browser.wait("error")
        self.assertIn("already running", errors[0]["message"])

    def test_bad_sample_rate_is_rejected_before_dialling_deepgram(self):
        self.browser.emit("start_meeting", {"title": "x", "sample_rate": 96000})
        errors = self.browser.wait("error")
        self.assertIn("96000", errors[0]["message"])
        self.assertIsNone(FakeWebSocketApp.latest)

    def test_missing_api_keys_block_the_meeting(self):
        config.DEEPGRAM_API_KEY = ""
        self.addCleanup(setattr, config, "DEEPGRAM_API_KEY", "dg-test")
        self.browser.emit("start_meeting", {"title": "x", "sample_rate": 16000})
        errors = self.browser.wait("error")
        self.assertIn("DEEPGRAM_API_KEY", errors[0]["message"])

    def test_malformed_frames_do_not_kill_the_connection(self):
        self.browser.ws.send("not json at all")
        self.browser.ws.send(json.dumps({"event": "nonsense", "data": {}}))
        self.browser.emit("resync")
        self.browser.wait("snapshot")  # still talking to us


class TestModelFallback(unittest.TestCase):
    """A model that never returns a result must hand over to the next one."""

    def setUp(self):
        dg.websocket.WebSocketApp = FailFirstModelWS
        FailFirstModelWS.urls.clear()

    def test_falls_back_to_the_next_model(self):
        statuses: list[dict] = []
        stt = dg.DeepgramLiveSTT(
            api_key="k",
            sample_rate=16000,
            language="zh-HK",
            models=["nova-3", "nova-2"],
            on_status=lambda **kw: statuses.append(kw),
            on_error=lambda msg: statuses.append({"state": "error", "detail": msg}),
        )
        self.addCleanup(stt.stop)
        stt.start()
        self.assertTrue(
            wait_for(lambda: any("model=nova-2" in u for u in FailFirstModelWS.urls)),
            f"never tried nova-2; tried {FailFirstModelWS.urls}",
        )
        self.assertTrue(any(s.get("state") == "fallback" for s in statuses), statuses)
        self.assertEqual(stt.model, "nova-2")

    def test_gives_up_with_an_error_when_every_model_is_rejected(self):
        FailFirstModelWS.reject_all = True
        self.addCleanup(setattr, FailFirstModelWS, "reject_all", False)
        errors: list[str] = []
        stt = dg.DeepgramLiveSTT(
            api_key="k",
            sample_rate=16000,
            models=["nova-3", "nova-2"],
            on_error=errors.append,
        )
        self.addCleanup(stt.stop)
        stt.start()
        self.assertTrue(wait_for(lambda: errors), "must report an unusable configuration")
        self.assertIn("rejected every configured model", errors[0])


class FailFirstModelWS(FakeWebSocketApp):
    """Rejects nova-3 outright; behaves normally for anything else."""

    urls: list[str] = []
    reject_all = False

    def __init__(self, url, **kwargs):
        super().__init__(url, **kwargs)
        FailFirstModelWS.urls.append(url)
        self._reject = FailFirstModelWS.reject_all or "model=nova-3" in url

    def run_forever(self, **_kwargs):
        self.on_open(self)
        self.opened.set()
        if self._reject:
            self.on_close(self, 1008, "model not available for this language")
            return
        self.push(_results("hello", is_final=False))
        self._closed.wait(3)
        self.on_close(self, 1000, "done")


if __name__ == "__main__":
    unittest.main(verbosity=2)
