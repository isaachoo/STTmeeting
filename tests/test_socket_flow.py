"""End-to-end test of the WebSocket wiring, with no external network.

Runs the real Flask app on a real port and talks to it with a real WebSocket
client, exactly as the browser does -- start_meeting, binary audio frames,
Deepgram result frames, naming a speaker, stop_meeting -- with Deepgram's socket
and OpenRouter replaced by fakes. This is what catches breakage in app.py, which
the unit tests never touch.

Run with:  python -m unittest discover -s tests -v
"""

import json
import socket as socketlib
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
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

THINK_JSON = {
    "key_point": "Alan 想加兩個 headcount",
    "suggested_questions": ["每個 head 的 fully loaded cost 係幾多?"],
    "watch_out": "budget model 未有人負責",
    "attendee": {"should_speak": False, "kind": "info", "urgency": "normal",
                 "say": "", "why": "", "needs_web": False, "search_query": ""},
}

NOTES_JSON = {
    "summary": "討論 Q3 headcount。",
    "decisions": ["Q3 加一個 head"],
    "action_items": [{"who": "Wing", "what": "update budget model", "due": "Friday"}],
    "open_questions": ["財務部幾時批?"],
    "topics": ["headcount", "budget"],
}

BRIEF = {
    "title": "Q3 planning",
    "agenda": "budget, headcount",
    "my_goal": "爭取兩個 headcount",
    "context": "上次財務部話等 Q4",
    "attendees": [
        {"name": "Isaac", "role": "me", "is_me": True},
        {"name": "Alan", "role": "ops"},
        {"name": "Wing", "role": "finance"},
    ],
    "glossary": ["Falcon", "NocolyHAP"],
}


class FakeLLM:
    def __init__(self):
        from copilot.llm import Usage

        self.usage = Usage()
        self.prompts: list[str] = []
        self.think_reply = None   # set by a test to override
        self.speaker_reply = {"mapping": [], "note": ""}
        self._lock = threading.Lock()

    def chat_json(self, messages, **_kwargs):
        system = messages[0]["content"]
        with self._lock:
            self.prompts.append(messages[1]["content"])
        self.usage.add({"prompt_tokens": 900, "completion_tokens": 120, "cost": 0.0004})
        if "note taker" in system:
            return dict(NOTES_JSON)
        if "match anonymous voices" in system:
            return json.loads(json.dumps(self.speaker_reply))
        return json.loads(json.dumps(self.think_reply or THINK_JSON))

    def chat(self, messages, **_kwargs):
        with self._lock:
            self.prompts.append(messages[1]["content"])
        self.usage.add({"prompt_tokens": 500, "completion_tokens": 90, "cost": 0.0002})
        return "A short answer."

    def saw(self, needle: str) -> bool:
        with self._lock:
            return any(needle in prompt for prompt in self.prompts)


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
    """A real WebSocket client that collects server events in the background.

    Two allowances are made for the client library, neither of which reflects
    anything about the app (a real browser has neither problem, which is what
    the Chromium check covers):

    1. simple_websocket.Client is not thread-safe -- send and receive share one
       socket and one wsproto connection -- so every socket call takes `_sock`.
    2. Its handshake reads until it sees AcceptConnection and takes only that
       one wsproto event. When the HTTP 101 and the server's first data frame
       arrive in the same recv(), the frame stays queued inside wsproto while
       the reader thread blocks in recv() waiting for bytes that never come, so
       the first message is lost until the server happens to write again
       (measured: ~1.5% of connections). Sending `resync` on connect guarantees
       that second write, which flushes both frames through.
    """

    def __init__(self, port: int):
        self.ws = simple_websocket.Client(f"ws://127.0.0.1:{port}/ws")
        self.events: dict[str, list[dict]] = {}
        self.reader_error: str | None = None
        self.receive_calls = 0
        self._events_lock = threading.Lock()
        self._sock = threading.Lock()
        self._closed = False
        threading.Thread(target=self._read, daemon=True).start()
        # See (2) above: force a second server write so the snapshot cannot be
        # stranded inside the client library.
        self.emit("resync")

    def _read(self):
        while not self._closed:
            try:
                self.receive_calls += 1
                with self._sock:
                    raw = self.ws.receive(timeout=0.02)
            except Exception as exc:  # noqa: BLE001 - surfaced in assertions
                if not self._closed:
                    self.reader_error = f"{exc.__class__.__name__}: {exc}"
                return
            if raw is None:
                time.sleep(0.005)  # let a sender in
                continue
            message = json.loads(raw)
            with self._events_lock:
                self.events.setdefault(message["event"], []).append(message.get("data") or {})

    def _send(self, payload):
        with self._sock:
            self.ws.send(payload)

    def emit(self, event: str, data: dict | None = None):
        self._send(json.dumps({"event": event, "data": data or {}}))

    def send_audio(self, chunk: bytes):
        self._send(chunk)

    def got(self, event: str) -> list[dict]:
        with self._events_lock:
            return list(self.events.get(event, []))

    def wait(self, event: str, count: int = 1, timeout: float = 10.0) -> list[dict]:
        if not wait_for(lambda: len(self.got(event)) >= count, timeout):
            with self._events_lock:
                seen = {k: len(v) for k, v in self.events.items()}
            raise AssertionError(
                f"never received {count}x {event!r}; saw {seen}; "
                f"reader_error={self.reader_error}; "
                f"ws.connected={getattr(self.ws, 'connected', '?')}; "
                f"lib_thread_alive={self.ws.thread.is_alive()}; "
                f"input_buffer={len(getattr(self.ws, 'input_buffer', []))}; "
                f"receive_calls={self.receive_calls}"
            )
        return self.got(event)

    def request_snapshot(self, timeout: float = 10.0) -> dict:
        """Ask for a fresh snapshot and return that one.

        Discards any earlier snapshots first, so a caller can never assert
        against the one delivered at connect time.
        """
        with self._events_lock:
            self.events.pop("snapshot", None)
        self.emit("resync")
        return self.wait("snapshot", timeout=timeout)[-1]

    def clear(self):
        with self._events_lock:
            self.events.clear()

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
    config.THINK_MIN_INTERVAL = 0
    config.THINK_MIN_NEW_CHARS = 1
    config.THINK_URGENT_INTERVAL = 0
    config.NOTES_INTERVAL = 0
    config.SPEAKER_GUESS_INTERVAL = 9999  # off unless a test asks for it
    config.ATTENDEE_ENABLED = True

    import app as app_module
    from storage import db

    db.init()
    port = free_port()
    threading.Thread(
        target=app_module.app.run,
        kwargs={"host": "127.0.0.1", "port": port, "threaded": True,
                "debug": False, "use_reloader": False},
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

    def start_meeting(self, brief=None, rate=16000, language=None, model=None):
        payload = {"brief": brief if brief is not None else BRIEF, "sample_rate": rate}
        if language:
            payload["language"] = language
        if model:
            payload["model"] = model
        self.browser.emit("start_meeting", payload)
        self.browser.wait("meeting_started")
        self.assertTrue(wait_for(lambda: FakeWebSocketApp.latest is not None))
        ws = FakeWebSocketApp.latest
        self.assertTrue(ws.opened.wait(5))
        return ws

    def get(self, path: str) -> tuple[int, str, dict]:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
            return r.status, r.read().decode(), dict(r.headers)


# ----------------------------------------------------------------------- tests


class TestMeetingLifecycle(LiveServerCase):
    def test_full_meeting_lifecycle(self):
        ws = self.start_meeting()

        # The connection Deepgram was asked for reflects the brief and config.
        self.assertIn("language=zh-HK", ws.url)
        self.assertIn("sample_rate=16000", ws.url)
        self.assertIn("model=nova-3", ws.url)
        self.assertIn("diarize=true", ws.url)
        self.assertIn("keyterm=Falcon", ws.url, "glossary must be boosted")
        self.assertIn("keyterm=Alan", ws.url, "names must be boosted")
        self.assertEqual(ws.header["Authorization"], "Token dg-test")

        statuses = self.browser.wait("status")
        self.assertTrue(any(s.get("state") == "listening" for s in statuses), statuses)

        # Binary audio frames from the browser reach Deepgram.
        chunk = b"\x11\x22" * 2048
        for _ in range(5):
            self.browser.send_audio(chunk)
        self.assertTrue(wait_for(lambda: ws.audio_bytes >= len(chunk) * 5))

        # Deepgram speaks; transcript, coaching and notes follow.
        ws.push_utterance("我想爭取多兩個 headcount", speaker=0)

        segments = self.browser.wait("segment")
        self.assertEqual(segments[0]["text"], "我想爭取多兩個 headcount")
        self.assertEqual(segments[0]["speaker_label"], "S1")

        advice = self.browser.wait("advice")
        self.assertEqual(advice[0]["key_point"], "Alan 想加兩個 headcount")

        notes = self.browser.wait("notes")
        self.assertEqual(notes[-1]["notes"]["decisions"], ["Q3 加一個 head"])

        # The whole brief reached the model.
        self.assertTrue(self.llm.saw("爭取兩個 headcount"))
        self.assertTrue(self.llm.saw("Alan (ops)"))

        def priced():
            costs = self.browser.got("cost")
            return costs and costs[-1]["llm_usd"] > 0

        self.assertTrue(wait_for(priced), "LLM cost never appeared")
        latest = self.browser.got("cost")[-1]
        self.assertAlmostEqual(
            latest["total_usd"], latest["stt_usd"] + latest["llm_usd"], places=4
        )

        meeting_id = self.app_module._session.meeting_id
        self.browser.emit("stop_meeting")
        self.browser.wait("meeting_stopped", timeout=20)
        self.assertTrue(any('"CloseStream"' in f for f in ws.text_frames))

        stored = self.db.get_meeting(meeting_id)
        self.assertIsNotNone(stored["ended_at"])
        self.assertEqual([s["text"] for s in stored["segments"]], ["我想爭取多兩個 headcount"])
        self.assertEqual(stored["notes_json"]["decisions"], ["Q3 加一個 head"])
        self.assertEqual(stored["brief_json"]["my_goal"], "爭取兩個 headcount")

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

    def test_a_meeting_with_no_brief_at_all_still_works(self):
        ws = self.start_meeting(brief={})
        self.assertNotIn("keyterm", ws.url)
        ws.push_utterance("開會啦")
        self.assertEqual(self.browser.wait("segment")[0]["text"], "開會啦")


class TestAttendeeOverTheWire(LiveServerCase):
    def test_the_attendee_turn_reaches_the_browser_and_the_database(self):
        self.llm.think_reply = {
            **THINK_JSON,
            "attendee": {
                "should_speak": True, "kind": "question", "urgency": "high",
                "say": "我想問一句，兩個 headcount 係全年計嗎?", "why": "cost differs",
                "needs_web": False, "search_query": "",
            },
        }
        ws = self.start_meeting()
        meeting_id = self.app_module._session.meeting_id
        ws.push_utterance("加兩個人")

        turn = self.browser.wait("attendee")[0]
        self.assertIn("全年計", turn["say"])
        self.assertEqual(turn["kind"], "question")
        self.assertEqual(turn["urgency"], "high")
        self.assertEqual(turn["why"], "cost differs")

        self.assertTrue(
            wait_for(lambda: any(
                e["kind"] == "attendee"
                for e in self.db.get_meeting(meeting_id)["events"]
            )),
            "the attendee's turns belong in the saved session",
        )

    def test_a_quiet_attendee_emits_nothing(self):
        ws = self.start_meeting()
        ws.push_utterance("我覺得 ok")
        self.browser.wait("advice")
        time.sleep(0.3)
        self.assertEqual(self.browser.got("attendee"), [])

    def test_user_can_ask_privately_mid_meeting(self):
        self.start_meeting()
        self.browser.emit("ask", {"question": "點樣講服財務部?"})
        answers = self.browser.wait("answer")
        self.assertTrue(answers[0]["from_user"])
        self.assertEqual(self.browser.got("attendee"), [], "private answers stay private")


class TestPauseAndSTTChoice(LiveServerCase):
    def test_pause_stops_audio_reaching_deepgram_then_resumes(self):
        ws = self.start_meeting()
        chunk = b"\x11\x22" * 2048

        self.browser.send_audio(chunk)
        self.assertTrue(wait_for(lambda: ws.audio_bytes >= len(chunk)))

        self.browser.emit("pause", {"paused": True})
        paused = self.browser.wait("paused")
        self.assertTrue(paused[-1]["paused"])

        before = ws.audio_bytes
        for _ in range(3):
            self.browser.send_audio(chunk)
        time.sleep(0.4)
        self.assertEqual(ws.audio_bytes, before, "paused audio must not be forwarded")

        # The socket stays open through the pause, so resuming needs no reconnect.
        self.assertFalse(ws._closed.is_set())

        self.browser.emit("pause", {"paused": False})
        self.assertTrue(wait_for(lambda: len(self.browser.got("paused")) >= 2))
        self.assertFalse(self.browser.got("paused")[-1]["paused"])

        self.browser.send_audio(chunk)
        self.assertTrue(wait_for(lambda: ws.audio_bytes > before))

    def test_pause_state_survives_a_reconnect(self):
        self.start_meeting()
        self.browser.emit("pause", {"paused": True})
        self.browser.wait("paused")
        self.assertTrue(self.browser.request_snapshot()["paused"])

    def test_the_chosen_language_and_model_reach_deepgram(self):
        ws = self.start_meeting(language="en", model="nova-2")
        self.assertIn("language=en", ws.url)
        self.assertIn("model=nova-2", ws.url)
        self.assertEqual(self.app_module._session.language, "en")

    def test_an_unknown_model_falls_back_instead_of_failing(self):
        ws = self.start_meeting(model="does-not-exist")
        self.assertIn("model=nova-3", ws.url)

    def test_the_language_is_stored_with_the_meeting(self):
        self.start_meeting(language="zh-TW")
        meeting_id = self.app_module._session.meeting_id
        self.assertEqual(self.db.get_meeting(meeting_id)["language"], "zh-TW")

    def test_my_role_reaches_the_model(self):
        brief = dict(BRIEF)
        brief["my_role"] = "IT manager asking finance to approve two headcount"
        ws = self.start_meeting(brief=brief)
        ws.push_utterance("我想爭取多兩個 headcount")
        self.browser.wait("advice")
        self.assertTrue(
            self.llm.saw("IT manager asking finance"),
            "the user's role must be in the prompt the copilot reasons from",
        )


class TestSpeakerNaming(LiveServerCase):
    def test_naming_a_voice_relabels_and_persists(self):
        ws = self.start_meeting()
        meeting_id = self.app_module._session.meeting_id
        ws.push_utterance("我係 Alan", speaker=0)
        self.browser.wait("segment")

        self.browser.emit("name_speaker", {"speaker": 0, "name": "Alan"})
        speakers = self.browser.wait("speakers")
        self.assertEqual(speakers[-1]["speaker_names"], {"0": "Alan"})

        snapshot = self.browser.request_snapshot()
        self.assertEqual(snapshot["segments"][0]["speaker_label"], "Alan")
        self.assertTrue(
            wait_for(lambda: self.db.get_meeting(meeting_id)["speaker_names"] == {0: "Alan"})
        )

    def test_a_name_can_be_cleared_again(self):
        ws = self.start_meeting()
        ws.push_utterance("hello", speaker=0)
        self.browser.wait("segment")
        self.browser.emit("name_speaker", {"speaker": 0, "name": "Alan"})
        self.browser.wait("speakers")
        self.browser.emit("name_speaker", {"speaker": 0, "name": ""})
        self.assertTrue(wait_for(lambda: self.browser.got("speakers")[-1]["speaker_names"] == {}))

    def test_a_nonsense_speaker_index_is_ignored(self):
        self.start_meeting()
        for bad in ({"speaker": "abc", "name": "X"}, {"speaker": -1, "name": "X"},
                    {"speaker": 999, "name": "X"}, {"name": "X"}):
            self.browser.emit("name_speaker", bad)
        time.sleep(0.3)
        self.assertEqual(self.browser.got("speakers"), [])
        self.browser.emit("resync")
        self.browser.wait("snapshot")  # connection still healthy

    def test_a_suggested_mapping_can_be_accepted(self):
        config.SPEAKER_GUESS_INTERVAL = 0
        config.SPEAKER_GUESS_MIN_SEGMENTS = 2
        self.addCleanup(setattr, config, "SPEAKER_GUESS_INTERVAL", 9999)
        self.llm.speaker_reply = {
            "mapping": [
                {"speaker": "S1", "name": "Alan", "confidence": "high", "evidence": "我係 Alan"},
                {"speaker": "S2", "name": "Wing", "confidence": "high", "evidence": "Wing 講"},
            ],
            "note": "",
        }
        ws = self.start_meeting()
        ws.push_utterance("我係 Alan", speaker=0)
        ws.push_utterance("我係 Wing", speaker=1)

        suggestion = self.browser.wait("speaker_suggestion")[0]
        self.assertEqual(
            [(p["label"], p["name"]) for p in suggestion["proposals"]],
            [("S1", "Alan"), ("S2", "Wing")],
        )

        self.browser.emit("speaker_suggestion", {"accept": True})
        self.assertTrue(
            wait_for(
                lambda: self.browser.got("speakers")
                and self.browser.got("speakers")[-1]["speaker_names"] == {"0": "Alan", "1": "Wing"}
            )
        )


class TestSaveTheSession(LiveServerCase):
    def test_markdown_and_json_downloads_work_mid_meeting(self):
        ws = self.start_meeting()
        meeting_id = self.app_module._session.meeting_id
        ws.push_utterance("我想爭取多兩個 headcount", speaker=0)
        self.browser.wait("notes")  # make sure notes exist before exporting
        self.browser.emit("name_speaker", {"speaker": 0, "name": "Alan"})
        self.browser.wait("speakers")

        status, body, headers = self.get(f"/api/meetings/{meeting_id}/export.md")
        self.assertEqual(status, 200)
        self.assertIn("attachment;", headers["Content-Disposition"])
        self.assertIn(".md", headers["Content-Disposition"])
        self.assertIn("# Q3 planning", body)
        self.assertIn("**Alan**: 我想爭取多兩個 headcount", body)
        self.assertIn("爭取兩個 headcount", body)
        self.assertIn("Q3 加一個 head", body, "live notes must be in a mid-meeting export")

        status, body, headers = self.get(f"/api/meetings/{meeting_id}/export.json")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["transcript"][0]["speaker_label"], "Alan")
        self.assertEqual(payload["brief"]["glossary"], ["Falcon", "NocolyHAP"])
        self.assertEqual(payload["notes"]["decisions"], ["Q3 加一個 head"])

    def test_history_lists_the_meeting(self):
        ws = self.start_meeting()
        meeting_id = self.app_module._session.meeting_id
        ws.push_utterance("一句話")
        self.browser.wait("segment")

        status, body, _ = self.get("/api/meetings")
        self.assertEqual(status, 200)
        rows = json.loads(body)
        row = next(r for r in rows if r["id"] == meeting_id)
        self.assertEqual(row["title"], "Q3 planning")
        self.assertGreaterEqual(row["segments"], 1)

    def test_a_missing_meeting_is_a_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/api/meetings/999999/export.md")
        self.assertEqual(caught.exception.code, 404)


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
        self.assertEqual(snapshot["brief"]["title"], "Q3 planning")

        ws.push_utterance("第二句話")
        self.assertEqual(second.wait("segment")[-1]["text"], "第二句話")

    def test_a_second_meeting_is_refused_while_one_runs(self):
        self.start_meeting()
        self.browser.emit("start_meeting", {"brief": BRIEF, "sample_rate": 16000})
        self.assertIn("already running", self.browser.wait("error")[0]["message"])

    def test_bad_sample_rate_is_rejected_before_dialling_deepgram(self):
        self.browser.emit("start_meeting", {"brief": BRIEF, "sample_rate": 96000})
        self.assertIn("96000", self.browser.wait("error")[0]["message"])
        self.assertIsNone(FakeWebSocketApp.latest)

    def test_missing_api_keys_block_the_meeting(self):
        config.DEEPGRAM_API_KEY = ""
        self.addCleanup(setattr, config, "DEEPGRAM_API_KEY", "dg-test")
        self.browser.emit("start_meeting", {"brief": BRIEF, "sample_rate": 16000})
        self.assertIn("DEEPGRAM_API_KEY", self.browser.wait("error")[0]["message"])

    def test_malformed_frames_do_not_kill_the_connection(self):
        self.browser.ws.send("not json at all")
        self.browser.ws.send(json.dumps({"event": "nonsense", "data": {}}))
        self.browser.ws.send(json.dumps({"event": "start_meeting", "data": "not-a-dict"}))
        self.browser.request_snapshot()  # still talking to us

    def test_user_notes_persist(self):
        self.start_meeting()
        meeting_id = self.app_module._session.meeting_id
        self.browser.emit("user_notes", {"text": "我自己嘅筆記"})
        self.assertTrue(
            wait_for(lambda: self.db.get_meeting(meeting_id)["user_notes"] == "我自己嘅筆記")
        )


class TestModelFallback(unittest.TestCase):
    """A model that never returns a result must hand over to the next one.

    Each test gets its own freshly built socket class and tears its STT down
    before returning. Sharing one class across tests let a lingering supervisor
    thread keep constructing sockets after its test had finished, which reached
    into the next test's state -- these tests must not leak threads.
    """

    def _stt(self, ws_class, **kwargs):
        real = dg.websocket.WebSocketApp
        dg.websocket.WebSocketApp = ws_class
        self.addCleanup(setattr, dg.websocket, "WebSocketApp", real)

        stt = dg.DeepgramLiveSTT(api_key="k", sample_rate=16000, **kwargs)
        # Order matters: stop first, then wait for the thread to actually exit.
        self.addCleanup(stt.join)
        self.addCleanup(stt.stop)
        stt.start()
        return stt

    def test_falls_back_to_the_next_model(self):
        ws_class = make_rejecting_ws()
        statuses: list[dict] = []
        stt = self._stt(
            ws_class, language="zh-HK", models=["nova-3", "nova-2"],
            on_status=lambda **kw: statuses.append(kw),
            on_error=lambda msg: statuses.append({"state": "error", "detail": msg}),
        )
        self.assertTrue(
            wait_for(lambda: any("model=nova-2" in u for u in ws_class.urls)),
            f"never tried nova-2; tried {ws_class.urls}",
        )
        self.assertTrue(any(s.get("state") == "fallback" for s in statuses), statuses)
        self.assertEqual(stt.model, "nova-2")

    def test_term_boosting_is_dropped_before_the_model_is_abandoned(self):
        ws_class = make_rejecting_ws()
        statuses: list[dict] = []
        self._stt(
            ws_class, language="zh-HK", models=["nova-3", "nova-2"], keyterms=["Falcon"],
            on_status=lambda **kw: statuses.append(kw), on_error=lambda msg: None,
        )
        # First attempt carries the terms, then the same model is retried without.
        self.assertTrue(
            wait_for(lambda: any(
                "model=nova-3" in u and "keyterm" not in u for u in ws_class.urls
            )),
            f"never retried nova-3 without boosting; tried {ws_class.urls}",
        )
        self.assertTrue(any(s.get("state") == "degraded" for s in statuses), statuses)
        self.assertIn("keyterm=Falcon", ws_class.urls[0])

    def test_gives_up_with_an_error_when_every_model_is_rejected(self):
        errors: list[str] = []
        self._stt(
            make_rejecting_ws(reject_all=True), models=["nova-3", "nova-2"],
            on_error=errors.append,
        )
        self.assertTrue(wait_for(lambda: errors), "must report an unusable configuration")
        self.assertIn("rejected every configured model", errors[0])


def make_rejecting_ws(reject_all: bool = False):
    """A fresh socket class per test, with its own url log and no shared state."""

    class RejectingWS(FakeWebSocketApp):
        urls: list[str] = []

        def __init__(self, url, **kwargs):
            super().__init__(url, **kwargs)
            RejectingWS.urls.append(url)
            self._reject = reject_all or "model=nova-3" in url

        def run_forever(self, **_kwargs):
            self.on_open(self)
            self.opened.set()
            if self._reject:
                self.on_close(self, 1008, "model not available for this language")
                return
            self.push(_results("hello", is_final=False))
            self._closed.wait(3)
            self.on_close(self, 1000, "done")

    return RejectingWS


if __name__ == "__main__":
    unittest.main(verbosity=2)
