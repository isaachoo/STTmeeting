"""Tests for continuity: asking the copilot mid-meeting, carrying on after a
crash, and meeting backgrounds saved in advance.

The thread running through all three is that nothing the user has typed or
said should be lost or need retyping. No network; the LLM is a double.
"""

import json
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
config.DB_PATH = config.DATA_DIR / "continuity-tests.sqlite3"

from copilot.brief import Attendee, Brief  # noqa: E402
from copilot.engine import CopilotEngine  # noqa: E402
from copilot.llm import Usage  # noqa: E402
from copilot.state import MeetingState  # noqa: E402
from storage import db  # noqa: E402
from stt.base import Utterance  # noqa: E402


class FakeLLM:
    def __init__(self, text_reply="an answer", json_reply=None):
        self.usage = Usage()
        self.text_reply = text_reply
        self.json_reply = json_reply or {}
        self.text_calls: list[list[dict]] = []
        self.json_calls: list[list[dict]] = []
        self.lock = threading.Lock()

    def chat(self, messages, **_kw):
        with self.lock:
            self.text_calls.append(messages)
        self.usage.add({"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0001})
        return self.text_reply

    def chat_json(self, messages, **_kw):
        with self.lock:
            self.json_calls.append(messages)
        self.usage.add({"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0001})
        return json.loads(json.dumps(self.json_reply))

    def last_prompt(self) -> str:
        with self.lock:
            return "\n".join(m["content"] for m in self.text_calls[-1])

    def last_roles(self) -> list[str]:
        with self.lock:
            return [m["role"] for m in self.text_calls[-1]]


class Emitter:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def __call__(self, event, payload):
        with self.lock:
            self.events.append((event, payload))

    def of(self, kind):
        with self.lock:
            return [p for e, p in self.events if e == kind]


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


LINES = [
    (0, "我們今日主要傾 Q3 budget 同 headcount"),
    (1, "Falcon 個 timeline delay 咗兩次，客戶已經投訴"),
    (0, "我要多兩個 headcount，唔係 Q3 做唔完"),
    (2, "財務部立場係 budget 最多加 50 萬，唔可以再多"),
    (1, "咁不如先請一個，Falcon 用 contractor 頂住"),
    (2, "我下星期五之前出一個 revised budget model"),
]


def brief() -> Brief:
    return Brief(
        title="Q3 planning",
        agenda="budget, headcount",
        attendees=[Attendee(name="Alan", is_me=True), Attendee(name="Bella")],
        glossary=["Falcon"],
    )


def stored_meeting(with_summary_marker=True, finished=False) -> dict:
    """A meeting as the database would hold it after a crash mid-way."""
    db.init()
    mid = db.create_meeting("Q3 planning", brief().as_dict(), "zh-HK", "nova-3")
    for i, (speaker, text) in enumerate(LINES):
        db.add_segment(mid, i, 10.0 * i, speaker, text)
    db.save_speakers(mid, {0: "Alan", 1: "Bella"})
    db.save_notes(mid, {"summary": "Budget and headcount.", "decisions": ["hire one"],
                        "action_items": [], "open_questions": [], "topics": ["budget"]})
    db.save_user_notes(mid, "my own note")
    if with_summary_marker:
        db.save_summary(mid, "Alan wants two heads; finance offers 500k.", upto=4)
    else:
        db.save_summary(mid, "Alan wants two heads; finance offers 500k.")
    db.add_event(mid, "answer", {"question": "budget?", "answer": "50 萬 [#3]", "from_user": True})
    db.add_event(mid, "advice", {"key_point": "finance is firm"})
    if finished:
        db.finish_meeting(mid, audio_seconds=120, usage={}, stt_model="nova-3")
    return db.get_meeting(mid)


# -------------------------------------------------------- asking mid-meeting


class TestAskingTheCopilot(unittest.TestCase):
    def setUp(self):
        db.init()
        mid = db.create_meeting("Q3 planning", brief().as_dict(), "zh-HK", "nova-3")
        self.state = MeetingState(meeting_id=mid, brief=brief())
        for speaker, text in LINES:
            self.state.add_utterance(text, speaker)
        self.state.set_speaker_name(0, "Alan")
        self.state.rolling_summary = "Alan asked for two heads."
        self.state.notes = {**self.state.notes, "decisions": ["hire one head"]}
        self.emitter = Emitter()

    def _engine(self, llm):
        engine = CopilotEngine(self.state, llm, self.emitter)
        self.addCleanup(engine.close, False)
        return engine

    def _ask(self, llm, question, web=False):
        engine = self._engine(llm)
        engine.ask(question, web=web)
        self.assertTrue(wait_for(lambda: self.emitter.of("answer")), "no answer arrived")
        return self.emitter.of("answer")[0]

    def test_the_answer_sees_the_whole_meeting_not_just_the_summary(self):
        llm = FakeLLM(text_reply="財務部最多加 50 萬 [#3]。")
        self._ask(llm, "budget 加幾多？")
        sent = llm.last_prompt()
        self.assertIn("Alan asked for two heads", sent, "the rolling summary")
        self.assertIn("hire one head", sent, "the live notes")
        self.assertIn("[#3]", sent, "numbered transcript lines the answer can cite")
        self.assertIn("50 萬", sent, "the actual words")

    def test_citations_come_back_as_line_numbers(self):
        answer = self._ask(FakeLLM(text_reply="係 [#3] 講嘅，仲有 [#5]。"), "budget?")
        self.assertEqual(answer["cited"], [3, 5])
        self.assertTrue(answer["from_user"])

    def test_an_invented_line_number_is_dropped(self):
        answer = self._ask(FakeLLM(text_reply="see [#3] and [#4444]"), "budget?")
        self.assertEqual(answer["cited"], [3])

    def test_no_web_search_unless_asked(self):
        from copilot import search

        called = []
        original_available, original_search = search.available, search.search
        search.available = lambda: True
        search.search = lambda q: called.append(q) or []
        self.addCleanup(setattr, search, "available", original_available)
        self.addCleanup(setattr, search, "search", original_search)

        answer = self._ask(FakeLLM(), "總結一下", web=False)
        self.assertEqual(called, [], "a meeting question must not hit the web")
        self.assertFalse(answer["web_requested"])

        self.emitter.events.clear()
        answer = self._ask(FakeLLM(), "market rate for a PM in HK", web=True)
        self.assertEqual(called, ["market rate for a PM in HK"])
        self.assertTrue(answer["web_requested"])

    def test_a_follow_up_sees_the_previous_exchange(self):
        llm = FakeLLM(text_reply="first")
        engine = self._engine(llm)
        engine.ask("第一個問題")
        self.assertTrue(wait_for(lambda: len(self.emitter.of("answer")) == 1))
        engine.ask("跟住呢？")
        self.assertTrue(wait_for(lambda: len(self.emitter.of("answer")) == 2))
        self.assertEqual(llm.last_roles(), ["system", "user", "assistant", "user"])
        self.assertEqual(self.state.qa_history[-1]["question"], "跟住呢？")

    def test_the_prompt_tells_the_model_how_to_summarise(self):
        llm = FakeLLM()
        self._ask(llm, "總結一下到目前為止")
        system = llm.text_calls[-1][0]["content"]
        self.assertIn("summarise", system)
        self.assertIn("[#42]", system, "the citation format is spelled out")

    def test_an_empty_question_is_ignored(self):
        engine = self._engine(FakeLLM())
        engine.ask("   ")
        time.sleep(0.1)
        self.assertEqual(self.emitter.of("answer"), [])


# ------------------------------------------------------------- the database


class TestInterruptedMeetingsInTheDatabase(unittest.TestCase):
    def test_a_meeting_never_stopped_is_listed_as_unfinished(self):
        meeting = stored_meeting()
        self.assertIn(meeting["id"], [m["id"] for m in db.unfinished_meetings()])

    def test_a_finished_meeting_is_not(self):
        meeting = stored_meeting(finished=True)
        self.assertNotIn(meeting["id"], [m["id"] for m in db.unfinished_meetings()])

    def test_closing_uses_the_last_words_not_the_clock(self):
        """It did not run until the moment someone pressed Close."""
        meeting = stored_meeting()
        self.assertTrue(db.close_interrupted(meeting["id"]))
        closed = db.get_meeting(meeting["id"])
        self.assertAlmostEqual(closed["ended_at"], meeting["started_at"] + 50.0, places=3)
        self.assertFalse(db.close_interrupted(meeting["id"]), "already closed")

    def test_the_summary_marker_survives(self):
        meeting = stored_meeting()
        self.assertEqual(meeting["summarised_upto"], 4)

    def test_save_summary_without_a_marker_leaves_the_old_one(self):
        meeting = stored_meeting()
        db.save_summary(meeting["id"], "newer text")
        again = db.get_meeting(meeting["id"])
        self.assertEqual(again["summary"], "newer text")
        self.assertEqual(again["summarised_upto"], 4)


# --------------------------------------------------------- restoring the state


class TestRestoringState(unittest.TestCase):
    def test_everything_the_copilot_had_comes_back(self):
        meeting = stored_meeting()
        state = MeetingState(meeting_id=meeting["id"], brief=brief())
        state.load_stored(meeting)

        self.assertEqual(len(state.segments), len(LINES))
        self.assertEqual(state.segments[3].text, LINES[3][1])
        self.assertEqual(state.speaker_names, {0: "Alan", 1: "Bella"})
        self.assertEqual(state.notes["decisions"], ["hire one"])
        self.assertEqual(state.user_notes, "my own note")
        self.assertIn("two heads", state.rolling_summary)
        self.assertEqual(state.summarised_upto, 4)

    def test_nothing_before_the_break_is_advised_on_again(self):
        meeting = stored_meeting()
        state = MeetingState(meeting_id=meeting["id"], brief=brief())
        state.load_stored(meeting)
        self.assertEqual(state.thought_upto, len(LINES))
        self.assertEqual(state.noted_upto, len(LINES))
        self.assertFalse(state.should_take_notes(now=time.time() + 10_000))

    def test_earlier_questions_come_back_as_history(self):
        meeting = stored_meeting()
        state = MeetingState(meeting_id=meeting["id"], brief=brief())
        state.load_stored(meeting)
        self.assertEqual(state.qa_history, [{"question": "budget?", "answer": "50 萬 [#3]"}])

    def test_an_old_row_without_the_marker_keeps_a_recent_tail_verbatim(self):
        meeting = stored_meeting(with_summary_marker=False)
        original = config.RECENT_WINDOW_CHARS
        config.RECENT_WINDOW_CHARS = 40
        self.addCleanup(setattr, config, "RECENT_WINDOW_CHARS", original)
        state = MeetingState(meeting_id=meeting["id"], brief=brief())
        state.load_stored(meeting)
        self.assertGreater(state.summarised_upto, 0)
        self.assertLess(state.summarised_upto, len(LINES))
        self.assertTrue(state.recent_text(), "something recent stays verbatim")

    def test_the_next_line_continues_the_numbering(self):
        meeting = stored_meeting()
        state = MeetingState(meeting_id=meeting["id"], brief=brief())
        state.load_stored(meeting)
        seg = state.add_utterance("新嘅一句", 0)
        self.assertEqual(seg.index, len(LINES))


# ------------------------------------------------------------- the session


class TestResumingASession(unittest.TestCase):
    def setUp(self):
        self.saved = {n: getattr(config, n) for n in
                      ("DEEPGRAM_API_KEY", "THINK_MIN_INTERVAL", "NOTES_INTERVAL",
                       "SPEAKER_GUESS_INTERVAL", "SUMMARY_TRIGGER_CHARS")}
        self.addCleanup(lambda: [setattr(config, n, v) for n, v in self.saved.items()])
        config.DEEPGRAM_API_KEY = "test-key"
        config.THINK_MIN_INTERVAL = 9999
        config.NOTES_INTERVAL = 9999
        config.SPEAKER_GUESS_INTERVAL = 9999
        config.SUMMARY_TRIGGER_CHARS = 10**9

        import session as session_module

        self.session_module = session_module
        self.llm = FakeLLM()
        original = session_module.OpenRouterClient
        session_module.OpenRouterClient = lambda **_kw: self.llm
        self.addCleanup(setattr, session_module, "OpenRouterClient", original)

        self.meeting = stored_meeting()
        # Money already spent before the crash.
        db.finish_meeting(self.meeting["id"], audio_seconds=600,
                          usage={"calls": 7, "prompt_tokens": 700, "completion_tokens": 70,
                                 "cost_usd": 0.05}, stt_model="nova-3")
        with db._connect() as conn:  # reopen it: finish_meeting is the only setter
            conn.execute("UPDATE meetings SET ended_at = NULL WHERE id = ?", (self.meeting["id"],))
        self.meeting = db.get_meeting(self.meeting["id"])
        self.emitter = Emitter()

    def _resume(self):
        session = self.session_module.MeetingSession(
            brief=Brief.from_payload(self.meeting["brief_json"]),
            sample_rate=16000,
            emit=self.emitter,
            resume=self.meeting,
        )
        self.addCleanup(session.engine.close, False)
        return session

    def test_it_is_the_same_meeting(self):
        session = self._resume()
        self.assertEqual(session.meeting_id, self.meeting["id"])
        self.assertTrue(session.resumed)
        self.assertEqual(db.unfinished_meetings()[0]["id"], self.meeting["id"],
                         "resuming does not create a second row")

    def test_the_page_gets_the_old_transcript_and_cards_back(self):
        snap = self._resume().snapshot()
        self.assertEqual(len(snap["segments"]), len(LINES))
        self.assertEqual(snap["segments"][0]["speaker_label"], "Alan")
        self.assertTrue(snap["resumed"])
        kinds = {c["kind"] for c in snap["cards"]}
        self.assertEqual(kinds, {"answer", "advice"})
        self.assertEqual(snap["notes"]["decisions"], ["hire one"])
        self.assertEqual(snap["user_notes"], "my own note")

    def test_new_speech_continues_the_transcript(self):
        session = self._resume()
        session._on_utterance(Utterance(text="繼續講", speaker=1))
        stored = db.get_meeting(self.meeting["id"])["segments"]
        self.assertEqual(len(stored), len(LINES) + 1)
        self.assertEqual(stored[-1]["idx"], len(LINES))
        self.assertEqual(self.emitter.of("segment")[0]["speaker_label"], "Bella")

    def test_the_cost_carries_on_from_before_the_crash(self):
        session = self._resume()
        session.stt._bytes_sent = 16000 * 2 * 60  # one new minute
        cost = session.cost()
        self.assertAlmostEqual(cost["audio_minutes"], 11.0, places=2)
        self.assertEqual(cost["llm_calls"], 7)
        self.assertAlmostEqual(cost["llm_usd"], 0.05, places=4)

    def test_starting_marks_the_seam_in_the_transcript(self):
        session = self._resume()
        session.stt.start = lambda: None  # the transcriber is not what is under test
        session.start()
        marks = self.emitter.of("paused")
        self.assertEqual(len(marks), 1)
        self.assertTrue(marks[0]["resumed"])
        kinds = [e["kind"] for e in db.get_meeting(self.meeting["id"])["events"]]
        self.assertIn("resume", kinds)
        # A page that loads later gets the seam from the snapshot, not only from
        # the event it may have missed.
        self.assertEqual(session.snapshot()["markers"], marks)

    def test_a_second_interruption_keeps_the_first_seam(self):
        session = self._resume()
        session.stt.start = lambda: None
        session.start()
        db_meeting = db.get_meeting(self.meeting["id"])
        again = self.session_module.MeetingSession(
            brief=Brief.from_payload(db_meeting["brief_json"]), sample_rate=16000,
            emit=Emitter(), resume=db_meeting,
        )
        self.addCleanup(again.engine.close, False)
        self.assertEqual(len(again.snapshot()["markers"]), 1, "the earlier seam is restored")

    def test_stopping_finishes_the_meeting_with_the_full_totals(self):
        session = self._resume()
        session.stt._bytes_sent = 16000 * 2 * 60
        session.stop()
        finished = db.get_meeting(self.meeting["id"])
        self.assertIsNotNone(finished["ended_at"])
        self.assertAlmostEqual(finished["audio_seconds"], 660.0, places=1)
        self.assertEqual(finished["summarised_upto"], 4)
        self.assertNotIn(self.meeting["id"], [m["id"] for m in db.unfinished_meetings()])

    def test_transcriber_settings_default_to_the_stored_ones(self):
        with db._connect() as conn:
            conn.execute("UPDATE meetings SET language = 'en', provider = 'deepgram' WHERE id = ?",
                         (self.meeting["id"],))
        self.meeting = db.get_meeting(self.meeting["id"])
        session = self._resume()
        self.assertEqual(session.language, "en")


# ------------------------------------------------------------- saved briefs


class TestSavedBriefs(unittest.TestCase):
    def setUp(self):
        db.init()
        for item in db.list_briefs():
            db.delete_brief(item["id"])

    def test_save_load_and_list(self):
        saved = db.save_brief("Monthly ops", brief().as_dict())
        self.assertEqual(saved["name"], "Monthly ops")
        listed = db.list_briefs()
        self.assertEqual([b["name"] for b in listed], ["Monthly ops"])
        self.assertEqual(listed[0]["brief"]["attendees"][0]["name"], "Alan")

    def test_saving_the_same_name_replaces(self):
        db.save_brief("Monthly ops", {"title": "v1"})
        db.save_brief("Monthly ops", {"title": "v2"})
        listed = db.list_briefs()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["brief"]["title"], "v2")

    def test_delete(self):
        saved = db.save_brief("Once", {"title": "x"})
        self.assertTrue(db.delete_brief(saved["id"]))
        self.assertEqual(db.list_briefs(), [])
        self.assertFalse(db.delete_brief(saved["id"]))

    def test_a_blank_name_is_refused(self):
        with self.assertRaises(ValueError):
            db.save_brief("   ", {})

    def test_newest_first(self):
        db.save_brief("older", {"title": "a"})
        time.sleep(0.01)
        db.save_brief("newer", {"title": "b"})
        self.assertEqual([b["name"] for b in db.list_briefs()], ["newer", "older"])


# --------------------------------------------------------------- the HTTP API


class TestContinuityApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as app_module

        cls.app_module = app_module
        app_module.app.config["TESTING"] = True

    def setUp(self):
        self.client = self.app_module.app.test_client()
        self.app_module._session = None
        for item in db.list_briefs():
            db.delete_brief(item["id"])

    def _json(self, response):
        return json.loads(response.data.decode())

    def test_interrupted_meetings_are_flagged_in_the_list(self):
        open_one = stored_meeting()
        done_one = stored_meeting(finished=True)
        rows = {m["id"]: m for m in self._json(self.client.get("/api/meetings"))}
        self.assertTrue(rows[open_one["id"]]["interrupted"])
        self.assertFalse(rows[done_one["id"]]["interrupted"])

    def test_the_running_meeting_is_not_called_interrupted(self):
        meeting = stored_meeting()

        class Live:
            meeting_id = meeting["id"]
            stopped = False

        self.app_module._session = Live()
        self.addCleanup(setattr, self.app_module, "_session", None)
        rows = {m["id"]: m for m in self._json(self.client.get("/api/meetings"))}
        self.assertFalse(rows[meeting["id"]]["interrupted"])
        self.assertTrue(rows[meeting["id"]]["running"])

    def test_closing_an_interrupted_meeting(self):
        meeting = stored_meeting()
        response = self.client.post(f"/api/meetings/{meeting['id']}/finish")
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(db.get_meeting(meeting["id"])["ended_at"])
        self.assertEqual(self.client.post(f"/api/meetings/{meeting['id']}/finish").status_code, 404)

    def test_a_running_meeting_cannot_be_closed_from_here(self):
        meeting = stored_meeting()

        class Live:
            meeting_id = meeting["id"]
            stopped = False

        self.app_module._session = Live()
        self.addCleanup(setattr, self.app_module, "_session", None)
        self.assertEqual(self.client.post(f"/api/meetings/{meeting['id']}/finish").status_code, 409)

    def test_briefs_round_trip_over_http(self):
        saved = self._json(self.client.post(
            "/api/briefs", json={"name": "Board meeting", "brief": brief().as_dict()}
        ))
        self.assertEqual(saved["saved"]["name"], "Board meeting")
        listed = self._json(self.client.get("/api/briefs"))["briefs"]
        self.assertEqual(listed[0]["brief"]["glossary"], ["Falcon"])
        deleted = self.client.delete(f"/api/briefs/{saved['saved']['id']}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self._json(self.client.get("/api/briefs"))["briefs"], [])

    def test_the_name_defaults_to_the_title(self):
        saved = self._json(self.client.post("/api/briefs", json={"brief": {"title": "Weekly sync"}}))
        self.assertEqual(saved["saved"]["name"], "Weekly sync")

    def test_no_name_and_no_title_is_refused(self):
        response = self.client.post("/api/briefs", json={"brief": {"agenda": "x"}})
        self.assertEqual(response.status_code, 400)

    # ------------------------------------------------------ the assistant

    def _fake_assistant(self, reply="an answer"):
        llm = FakeLLM(text_reply=reply)
        original = self.app_module.OpenRouterClient
        self.app_module.OpenRouterClient = lambda *a, **k: llm
        self.addCleanup(setattr, self.app_module, "OpenRouterClient", original)
        saved_key = config.OPENROUTER_API_KEY
        config.OPENROUTER_API_KEY = "or-test"
        self.addCleanup(setattr, config, "OPENROUTER_API_KEY", saved_key)
        return llm

    def test_the_assistant_answers_without_a_meeting(self):
        llm = self._fake_assistant("強積金上限係 $1,500。")
        payload = self._json(self.client.post("/api/chat", json={"question": "MPF 上限幾多？"}))
        self.assertEqual(payload["answer"], "強積金上限係 $1,500。")
        self.assertFalse(payload["searched"])
        self.assertGreater(payload["cost_usd"], 0)
        system = llm.text_calls[-1][0]["content"]
        self.assertIn("general-purpose assistant", system)
        self.assertNotIn("What the person is in the middle of", llm.last_prompt())

    def test_the_conversation_travels_with_the_question(self):
        llm = self._fake_assistant()
        self.client.post("/api/chat", json={
            "question": "同英文講一次",
            "history": [{"role": "user", "content": "MPF 上限？"},
                        {"role": "assistant", "content": "$1,500"},
                        {"role": "system", "content": "ignored: only user/assistant turns"}],
        })
        self.assertEqual(llm.last_roles(), ["system", "user", "assistant", "user"])

    def test_web_search_only_when_asked_for(self):
        from copilot import search

        llm = self._fake_assistant("上限係 $1,500 [1]。")
        called = []
        saved = (search.available, search.search)
        search.available = lambda: True
        search.search = lambda q, max_results=4: called.append(q) or [
            {"title": "MPFA", "url": "https://mpfa.example", "content": "cap 1500"}]
        self.addCleanup(setattr, search, "available", saved[0])
        self.addCleanup(setattr, search, "search", saved[1])

        off = self._json(self.client.post("/api/chat", json={"question": "MPF cap?", "web": False}))
        self.assertEqual(called, [])
        self.assertEqual(off["sources"], [])

        on = self._json(self.client.post("/api/chat", json={"question": "MPF cap?", "web": True}))
        self.assertEqual(called, ["MPF cap?"])
        self.assertEqual(on["sources"][0]["title"], "MPFA")
        self.assertTrue(on["searched"])
        self.assertIn("Web search results", llm.last_prompt())

    def test_a_running_meeting_lends_context_but_not_the_transcript(self):
        llm = self._fake_assistant()
        meeting = stored_meeting()
        from copilot.state import MeetingState

        class Live:
            meeting_id = meeting["id"]
            stopped = False
            state = MeetingState(meeting_id=meeting["id"], brief=brief())

        Live.state.add_utterance("秘密：財務部私底下講過 70 萬", 2)
        self.app_module._session = Live()
        self.addCleanup(setattr, self.app_module, "_session", None)
        self.client.post("/api/chat", json={"question": "what is a headcount?"})
        sent = llm.last_prompt()
        self.assertIn("Q3 planning", sent, "the brief is context")
        self.assertNotIn("70 萬", sent, "what was said stays in the Copilot panel")

    def test_no_key_and_no_question_are_refused(self):
        self._fake_assistant()
        self.assertEqual(self.client.post("/api/chat", json={"question": "  "}).status_code, 400)
        config.OPENROUTER_API_KEY = ""
        self.assertEqual(self.client.post("/api/chat", json={"question": "x"}).status_code, 400)

    def test_a_model_failure_is_a_readable_error(self):
        from copilot.llm import LLMError

        class Broken:
            usage = Usage()

            def chat(self, *a, **k):
                raise LLMError("upstream is down")

        original = self.app_module.OpenRouterClient
        self.app_module.OpenRouterClient = lambda *a, **k: Broken()
        self.addCleanup(setattr, self.app_module, "OpenRouterClient", original)
        saved_key = config.OPENROUTER_API_KEY
        config.OPENROUTER_API_KEY = "or-test"
        self.addCleanup(setattr, config, "OPENROUTER_API_KEY", saved_key)
        response = self.client.post("/api/chat", json={"question": "x"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("upstream is down", self._json(response)["error"])

    def test_junk_in_a_saved_brief_is_cleaned_like_any_other(self):
        saved = self._json(self.client.post(
            "/api/briefs", json={"name": "n", "brief": {"title": "t", "attendees": "Alan, Bella", "bogus": 1}}
        ))
        self.assertEqual([a["name"] for a in saved["saved"]["brief"]["attendees"]], ["Alan", "Bella"])
        self.assertNotIn("bogus", saved["saved"]["brief"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
