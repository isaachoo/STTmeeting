"""Offline tests: no Deepgram, no OpenRouter, no network.

Covers the parts most likely to break silently in a live meeting -- the JSON
coercion around model output, the rate limiting that decides when the copilot
thinks, the rolling-summary window, and the Deepgram message parser.

Run with:  python -m unittest discover -s tests -v
"""

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# Point storage at a scratch directory before anything opens the database.
_TMP = tempfile.TemporaryDirectory()
config.DATA_DIR = Path(_TMP.name)
config.AUDIO_DIR = config.DATA_DIR / "audio"
config.DB_PATH = config.DATA_DIR / "test.sqlite3"

from copilot.engine import CopilotEngine, _normalise_notes  # noqa: E402
from copilot.llm import Usage, extract_json  # noqa: E402
from copilot.state import MeetingState  # noqa: E402
from stt.base import Utterance  # noqa: E402
from stt.deepgram_live import DeepgramLiveSTT  # noqa: E402
from storage import db  # noqa: E402


class FakeLLM:
    """Stands in for OpenRouterClient, recording what it was asked."""

    def __init__(self, json_reply=None, text_reply="a plain answer"):
        self.usage = Usage()
        self.json_reply = json_reply or {}
        self.text_reply = text_reply
        self.json_calls: list[list[dict]] = []
        self.text_calls: list[list[dict]] = []
        self.lock = threading.Lock()

    def chat_json(self, messages, **_kwargs):
        with self.lock:
            self.json_calls.append(messages)
        self.usage.add({"prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0})
        reply = self.json_reply
        return reply(messages) if callable(reply) else dict(reply)

    def chat(self, messages, **_kwargs):
        with self.lock:
            self.text_calls.append(messages)
        self.usage.add({"prompt_tokens": 100, "completion_tokens": 20})
        return self.text_reply


def wait_for(predicate, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


TUNABLES = (
    "ADVICE_MIN_INTERVAL",
    "ADVICE_MIN_NEW_CHARS",
    "NOTES_INTERVAL",
    "SUMMARY_TRIGGER_CHARS",
    "RECENT_WINDOW_CHARS",
    "DEEPGRAM_API_KEY",
)


class ConfigGuard(unittest.TestCase):
    """Restores the module-level tunables so tests cannot leak into each other."""

    def setUp(self):
        saved = {name: getattr(config, name) for name in TUNABLES}

        def restore():
            for name, value in saved.items():
                setattr(config, name, value)

        self.addCleanup(restore)
        super().setUp()


class Emitter:
    """Collects what the engine would have sent to the browser."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []
        self.lock = threading.Lock()

    def __call__(self, event, payload):
        with self.lock:
            self.events.append((event, payload))

    def of(self, kind):
        with self.lock:
            return [payload for name, payload in self.events if name == kind]


# --------------------------------------------------------------------- JSON


class TestExtractJSON(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_prose_around_object(self):
        text = 'Sure! Here you go:\n{"a": 1, "b": {"c": 2}}\nHope that helps.'
        self.assertEqual(extract_json(text), {"a": 1, "b": {"c": 2}})

    def test_braces_inside_strings_do_not_confuse_the_scan(self):
        text = 'noise {"a": "a } brace", "b": 2} trailing'
        self.assertEqual(extract_json(text), {"a": "a } brace", "b": 2})

    def test_rejects_non_objects(self):
        self.assertIsNone(extract_json("[1, 2, 3]"))
        self.assertIsNone(extract_json(""))
        self.assertIsNone(extract_json("no json at all"))


class TestNormaliseNotes(unittest.TestCase):
    def test_fills_missing_keys(self):
        notes = _normalise_notes({})
        self.assertEqual(
            set(notes),
            {"summary", "decisions", "action_items", "open_questions", "topics"},
        )
        self.assertEqual(notes["decisions"], [])

    def test_action_items_from_alternative_key_names(self):
        notes = _normalise_notes(
            {"action_items": [{"owner": "Alan", "task": "send budget", "deadline": "Friday"}]}
        )
        self.assertEqual(
            notes["action_items"],
            [{"who": "Alan", "what": "send budget", "due": "Friday"}],
        )

    def test_action_items_as_bare_strings(self):
        notes = _normalise_notes({"action_items": ["book the room"]})
        self.assertEqual(
            notes["action_items"], [{"who": "unassigned", "what": "book the room", "due": ""}]
        )

    def test_drops_items_with_no_task(self):
        notes = _normalise_notes({"action_items": [{"who": "Alan"}, {}, None]})
        self.assertEqual(notes["action_items"], [])

    def test_string_null_is_treated_as_empty(self):
        notes = _normalise_notes({"summary": "null", "decisions": "one decision"})
        self.assertEqual(notes["summary"], "")
        self.assertEqual(notes["decisions"], ["one decision"])


# -------------------------------------------------------------------- state


class TestMeetingState(ConfigGuard):
    def setUp(self):
        super().setUp()
        self.state = MeetingState(meeting_id=1, brief="test brief")

    def test_speaker_labels(self):
        seg = self.state.add_utterance("hello", 0)
        self.assertEqual(seg.speaker_label, "S1")
        self.assertEqual(self.state.add_utterance("hi", None).speaker_label, "?")

    def test_recent_text_respects_the_char_budget(self):
        for i in range(50):
            self.state.add_utterance(f"utterance number {i}", 0)
        recent = self.state.recent_text(max_chars=120)
        self.assertLess(len(recent), 200)
        self.assertIn("utterance number 49", recent)
        self.assertNotIn("utterance number 0:", recent)

    def test_recent_text_never_reaches_behind_the_summary(self):
        for i in range(10):
            self.state.add_utterance(f"line {i}", 0)
        self.state.summarised_upto = 8
        recent = self.state.recent_text(max_chars=10_000)
        self.assertEqual(recent, "S1: line 8\nS1: line 9")

    def test_recent_text_keeps_one_segment_even_when_over_budget(self):
        self.state.add_utterance("x" * 500, 0)
        self.assertIn("x" * 500, self.state.recent_text(max_chars=10))

    def test_advice_is_rate_limited_by_time_and_by_new_speech(self):
        config.ADVICE_MIN_INTERVAL = 15
        config.ADVICE_MIN_NEW_CHARS = 20
        self.state.add_utterance("x" * 50, 0)
        self.assertTrue(self.state.should_advise())

        self.state.last_advice_at = time.time()
        self.assertFalse(self.state.should_advise(), "too soon after the last call")

        self.state.last_advice_at = 0
        self.state.advised_upto = len(self.state.segments)
        self.assertFalse(self.state.should_advise(), "no new speech to advise on")

        self.state.add_utterance("y" * 5, 0)
        self.assertFalse(self.state.should_advise(), "5 chars is below the threshold")
        self.state.add_utterance("z" * 30, 0)
        self.assertTrue(self.state.should_advise())

    def test_first_notes_pass_waits_for_the_interval(self):
        config.NOTES_INTERVAL = 90
        self.state.add_utterance("第一句", 0)
        self.assertFalse(
            self.state.should_take_notes(),
            "must not summarise a meeting that is one sentence old",
        )
        self.state.last_notes_at = time.time() - 91
        self.assertTrue(self.state.should_take_notes())
        self.state.noted_upto = len(self.state.segments)
        self.assertFalse(self.state.should_take_notes(), "nothing new to note")

    def test_answered_question_dedupe_matches_loosely(self):
        self.state.remember_answered("What is the Q3 budget?")
        self.assertTrue(self.state.already_answered("what is the q3 budget"))
        self.assertTrue(self.state.already_answered("What is the Q3 budget???"))
        self.assertFalse(self.state.already_answered("Who owns the Falcon project?"))


# ------------------------------------------------------------------- engine


class TestCopilotEngine(ConfigGuard):
    def setUp(self):
        super().setUp()
        config.ADVICE_MIN_INTERVAL = 0
        config.ADVICE_MIN_NEW_CHARS = 1
        config.NOTES_INTERVAL = 0
        config.SUMMARY_TRIGGER_CHARS = 10_000
        self.state = MeetingState(meeting_id=1, brief="Q3 budget meeting")
        self.emitter = Emitter()

    def _engine(self, llm):
        engine = CopilotEngine(self.state, llm, self.emitter)
        self.addCleanup(engine.close, False)
        return engine

    def test_advice_is_emitted_and_remembered(self):
        llm = FakeLLM(
            {
                "key_point": "Alan wants two more headcount",
                "suggested_questions": ["What is the cost per head?", ""],
                "watch_out": "No owner named for the budget model",
                "question_to_answer": None,
            }
        )
        engine = self._engine(llm)
        seg = self.state.add_utterance("我想爭取多兩個 headcount", 0)
        engine.on_utterance(seg)

        self.assertTrue(wait_for(lambda: self.emitter.of("advice")))
        advice = self.emitter.of("advice")[0]
        self.assertEqual(advice["key_point"], "Alan wants two more headcount")
        self.assertEqual(advice["questions"], ["What is the cost per head?"])
        self.assertIn("No owner named", advice["watch_out"])
        # Anti-repetition history is what stops the panel filling with clones.
        self.assertIn("Alan wants two more headcount", self.state.advice_history)
        self.assertEqual(self.state.advised_upto, 1)

    def test_empty_advice_emits_nothing(self):
        llm = FakeLLM(
            {"key_point": None, "suggested_questions": [], "watch_out": None,
             "question_to_answer": None}
        )
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("咁樣啦", 0))
        self.assertTrue(wait_for(lambda: llm.json_calls))
        time.sleep(0.2)
        self.assertEqual(self.emitter.of("advice"), [])

    def test_detected_question_triggers_one_answer_only(self):
        llm = FakeLLM(
            {
                "key_point": "someone asked about the levy",
                "suggested_questions": [],
                "watch_out": None,
                "question_to_answer": {
                    "question": "香港 2026 年的最低工資係幾多?",
                    "needs_web": True,
                    "search_query": "Hong Kong statutory minimum wage 2026",
                },
            },
            text_reply="HK$42.10 per hour. [1]",
        )
        engine = self._engine(llm)

        engine.on_utterance(self.state.add_utterance("最低工資係幾多?", 0))
        self.assertTrue(wait_for(lambda: self.emitter.of("answer")))
        answer = self.emitter.of("answer")[0]
        self.assertEqual(answer["answer"], "HK$42.10 per hour. [1]")
        self.assertFalse(answer["from_user"])
        # No TAVILY_API_KEY in the test environment, so it must say it did not search.
        self.assertFalse(answer["searched"])

        # The same question coming back next cycle must not be looked up again.
        self.state.last_advice_at = 0
        engine.on_utterance(self.state.add_utterance("係幾多呀?", 1))
        self.assertTrue(wait_for(lambda: len(self.emitter.of("advice")) == 2))
        time.sleep(0.3)
        self.assertEqual(len(self.emitter.of("answer")), 1)

    def test_user_question_is_always_answered(self):
        llm = FakeLLM(text_reply="Here is the answer.")
        engine = self._engine(llm)
        engine.ask("點樣講服財務部?")
        self.assertTrue(wait_for(lambda: self.emitter.of("answer")))
        self.assertTrue(self.emitter.of("answer")[0]["from_user"])

    def test_notes_replace_wholesale_and_advance_the_cursor(self):
        llm = FakeLLM(
            {
                "summary": "Discussed headcount.",
                "decisions": ["Add one head in Q3"],
                "action_items": [{"who": "Wing", "what": "update the model", "due": ""}],
                "open_questions": [],
                "topics": ["headcount"],
            }
        )
        engine = self._engine(llm)
        self.state.add_utterance("一句", 0)
        engine._run_notes()

        notes = self.emitter.of("notes")[0]["notes"]
        self.assertEqual(notes["decisions"], ["Add one head in Q3"])
        self.assertEqual(self.state.noted_upto, 1)
        self.assertEqual(self.state.notes["summary"], "Discussed headcount.")

    def test_summary_folds_old_segments_and_keeps_the_recent_window(self):
        config.RECENT_WINDOW_CHARS = 40
        llm = FakeLLM(text_reply="A merged summary of the early discussion.")
        engine = self._engine(llm)
        for i in range(20):
            self.state.add_utterance(f"segment {i} text", 0)

        engine._run_summary()
        self.assertEqual(
            self.state.rolling_summary, "A merged summary of the early discussion."
        )
        self.assertGreater(self.state.summarised_upto, 0)
        self.assertLess(self.state.summarised_upto, 20, "must not fold the recent window")
        self.assertIn("segment 19", self.state.recent_text())

    def test_a_failing_llm_reports_instead_of_dying(self):
        class Boom(FakeLLM):
            def chat_json(self, messages, **kwargs):
                raise RuntimeError("model exploded")

        engine = self._engine(Boom())
        engine.on_utterance(self.state.add_utterance("測試", 0))
        self.assertTrue(wait_for(lambda: self.emitter.of("copilot_error")))
        self.assertIn("model exploded", self.emitter.of("copilot_error")[0]["message"])

    def test_only_one_advice_call_runs_at_a_time(self):
        config.NOTES_INTERVAL = 9999  # keep note-taking out of the call count
        started = threading.Event()
        release = threading.Event()

        class Slow(FakeLLM):
            def chat_json(self, messages, **kwargs):
                with self.lock:
                    self.json_calls.append(messages)
                started.set()
                release.wait(3)
                return {
                    "key_point": "k", "suggested_questions": [], "watch_out": None,
                    "question_to_answer": None,
                }

        llm = Slow()
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("first", 0))
        self.assertTrue(started.wait(2))

        self.state.last_advice_at = 0  # rate limit would otherwise hide the guard
        for i in range(5):
            engine.on_utterance(self.state.add_utterance(f"more {i}", 0))
        time.sleep(0.2)
        self.assertEqual(len(llm.json_calls), 1, "advisor calls must not stack up")
        release.set()


# ---------------------------------------------------- Deepgram message parsing


class TestDeepgramParsing(unittest.TestCase):
    def setUp(self):
        self.interims: list[str] = []
        self.utterances: list[Utterance] = []
        self.errors: list[str] = []
        self.stt = DeepgramLiveSTT(
            api_key="test",
            sample_rate=16000,
            on_interim=self.interims.append,
            on_utterance=self.utterances.append,
            on_error=self.errors.append,
        )

    @staticmethod
    def result(text, is_final, speech_final=False, speaker=0, start=0.0, duration=1.0):
        return {
            "type": "Results",
            "is_final": is_final,
            "speech_final": speech_final,
            "start": start,
            "duration": duration,
            "channel": {
                "alternatives": [
                    {
                        "transcript": text,
                        "words": [{"word": w, "speaker": speaker} for w in text.split()],
                    }
                ]
            },
        }

    def test_interim_then_final_builds_one_utterance(self):
        self.stt._handle_message(self.result("我想", False))
        self.stt._handle_message(self.result("我想爭取", False))
        self.stt._handle_message(self.result("我想爭取多兩個 headcount", True))
        self.assertEqual(self.utterances, [])  # not closed yet

        self.stt._handle_message(self.result("係呀", True, speech_final=True))
        self.assertEqual(len(self.utterances), 1)
        self.assertEqual(self.utterances[0].text, "我想爭取多兩個 headcount 係呀")
        self.assertEqual(self.utterances[0].speaker, 0)

    def test_utterance_end_flushes_when_no_endpoint_was_detected(self):
        self.stt._handle_message(self.result("開會啦", True))
        self.stt._handle_message({"type": "UtteranceEnd", "last_word_end": 1.2})
        self.assertEqual(len(self.utterances), 1)
        self.assertEqual(self.utterances[0].text, "開會啦")

    def test_utterance_end_with_nothing_pending_is_a_no_op(self):
        self.stt._handle_message({"type": "UtteranceEnd"})
        self.stt._handle_message({"type": "UtteranceEnd"})
        self.assertEqual(self.utterances, [])

    def test_empty_transcripts_are_ignored(self):
        self.stt._handle_message(self.result("", False))
        self.stt._handle_message(self.result("", True, speech_final=True))
        self.assertEqual(self.utterances, [])
        self.assertEqual([i for i in self.interims if i], [])

    def test_interim_shows_settled_text_plus_the_live_tail(self):
        self.stt._handle_message(self.result("第一句", True))
        self.stt._handle_message(self.result("第二", False))
        self.assertEqual(self.interims[-1], "第一句 第二")

    def test_speaker_comes_from_the_first_word_of_the_utterance(self):
        self.stt._handle_message(self.result("你好", True, speaker=2))
        self.stt._handle_message({"type": "UtteranceEnd"})
        self.assertEqual(self.utterances[0].speaker, 2)

    def test_error_frames_are_surfaced(self):
        self.stt._handle_message({"type": "Error", "description": "bad model"})
        self.assertEqual(self.errors, ["Deepgram: bad model"])

    def test_results_prove_the_model_works(self):
        self.assertFalse(self.stt._proven)
        self.stt._handle_message(self.result("hi", False))
        self.assertTrue(self.stt._proven, "a result must stop the model fallback")

    def test_audio_seconds_tracks_bytes_at_the_declared_rate(self):
        self.stt._bytes_sent = 16000 * 2 * 3
        self.assertAlmostEqual(self.stt.audio_seconds, 3.0)

    def test_send_audio_drops_the_oldest_chunk_when_backed_up(self):
        from stt.deepgram_live import MAX_QUEUED_CHUNKS

        for i in range(MAX_QUEUED_CHUNKS + 10):
            self.stt.send_audio(bytes([i % 256]) * 4)
        self.assertEqual(self.stt._audio.qsize(), MAX_QUEUED_CHUNKS)
        self.assertEqual(self.stt._dropped_chunks, 10)

    def test_stop_flushes_whatever_was_pending(self):
        self.stt._handle_message(self.result("最後一句", True))
        self.stt.stop()
        self.assertEqual(len(self.utterances), 1)
        self.assertEqual(self.utterances[0].text, "最後一句")

    def test_url_declares_the_sample_rate_and_language(self):
        url = self.stt._url()
        self.assertIn("sample_rate=16000", url)
        self.assertIn("language=zh-HK", url)
        self.assertIn("encoding=linear16", url)
        self.assertIn("interim_results=true", url)
        self.assertIn("diarize=true", url)


# ------------------------------------------------------------------ storage


class TestStorage(unittest.TestCase):
    def setUp(self):
        db.init()

    def test_meeting_round_trip(self):
        mid = db.create_meeting("Budget", "the brief", "zh-HK", "nova-3")
        db.add_segment(mid, 0, 1.5, 0, "第一句")
        db.add_segment(mid, 1, 4.0, None, "第二句")
        db.add_event(mid, "advice", {"key_point": "something"})
        db.save_notes(mid, {"summary": "a summary", "decisions": ["d1"]})
        db.save_user_notes(mid, "my own notes")
        db.save_summary(mid, "rolling")
        db.finish_meeting(mid, 300.0, {"cost_usd": 0.12}, "nova-2")

        meeting = db.get_meeting(mid)
        self.assertEqual(meeting["title"], "Budget")
        self.assertEqual(meeting["stt_model"], "nova-2")
        self.assertEqual(meeting["audio_seconds"], 300.0)
        self.assertEqual(meeting["user_notes"], "my own notes")
        self.assertEqual(meeting["summary"], "rolling")
        self.assertEqual([s["text"] for s in meeting["segments"]], ["第一句", "第二句"])
        self.assertEqual(meeting["segments"][1]["speaker"], None)
        self.assertEqual(meeting["events"][0]["payload"]["key_point"], "something")
        self.assertEqual(meeting["notes_json"]["decisions"], ["d1"])
        self.assertEqual(meeting["usage_json"]["cost_usd"], 0.12)
        self.assertIsNotNone(meeting["ended_at"])

    def test_missing_meeting_is_none(self):
        self.assertIsNone(db.get_meeting(999_999))


# ------------------------------------------------------------------- session


class TestSessionCostAndSnapshot(ConfigGuard):
    def setUp(self):
        super().setUp()
        db.init()
        config.DEEPGRAM_API_KEY = "test-key"
        config.ADVICE_MIN_INTERVAL = 9999  # keep the copilot out of this test
        config.NOTES_INTERVAL = 9999
        from session import MeetingSession, _validate_sample_rate

        self.MeetingSession = MeetingSession
        self.validate = _validate_sample_rate
        self.emitter = Emitter()
        self.session = MeetingSession(
            title="Test", brief="a brief", sample_rate=16000, emit=self.emitter
        )
        self.addCleanup(self.session.engine.close, False)

    def test_sample_rate_validation(self):
        self.assertEqual(self.validate("48000"), 48000)
        for bad in ("abc", None, 4000, 96000):
            with self.assertRaises(ValueError):
                self.validate(bad)

    def test_utterance_is_stored_emitted_and_priced(self):
        self.session._on_utterance(Utterance(text="第一句", speaker=0))
        self.assertEqual(self.emitter.of("segment")[0]["text"], "第一句")

        stored = db.get_meeting(self.session.meeting_id)
        self.assertEqual([s["text"] for s in stored["segments"]], ["第一句"])

        self.session.stt._bytes_sent = 16000 * 2 * 600  # ten minutes
        cost = self.session.cost()
        self.assertAlmostEqual(cost["audio_minutes"], 10.0)
        self.assertAlmostEqual(
            cost["stt_usd"], 10 * config.DEEPGRAM_USD_PER_MINUTE, places=4
        )
        self.assertAlmostEqual(cost["total_usd"], cost["stt_usd"] + cost["llm_usd"], places=4)

    def test_five_hour_cost_estimate_is_in_the_expected_range(self):
        """Guards the number quoted to the user: STT for 5h should be ~$2.31."""
        self.session.stt._bytes_sent = 16000 * 2 * 5 * 3600
        cost = self.session.cost()
        self.assertAlmostEqual(cost["audio_minutes"], 300.0)
        self.assertTrue(2.0 < cost["stt_usd"] < 3.0, cost["stt_usd"])

    def test_snapshot_carries_everything_the_page_needs(self):
        self.session._on_utterance(Utterance(text="一句", speaker=1))
        self.session.set_user_notes("my notes")
        snap = self.session.snapshot()
        for key in (
            "meeting_id", "running", "status", "segments", "cards", "notes",
            "user_notes", "summary", "cost", "web_search", "brief",
        ):
            self.assertIn(key, snap)
        self.assertTrue(snap["running"])
        self.assertEqual(snap["user_notes"], "my notes")
        self.assertEqual(snap["segments"][0]["speaker_label"], "S2")
        self.assertEqual(
            db.get_meeting(self.session.meeting_id)["user_notes"], "my notes"
        )

    def test_feeding_audio_after_stop_is_ignored(self):
        self.session.stopped = True
        self.session.feed_audio(b"\x00\x00" * 100)
        self.assertEqual(self.session.stt.audio_seconds, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
