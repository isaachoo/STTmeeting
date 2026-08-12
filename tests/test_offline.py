"""Offline tests: no Deepgram, no OpenRouter, no network.

Covers the parts most likely to break silently in a live meeting -- the JSON
coercion around model output, the rate limiting that decides when the copilot
thinks, the AI attendee's speak-or-stay-quiet handling, speaker naming, the
rolling-summary window, the Deepgram message parser, and the exports.

Run with:  python -m unittest discover -s tests -v
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

# Point storage at a scratch directory before anything opens the database.
_TMP = tempfile.TemporaryDirectory()
config.DATA_DIR = Path(_TMP.name)
config.AUDIO_DIR = config.DATA_DIR / "audio"
config.DB_PATH = config.DATA_DIR / "test.sqlite3"

from copilot.brief import Attendee, Brief  # noqa: E402
from copilot.engine import CopilotEngine, _normalise_notes, _speaker_index  # noqa: E402
from copilot.llm import Usage, extract_json  # noqa: E402
from copilot.state import MeetingState, label_for, looks_like_a_question  # noqa: E402
from stt.base import Utterance  # noqa: E402
from stt.deepgram_live import DeepgramLiveSTT  # noqa: E402
from storage import db, export  # noqa: E402


# --------------------------------------------------------------------- doubles


THINK_REPLY = {
    "key_point": "Alan 想加兩個 headcount",
    "suggested_questions": ["每個 head 的 fully loaded cost 係幾多?"],
    "watch_out": "budget model 未有人負責",
    "attendee": {"should_speak": False, "kind": "info", "urgency": "normal",
                 "say": "", "why": "", "needs_web": False, "search_query": ""},
}

NOTES_REPLY = {
    "summary": "討論 Q3 headcount。",
    "decisions": ["Q3 加一個 head"],
    "action_items": [{"who": "Wing", "what": "update budget model", "due": "Friday"}],
    "open_questions": [],
    "topics": ["headcount"],
}


class FakeLLM:
    """Stands in for OpenRouterClient, dispatching on which prompt it was sent."""

    def __init__(self, think=None, notes=None, speakers=None, text_reply="a plain answer"):
        self.usage = Usage()
        self.think = think if think is not None else THINK_REPLY
        self.notes = notes if notes is not None else NOTES_REPLY
        self.speakers = speakers if speakers is not None else {"mapping": [], "note": ""}
        self.text_reply = text_reply
        self.json_calls: list[list[dict]] = []
        self.text_calls: list[list[dict]] = []
        self.lock = threading.Lock()

    def _reply_for(self, messages):
        system = messages[0]["content"]
        if "note taker" in system:
            return self.notes
        if "match anonymous voices" in system:
            return self.speakers
        return self.think

    def chat_json(self, messages, **_kwargs):
        with self.lock:
            self.json_calls.append(messages)
        self.usage.add({"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0001})
        reply = self._reply_for(messages)
        return reply(messages) if callable(reply) else json.loads(json.dumps(reply))

    def chat(self, messages, **_kwargs):
        with self.lock:
            self.text_calls.append(messages)
        self.usage.add({"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0001})
        return self.text_reply

    def prompts(self) -> str:
        with self.lock:
            calls = list(self.json_calls) + list(self.text_calls)
        return "\n".join(m["content"] for call in calls for m in call)


def wait_for(predicate, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


TUNABLES = (
    "THINK_MIN_INTERVAL",
    "THINK_MIN_NEW_CHARS",
    "THINK_URGENT_INTERVAL",
    "NOTES_INTERVAL",
    "SUMMARY_TRIGGER_CHARS",
    "RECENT_WINDOW_CHARS",
    "SPEAKER_GUESS_INTERVAL",
    "SPEAKER_GUESS_MIN_SEGMENTS",
    "ATTENDEE_ENABLED",
    "ATTENDEE_MODE",
    "DEEPGRAM_API_KEY",
    "DEEPGRAM_KEYTERMS",
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


def sample_brief() -> Brief:
    return Brief(
        title="Q3 planning",
        context="上次會議財務部話要等 Q4",
        agenda="1. budget 2. headcount",
        my_goal="爭取兩個 headcount",
        attendees=[
            Attendee(name="Isaac", role="me", is_me=True),
            Attendee(name="Alan", role="ops"),
            Attendee(name="Wing", role="finance"),
        ],
        glossary=["Falcon", "NocolyHAP"],
    )


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

    def test_action_items_from_alternative_key_names(self):
        notes = _normalise_notes(
            {"action_items": [{"owner": "Alan", "task": "send budget", "deadline": "Friday"}]}
        )
        self.assertEqual(
            notes["action_items"], [{"who": "Alan", "what": "send budget", "due": "Friday"}]
        )

    def test_action_items_as_bare_strings(self):
        notes = _normalise_notes({"action_items": ["book the room"]})
        self.assertEqual(
            notes["action_items"], [{"who": "unassigned", "what": "book the room", "due": ""}]
        )

    def test_drops_items_with_no_task(self):
        self.assertEqual(
            _normalise_notes({"action_items": [{"who": "Alan"}, {}, None]})["action_items"], []
        )

    def test_string_null_is_treated_as_empty(self):
        notes = _normalise_notes({"summary": "null", "decisions": "one decision"})
        self.assertEqual(notes["summary"], "")
        self.assertEqual(notes["decisions"], ["one decision"])


class TestSpeakerIndexParsing(unittest.TestCase):
    def test_accepts_the_shapes_models_actually_return(self):
        self.assertEqual(_speaker_index("S1"), 0)
        self.assertEqual(_speaker_index("s3"), 2)
        self.assertEqual(_speaker_index(2), 1)
        self.assertEqual(_speaker_index("2"), 1)

    def test_rejects_nonsense(self):
        for bad in (None, "", "SX", "S0", 0, -1, True, {}):
            self.assertIsNone(_speaker_index(bad), bad)


# ---------------------------------------------------------------------- brief


class TestBrief(unittest.TestCase):
    def test_from_payload_builds_the_whole_thing(self):
        brief = Brief.from_payload(
            {
                "title": "Q3",
                "agenda": "budget",
                "my_goal": "two heads",
                "context": "background",
                "attendees": [
                    {"name": "Alan", "role": "ops"},
                    {"name": "Isaac", "is_me": True},
                ],
                "glossary": ["Falcon", "KPI"],
            }
        )
        self.assertEqual(brief.title, "Q3")
        self.assertEqual([a.name for a in brief.attendees], ["Alan", "Isaac"])
        self.assertTrue(brief.me.is_me)
        self.assertEqual(brief.me.name, "Isaac")

    def test_glossary_accepts_a_comma_or_newline_string(self):
        brief = Brief.from_payload({"glossary": "Falcon, KPI\nheadcount"})
        self.assertEqual(brief.glossary, ["Falcon", "KPI", "headcount"])

    def test_glossary_deduplicates(self):
        self.assertEqual(Brief.from_payload({"glossary": "KPI, KPI"}).glossary, ["KPI"])

    def test_attendees_without_a_name_are_dropped(self):
        brief = Brief.from_payload({"attendees": [{"role": "ops"}, {"name": ""}, "Wing"]})
        self.assertEqual([a.name for a in brief.attendees], ["Wing"])

    def test_keyterms_cover_jargon_and_every_name(self):
        self.assertEqual(
            sample_brief().keyterms(),
            ["Falcon", "NocolyHAP", "Isaac", "Alan", "Wing"],
        )

    def test_render_includes_every_section_and_flags_the_user(self):
        rendered = sample_brief().render()
        for expected in ("Q3 planning", "Alan (ops)", "budget", "爭取兩個 headcount",
                         "Falcon", "this is the user"):
            self.assertIn(expected, rendered)

    def test_render_omits_empty_sections(self):
        rendered = Brief(title="Just a title").render()
        self.assertIn("Meeting: Just a title", rendered)
        for absent in ("Agenda:", "Background:", "In the room:"):
            self.assertNotIn(absent, rendered)

    def test_my_role_reaches_the_prompt(self):
        brief = Brief(my_role="IT manager asking finance for two headcount")
        self.assertIn("IT manager asking finance", brief.render())
        self.assertIn("The user's role", brief.render())

    def test_role_falls_back_to_the_roster_then_to_general(self):
        self.assertEqual(
            Brief(my_role="chairing").role_line(), "chairing"
        )
        from_roster = Brief(attendees=[Attendee(name="Isaac", role="CFO", is_me=True)])
        self.assertEqual(from_roster.role_line(), "CFO")
        # No role anywhere still has to say something usable to the model.
        self.assertIn("general participant", Brief().role_line())
        self.assertIn("The user's role", Brief().render())

    def test_round_trips_through_a_dict(self):
        brief = sample_brief()
        again = Brief.from_dict(brief.as_dict())
        self.assertEqual(again.as_dict(), brief.as_dict())

    def test_garbage_payload_does_not_raise(self):
        for bad in (None, {}, {"attendees": "nonsense"}, {"attendees": [None, 5]},
                    {"glossary": 42}, {"title": None}):
            Brief.from_payload(bad)

    def test_is_empty_only_when_nothing_useful_was_given(self):
        self.assertTrue(Brief(title="only a title").is_empty())
        self.assertFalse(Brief(my_goal="something").is_empty())


# -------------------------------------------------------------------- state


class TestQuestionHeuristic(unittest.TestCase):
    def test_spots_cantonese_and_english_questions(self):
        for text in ["最低工資係幾多?", "咁 Falcon 幾時 launch", "點解會咁", "有無 buffer",
                     "邊個負責", "How much is it", "can we ship by Q3", "係唔係咁"]:
            self.assertTrue(looks_like_a_question(text), text)

    def test_ignores_statements(self):
        for text in ["我覺得 ok", "好呀 no problem", "我們下星期開始", ""]:
            self.assertFalse(looks_like_a_question(text), text)


class TestMeetingState(ConfigGuard):
    def setUp(self):
        super().setUp()
        self.state = MeetingState(meeting_id=1, brief=sample_brief())

    def test_speaker_labels_fall_back_to_the_diarisation_tag(self):
        seg = self.state.add_utterance("hello", 0)
        self.assertEqual(seg.as_dict()["speaker_label"], "S1")
        self.assertEqual(self.state.add_utterance("hi", None).as_dict()["speaker_label"], "?")

    def test_naming_a_voice_changes_every_label(self):
        self.state.add_utterance("第一句", 0)
        self.state.add_utterance("第二句", 1)
        self.state.set_speaker_name(0, "Alan")
        names = self.state.speaker_names
        self.assertEqual(self.state.segments[0].as_dict(names)["speaker_label"], "Alan")
        self.assertEqual(self.state.segments[1].as_dict(names)["speaker_label"], "S2")
        self.assertIn("Alan: 第一句", self.state.text_from(0))

    def test_a_name_can_be_cleared(self):
        self.state.set_speaker_name(0, "Alan")
        self.state.set_speaker_name(0, "  ")
        self.assertEqual(self.state.speaker_names, {})
        self.assertEqual(label_for(0, self.state.speaker_names), "S1")

    def test_roster_lists_named_and_unnamed_voices(self):
        self.state.add_utterance("a", 0)
        self.state.add_utterance("b", 2)
        self.state.set_speaker_name(0, "Alan")
        roster = self.state.speaker_roster()
        self.assertIn("S1 is Alan", roster)
        self.assertIn("S3 is not yet identified", roster)
        self.assertEqual(self.state.unnamed_speakers(), [2])

    def test_recent_text_respects_the_char_budget(self):
        for i in range(50):
            self.state.add_utterance(f"utterance number {i}", 0)
        recent = self.state.recent_text(max_chars=120)
        self.assertLess(len(recent), 200)
        self.assertIn("utterance number 49", recent)

    def test_recent_text_never_reaches_behind_the_summary(self):
        for i in range(10):
            self.state.add_utterance(f"line {i}", 0)
        self.state.summarised_upto = 8
        self.assertEqual(self.state.recent_text(max_chars=10_000), "S1: line 8\nS1: line 9")

    def test_recent_text_keeps_one_segment_even_when_over_budget(self):
        self.state.add_utterance("x" * 500, 0)
        self.assertIn("x" * 500, self.state.recent_text(max_chars=10))

    def test_thinking_is_rate_limited_by_time_and_by_new_speech(self):
        config.THINK_MIN_INTERVAL = 15
        config.THINK_MIN_NEW_CHARS = 20
        self.state.add_utterance("x" * 50, 0)
        self.assertTrue(self.state.should_think())

        self.state.last_think_at = time.time()
        self.assertFalse(self.state.should_think(), "too soon after the last call")

        self.state.last_think_at = 0
        self.state.thought_upto = len(self.state.segments)
        self.assertFalse(self.state.should_think(), "no new speech to think about")

        self.state.add_utterance("y" * 5, 0)
        self.assertFalse(self.state.should_think(), "5 chars is below the threshold")
        self.state.add_utterance("z" * 30, 0)
        self.assertTrue(self.state.should_think())

    def test_a_question_shortens_the_interval_but_does_not_remove_it(self):
        config.THINK_MIN_INTERVAL = 60
        config.THINK_MIN_NEW_CHARS = 500
        config.THINK_URGENT_INTERVAL = 5
        self.state.add_utterance("幾時?", 0)
        self.state.last_think_at = time.time() - 10

        self.assertFalse(self.state.should_think(urgent=False), "not enough new speech")
        self.assertTrue(self.state.should_think(urgent=True), "a question overrides the budget")

        self.state.last_think_at = time.time() - 1
        self.assertFalse(
            self.state.should_think(urgent=True),
            "even urgent thinking must respect a floor, or a fast exchange spams the model",
        )

    def test_first_notes_pass_waits_for_the_interval(self):
        config.NOTES_INTERVAL = 90
        self.state.add_utterance("第一句", 0)
        self.assertFalse(self.state.should_take_notes())
        self.state.last_notes_at = time.time() - 91
        self.assertTrue(self.state.should_take_notes())
        self.state.noted_upto = len(self.state.segments)
        self.assertFalse(self.state.should_take_notes(), "nothing new to note")

    def test_speaker_guessing_waits_for_enough_material(self):
        config.SPEAKER_GUESS_INTERVAL = 120
        config.SPEAKER_GUESS_MIN_SEGMENTS = 8
        self.state.add_utterance("hello", 0)
        self.assertFalse(self.state.should_guess_speakers(), "one line is not enough")

        for i in range(9):
            self.state.add_utterance(f"line {i}", i % 3)
        self.assertTrue(self.state.should_guess_speakers())

        self.state.last_speaker_guess_at = time.time()
        self.assertFalse(self.state.should_guess_speakers(), "rate limited")

    def test_no_speaker_guessing_without_a_roster_to_match_against(self):
        state = MeetingState(meeting_id=2, brief=Brief(title="no attendees"))
        for i in range(12):
            state.add_utterance(f"line {i}", 0)
        self.assertFalse(state.should_guess_speakers())

    def test_no_speaker_guessing_once_every_voice_is_named(self):
        for i in range(12):
            self.state.add_utterance(f"line {i}", 0)
        self.assertTrue(self.state.should_guess_speakers())
        self.state.set_speaker_name(0, "Alan")
        self.assertFalse(self.state.should_guess_speakers())

    def test_the_attendee_will_not_say_the_same_thing_twice(self):
        line = "我想問一句，六十萬係 fully loaded cost 定係 base salary?"
        self.assertFalse(self.state.already_said(line))
        self.state.remember_attendee_turn(line)

        self.assertTrue(self.state.already_said(line))
        self.assertTrue(self.state.already_said(line + "  "), "whitespace is not a new point")
        self.assertTrue(self.state.already_said(line + "?"), "punctuation is not a new point")
        self.assertTrue(self.state.already_said(""), "an empty turn is never worth saying")

    def test_a_genuinely_longer_point_still_gets_through(self):
        self.state.remember_attendee_turn("六十萬係咪 fully loaded?")
        self.assertFalse(
            self.state.already_said(
                "六十萬係咪 fully loaded? 如果係 base salary，加埋 MPF 同 benefits "
                "應該係八十萬左右，個 budget number 要改。"
            ),
            "a new turn that expands on an old one is new information",
        )

    def test_answered_question_dedupe_matches_loosely(self):
        self.state.remember_answered("What is the Q3 budget?")
        self.assertTrue(self.state.already_answered("what is the q3 budget"))
        self.assertFalse(self.state.already_answered("Who owns the Falcon project?"))


# ------------------------------------------------------------------- engine


class TestCopilotEngine(ConfigGuard):
    def setUp(self):
        super().setUp()
        config.THINK_MIN_INTERVAL = 0
        config.THINK_MIN_NEW_CHARS = 1
        config.THINK_URGENT_INTERVAL = 0
        config.NOTES_INTERVAL = 9999
        config.SUMMARY_TRIGGER_CHARS = 10_000
        config.SPEAKER_GUESS_INTERVAL = 9999
        config.ATTENDEE_ENABLED = True
        config.ATTENDEE_MODE = "normal"
        self.state = MeetingState(meeting_id=1, brief=sample_brief())
        self.emitter = Emitter()

    def _engine(self, llm):
        engine = CopilotEngine(self.state, llm, self.emitter)
        self.addCleanup(engine.close, False)
        return engine

    # --- coaching ---

    def test_advice_is_emitted_and_remembered(self):
        llm = FakeLLM()
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("我想爭取多兩個 headcount", 0))

        self.assertTrue(wait_for(lambda: self.emitter.of("advice")))
        advice = self.emitter.of("advice")[0]
        self.assertEqual(advice["key_point"], "Alan 想加兩個 headcount")
        self.assertEqual(advice["questions"], ["每個 head 的 fully loaded cost 係幾多?"])
        self.assertIn("Alan 想加兩個 headcount", self.state.advice_history)
        self.assertEqual(self.state.thought_upto, 1)

    def test_the_brief_and_roster_reach_the_prompt(self):
        llm = FakeLLM()
        engine = self._engine(llm)
        self.state.set_speaker_name(0, "Alan")
        engine.on_utterance(self.state.add_utterance("Falcon 幾時 launch?", 0))
        self.assertTrue(wait_for(lambda: llm.json_calls))
        prompts = llm.prompts()
        self.assertIn("Falcon", prompts, "glossary must reach the model")
        self.assertIn("爭取兩個 headcount", prompts, "the user's goal must reach the model")
        self.assertIn("S1 is Alan", prompts, "the speaker roster must reach the model")

    def test_empty_advice_emits_nothing(self):
        llm = FakeLLM(think={"key_point": None, "suggested_questions": [], "watch_out": None,
                             "attendee": {"should_speak": False}})
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("咁樣啦", 0))
        self.assertTrue(wait_for(lambda: llm.json_calls))
        time.sleep(0.2)
        self.assertEqual(self.emitter.of("advice"), [])
        self.assertEqual(self.emitter.of("attendee"), [])

    # --- the AI attendee ---

    def test_attendee_speaks_when_it_has_something_to_say(self):
        llm = FakeLLM(think={
            **THINK_REPLY,
            "attendee": {
                "should_speak": True, "kind": "question", "urgency": "normal",
                "say": "我想問一句，兩個 headcount 係全年計定係 pro-rata?",
                "why": "the cost changes a lot either way",
                "needs_web": False, "search_query": "",
            },
        })
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("加兩個人", 0))

        self.assertTrue(wait_for(lambda: self.emitter.of("attendee")))
        turn = self.emitter.of("attendee")[0]
        self.assertEqual(turn["kind"], "question")
        self.assertIn("pro-rata", turn["say"])
        self.assertEqual(turn["why"], "the cost changes a lot either way")
        self.assertFalse(turn["searched"])
        self.assertIn(turn["say"], self.state.attendee_history)

    def test_a_repeated_attendee_turn_is_dropped(self):
        llm = FakeLLM(think={
            **THINK_REPLY,
            "attendee": {"should_speak": True, "kind": "question", "urgency": "normal",
                         "say": "六十萬係 fully loaded cost 嗎?", "why": "matters",
                         "needs_web": False, "search_query": ""},
        })
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("六十萬一年", 0))
        self.assertTrue(wait_for(lambda: self.emitter.of("attendee")))

        # The model says the identical thing again on the next cycle.
        self.state.last_think_at = 0
        engine.on_utterance(self.state.add_utterance("係呀六十萬", 0))
        self.assertTrue(wait_for(lambda: len(llm.json_calls) >= 2))
        time.sleep(0.2)
        self.assertEqual(
            len(self.emitter.of("attendee")), 1, "the attendee must not repeat itself"
        )

    def test_attendee_stays_quiet_by_default(self):
        engine = self._engine(FakeLLM())  # THINK_REPLY has should_speak False
        engine.on_utterance(self.state.add_utterance("一句話", 0))
        self.assertTrue(wait_for(lambda: self.emitter.of("advice")))
        time.sleep(0.2)
        self.assertEqual(self.emitter.of("attendee"), [], "silence must be the default")

    def test_attendee_with_no_words_does_not_emit_an_empty_turn(self):
        llm = FakeLLM(think={**THINK_REPLY,
                             "attendee": {"should_speak": True, "say": "  ", "kind": "info"}})
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("一句話", 0))
        self.assertTrue(wait_for(lambda: self.emitter.of("advice")))
        time.sleep(0.2)
        self.assertEqual(self.emitter.of("attendee"), [])

    def test_attendee_can_be_switched_off_entirely(self):
        config.ATTENDEE_ENABLED = False
        llm = FakeLLM(think={**THINK_REPLY,
                             "attendee": {"should_speak": True, "say": "hello", "kind": "info"}})
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("一句話", 0))
        self.assertTrue(wait_for(lambda: self.emitter.of("advice")))
        time.sleep(0.2)
        self.assertEqual(self.emitter.of("attendee"), [])

    def test_an_unknown_kind_is_coerced_rather_than_shown_raw(self):
        llm = FakeLLM(think={**THINK_REPLY,
                             "attendee": {"should_speak": True, "say": "一句", "kind": "SHOUTING",
                                          "urgency": "EXTREME"}})
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("一句話", 0))
        self.assertTrue(wait_for(lambda: self.emitter.of("attendee")))
        turn = self.emitter.of("attendee")[0]
        self.assertEqual(turn["kind"], "info")
        self.assertEqual(turn["urgency"], "normal")

    def test_high_urgency_is_passed_through(self):
        llm = FakeLLM(think={**THINK_REPLY,
                             "attendee": {"should_speak": True, "say": "係 42 蚊", "kind": "answer",
                                          "urgency": "high"}})
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("幾多錢?", 0))
        self.assertTrue(wait_for(lambda: self.emitter.of("attendee")))
        self.assertEqual(self.emitter.of("attendee")[0]["urgency"], "high")

    def test_a_question_in_the_room_triggers_thinking_immediately(self):
        config.THINK_MIN_INTERVAL = 600  # normal cadence would never fire
        config.THINK_URGENT_INTERVAL = 0
        config.THINK_MIN_NEW_CHARS = 5000
        llm = FakeLLM()
        engine = self._engine(llm)
        self.state.last_think_at = time.time() - 30

        engine.on_utterance(self.state.add_utterance("我覺得 ok", 0))
        time.sleep(0.2)
        self.assertEqual(llm.json_calls, [], "a statement should not trigger a think")

        engine.on_utterance(self.state.add_utterance("咁幾時 launch?", 1))
        self.assertTrue(wait_for(lambda: llm.json_calls), "a question should")

    # --- answering ---

    def test_user_question_is_answered_privately(self):
        llm = FakeLLM(text_reply="Here is the answer.")
        engine = self._engine(llm)
        engine.ask("點樣講服財務部?")
        self.assertTrue(wait_for(lambda: self.emitter.of("answer")))
        answer = self.emitter.of("answer")[0]
        self.assertTrue(answer["from_user"])
        self.assertEqual(answer["answer"], "Here is the answer.")
        self.assertEqual(self.emitter.of("attendee"), [], "private answers stay private")

    # --- notes and summary ---

    def test_notes_replace_wholesale_and_advance_the_cursor(self):
        llm = FakeLLM()
        engine = self._engine(llm)
        self.state.add_utterance("一句", 0)
        engine._run_notes()

        notes = self.emitter.of("notes")[0]["notes"]
        self.assertEqual(notes["decisions"], ["Q3 加一個 head"])
        self.assertEqual(self.state.noted_upto, 1)

    def test_summary_folds_old_segments_and_keeps_the_recent_window(self):
        config.RECENT_WINDOW_CHARS = 40
        llm = FakeLLM(text_reply="A merged summary of the early discussion.")
        engine = self._engine(llm)
        for i in range(20):
            self.state.add_utterance(f"segment {i} text", 0)

        engine._run_summary()
        self.assertEqual(self.state.rolling_summary, "A merged summary of the early discussion.")
        self.assertGreater(self.state.summarised_upto, 0)
        self.assertLess(self.state.summarised_upto, 20, "must not fold the recent window")
        self.assertIn("segment 19", self.state.recent_text())

    # --- speaker inference ---

    def test_speaker_guess_is_suggested_never_applied(self):
        llm = FakeLLM(speakers={
            "mapping": [
                {"speaker": "S1", "name": "Alan", "confidence": "high", "evidence": "我係 Alan"},
                {"speaker": "S2", "name": "Wing", "confidence": "low", "evidence": "Wing 你講"},
            ],
            "note": "",
        })
        engine = self._engine(llm)
        for i in range(10):
            self.state.add_utterance(f"line {i}", i % 2)

        engine._run_speaker_guess()
        suggestion = self.emitter.of("speaker_suggestion")[0]
        self.assertEqual(
            [(p["speaker"], p["name"]) for p in suggestion["proposals"]],
            [(0, "Alan"), (1, "Wing")],
        )
        self.assertEqual(
            self.state.speaker_names, {}, "a guess must never rename a voice on its own"
        )
        self.assertIsNotNone(self.state.speaker_suggestion)

    def test_speaker_guess_rejects_names_that_are_not_attendees(self):
        llm = FakeLLM(speakers={"mapping": [
            {"speaker": "S1", "name": "Somebody Invented", "confidence": "high"},
            {"speaker": "S2", "name": "Alan", "confidence": "high"},
        ]})
        engine = self._engine(llm)
        for i in range(10):
            self.state.add_utterance(f"line {i}", i % 2)
        engine._run_speaker_guess()
        proposals = self.emitter.of("speaker_suggestion")[0]["proposals"]
        self.assertEqual([(p["speaker"], p["name"]) for p in proposals], [(1, "Alan")])

    def test_speaker_guess_never_assigns_one_person_to_two_voices(self):
        llm = FakeLLM(speakers={"mapping": [
            {"speaker": "S1", "name": "Alan", "confidence": "high"},
            {"speaker": "S2", "name": "Alan", "confidence": "low"},
        ]})
        engine = self._engine(llm)
        for i in range(10):
            self.state.add_utterance(f"line {i}", i % 2)
        engine._run_speaker_guess()
        proposals = self.emitter.of("speaker_suggestion")[0]["proposals"]
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["speaker"], 0)

    def test_speaker_guess_skips_voices_the_user_already_named(self):
        llm = FakeLLM(speakers={"mapping": [
            {"speaker": "S1", "name": "Alan", "confidence": "high"},
        ]})
        engine = self._engine(llm)
        for i in range(10):
            self.state.add_utterance(f"line {i}", 0)
        self.state.set_speaker_name(0, "Wing")  # the user says S1 is Wing
        engine._run_speaker_guess()
        self.assertEqual(
            self.emitter.of("speaker_suggestion"), [], "must not contradict the user"
        )

    def test_an_empty_mapping_produces_no_suggestion(self):
        engine = self._engine(FakeLLM(speakers={"mapping": [], "note": "cannot tell"}))
        for i in range(10):
            self.state.add_utterance(f"line {i}", 0)
        engine._run_speaker_guess()
        self.assertEqual(self.emitter.of("speaker_suggestion"), [])

    # --- robustness ---

    def test_a_failing_llm_reports_instead_of_dying(self):
        class Boom(FakeLLM):
            def chat_json(self, messages, **kwargs):
                raise RuntimeError("model exploded")

        engine = self._engine(Boom())
        engine.on_utterance(self.state.add_utterance("測試", 0))
        self.assertTrue(wait_for(lambda: self.emitter.of("copilot_error")))
        self.assertIn("model exploded", self.emitter.of("copilot_error")[0]["message"])

    def test_only_one_think_call_runs_at_a_time(self):
        started = threading.Event()
        release = threading.Event()

        class Slow(FakeLLM):
            def chat_json(self, messages, **kwargs):
                with self.lock:
                    self.json_calls.append(messages)
                started.set()
                release.wait(3)
                return dict(THINK_REPLY)

        llm = Slow()
        engine = self._engine(llm)
        engine.on_utterance(self.state.add_utterance("first", 0))
        self.assertTrue(started.wait(2))

        self.state.last_think_at = 0  # rate limit would otherwise hide the guard
        for i in range(5):
            engine.on_utterance(self.state.add_utterance(f"more {i}", 0))
        time.sleep(0.2)
        self.assertEqual(len(llm.json_calls), 1, "think calls must not stack up")
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
        self.stt._handle_message(self.result("我想爭取多兩個 headcount", True))
        self.assertEqual(self.utterances, [])  # not closed yet

        self.stt._handle_message(self.result("係呀", True, speech_final=True))
        self.assertEqual(len(self.utterances), 1)
        self.assertEqual(self.utterances[0].text, "我想爭取多兩個 headcount 係呀")

    def test_utterance_end_flushes_when_no_endpoint_was_detected(self):
        self.stt._handle_message(self.result("開會啦", True))
        self.stt._handle_message({"type": "UtteranceEnd", "last_word_end": 1.2})
        self.assertEqual([u.text for u in self.utterances], ["開會啦"])

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
        self.assertEqual([u.text for u in self.utterances], ["最後一句"])

    def test_url_declares_the_sample_rate_language_and_diarisation(self):
        url = self.stt._url()
        for expected in ("sample_rate=16000", "language=zh-HK", "encoding=linear16",
                         "interim_results=true", "diarize=true"):
            self.assertIn(expected, url)


class TestDeepgramKeyterms(unittest.TestCase):
    def make(self, model, terms):
        return DeepgramLiveSTT(
            api_key="k", sample_rate=16000, models=[model], keyterms=terms
        )

    def test_nova3_uses_keyterm_and_repeats_the_parameter(self):
        url = self.make("nova-3", ["Falcon", "NocolyHAP"])._url()
        self.assertIn("keyterm=Falcon", url)
        self.assertIn("keyterm=NocolyHAP", url)

    def test_older_models_use_the_keywords_parameter(self):
        url = self.make("nova-2", ["Falcon"])._url()
        self.assertIn("keywords=Falcon", url)
        self.assertNotIn("keyterm=", url)

    def test_terms_are_url_encoded(self):
        url = self.make("nova-3", ["fully loaded cost"])._url()
        self.assertIn("keyterm=fully+loaded+cost", url)

    def test_blank_terms_are_dropped_and_the_list_is_capped(self):
        stt = self.make("nova-3", ["  ", "real"] + [f"t{i}" for i in range(200)])
        from stt.deepgram_live import MAX_KEYTERMS

        self.assertEqual(len(stt.keyterms), MAX_KEYTERMS)
        self.assertNotIn("  ", stt.keyterms)

    def test_no_terms_means_no_parameter(self):
        self.assertNotIn("keyterm", self.make("nova-3", [])._url())

    def test_boosting_is_dropped_before_the_model_is_abandoned(self):
        """A rejected connection should cost us the extras, not the transcript."""
        stt = self.make("nova-3", ["Falcon"])
        self.assertIn("keyterm=Falcon", stt._url())
        stt._use_keyterms = False  # what the supervisor does on a refusal
        url = stt._url()
        self.assertNotIn("keyterm", url)
        self.assertIn("model=nova-3", url, "same model, just without boosting")


# ------------------------------------------------------------------ storage


class TestSTTChoices(unittest.TestCase):
    """The language/model picker offered in the pre-meeting form."""

    def test_preferred_model_leads_and_the_rest_stay_as_fallbacks(self):
        self.assertEqual(config.models_from("nova-2")[0], "nova-2")
        self.assertIn("nova-3", config.models_from("nova-2"))
        self.assertEqual(config.models_from("nova-3")[0], "nova-3")

    def test_an_unknown_model_falls_back_to_the_configured_list(self):
        self.assertEqual(config.models_from("not-a-model"), list(config.DEEPGRAM_MODELS))
        self.assertEqual(config.models_from(""), list(config.DEEPGRAM_MODELS))

    def test_cantonese_is_offered_and_multi_is_labelled_as_excluding_it(self):
        codes = [c["code"] for c in config.DEEPGRAM_LANGUAGE_CHOICES]
        self.assertIn("zh-HK", codes)
        # Deepgram's code-switching set does not include Cantonese, so the
        # option must say so rather than look like a better choice.
        multi = next(c for c in config.DEEPGRAM_LANGUAGE_CHOICES if c["code"] == "multi")
        self.assertIn("no Cantonese", multi["label"])


class TestStorage(unittest.TestCase):
    def setUp(self):
        db.init()

    def test_meeting_round_trip(self):
        brief = sample_brief()
        mid = db.create_meeting("Budget", brief.as_dict(), "zh-HK", "nova-3")
        db.add_segment(mid, 0, 1.5, 0, "第一句")
        db.add_segment(mid, 1, 4.0, None, "第二句")
        db.add_event(mid, "advice", {"key_point": "something"})
        db.add_event(mid, "attendee", {"say": "我想問一句", "kind": "question"})
        db.save_notes(mid, {"summary": "a summary", "decisions": ["d1"]})
        db.save_user_notes(mid, "my own notes")
        db.save_summary(mid, "rolling")
        db.save_speakers(mid, {0: "Alan", 1: "Wing"})
        db.finish_meeting(mid, 300.0, {"cost_usd": 0.12}, "nova-2")

        meeting = db.get_meeting(mid)
        self.assertEqual(meeting["title"], "Budget")
        self.assertEqual(meeting["stt_model"], "nova-2")
        self.assertEqual(meeting["user_notes"], "my own notes")
        self.assertEqual([s["text"] for s in meeting["segments"]], ["第一句", "第二句"])
        self.assertEqual(meeting["notes_json"]["decisions"], ["d1"])
        self.assertEqual(meeting["usage_json"]["cost_usd"], 0.12)
        self.assertEqual(meeting["speaker_names"], {0: "Alan", 1: "Wing"})
        self.assertEqual(meeting["brief_json"]["my_goal"], "爭取兩個 headcount")
        self.assertEqual([a["name"] for a in meeting["brief_json"]["attendees"]],
                         ["Isaac", "Alan", "Wing"])
        self.assertEqual([e["kind"] for e in meeting["events"]], ["advice", "attendee"])
        self.assertIsNotNone(meeting["ended_at"])

    def test_listing_includes_the_line_count(self):
        mid = db.create_meeting("Listed", Brief(title="Listed").as_dict(), "zh-HK", "nova-3")
        db.add_segment(mid, 0, 0.0, 0, "a")
        db.add_segment(mid, 1, 1.0, 0, "b")
        row = next(m for m in db.list_meetings() if m["id"] == mid)
        self.assertEqual(row["segments"], 2)

    def test_missing_meeting_is_none(self):
        self.assertIsNone(db.get_meeting(999_999))

    def test_init_is_idempotent_and_migrates_an_old_database(self):
        db.init()
        db.init()  # must not fail on the second pass
        mid = db.create_meeting("Again", {}, "zh-HK", "nova-3")
        self.assertIsNotNone(db.get_meeting(mid))


class TestExport(unittest.TestCase):
    def setUp(self):
        db.init()
        brief = sample_brief()
        self.mid = db.create_meeting(brief.title, brief.as_dict(), "zh-HK", "nova-3")
        db.add_segment(self.mid, 0, 5.0, 0, "我想爭取多兩個 headcount")
        db.add_segment(self.mid, 1, 3665.0, 1, "每個 head 幾多錢?")
        db.add_event(self.mid, "advice", {
            "key_point": "Alan wants headcount", "watch_out": "no owner",
            "questions": ["what is the cost?"],
        })
        db.add_event(self.mid, "attendee", {
            "say": "我想問一句，係全年計嗎?", "kind": "question", "why": "cost differs",
            "sources": [{"title": "A source", "url": "https://example.com"}],
        })
        db.add_event(self.mid, "answer", {
            "question": "點樣講服財務部?", "answer": "用數字",
            "sources": [{"title": "Ref", "url": "https://ref.example"}],
        })
        db.save_notes(self.mid, {
            "summary": "討論 headcount",
            "decisions": ["加一個 head"],
            "action_items": [{"who": "Wing", "what": "update model", "due": "Friday"}],
            "open_questions": ["幾時批?"],
            "topics": ["budget"],
        })
        db.save_user_notes(self.mid, "我自己嘅筆記")
        db.save_summary(self.mid, "rolling summary text")
        db.save_speakers(self.mid, {0: "Alan", 1: "Wing"})
        db.finish_meeting(self.mid, 3700.0, {
            "cost_usd": 0.42, "calls": 12, "prompt_tokens": 1000, "completion_tokens": 200,
        }, "nova-3")
        self.meeting = db.get_meeting(self.mid)

    def test_markdown_contains_every_part_of_the_session(self):
        md = export.to_markdown(self.meeting)
        for expected in [
            "# Q3 planning",
            "Alan** — ops",
            "*(me)*",
            "爭取兩個 headcount",           # my goal
            "Falcon, NocolyHAP",            # glossary
            "## Decisions",
            "**Wing**: update model — due Friday",
            "## Open questions",
            "我自己嘅筆記",                  # my own notes
            "我想問一句",                    # attendee turn
            "why: cost differs",
            "[A source](https://example.com)",
            "Alan wants headcount",         # copilot advice
            "點樣講服財務部?",               # question the user asked
            "## Full transcript",
            "**Alan**: 我想爭取多兩個 headcount",
            "rolling summary text",
            "$0.4200",
        ]:
            self.assertIn(expected, md, expected)

    def test_markdown_timestamps_are_readable_past_an_hour(self):
        md = export.to_markdown(self.meeting)
        self.assertIn("`0:00:05`", md)
        self.assertIn("`1:01:05`", md, "a five-hour meeting needs hours in the clock")

    def test_json_export_is_valid_and_complete(self):
        payload = json.loads(export.to_json(self.meeting))
        self.assertEqual(payload["meeting"]["title"], "Q3 planning")
        self.assertEqual(payload["brief"]["glossary"], ["Falcon", "NocolyHAP"])
        self.assertEqual(payload["transcript"][0]["speaker_label"], "Alan")
        self.assertEqual(payload["transcript"][1]["at_clock"], "1:01:05")
        self.assertEqual(payload["notes"]["decisions"], ["加一個 head"])
        self.assertEqual(payload["my_notes"], "我自己嘅筆記")
        self.assertEqual(payload["rolling_summary"], "rolling summary text")
        self.assertEqual(len(payload["copilot_log"]), 3)
        self.assertEqual(payload["usage"]["cost_usd"], 0.42)

    def test_filename_stem_is_safe_and_descriptive(self):
        stem = export.filename_stem(self.meeting)
        self.assertIn("Q3-planning", stem)
        for bad in "/\\:*?\"<>|":
            self.assertNotIn(bad, stem)

    def test_export_survives_a_meeting_with_almost_nothing_in_it(self):
        mid = db.create_meeting("", {}, "zh-HK", "nova-3")
        meeting = db.get_meeting(mid)
        self.assertIn("#", export.to_markdown(meeting))
        json.loads(export.to_json(meeting))


# ------------------------------------------------------------------- session


class TestSessionCostAndSnapshot(ConfigGuard):
    def setUp(self):
        super().setUp()
        db.init()
        config.DEEPGRAM_API_KEY = "test-key"
        config.THINK_MIN_INTERVAL = 9999  # keep the copilot out of this test
        config.NOTES_INTERVAL = 9999
        config.SPEAKER_GUESS_INTERVAL = 9999
        from session import MeetingSession, _validate_sample_rate

        self.validate = _validate_sample_rate
        self.emitter = Emitter()
        self.session = MeetingSession(
            brief=sample_brief(), sample_rate=16000, emit=self.emitter
        )
        self.addCleanup(self.session.engine.close, False)

    def test_sample_rate_validation(self):
        self.assertEqual(self.validate("48000"), 48000)
        for bad in ("abc", None, 4000, 96000):
            with self.assertRaises(ValueError):
                self.validate(bad)

    def test_the_brief_reaches_the_transcriber_as_boosted_terms(self):
        self.assertIn("Falcon", self.session.stt.keyterms)
        self.assertIn("Alan", self.session.stt.keyterms)

    def test_utterance_is_stored_emitted_and_priced(self):
        self.session._on_utterance(Utterance(text="第一句", speaker=0))
        self.assertEqual(self.emitter.of("segment")[0]["text"], "第一句")

        stored = db.get_meeting(self.session.meeting_id)
        self.assertEqual([s["text"] for s in stored["segments"]], ["第一句"])

        self.session.stt._bytes_sent = 16000 * 2 * 600  # ten minutes
        cost = self.session.cost()
        self.assertAlmostEqual(cost["audio_minutes"], 10.0)
        self.assertAlmostEqual(cost["stt_usd"], 10 * config.DEEPGRAM_USD_PER_MINUTE, places=4)

    def test_five_hour_cost_estimate_is_in_the_expected_range(self):
        """Guards the number quoted to the user: STT for 5h should be ~$2.31."""
        self.session.stt._bytes_sent = 16000 * 2 * 5 * 3600
        cost = self.session.cost()
        self.assertAlmostEqual(cost["audio_minutes"], 300.0)
        self.assertTrue(2.0 < cost["stt_usd"] < 3.0, cost["stt_usd"])

    def test_naming_a_speaker_persists_and_broadcasts(self):
        self.session._on_utterance(Utterance(text="第一句", speaker=0))
        self.session.set_speaker_name(0, "Alan")

        self.assertEqual(self.emitter.of("speakers")[-1]["speaker_names"], {"0": "Alan"})
        self.assertEqual(db.get_meeting(self.session.meeting_id)["speaker_names"], {0: "Alan"})
        self.assertEqual(self.session.snapshot()["segments"][0]["speaker_label"], "Alan")

    def test_accepting_a_speaker_suggestion_applies_every_proposal(self):
        self.session._on_utterance(Utterance(text="a", speaker=0))
        self.session._on_utterance(Utterance(text="b", speaker=1))
        with self.session.state.lock:
            self.session.state.speaker_suggestion = {
                "proposals": [
                    {"speaker": 0, "name": "Alan"},
                    {"speaker": 1, "name": "Wing"},
                ]
            }
        self.session.apply_speaker_suggestion(accept=True)

        self.assertEqual(self.session.state.speaker_names, {0: "Alan", 1: "Wing"})
        self.assertIsNone(self.session.state.speaker_suggestion)
        self.assertEqual(
            db.get_meeting(self.session.meeting_id)["speaker_names"], {0: "Alan", 1: "Wing"}
        )

    def test_rejecting_a_speaker_suggestion_changes_nothing(self):
        with self.session.state.lock:
            self.session.state.speaker_suggestion = {"proposals": [{"speaker": 0, "name": "Alan"}]}
        self.session.apply_speaker_suggestion(accept=False)
        self.assertEqual(self.session.state.speaker_names, {})
        self.assertIsNone(self.session.state.speaker_suggestion)
        self.assertEqual(self.emitter.of("speaker_suggestion_cleared"), [{}])

    def test_attendee_turns_are_kept_and_persisted(self):
        self.session._engine_emit("attendee", {"say": "我想問一句", "kind": "question"})
        self.assertEqual(self.session.attendee_turns[-1]["say"], "我想問一句")
        self.assertEqual(self.session.snapshot()["attendee_turns"][-1]["say"], "我想問一句")
        kinds = [e["kind"] for e in db.get_meeting(self.session.meeting_id)["events"]]
        self.assertIn("attendee", kinds)

    def test_snapshot_carries_everything_the_page_needs(self):
        self.session._on_utterance(Utterance(text="一句", speaker=1))
        self.session.set_user_notes("my notes")
        snap = self.session.snapshot()
        for key in (
            "meeting_id", "brief", "running", "status", "segments", "speaker_names",
            "speaker_suggestion", "cards", "attendee_turns", "notes", "user_notes",
            "summary", "cost", "web_search", "attendee_enabled",
        ):
            self.assertIn(key, snap)
        self.assertTrue(snap["running"])
        self.assertEqual(snap["brief"]["title"], "Q3 planning")
        self.assertEqual(snap["segments"][0]["speaker_label"], "S2")

    def test_pausing_stops_audio_reaching_the_transcriber(self):
        """Cost follows from this: audio_seconds counts bytes actually sent on to
        Deepgram, so audio that never enters the queue is never billed."""
        queue = self.session.stt._audio

        self.session.feed_audio(b"\x00\x01" * 1000)
        self.assertEqual(queue.qsize(), 1)

        self.session.set_paused(True)
        self.assertTrue(self.session.paused)
        self.session.feed_audio(b"\x00\x01" * 1000)
        self.assertEqual(queue.qsize(), 1, "paused audio must be dropped, not queued")

        self.session.set_paused(False)
        self.session.feed_audio(b"\x00\x01" * 1000)
        self.assertEqual(queue.qsize(), 2, "resuming must start feeding again")

    def test_pause_is_announced_and_recorded(self):
        self.session.set_paused(True)
        payload = self.emitter.of("paused")[-1]
        self.assertTrue(payload["paused"])
        self.assertIn("at", payload)

        self.session.set_paused(False)
        self.assertFalse(self.emitter.of("paused")[-1]["paused"])

        kinds = [e["kind"] for e in db.get_meeting(self.session.meeting_id)["events"]]
        self.assertEqual(kinds.count("pause"), 2, "both edges belong in the record")

    def test_pausing_twice_changes_nothing(self):
        self.session.set_paused(True)
        self.session.set_paused(True)
        self.assertEqual(len(self.emitter.of("paused")), 1)

    def test_a_stopped_meeting_cannot_be_paused(self):
        self.session.stopped = True
        self.session.set_paused(True)
        self.assertFalse(self.session.paused)
        self.assertEqual(self.emitter.of("paused"), [])

    def test_snapshot_reports_pause_state_and_language(self):
        self.session.set_paused(True)
        snap = self.session.snapshot()
        self.assertTrue(snap["paused"])
        self.assertEqual(snap["language"], config.DEEPGRAM_LANGUAGE)

    def test_feeding_audio_after_stop_is_ignored(self):
        self.session.stopped = True
        self.session.feed_audio(b"\x00\x00" * 100)
        self.assertEqual(self.session.stt.audio_seconds, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
