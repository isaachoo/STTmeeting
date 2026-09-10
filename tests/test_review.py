"""Tests for the review workspace: retrieval, digest, reports, and the HTTP API.

No network. The LLM is a double that records what it was asked and replies with
whatever the test needs, so the things being checked are the ones that actually
go wrong: passages losing their line numbers, citations pointing at lines that
were never sent, a digest silently claiming to be complete when a section
failed, a report generation quietly re-reading a transcript that was already
read, and the endpoints that let a user fix all of it.
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
config.DB_PATH = config.DATA_DIR / "review-tests.sqlite3"

from copilot.llm import LLMError, Usage  # noqa: E402
from review import digest as digest_module  # noqa: E402
from review import jobs, qa, reports, retrieval  # noqa: E402
from storage import db  # noqa: E402

# ---------------------------------------------------------------- fixtures

# A short meeting that reads like a real one: Cantonese with English terms mixed
# in, numbers stated out loud, one commitment, one thing left open.
LINES = [
    (0, "我們今日主要傾 Q3 budget 同 headcount"),
    (1, "Falcon 個 timeline delay 咗兩次，客戶已經投訴"),
    (0, "我要多兩個 headcount，唔係 Q3 做唔完"),
    (2, "財務部立場係 budget 最多加 50 萬，唔可以再多"),
    (0, "50 萬只夠一個 head 加少少 contractor"),
    (1, "咁不如先請一個，Falcon 用 contractor 頂住"),
    (2, "我下星期五之前出一個 revised budget model"),
    (0, "Aircon 壞咗嗰件事，邊個跟？"),
    (1, "未有人跟，office admin 放假"),
]


class FakeLLM:
    """Stands in for OpenRouterClient. `json_reply` and `text_reply` may be
    callables, so a test can vary the answer per call or raise."""

    def __init__(self, json_reply=None, text_reply="an answer"):
        self.usage = Usage()
        self.json_reply = json_reply if json_reply is not None else {}
        self.text_reply = text_reply
        self.json_calls: list[list[dict]] = []
        self.text_calls: list[list[dict]] = []
        self.lock = threading.Lock()

    def chat_json(self, messages, **_kwargs):
        with self.lock:
            self.json_calls.append(messages)
        self.usage.add({"prompt_tokens": 200, "completion_tokens": 50, "cost": 0.0002})
        reply = self.json_reply
        if callable(reply):
            reply = reply(messages, len(self.json_calls))
        return json.loads(json.dumps(reply))

    def chat(self, messages, **_kwargs):
        with self.lock:
            self.text_calls.append(messages)
        self.usage.add({"prompt_tokens": 200, "completion_tokens": 50, "cost": 0.0002})
        reply = self.text_reply
        return reply(messages, len(self.text_calls)) if callable(reply) else reply

    def sent(self) -> str:
        with self.lock:
            calls = list(self.json_calls) + list(self.text_calls)
        return "\n".join(m["content"] for call in calls for m in call)


def make_meeting(lines=LINES, title="Q3 planning", names=None) -> dict:
    """A stored meeting, created through the real database layer."""
    db.init()
    meeting_id = db.create_meeting(
        title=title,
        brief={"title": title, "attendees": [{"name": "Alan", "is_me": True}]},
        language="zh-HK",
        stt_model="nova-3",
    )
    for index, (speaker, text) in enumerate(lines):
        db.add_segment(meeting_id, index, index * 4.0, speaker, text)
    db.save_speakers(meeting_id, names if names is not None else {0: "Alan", 1: "Bella"})
    db.save_notes(meeting_id, {"summary": "Budget and headcount.", "decisions": []})
    db.finish_meeting(meeting_id, audio_seconds=600, usage={}, stt_model="nova-3")
    return db.get_meeting(meeting_id)


DIGEST_REPLY = {
    "topics": [{"topic": "Q3 budget", "what_happened": "Alan asked for two heads.",
                "lines": [0, 2]}],
    "decisions": [{"decision": "Hire one head now", "by": "Alan", "lines": [5]}],
    "actions": [{"who": "Carmen", "what": "issue a revised budget model",
                 "due": "next Friday", "lines": [6]}],
    "questions": [{"question": "Who follows up the aircon?", "asked_by": "Alan",
                   "answered": False, "lines": [7, 8]}],
    "facts": [{"fact": "Finance will add at most HK$500k", "lines": [3]}],
    "quotes": [{"who": "Carmen", "said": "budget 最多加 50 萬", "line": 3}],
}


# ---------------------------------------------------------------- retrieval


class TestFeatures(unittest.TestCase):
    def test_chinese_produces_bigrams(self):
        self.assertIn("預算", retrieval.features("預算"))
        self.assertIn("加預", retrieval.features("加預算"))
        self.assertIn("預算", retrieval.features("加預算"))

    def test_single_characters_are_kept_but_tagged(self):
        """They are weak evidence, and the tag is what lets them be weighted
        down instead of counting the same as a bigram."""
        self.assertIn(retrieval.UNIGRAM + "錢", retrieval.features("錢 budget"))
        self.assertLess(retrieval.UNIGRAM_WEIGHT, 1.0)

    def test_a_word_reordered_still_matches_something(self):
        """"budget 加幾多" against "最多加 50 萬" shares no bigram at all; without
        single characters the question would match nothing."""
        question = set(retrieval.features("加幾多"))
        spoken = set(retrieval.features("最多加"))
        self.assertTrue(question & spoken)

    def test_english_words_are_kept_whole(self):
        self.assertIn("headcount", retrieval.features("more headcount please"))

    def test_filler_words_are_dropped(self):
        self.assertNotIn("what", retrieval.features("what did we say"))
        self.assertNotIn("the", retrieval.features("the budget"))

    def test_case_does_not_matter(self):
        self.assertEqual(retrieval.features("Falcon"), retrieval.features("FALCON"))

    def test_empty_input_is_fine(self):
        self.assertEqual(retrieval.features(""), [])
        self.assertEqual(retrieval.features(None), [])


class TestPassages(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {"idx": i, "at": i * 4.0, "speaker": s, "text": t, "speaker_name": ""}
            for i, (s, t) in enumerate(LINES)
        ]

    def test_every_line_carries_its_number(self):
        """The line number is what makes a citation clickable, so it has to be in
        the text the model sees, not just in the metadata."""
        passages = retrieval.build_passages(self.segments, {0: "Alan"}, max_chars=60)
        joined = "\n".join(p.text for p in passages)
        for i in range(len(LINES)):
            self.assertIn(f"[#{i}]", joined)

    def test_nothing_is_lost_and_nothing_is_duplicated(self):
        passages = retrieval.build_passages(self.segments, {}, max_chars=50)
        indices = [i for p in passages for i in p.indices]
        self.assertEqual(indices, list(range(len(LINES))))

    def test_names_are_used_when_known(self):
        passages = retrieval.build_passages(self.segments, {0: "Alan"}, max_chars=10_000)
        self.assertIn("Alan:", passages[0].text)
        self.assertIn("S2:", passages[0].text, "an unnamed voice keeps its tag")

    def test_a_line_level_name_wins(self):
        self.segments[0]["speaker_name"] = "Someone else"
        passages = retrieval.build_passages(self.segments, {0: "Alan"}, max_chars=10_000)
        self.assertIn("[#0] Someone else:", passages[0].text)

    def test_empty_lines_are_skipped(self):
        self.segments.insert(2, {"idx": 99, "at": 1, "speaker": 0, "text": "   "})
        passages = retrieval.build_passages(self.segments, {}, max_chars=10_000)
        self.assertNotIn("[#99]", passages[0].text)

    def test_passages_respect_the_size_limit(self):
        passages = retrieval.build_passages(self.segments, {}, max_chars=60)
        self.assertGreater(len(passages), 1)
        for passage in passages:
            # One line longer than the limit still gets its own passage; the
            # limit is a target, not a guarantee it can always meet.
            self.assertTrue(len(passage.text) <= 60 or len(passage.indices) == 1)

    def test_no_segments_means_no_passages(self):
        self.assertEqual(retrieval.build_passages([], {}), [])


class TestSearch(unittest.TestCase):
    def setUp(self):
        segments = [
            {"idx": i, "at": i * 4.0, "speaker": s, "text": t, "speaker_name": ""}
            for i, (s, t) in enumerate(LINES)
        ]
        self.index = retrieval.Index(retrieval.build_passages(segments, {}, max_chars=45))
        self.segments = segments

    def _top(self, query):
        hits = self.index.search(query, top_k=1)
        return hits[0][0] if hits else None

    def test_it_finds_the_line_about_the_number(self):
        self.assertIn("50 萬", self._top("budget 加幾多錢").text)

    def test_it_finds_an_english_term(self):
        self.assertIn("Falcon", self._top("Falcon timeline").text)

    def test_it_finds_a_cantonese_term(self):
        self.assertIn("Aircon", self._top("aircon 壞咗邊個跟").text)

    def test_results_come_back_in_meeting_order(self):
        hits = self.index.search("budget headcount Falcon", top_k=4)
        starts = [passage.start for passage, _ in hits]
        self.assertEqual(starts, sorted(starts))

    def test_a_query_matching_nothing_returns_nothing(self):
        self.assertEqual(self.index.search("zzzzz quantum tunnelling"), [])

    def test_an_empty_query_returns_nothing(self):
        self.assertEqual(self.index.search(""), [])

    def test_an_empty_meeting_returns_nothing(self):
        self.assertEqual(retrieval.Index([]).search("anything"), [])

    def test_find_is_the_whole_pipeline(self):
        passages = retrieval.find(self.segments, {0: "Alan"}, "headcount", top_k=2)
        self.assertTrue(passages)
        self.assertTrue(any("headcount" in p.text for p in passages))


class TestCitations(unittest.TestCase):
    def test_it_reads_the_numbers_out_of_an_answer(self):
        self.assertEqual(
            retrieval.cited_indices("財務部只加 50 萬 [#3]，Alan 想加兩個 [#2]"),
            [3, 2],
        )

    def test_duplicates_collapse_but_order_is_kept(self):
        self.assertEqual(retrieval.cited_indices("[#5] [#2] [#5]"), [5, 2])

    def test_an_invented_line_number_is_dropped(self):
        """A citation the user can click and land nowhere is worse than none."""
        self.assertEqual(
            retrieval.cited_indices("see [#3] and [#900]", valid={0, 1, 2, 3}), [3]
        )

    def test_no_citations_is_not_an_error(self):
        self.assertEqual(retrieval.cited_indices("no numbers here"), [])
        self.assertEqual(retrieval.cited_indices(""), [])

    def test_indices_in_reports_exactly_what_was_sent(self):
        segments = [
            {"idx": 0, "at": 0, "speaker": 0, "text": "one"},
            {"idx": 1, "at": 1, "speaker": 0, "text": "   "},  # skipped
            {"idx": 2, "at": 2, "speaker": 0, "text": "three"},
        ]
        passages = retrieval.build_passages(segments, {}, max_chars=10_000)
        self.assertEqual(retrieval.indices_in(passages), {0, 2})


# ------------------------------------------------------------------- digest


class TestChunking(unittest.TestCase):
    def test_the_whole_transcript_is_covered(self):
        segments = [
            {"idx": i, "at": i, "speaker": 0, "text": f"line number {i} " + "x" * 40}
            for i in range(40)
        ]
        chunks = digest_module.chunk_transcript(segments, {}, max_chars=400)
        self.assertGreater(len(chunks), 1)
        joined = "\n".join(chunks)
        for i in range(40):
            self.assertIn(f"[#{i}]", joined)

    def test_an_empty_meeting_produces_no_chunks(self):
        self.assertEqual(digest_module.chunk_transcript([], {}), [])


class TestDigestBuild(unittest.TestCase):
    def setUp(self):
        self.meeting = make_meeting()

    def test_it_reads_every_chunk_and_merges_the_results(self):
        llm = FakeLLM(json_reply=DIGEST_REPLY)
        original = config.REVIEW_CHUNK_CHARS
        config.REVIEW_CHUNK_CHARS = 80  # force several chunks out of a short meeting
        self.addCleanup(setattr, config, "REVIEW_CHUNK_CHARS", original)

        digest = digest_module.build(self.meeting, client=llm)
        sections = digest["sections"]
        self.assertGreater(sections, 1)
        self.assertEqual(len(llm.json_calls), sections)
        self.assertEqual(len(digest["decisions"]), sections, "one per section, merged")
        self.assertEqual(digest["failed_sections"], [])

    def test_progress_is_reported_for_every_section(self):
        seen = []
        original = config.REVIEW_CHUNK_CHARS
        config.REVIEW_CHUNK_CHARS = 80
        self.addCleanup(setattr, config, "REVIEW_CHUNK_CHARS", original)
        digest_module.build(
            self.meeting,
            client=FakeLLM(json_reply=DIGEST_REPLY),
            on_progress=lambda done, total: seen.append((done, total)),
        )
        self.assertTrue(seen)
        self.assertEqual(max(d for d, _ in seen), seen[0][1])

    def test_the_brief_and_the_roster_reach_the_prompt(self):
        llm = FakeLLM(json_reply=DIGEST_REPLY)
        digest_module.build(self.meeting, client=llm)
        sent = llm.sent()
        self.assertIn("Q3 planning", sent)
        self.assertIn("Alan", sent)

    def test_one_failed_section_does_not_lose_the_others(self):
        original = config.REVIEW_CHUNK_CHARS
        config.REVIEW_CHUNK_CHARS = 80
        self.addCleanup(setattr, config, "REVIEW_CHUNK_CHARS", original)

        def reply(_messages, call_number):
            if call_number == 1:
                raise LLMError("rate limited")
            return DIGEST_REPLY

        digest = digest_module.build(self.meeting, client=FakeLLM(json_reply=reply))
        self.assertTrue(digest["decisions"], "the sections that worked are kept")
        self.assertEqual(len(digest["failed_sections"]), 1)

    def test_a_failure_is_visible_in_the_rendered_digest(self):
        """A report written from an incomplete digest must be able to say so."""
        digest = digest_module.merge([{"_failed": "section 2 of 3"}], sections=3)
        self.assertIn("could not be read", digest_module.to_text(digest))

    def test_a_meeting_with_no_transcript_needs_no_calls(self):
        llm = FakeLLM(json_reply=DIGEST_REPLY)
        empty_meeting = dict(self.meeting, segments=[])
        digest = digest_module.build(empty_meeting, client=llm)
        self.assertEqual(llm.json_calls, [])
        self.assertEqual(digest["decisions"], [])

    def test_junk_in_the_reply_is_ignored(self):
        llm = FakeLLM(json_reply={"decisions": "not a list", "facts": [1, 2, "x"]})
        digest = digest_module.build(self.meeting, client=llm)
        self.assertEqual(digest["decisions"], [])
        self.assertEqual(digest["facts"], [])


class TestDigestCache(unittest.TestCase):
    def setUp(self):
        self.meeting = make_meeting()

    def test_the_second_call_costs_nothing(self):
        first = FakeLLM(json_reply=DIGEST_REPLY)
        _, built = digest_module.ensure(self.meeting, client=first)
        self.assertTrue(built)

        second = FakeLLM(json_reply=DIGEST_REPLY)
        digest, built = digest_module.ensure(self.meeting, client=second)
        self.assertFalse(built)
        self.assertEqual(second.json_calls, [], "a cached digest must not be re-read")
        self.assertTrue(digest["decisions"])

    def test_a_longer_transcript_invalidates_it(self):
        digest_module.ensure(self.meeting, client=FakeLLM(json_reply=DIGEST_REPLY))
        db.add_segment(self.meeting["id"], len(LINES), 99.0, 0, "one more thing")
        fresh = db.get_meeting(self.meeting["id"])

        llm = FakeLLM(json_reply=DIGEST_REPLY)
        _, built = digest_module.ensure(fresh, client=llm)
        self.assertTrue(built)
        self.assertTrue(llm.json_calls)

    def test_a_digest_with_gaps_is_rebuilt_next_time(self):
        db.save_digest(
            self.meeting["id"],
            {**digest_module.empty(), "sections": 2, "failed_sections": ["section 1 of 2"]},
            len(LINES),
        )
        llm = FakeLLM(json_reply=DIGEST_REPLY)
        _, built = digest_module.ensure(self.meeting, client=llm)
        self.assertTrue(built, "a partial read should be retried, not treated as done")


class TestDigestText(unittest.TestCase):
    def test_it_keeps_the_line_references(self):
        text = digest_module.to_text({**digest_module.empty(), **DIGEST_REPLY})
        self.assertIn("[#3]", text)
        self.assertIn("[#6]", text)

    def test_unanswered_questions_are_marked(self):
        text = digest_module.to_text({**digest_module.empty(), **DIGEST_REPLY})
        self.assertIn("NOT answered", text)

    def test_the_important_parts_come_first(self):
        """Truncation cuts from the end, so decisions and facts must be above the
        blow-by-blow discussion."""
        text = digest_module.to_text({**digest_module.empty(), **DIGEST_REPLY})
        self.assertLess(text.index("## Decisions"), text.index("## How the discussion went"))

    def test_it_is_truncated_rather_than_left_unbounded(self):
        big = {**digest_module.empty(),
               "facts": [{"fact": "x" * 200, "lines": [1]} for _ in range(200)]}
        text = digest_module.to_text(big, max_chars=1000)
        self.assertLessEqual(len(text), 1100)
        self.assertIn("truncated", text)

    def test_an_empty_digest_renders_to_nothing(self):
        self.assertEqual(digest_module.to_text({}), "")


# ------------------------------------------------------------------ reports


class TestReports(unittest.TestCase):
    def setUp(self):
        self.meeting = make_meeting()

    def _llm(self, body="# Minutes\n\nSomething happened."):
        return FakeLLM(json_reply=DIGEST_REPLY, text_reply=body)

    def test_every_offered_kind_can_be_generated(self):
        for kind in [k["code"] for k in config.REPORT_KINDS]:
            with self.subTest(kind=kind):
                row = reports.generate(self.meeting, kind, client=self._llm())
                self.assertEqual(row["kind"], kind)
                self.assertTrue(row["body"])
                self.assertTrue(row["title"])

    def test_the_prompts_module_offers_the_same_four(self):
        from copilot import prompts

        self.assertEqual(
            sorted(prompts.report_kinds()),
            sorted(k["code"] for k in config.REPORT_KINDS),
        )

    def test_an_unknown_kind_is_refused_before_any_spend(self):
        llm = self._llm()
        with self.assertRaises(ValueError):
            reports.generate(self.meeting, "haiku", client=llm)
        self.assertEqual(llm.text_calls, [])

    def test_it_is_saved_and_listed(self):
        row = reports.generate(self.meeting, "minutes", client=self._llm())
        listed = db.list_reports(self.meeting["id"])
        self.assertIn(row["id"], [r["id"] for r in listed])

    def test_regenerating_keeps_the_previous_version(self):
        """Someone may already have edited and sent the first one."""
        first = reports.generate(self.meeting, "minutes", client=self._llm("first"))
        second = reports.generate(self.meeting, "minutes", client=self._llm("second"))
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(db.get_report(first["id"])["body"], "first")

    def test_the_digest_is_read_once_for_four_reports(self):
        llm = self._llm()
        for kind in ("minutes", "actions", "summary", "email"):
            reports.generate(self.meeting, kind, client=llm)
        self.assertEqual(
            len(llm.json_calls), 1, "the digest is the expensive part; read it once"
        )
        self.assertEqual(len(llm.text_calls), 4)

    def test_the_user_action_list_is_given_to_the_writer(self):
        db.add_action(self.meeting["id"], who="Alan", what="book the room", due="Monday")
        llm = self._llm()
        reports.generate(self.meeting, "actions", client=llm)
        sent = llm.sent()
        self.assertIn("book the room", sent)
        self.assertIn("authoritative", sent)

    def test_the_date_is_supplied_because_a_transcript_has_none(self):
        llm = self._llm()
        reports.generate(self.meeting, "minutes", client=llm)
        self.assertIn("Date:", llm.sent())

    def test_an_empty_document_is_an_error_not_a_saved_blank(self):
        with self.assertRaises(LLMError):
            reports.generate(self.meeting, "minutes", client=self._llm("   "))
        self.assertEqual(db.list_reports(self.meeting["id"]), [])

    def test_the_filename_says_what_it_is(self):
        row = reports.generate(self.meeting, "email", client=self._llm())
        name = reports.filename_for(row, self.meeting)
        self.assertTrue(name.endswith("-email.md"))
        self.assertIn("Q3-planning", name)


class TestDraftActions(unittest.TestCase):
    def setUp(self):
        self.meeting = make_meeting()

    def _llm(self, actions):
        def reply(messages, _n):
            if "extract action items" in messages[0]["content"]:
                return {"actions": actions}
            return DIGEST_REPLY

        return FakeLLM(json_reply=reply)

    def test_it_proposes_without_saving(self):
        """A wrong owner in a list that gets emailed is a real problem, so
        proposals are the user's to accept."""
        result = reports.draft_actions(
            self.meeting,
            client=self._llm([{"who": "Carmen", "what": "revised budget model",
                               "due": "Friday", "lines": [6]}]),
        )
        self.assertEqual(len(result["proposals"]), 1)
        self.assertEqual(db.list_actions(self.meeting["id"]), [])

    def test_an_item_with_no_task_is_dropped(self):
        result = reports.draft_actions(
            self.meeting, client=self._llm([{"who": "Alan", "what": "  "}])
        )
        self.assertEqual(result["proposals"], [])

    def test_a_missing_owner_becomes_unassigned_rather_than_blank(self):
        result = reports.draft_actions(
            self.meeting, client=self._llm([{"what": "chase the aircon"}])
        )
        self.assertEqual(result["proposals"][0]["who"], "unassigned")

    def test_low_confidence_is_carried_through_to_the_user(self):
        result = reports.draft_actions(
            self.meeting,
            client=self._llm([{"what": "maybe hire", "confidence": "low"}]),
        )
        self.assertEqual(result["proposals"][0]["confidence"], "low")

    def test_junk_line_numbers_are_dropped(self):
        result = reports.draft_actions(
            self.meeting,
            client=self._llm([{"what": "do it", "lines": [3, "x", None, 5]}]),
        )
        self.assertEqual(result["proposals"][0]["lines"], [3, 5])

    def test_what_the_user_already_has_is_shown_to_the_model(self):
        db.add_action(self.meeting["id"], who="Alan", what="already on my list")
        llm = self._llm([])
        reports.draft_actions(self.meeting, client=llm)
        self.assertIn("already on my list", llm.sent())

    def test_nothing_agreed_is_an_acceptable_answer(self):
        result = reports.draft_actions(self.meeting, client=self._llm([]))
        self.assertEqual(result["proposals"], [])


# ------------------------------------------------------------ ask the meeting


class TestAsk(unittest.TestCase):
    def setUp(self):
        self.meeting = make_meeting()

    def test_the_matching_passage_is_what_gets_sent(self):
        llm = FakeLLM(text_reply="財務部最多加 50 萬 [#3]。")
        qa.ask(self.meeting, "budget 加幾多？", client=llm)
        sent = llm.sent()
        self.assertIn("50 萬", sent)
        self.assertIn("[#3]", sent)

    def test_the_answer_and_its_citations_come_back(self):
        llm = FakeLLM(text_reply="財務部最多加 50 萬 [#3]。")
        result = qa.ask(self.meeting, "budget 加幾多？", client=llm)
        self.assertEqual(result["cited"], [3])
        self.assertTrue(result["passages"])
        self.assertGreater(result["cost_usd"], 0)

    def test_a_citation_outside_the_passages_is_dropped(self):
        llm = FakeLLM(text_reply="see [#3] and [#4321]")
        result = qa.ask(self.meeting, "budget", client=llm)
        self.assertNotIn(4321, result["cited"])

    def test_it_is_saved_so_a_reload_keeps_the_thread(self):
        qa.ask(self.meeting, "第一個問題", client=FakeLLM(text_reply="答案 [#0]"))
        turns = db.list_chat(self.meeting["id"])
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["question"], "第一個問題")
        self.assertEqual(turns[0]["cited"], [0])

    def test_earlier_turns_go_in_as_conversation(self):
        """So "and who objected to that?" resolves against the last answer."""
        llm = FakeLLM(text_reply="第二個答案")
        qa.ask(
            self.meeting,
            "跟住呢？",
            history=[{"question": "第一個問題", "answer": "第一個答案"}],
            client=llm,
        )
        roles = [m["role"] for m in llm.text_calls[0]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_without_a_digest_it_says_so_and_uses_the_live_notes(self):
        llm = FakeLLM(text_reply="answer")
        result = qa.ask(self.meeting, "budget", client=llm)
        self.assertFalse(result["used_digest"])
        self.assertIn("No full-meeting digest", llm.sent())

    def test_asking_never_silently_builds_the_digest(self):
        """It is a dozen LLM calls and a minute of waiting; the user chooses."""
        llm = FakeLLM(text_reply="answer")
        qa.ask(self.meeting, "budget", client=llm)
        self.assertEqual(llm.json_calls, [], "no chunk reads should have happened")
        self.assertEqual(db.get_digest(self.meeting["id"])[0], {})

    def test_with_a_digest_it_is_used(self):
        digest_module.ensure(self.meeting, client=FakeLLM(json_reply=DIGEST_REPLY))
        llm = FakeLLM(text_reply="answer")
        result = qa.ask(self.meeting, "budget", client=llm)
        self.assertTrue(result["used_digest"])
        self.assertIn("Finance will add at most", llm.sent())

    def test_an_empty_question_is_refused(self):
        with self.assertRaises(ValueError):
            qa.ask(self.meeting, "   ", client=FakeLLM())

    def test_a_question_matching_nothing_still_gets_an_answer(self):
        llm = FakeLLM(text_reply="Nothing in the meeting covers that.")
        result = qa.ask(self.meeting, "zzzz quantum tunnelling", client=llm)
        self.assertEqual(result["passages"], [])
        self.assertIn("nothing in the transcript matched", llm.sent())


# -------------------------------------------------------------- action items


class TestActionItems(unittest.TestCase):
    def setUp(self):
        self.meeting_id = make_meeting()["id"]

    def test_add_and_list_in_order(self):
        first = db.add_action(self.meeting_id, "Alan", "book the room")
        second = db.add_action(self.meeting_id, "Bella", "send the deck")
        listed = db.list_actions(self.meeting_id)
        self.assertEqual([a["id"] for a in listed], [first["id"], second["id"]])
        self.assertEqual(listed[0]["status"], "open")

    def test_edit_one_field_leaves_the_rest(self):
        action = db.add_action(self.meeting_id, "Alan", "book the room", "Monday")
        updated = db.update_action(self.meeting_id, action["id"], {"who": "Bella"})
        self.assertEqual(updated["who"], "Bella")
        self.assertEqual(updated["what"], "book the room")
        self.assertEqual(updated["due"], "Monday")

    def test_ticking_it_off(self):
        action = db.add_action(self.meeting_id, "Alan", "book the room")
        updated = db.update_action(self.meeting_id, action["id"], {"status": "done"})
        self.assertEqual(updated["status"], "done")

    def test_columns_a_user_should_not_set_are_ignored(self):
        action = db.add_action(self.meeting_id, "Alan", "book the room")
        db.update_action(self.meeting_id, action["id"], {"meeting_id": 9999, "id": 1})
        self.assertEqual(db.get_action(self.meeting_id, action["id"])["meeting_id"],
                         self.meeting_id)

    def test_delete(self):
        action = db.add_action(self.meeting_id, "Alan", "book the room")
        self.assertTrue(db.delete_action(self.meeting_id, action["id"]))
        self.assertEqual(db.list_actions(self.meeting_id), [])
        self.assertFalse(db.delete_action(self.meeting_id, action["id"]))

    def test_items_belong_to_their_meeting(self):
        other = make_meeting(title="Another")["id"]
        action = db.add_action(self.meeting_id, "Alan", "mine")
        self.assertIsNone(db.get_action(other, action["id"]))
        self.assertFalse(db.delete_action(other, action["id"]))


class TestReportStorage(unittest.TestCase):
    def setUp(self):
        self.meeting_id = make_meeting()["id"]

    def test_edit_keeps_the_title_unless_asked(self):
        row = db.save_report(self.meeting_id, "minutes", "Minutes", "body", "m")
        updated = db.update_report(row["id"], "edited body")
        self.assertEqual(updated["body"], "edited body")
        self.assertEqual(updated["title"], "Minutes")

    def test_listing_can_leave_the_bodies_out(self):
        db.save_report(self.meeting_id, "minutes", "Minutes", "a long body", "m")
        light = db.list_reports(self.meeting_id, with_body=False)
        self.assertNotIn("body", light[0])

    def test_newest_first(self):
        db.save_report(self.meeting_id, "minutes", "one", "a", "m")
        second = db.save_report(self.meeting_id, "summary", "two", "b", "m")
        self.assertEqual(db.list_reports(self.meeting_id)[0]["id"], second["id"])

    def test_delete(self):
        row = db.save_report(self.meeting_id, "minutes", "Minutes", "body", "m")
        self.assertTrue(db.delete_report(row["id"]))
        self.assertIsNone(db.get_report(row["id"]))


class TestChatStorage(unittest.TestCase):
    def setUp(self):
        self.meeting_id = make_meeting()["id"]

    def test_a_turn_round_trips(self):
        db.add_chat_turn(self.meeting_id, "q", "a [#1]", [1], 0.0003)
        turn = db.list_chat(self.meeting_id)[0]
        self.assertEqual(turn["cited"], [1])
        self.assertAlmostEqual(turn["cost_usd"], 0.0003)

    def test_clearing_only_touches_one_meeting(self):
        other = make_meeting(title="Another")["id"]
        db.add_chat_turn(self.meeting_id, "q", "a", [], 0)
        db.add_chat_turn(other, "q", "a", [], 0)
        self.assertEqual(db.clear_chat(self.meeting_id), 1)
        self.assertEqual(len(db.list_chat(other)), 1)


# -------------------------------------------------------------------- jobs


class TestJobs(unittest.TestCase):
    def test_a_job_runs_and_carries_its_result_back(self):
        job = jobs.start(1, "test", "Testing", lambda job: {"answer": 42})
        self.assertTrue(_wait(lambda: job.status != "running"))
        self.assertEqual(job.status, "done")
        self.assertEqual(job.result, {"answer": 42})

    def test_a_failure_is_reported_rather_than_swallowed(self):
        def broken(job):
            raise LLMError("model said no")

        job = jobs.start(2, "test", "Testing", broken)
        self.assertTrue(_wait(lambda: job.status != "running"))
        self.assertEqual(job.status, "error")
        self.assertIn("model said no", job.error)

    def test_progress_is_visible_while_it_runs(self):
        release = threading.Event()

        def slow(job):
            job.progress("reading", 3, 7)
            release.wait(5)
            return "ok"

        job = jobs.start(3, "test", "Testing", slow)
        self.assertTrue(_wait(lambda: job.done == 3))
        self.assertEqual(job.as_dict()["total"], 7)
        release.set()
        self.assertTrue(_wait(lambda: job.status == "done"))

    def test_two_jobs_for_one_meeting_are_refused(self):
        """Both would read the transcript, and the user would pay twice."""
        release = threading.Event()
        first = jobs.start(4, "test", "Testing", lambda job: release.wait(5))
        with self.assertRaises(RuntimeError):
            jobs.start(4, "test", "Testing", lambda job: None)
        release.set()
        self.assertTrue(_wait(lambda: first.status == "done"))

    def test_a_different_meeting_is_not_blocked(self):
        release = threading.Event()
        first = jobs.start(5, "test", "Testing", lambda job: release.wait(5))
        second = jobs.start(6, "test", "Testing", lambda job: "fine")
        self.assertTrue(_wait(lambda: second.status == "done"))
        release.set()
        self.assertTrue(_wait(lambda: first.status == "done"))

    def test_a_finished_job_is_still_findable(self):
        job = jobs.start(7, "test", "Testing", lambda job: "done")
        self.assertTrue(_wait(lambda: job.status == "done"))
        self.assertIsNotNone(jobs.get(job.id))

    def test_a_cancelled_job_frees_the_meeting_and_drops_its_late_result(self):
        release = threading.Event()
        first = jobs.start(8, "test", "Testing", lambda job: release.wait(5) or "late")
        cancelled = jobs.cancel(first.id)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertIsNone(jobs.running_for(8), "the meeting is free at once")
        second = jobs.start(8, "test", "Testing", lambda job: "fresh")
        self.assertTrue(_wait(lambda: second.status == "done"))
        release.set()
        time.sleep(0.05)
        self.assertEqual(first.status, "cancelled", "a cancelled job never turns into done")
        self.assertIsNone(first.result)

    def test_cancelling_an_unknown_or_finished_job(self):
        self.assertIsNone(jobs.cancel("nope"))
        job = jobs.start(9, "test", "Testing", lambda job: "ok")
        self.assertTrue(_wait(lambda: job.status == "done"))
        self.assertEqual(jobs.cancel(job.id).status, "done", "finished stays finished")

    def test_old_jobs_are_forgotten(self):
        ids = []
        for i in range(jobs.KEEP + 8):
            job = jobs.start(1000 + i, "test", "Testing", lambda job: i)
            self.assertTrue(_wait(lambda: job.status == "done"))
            ids.append(job.id)
        alive = [i for i in ids if jobs.get(i)]
        self.assertLessEqual(len(alive), jobs.KEEP + 1)


# -------------------------------------------------------------- the HTTP API


class TestReviewApi(unittest.TestCase):
    """The endpoints the review page actually calls, through Flask's test client."""

    @classmethod
    def setUpClass(cls):
        import app as app_module

        cls.app_module = app_module
        app_module.app.config["TESTING"] = True

    def setUp(self):
        self.meeting = make_meeting()
        self.meeting_id = self.meeting["id"]
        self.client = self.app_module.app.test_client()
        # The endpoints refuse to spend money without a key; these tests never
        # reach the network because the LLM entry points are patched.
        original = config.OPENROUTER_API_KEY
        self.addCleanup(setattr, config, "OPENROUTER_API_KEY", original)
        config.OPENROUTER_API_KEY = "test-key"

    def _json(self, response):
        return json.loads(response.data.decode())

    # ---------------------------------------------------------------- page

    def test_the_page_loads(self):
        response = self.client.get(f"/review/{self.meeting_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Ask the meeting", response.data.decode())

    def test_an_unknown_meeting_says_so_rather_than_crashing(self):
        response = self.client.get("/review/999999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("No meeting", response.data.decode())

    # ---------------------------------------------------------------- data

    def test_one_request_returns_everything_the_page_needs(self):
        db.add_action(self.meeting_id, "Alan", "book the room")
        db.save_report(self.meeting_id, "minutes", "Minutes", "body", "m")
        db.add_chat_turn(self.meeting_id, "q", "a", [1], 0.001)

        payload = self._json(self.client.get(f"/api/meetings/{self.meeting_id}/review"))
        self.assertEqual(payload["meeting"]["title"], "Q3 planning")
        self.assertEqual(len(payload["segments"]), len(LINES))
        self.assertEqual(payload["speaker_names"]["0"], "Alan")
        self.assertEqual(len(payload["actions"]), 1)
        self.assertEqual(len(payload["reports"]), 1)
        self.assertEqual(len(payload["chat"]), 1)
        self.assertFalse(payload["running"])

    def test_it_says_whether_the_meeting_has_been_read_and_what_that_costs(self):
        payload = self._json(self.client.get(f"/api/meetings/{self.meeting_id}/review"))
        self.assertFalse(payload["digest"]["built"])
        self.assertGreaterEqual(payload["digest"]["estimated_sections"], 1)

        digest_module.ensure(self.meeting, client=FakeLLM(json_reply=DIGEST_REPLY))
        payload = self._json(self.client.get(f"/api/meetings/{self.meeting_id}/review"))
        self.assertTrue(payload["digest"]["built"])
        self.assertFalse(payload["digest"]["stale"])

    def test_a_missing_meeting_is_a_404_not_a_500(self):
        self.assertEqual(self.client.get("/api/meetings/999999/review").status_code, 404)

    # ------------------------------------------------------------ speakers

    def test_renaming_one_line(self):
        response = self.client.post(
            f"/api/meetings/{self.meeting_id}/segments/2/speaker",
            json={"name": "Carmen"},
        )
        self.assertEqual(response.status_code, 200)
        segments = db.get_meeting(self.meeting_id)["segments"]
        self.assertEqual(segments[2]["speaker_name"], "Carmen")

    def test_renaming_a_voice_everywhere(self):
        response = self.client.post(
            f"/api/meetings/{self.meeting_id}/speakers",
            json={"speaker": 2, "name": "Carmen"},
        )
        self.assertEqual(self._json(response)["speaker_names"]["2"], "Carmen")
        self.assertEqual(db.get_meeting(self.meeting_id)["speaker_names"][2], "Carmen")

    def test_an_empty_name_clears_it(self):
        self.client.post(f"/api/meetings/{self.meeting_id}/speakers",
                         json={"speaker": 2, "name": "Carmen"})
        self.client.post(f"/api/meetings/{self.meeting_id}/speakers",
                         json={"speaker": 2, "name": ""})
        self.assertNotIn(2, db.get_meeting(self.meeting_id)["speaker_names"])

    def test_a_non_numeric_voice_is_rejected(self):
        response = self.client.post(
            f"/api/meetings/{self.meeting_id}/speakers", json={"speaker": "S2"}
        )
        self.assertEqual(response.status_code, 400)

    def test_a_running_meeting_is_edited_in_the_live_view_not_here(self):
        """Two writers on one meeting would leave the live session's names and
        the database disagreeing."""

        class LiveSession:
            meeting_id = self.meeting_id
            stopped = False

        original = self.app_module._session
        self.addCleanup(setattr, self.app_module, "_session", original)
        self.app_module._session = LiveSession()

        response = self.client.post(
            f"/api/meetings/{self.meeting_id}/segments/0/speaker", json={"name": "X"}
        )
        self.assertEqual(response.status_code, 409)
        payload = self._json(self.client.get(f"/api/meetings/{self.meeting_id}/review"))
        self.assertTrue(payload["running"])

    # -------------------------------------------------------- action items

    def test_add_edit_and_delete_over_http(self):
        added = self._json(self.client.post(
            f"/api/meetings/{self.meeting_id}/actions",
            json={"who": "Alan", "what": "book the room", "due": "Monday"},
        ))
        action_id = added["added"][0]["id"]

        patched = self._json(self.client.patch(
            f"/api/meetings/{self.meeting_id}/actions/{action_id}",
            json={"status": "done"},
        ))
        self.assertEqual(patched["action"]["status"], "done")

        self.assertEqual(
            self.client.delete(
                f"/api/meetings/{self.meeting_id}/actions/{action_id}"
            ).status_code,
            200,
        )
        self.assertEqual(db.list_actions(self.meeting_id), [])

    def test_accepting_a_batch_of_proposals(self):
        payload = self._json(self.client.post(
            f"/api/meetings/{self.meeting_id}/actions",
            json={"items": [
                {"who": "Alan", "what": "one", "source": "copilot"},
                {"who": "Bella", "what": "two", "source": "copilot"},
            ]},
        ))
        self.assertEqual(len(payload["added"]), 2)
        self.assertEqual(payload["actions"][0]["source"], "copilot")

    def test_an_item_with_no_task_is_refused(self):
        response = self.client.post(
            f"/api/meetings/{self.meeting_id}/actions", json={"who": "Alan", "what": "  "}
        )
        self.assertEqual(response.status_code, 400)

    def test_an_unknown_status_becomes_open_rather_than_stored_verbatim(self):
        action = db.add_action(self.meeting_id, "Alan", "book the room")
        patched = self._json(self.client.patch(
            f"/api/meetings/{self.meeting_id}/actions/{action['id']}",
            json={"status": "nonsense"},
        ))
        self.assertEqual(patched["action"]["status"], "open")

    def test_patching_a_missing_item_is_a_404(self):
        response = self.client.patch(
            f"/api/meetings/{self.meeting_id}/actions/999999", json={"who": "x"}
        )
        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------- reports

    def test_saving_editing_downloading_and_deleting_a_report(self):
        row = db.save_report(self.meeting_id, "minutes", "Minutes", "# original", "m")

        saved = self._json(self.client.put(
            f"/api/reports/{row['id']}", json={"body": "# edited by hand"}
        ))
        self.assertEqual(saved["report"]["body"], "# edited by hand")

        download = self.client.get(f"/api/reports/{row['id']}/download.md")
        self.assertEqual(download.status_code, 200)
        self.assertIn("edited by hand", download.data.decode())
        self.assertIn("-minutes.md", download.headers["Content-Disposition"])

        self.assertEqual(self.client.delete(f"/api/reports/{row['id']}").status_code, 200)
        self.assertIsNone(db.get_report(row["id"]))

    def test_a_put_with_no_body_is_refused(self):
        row = db.save_report(self.meeting_id, "minutes", "Minutes", "# original", "m")
        response = self.client.put(f"/api/reports/{row['id']}", json={"title": "new"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(db.get_report(row["id"])["body"], "# original")

    def test_an_unknown_report_kind_is_refused(self):
        response = self.client.post(
            f"/api/meetings/{self.meeting_id}/reports", json={"kind": "haiku"}
        )
        self.assertEqual(response.status_code, 400)

    def test_generating_a_report_runs_as_a_job(self):
        llm = FakeLLM(json_reply=DIGEST_REPLY, text_reply="# Minutes\n\nbody")
        self._patch_client(llm)

        started = self._json(self.client.post(
            f"/api/meetings/{self.meeting_id}/reports", json={"kind": "minutes"}
        ))
        job_id = started["job"]["id"]
        self.assertTrue(_wait(lambda: jobs.get(job_id).status != "running", timeout=10))

        job = self._json(self.client.get(f"/api/jobs/{job_id}"))["job"]
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["result"]["kind"], "minutes")
        self.assertIn("Minutes", job["result"]["body"])

    def test_a_second_job_for_the_same_meeting_is_refused(self):
        release = threading.Event()
        self.addCleanup(release.set)
        first = jobs.start(
            self.meeting_id, "test", "Testing", lambda job: release.wait(10)
        )
        response = self.client.post(
            f"/api/meetings/{self.meeting_id}/reports", json={"kind": "minutes"}
        )
        self.assertEqual(response.status_code, 409)
        release.set()
        self.assertTrue(_wait(lambda: first.status == "done"))

    def test_a_missing_key_stops_a_job_before_it_starts(self):
        config.OPENROUTER_API_KEY = ""
        response = self.client.post(
            f"/api/meetings/{self.meeting_id}/reports", json={"kind": "minutes"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("OPENROUTER_API_KEY", self._json(response)["error"])

    def test_building_the_digest_reports_what_it_found(self):
        self._patch_client(FakeLLM(json_reply=DIGEST_REPLY))
        started = self._json(
            self.client.post(f"/api/meetings/{self.meeting_id}/digest")
        )
        job_id = started["job"]["id"]
        self.assertTrue(_wait(lambda: jobs.get(job_id).status != "running", timeout=10))
        result = self._json(self.client.get(f"/api/jobs/{job_id}"))["job"]
        self.assertEqual(result["status"], "done", result)
        self.assertTrue(result["result"]["built"])
        self.assertGreaterEqual(result["result"]["counts"]["decisions"], 1)

    def test_a_running_job_can_be_cancelled_over_http(self):
        release = threading.Event()
        self.addCleanup(release.set)
        job = jobs.start(self.meeting_id, "test", "Testing", lambda j: release.wait(10))
        payload = self._json(self.client.post(f"/api/jobs/{job.id}/cancel"))
        self.assertEqual(payload["job"]["status"], "cancelled")
        # and a new report job can start straight away
        self._patch_client(FakeLLM(json_reply=DIGEST_REPLY, text_reply="# Minutes"))
        response = self.client.post(f"/api/meetings/{self.meeting_id}/reports", json={"kind": "minutes"})
        self.assertEqual(response.status_code, 202)

    def test_an_unknown_job_is_a_404(self):
        self.assertEqual(self.client.get("/api/jobs/nope").status_code, 404)

    # ----------------------------------------------------------------- ask

    def test_asking_a_question(self):
        self._patch_client(FakeLLM(text_reply="財務部最多加 50 萬 [#3]。"))
        payload = self._json(self.client.post(
            f"/api/meetings/{self.meeting_id}/ask", json={"question": "budget 加幾多？"}
        ))
        self.assertEqual(payload["cited"], [3])
        self.assertEqual(len(db.list_chat(self.meeting_id)), 1)

    def test_an_empty_question_is_refused(self):
        response = self.client.post(
            f"/api/meetings/{self.meeting_id}/ask", json={"question": "  "}
        )
        self.assertEqual(response.status_code, 400)

    def test_a_model_failure_becomes_a_readable_error(self):
        original = qa.OpenRouterClient
        qa.OpenRouterClient = lambda *a, **k: _Broken()
        self.addCleanup(setattr, qa, "OpenRouterClient", original)

        response = self.client.post(
            f"/api/meetings/{self.meeting_id}/ask", json={"question": "anything"}
        )
        self.assertEqual(response.status_code, 502)
        self.assertIn("upstream is down", self._json(response)["error"])

    def test_clearing_the_thread(self):
        db.add_chat_turn(self.meeting_id, "q", "a", [], 0)
        payload = self._json(
            self.client.delete(f"/api/meetings/{self.meeting_id}/chat")
        )
        self.assertEqual(payload["removed"], 1)
        self.assertEqual(db.list_chat(self.meeting_id), [])

    # -------------------------------------------------------------- helper

    def _patch_client(self, llm):
        """Point every review module at one fake LLM for the duration of a test."""
        for module in (digest_module, qa, reports):
            original = module.OpenRouterClient
            self.addCleanup(setattr, module, "OpenRouterClient", original)
            module.OpenRouterClient = lambda *a, **k: llm


class _Broken:
    usage = Usage()

    def chat(self, *_args, **_kwargs):
        raise LLMError("upstream is down")

    def chat_json(self, *_args, **_kwargs):
        raise LLMError("upstream is down")


def _wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


if __name__ == "__main__":
    unittest.main(verbosity=2)
