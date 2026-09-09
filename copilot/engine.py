"""The copilot orchestrator.

Decides when to think, and never lets thinking block transcription. Every LLM
call runs on a worker thread; at most one of each kind (think, notes, summary,
speakers) is ever in flight, so a slow model response throttles the copilot
instead of queueing a backlog that arrives all at once minutes later.

Triggers:
  think     -- on utterance end, rate limited by time AND by new speech; the
               limit shortens when someone in the room just asked a question
  answer    -- when the attendee's turn needs a fact looked up first
  notes     -- on a timer
  summary   -- when the unsummarised transcript grows past a threshold
  speakers  -- occasionally, while diarised voices are still unnamed

One think cycle produces both outputs: the private coaching in the Copilot panel
and the AI attendee's spoken turn. Deliberately one call, not two -- it halves
the cost and the two panels can never contradict each other.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import config

from . import prompts, search
from .llm import LLMError, OpenRouterClient
from .state import EMPTY_NOTES, MeetingState, Segment, looks_like_a_question

log = logging.getLogger(__name__)

VALID_KINDS = ("answer", "question", "clarification", "challenge", "info")


class CopilotEngine:
    def __init__(self, state: MeetingState, llm: OpenRouterClient, emit):
        self.state = state
        self.llm = llm
        self.emit = emit  # emit(event_name: str, payload: dict) -> None

        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="copilot")
        self._flags_lock = threading.Lock()
        self._busy: set[str] = set()
        self._closed = False

    # ------------------------------------------------------------------ hooks

    def on_utterance(self, seg: Segment) -> None:
        """Called for every finalised utterance."""
        if self._closed:
            return
        urgent = looks_like_a_question(seg.text)
        if self.state.should_think(urgent=urgent):
            self._submit("think", self._run_think)
        if self.state.should_take_notes():
            self._submit("notes", self._run_notes)
        if self.state.needs_summary():
            self._submit("summary", self._run_summary)
        if self.state.should_guess_speakers():
            self._submit("speakers", self._run_speaker_guess)

    def ask(self, question: str, web: bool = False) -> None:
        """A question typed by the user: always answered, never deduplicated.

        `web` is the user's choice, not a guess: most questions asked mid-meeting
        are about the meeting ("what did she just say", "summarise so far") and
        a web search for those is noise and cost.
        """
        if self._closed or not question.strip():
            return
        self._pool.submit(self._guarded, self._run_user_question, question.strip(), web)

    def run_notes_now(self) -> None:
        """Take a note-taking pass on the calling thread and wait for it.

        Used once, when the meeting ends, so the closing minutes make it into
        the saved notes before we tear the session down.
        """
        self._guarded(self._run_notes)

    def close(self, wait: bool = True) -> None:
        self._closed = True
        self._pool.shutdown(wait=wait, cancel_futures=not wait)

    # -------------------------------------------------------------- machinery

    def _submit(self, kind: str, fn, *args) -> None:
        with self._flags_lock:
            if kind in self._busy:
                return
            self._busy.add(kind)

        def release_and_run():
            try:
                self._guarded(fn, *args)
            finally:
                with self._flags_lock:
                    self._busy.discard(kind)

        try:
            self._pool.submit(release_and_run)
        except RuntimeError:  # pool already shut down
            with self._flags_lock:
                self._busy.discard(kind)

    def _guarded(self, fn, *args) -> None:
        """Run a job; report failures to the UI instead of dying silently."""
        try:
            fn(*args)
        except LLMError as exc:
            log.warning("copilot job %s failed: %s", fn.__name__, exc)
            self.emit("copilot_error", {"where": fn.__name__, "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - a bad job must not kill the meeting
            log.exception("copilot job %s crashed", fn.__name__)
            self.emit(
                "copilot_error",
                {"where": fn.__name__, "message": f"{exc.__class__.__name__}: {exc}"},
            )
        finally:
            self.emit("usage", self.llm.usage.snapshot())

    def _context(self) -> tuple[str, str, str]:
        state = self.state
        with state.lock:
            return state.brief.render(), state.speaker_roster(), state.rolling_summary

    # ------------------------------------------------------------------- jobs

    def _run_think(self) -> None:
        state = self.state
        with state.lock:
            upto = len(state.segments)
            state.last_think_at = time.time()
            advice_history = list(state.advice_history)
            attendee_history = list(state.attendee_history)
        brief_text, roster, summary = self._context()
        recent = state.recent_text()
        if not recent.strip():
            return

        data = self.llm.chat_json(
            prompts.think_messages(
                brief_text, roster, summary, recent, advice_history, attendee_history
            ),
            model=config.OPENROUTER_MODEL,
            temperature=0.4,
            max_tokens=900,
        )

        with state.lock:
            state.thought_upto = upto

        self._emit_advice(data)
        self._handle_attendee(data.get("attendee"))

    def _emit_advice(self, data: dict) -> None:
        key_point = _as_text(data.get("key_point"))
        watch_out = _as_text(data.get("watch_out"))
        questions = _as_list(data.get("suggested_questions"))[:3]

        self.state.remember_advice([t for t in (key_point, watch_out, *questions) if t])
        if key_point or watch_out or questions:
            self.emit(
                "advice",
                {
                    "at": time.time(),
                    "key_point": key_point,
                    "watch_out": watch_out,
                    "questions": questions,
                },
            )

    def _handle_attendee(self, attendee) -> None:
        if not config.ATTENDEE_ENABLED or not isinstance(attendee, dict):
            return
        if not attendee.get("should_speak"):
            return

        say = _as_text(attendee.get("say"))
        if not say or self.state.already_said(say):
            return

        kind = _as_text(attendee.get("kind")).lower()
        if kind not in VALID_KINDS:
            kind = "info"
        urgency = "high" if _as_text(attendee.get("urgency")).lower() == "high" else "normal"
        why = _as_text(attendee.get("why"))

        # A turn that rests on a fact we should check goes through search first,
        # so the attendee cites rather than asserts.
        if attendee.get("needs_web") and search.available():
            query = _as_text(attendee.get("search_query")) or say
            if not self.state.already_answered(query):
                self.state.remember_answered(query)
                self._pool.submit(
                    self._guarded, self._run_answer, say, True, query, "attendee",
                    {"kind": kind, "urgency": urgency, "why": why},
                )
                return

        self._emit_attendee_turn(say, kind, urgency, why, [], searched=False)

    def _emit_attendee_turn(
        self, say: str, kind: str, urgency: str, why: str, sources: list[dict], searched: bool
    ) -> None:
        self.state.remember_attendee_turn(say)
        self.emit(
            "attendee",
            {
                "at": time.time(),
                "say": say,
                "kind": kind,
                "urgency": urgency,
                "why": why,
                "searched": searched,
                "web_enabled": search.available(),
                "sources": [{"title": s["title"], "url": s["url"]} for s in sources],
            },
        )

    def _run_user_question(self, question: str, web: bool) -> None:
        """Answer the user's own question, privately.

        The whole meeting so far is in reach: the rolling summary and live notes
        for shape, the recent transcript verbatim, and the passages anywhere in
        the meeting that match the question (the same retrieval the review
        workspace uses), so "what did Carmen say about the budget an hour ago"
        works as well as "what did I just miss". Answers cite line numbers the
        page can jump to.
        """
        from review import retrieval  # local import: review is the heavier package

        sources: list[dict] = []
        if web and search.available():
            sources = search.search(question)

        state = self.state
        brief_text, roster, summary = self._context()
        with state.lock:
            notes = dict(state.notes)
            history = list(state.qa_history)
            names = dict(state.speaker_names)
        recent = state.recent_text()
        passages = retrieval.find(state.segments_as_dicts(), names, question)
        # Lines already shown verbatim need not be repeated as passages.
        covered = retrieval.indices_in(passages)

        answer = self.llm.chat(
            prompts.live_ask_messages(
                question=question,
                brief_text=brief_text,
                roster=roster,
                rolling_summary=summary,
                notes=notes,
                recent_transcript=recent,
                passages="\n\n".join(p.text for p in passages),
                sources=sources,
                history=history,
            ),
            model=config.OPENROUTER_MODEL,
            temperature=0.2,
            max_tokens=700,
        ).strip()
        if not answer:
            return

        with state.lock:
            valid = covered | {s.index for s in state.segments}
        cited = retrieval.cited_indices(answer, valid=valid)
        state.remember_qa(question, answer)
        self.emit(
            "answer",
            {
                "at": time.time(),
                "question": question,
                "answer": answer,
                "cited": cited,
                "from_user": True,
                "searched": bool(sources),
                "web_requested": web,
                "web_enabled": search.available(),
                "sources": [{"title": s["title"], "url": s["url"]} for s in sources],
            },
        )

    def _run_answer(
        self, question: str, needs_web: bool, query: str, target: str, turn: dict | None
    ) -> None:
        """Answer a question the room asked, as the attendee, looking a fact up
        first when the think cycle said one was needed."""
        sources: list[dict] = []
        if needs_web and search.available():
            sources = search.search(query)

        brief_text, roster, summary = self._context()
        answer = self.llm.chat(
            prompts.answer_messages(
                question, brief_text, roster, summary, sources,
                as_attendee=(target == "attendee"),
            ),
            model=config.OPENROUTER_MODEL,
            temperature=0.2,
            max_tokens=450,
        ).strip()
        if not answer:
            return

        turn = turn or {}
        self._emit_attendee_turn(
            answer,
            turn.get("kind", "answer"),
            turn.get("urgency", "normal"),
            turn.get("why", ""),
            sources,
            searched=bool(sources),
        )

    def _run_notes(self) -> None:
        state = self.state
        with state.lock:
            upto = len(state.segments)
            since = state.noted_upto
            state.last_notes_at = time.time()
            current = dict(state.notes)
        if upto <= since:
            return
        new_text = state.text_from(since)
        if not new_text.strip():
            return
        brief_text, roster, summary = self._context()

        data = self.llm.chat_json(
            prompts.notes_messages(brief_text, roster, summary, current, new_text),
            model=config.OPENROUTER_NOTES_MODEL,
            temperature=0.2,
            max_tokens=1200,
        )
        notes = _normalise_notes(data)

        with state.lock:
            state.notes = notes
            state.noted_upto = upto
        self.emit("notes", {"at": time.time(), "notes": notes})

    def _run_summary(self) -> None:
        state = self.state
        with state.lock:
            # Keep the recent window verbatim; fold everything before it.
            cut = len(state.segments)
            kept = 0
            for seg in reversed(state.segments[state.summarised_upto :]):
                if kept >= config.RECENT_WINDOW_CHARS:
                    break
                kept += len(seg.text)
                cut -= 1
            if cut <= state.summarised_upto:
                return
            to_fold = state.render(state.segments[state.summarised_upto : cut])
            existing = state.rolling_summary
            brief_text = state.brief.render()

        if not to_fold.strip():
            return

        merged = self.llm.chat(
            prompts.summary_messages(brief_text, existing, to_fold),
            model=config.OPENROUTER_NOTES_MODEL,
            temperature=0.2,
            max_tokens=600,
        ).strip()
        if not merged:
            return

        with state.lock:
            state.rolling_summary = merged
            state.summarised_upto = cut
        self.emit("summary", {"summary": merged, "upto": cut})

    def _run_speaker_guess(self) -> None:
        """Propose which diarised voice is which person. Never auto-applied:
        a wrong name the user trusts is worse than an unnamed voice."""
        state = self.state
        with state.lock:
            state.last_speaker_guess_at = time.time()
            attendees = [a.label() for a in state.brief.attendees]
            known = dict(state.speaker_names)
        unnamed = state.unnamed_speakers()
        if not attendees or not unnamed:
            return

        transcript = state.recent_text(max_chars=6000)
        if not transcript.strip():
            return

        data = self.llm.chat_json(
            prompts.speaker_guess_messages(
                attendees, [f"S{s + 1}" for s in unnamed], transcript
            ),
            model=config.OPENROUTER_MODEL,
            temperature=0.1,
            max_tokens=500,
        )

        valid_names = {a.name for a in state.brief.attendees}
        taken = set(known.values())
        proposals = []
        for row in _as_dicts(data.get("mapping"))[:8]:
            speaker = _speaker_index(row.get("speaker"))
            name = _as_text(row.get("name"))
            if speaker is None or speaker not in unnamed:
                continue
            if name not in valid_names or name in taken:
                continue  # only real attendees, and never the same person twice
            taken.add(name)
            proposals.append(
                {
                    "speaker": speaker,
                    "label": f"S{speaker + 1}",
                    "name": name,
                    "confidence": (
                        "high" if _as_text(row.get("confidence")).lower() == "high" else "low"
                    ),
                    "evidence": _as_text(row.get("evidence"))[:300],
                }
            )

        if not proposals:
            return
        suggestion = {"at": time.time(), "proposals": proposals}
        with state.lock:
            state.speaker_suggestion = suggestion
        self.emit("speaker_suggestion", suggestion)


# ------------------------------------------------------------------ coercion


def _as_text(value) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        return "" if stripped.lower() in ("null", "none", "n/a", "") else stripped
    return str(value).strip()


def _as_list(value) -> list[str]:
    if isinstance(value, str):
        return [_as_text(value)] if _as_text(value) else []
    if not isinstance(value, list):
        return []
    return [_as_text(v) for v in value if _as_text(v)]


def _as_dicts(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, dict)]


def _speaker_index(value) -> int | None:
    """Accept "S2", "s2", 2 or 1 and return the zero-based diarisation index."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value - 1 if value > 0 else None
    text = _as_text(value).upper().lstrip("S")
    if not text.isdigit():
        return None
    number = int(text)
    return number - 1 if number > 0 else None


def _normalise_notes(data: dict) -> dict:
    """Models drift on shape; the UI should never have to defend itself."""
    notes = dict(EMPTY_NOTES)
    notes["summary"] = _as_text(data.get("summary"))
    notes["decisions"] = _as_list(data.get("decisions"))
    notes["open_questions"] = _as_list(data.get("open_questions"))
    notes["topics"] = _as_list(data.get("topics"))

    items = []
    raw_items = data.get("action_items")
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                what = _as_text(item.get("what") or item.get("task") or item.get("action"))
                if not what:
                    continue
                items.append(
                    {
                        "who": _as_text(item.get("who") or item.get("owner")) or "unassigned",
                        "what": what,
                        "due": _as_text(item.get("due") or item.get("deadline")),
                    }
                )
            elif _as_text(item):
                items.append({"who": "unassigned", "what": _as_text(item), "due": ""})
    notes["action_items"] = items
    return notes
