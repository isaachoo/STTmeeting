"""The copilot orchestrator.

Decides when to think, and never lets thinking block transcription. Every LLM
call runs on a worker thread; at most one of each kind (advice, notes, summary)
is ever in flight, so a slow model response throttles the copilot instead of
queueing up a backlog that arrives all at once minutes later.

Triggers:
  advice   -- on utterance end, rate limited by time AND by new speech
  answer   -- when the advisor spots an unanswered factual question
  notes    -- on a timer
  summary  -- when the unsummarised transcript grows past a threshold
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import config

from . import prompts, search
from .llm import LLMError, OpenRouterClient
from .state import EMPTY_NOTES, MeetingState, Segment

log = logging.getLogger(__name__)


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
        if self.state.should_advise():
            self._submit("advice", self._run_advice)
        if self.state.should_take_notes():
            self._submit("notes", self._run_notes)
        if self.state.needs_summary():
            self._submit("summary", self._run_summary)

    def ask(self, question: str) -> None:
        """A question typed by the user; always answered, never deduplicated."""
        if self._closed or not question.strip():
            return
        self._pool.submit(self._guarded, self._run_answer, question.strip(), True, question.strip(), True)

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

    def _submit(self, kind: str, fn, *args, force: bool = False) -> None:
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

    # ------------------------------------------------------------------- jobs

    def _run_advice(self) -> None:
        state = self.state
        with state.lock:
            upto = len(state.segments)
            state.last_advice_at = time.time()
            brief, summary = state.brief, state.rolling_summary
            history = list(state.advice_history)
        recent = state.recent_text()
        if not recent.strip():
            return

        data = self.llm.chat_json(
            prompts.advisor_messages(brief, summary, recent, history),
            model=config.OPENROUTER_MODEL,
            temperature=0.4,
            max_tokens=600,
        )

        key_point = _as_text(data.get("key_point"))
        watch_out = _as_text(data.get("watch_out"))
        questions = [q for q in (_as_list(data.get("suggested_questions"))) if q][:3]

        with state.lock:
            state.advised_upto = upto
        state.remember_advice([t for t in (key_point, watch_out, *questions) if t])

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

        pending = data.get("question_to_answer")
        if isinstance(pending, dict):
            question = _as_text(pending.get("question"))
            if question and not state.already_answered(question):
                state.remember_answered(question)
                query = _as_text(pending.get("search_query")) or question
                needs_web = bool(pending.get("needs_web"))
                self._pool.submit(
                    self._guarded, self._run_answer, question, needs_web, query, False
                )

    def _run_answer(
        self, question: str, needs_web: bool, query: str, from_user: bool
    ) -> None:
        sources: list[dict] = []
        if needs_web and search.available():
            sources = search.search(query)

        with self.state.lock:
            brief, summary = self.state.brief, self.state.rolling_summary

        answer = self.llm.chat(
            prompts.answer_messages(question, brief, summary, sources),
            model=config.OPENROUTER_MODEL,
            temperature=0.2,
            max_tokens=450,
        ).strip()

        if not answer:
            return
        self.emit(
            "answer",
            {
                "at": time.time(),
                "question": question,
                "answer": answer,
                "from_user": from_user,
                "searched": bool(sources),
                "web_enabled": search.available(),
                "sources": [
                    {"title": s["title"], "url": s["url"]} for s in sources
                ],
            },
        )

    def _run_notes(self) -> None:
        state = self.state
        with state.lock:
            upto = len(state.segments)
            since = state.noted_upto
            state.last_notes_at = time.time()
            brief, summary = state.brief, state.rolling_summary
            current = dict(state.notes)
        if upto <= since:
            return
        new_text = state.text_from(since)
        if not new_text.strip():
            return

        data = self.llm.chat_json(
            prompts.notes_messages(brief, summary, current, new_text),
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
            brief = state.brief

        if not to_fold.strip():
            return

        merged = self.llm.chat(
            prompts.summary_messages(brief, existing, to_fold),
            model=config.OPENROUTER_NOTES_MODEL,
            temperature=0.2,
            max_tokens=600,
        ).strip()
        if not merged:
            return

        with state.lock:
            state.rolling_summary = merged
            state.summarised_upto = cut
        self.emit("summary", {"summary": merged})


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
