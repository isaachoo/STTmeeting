"""In-memory state for one meeting.

The transcript grows without bound but what the LLM sees does not: recent
material goes in verbatim, and everything older is folded into a rolling
summary. That keeps each prompt a predictable size, so cost per advisor call
stays flat whether you are ten minutes or four hours into a meeting.
"""

import threading
import time
from dataclasses import dataclass, field

import config

EMPTY_NOTES: dict = {
    "summary": "",
    "decisions": [],
    "action_items": [],
    "open_questions": [],
    "topics": [],
}


@dataclass
class Segment:
    index: int
    text: str
    speaker: int | None
    at: float

    @property
    def speaker_label(self) -> str:
        return f"S{self.speaker + 1}" if self.speaker is not None else "?"

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "speaker": self.speaker,
            "speaker_label": self.speaker_label,
            "at": self.at,
        }


@dataclass
class MeetingState:
    meeting_id: int
    title: str = ""
    brief: str = ""
    started_at: float = field(default_factory=time.time)

    segments: list[Segment] = field(default_factory=list)
    rolling_summary: str = ""
    summarised_upto: int = 0  # segments[:summarised_upto] are in the summary

    notes: dict = field(default_factory=lambda: dict(EMPTY_NOTES))
    user_notes: str = ""

    advice_history: list[str] = field(default_factory=list)
    answered_questions: list[str] = field(default_factory=list)

    last_advice_at: float = 0.0
    advised_upto: int = 0
    last_notes_at: float = 0.0
    noted_upto: int = 0

    lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self):
        # Anchor the note-taking clock to the start of the meeting, otherwise a
        # zero timestamp reads as "last noted in 1970" and the first utterance
        # triggers a pointless pass over a single sentence.
        self.last_notes_at = self.started_at

    # ------------------------------------------------------------- transcript

    def add_utterance(self, text: str, speaker: int | None) -> Segment:
        with self.lock:
            seg = Segment(
                index=len(self.segments),
                text=text,
                speaker=speaker,
                at=time.time() - self.started_at,
            )
            self.segments.append(seg)
            return seg

    def render(self, segments: list[Segment]) -> str:
        return "\n".join(f"{s.speaker_label}: {s.text}" for s in segments)

    def text_from(self, index: int) -> str:
        with self.lock:
            return self.render(self.segments[index:])

    def recent_text(self, max_chars: int | None = None) -> str:
        """Verbatim tail of the transcript, never older than the summary."""
        limit = max_chars or config.RECENT_WINDOW_CHARS
        with self.lock:
            picked: list[Segment] = []
            total = 0
            for seg in reversed(self.segments[self.summarised_upto :]):
                line_len = len(seg.text) + 6
                if picked and total + line_len > limit:
                    break
                picked.append(seg)
                total += line_len
            picked.reverse()
            return self.render(picked)

    def unsummarised_chars(self) -> int:
        with self.lock:
            return sum(len(s.text) for s in self.segments[self.summarised_upto :])

    def needs_summary(self) -> bool:
        return self.unsummarised_chars() > config.SUMMARY_TRIGGER_CHARS

    # ---------------------------------------------------------------- advice

    def new_chars_since_advice(self) -> int:
        with self.lock:
            return sum(len(s.text) for s in self.segments[self.advised_upto :])

    def should_advise(self, now: float | None = None) -> bool:
        now = now or time.time()
        with self.lock:
            if now - self.last_advice_at < config.ADVICE_MIN_INTERVAL:
                return False
            return self.new_chars_since_advice() >= config.ADVICE_MIN_NEW_CHARS

    def should_take_notes(self, now: float | None = None) -> bool:
        now = now or time.time()
        with self.lock:
            if len(self.segments) <= self.noted_upto:
                return False
            return now - self.last_notes_at >= config.NOTES_INTERVAL

    def remember_advice(self, lines: list[str]) -> None:
        with self.lock:
            for line in lines:
                if line and line not in self.advice_history:
                    self.advice_history.append(line)
            del self.advice_history[:-30]

    def already_answered(self, question: str) -> bool:
        """Crude duplicate guard so one unanswered question is not looked up
        again every advisor cycle while it stays on the table."""
        needle = _normalise(question)
        if not needle:
            return True
        with self.lock:
            for seen in self.answered_questions:
                if needle == seen or needle in seen or seen in needle:
                    return True
            return False

    def remember_answered(self, question: str) -> None:
        with self.lock:
            self.answered_questions.append(_normalise(question))
            del self.answered_questions[:-40]

    # ------------------------------------------------------------------ misc

    def elapsed(self) -> float:
        return time.time() - self.started_at


def _normalise(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())
