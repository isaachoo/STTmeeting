"""In-memory state for one meeting.

The transcript grows without bound but what the LLM sees does not: recent
material goes in verbatim, and everything older is folded into a rolling
summary. That keeps each prompt a predictable size, so cost per think cycle
stays flat whether you are ten minutes or four hours into a meeting.
"""

import re
import threading
import time
from dataclasses import dataclass, field

import config

from .brief import Brief

EMPTY_NOTES: dict = {
    "summary": "",
    "decisions": [],
    "action_items": [],
    "open_questions": [],
    "topics": [],
}

# Cantonese and English question markers. A crude test on purpose: it only
# decides whether the copilot thinks *now* instead of at the next interval, so a
# false positive costs one cheap call and a false negative costs a few seconds.
_QUESTION_PATTERNS = re.compile(
    r"[?？]|嗎|呢|咩|乜|幾時|幾多|點樣|點解|係唔係|有無|有冇|可唔可以|得唔得|邊個|邊位|邊度"
    r"|\bwhat\b|\bwhen\b|\bwhy\b|\bhow\b|\bwho\b|\bwhere\b|\bwhich\b|\bcan we\b|\bshould we\b",
    re.IGNORECASE,
)


def looks_like_a_question(text: str) -> bool:
    return bool(_QUESTION_PATTERNS.search(text or ""))


@dataclass
class Segment:
    index: int
    text: str
    speaker: int | None
    at: float
    # A name set on this line specifically, which beats the voice-level name.
    # Diarisation splits and merges voices, so some lines need fixing one by one.
    speaker_name: str = ""

    def label(self, speaker_names: dict[int, str] | None = None) -> str:
        return self.speaker_name or label_for(self.speaker, speaker_names)

    def as_dict(self, speaker_names: dict[int, str] | None = None) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "speaker": self.speaker,
            "speaker_label": self.label(speaker_names),
            "speaker_name": self.speaker_name,
            "at": self.at,
        }


def label_for(speaker: int | None, speaker_names: dict[int, str] | None = None) -> str:
    """A name once we know one, otherwise the diarisation tag."""
    if speaker is None:
        return "?"
    named = (speaker_names or {}).get(speaker)
    return named or f"S{speaker + 1}"


@dataclass
class MeetingState:
    meeting_id: int
    brief: Brief = field(default_factory=Brief)
    started_at: float = field(default_factory=time.time)

    segments: list[Segment] = field(default_factory=list)
    rolling_summary: str = ""
    summarised_upto: int = 0  # segments[:summarised_upto] are in the summary

    # Diarisation gives us anonymous voice groups; these are the names the user
    # (or an accepted copilot suggestion) has attached to them.
    speaker_names: dict[int, str] = field(default_factory=dict)
    speaker_suggestion: dict | None = None

    notes: dict = field(default_factory=lambda: dict(EMPTY_NOTES))
    user_notes: str = ""

    advice_history: list[str] = field(default_factory=list)
    attendee_history: list[str] = field(default_factory=list)
    answered_questions: list[str] = field(default_factory=list)

    last_think_at: float = 0.0
    thought_upto: int = 0
    last_notes_at: float = 0.0
    noted_upto: int = 0
    last_speaker_guess_at: float = 0.0

    lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self):
        # Anchor the note-taking clock to the start of the meeting, otherwise a
        # zero timestamp reads as "last noted in 1970" and the first utterance
        # triggers a pointless pass over a single sentence.
        self.last_notes_at = self.started_at

    @property
    def title(self) -> str:
        return self.brief.title

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
        with self.lock:
            names = dict(self.speaker_names)
        return "\n".join(f"{s.label(names)}: {s.text}" for s in segments)

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

    # ---------------------------------------------------------------- speakers

    def observed_speakers(self) -> list[int]:
        with self.lock:
            return sorted({s.speaker for s in self.segments if s.speaker is not None})

    def set_segment_speaker(self, index: int, name: str) -> Segment | None:
        """Name one line. Returns the segment, or None if the index is unknown."""
        cleaned = (name or "").strip()[:80]
        with self.lock:
            if not 0 <= index < len(self.segments):
                return None
            self.segments[index].speaker_name = cleaned
            return self.segments[index]

    def set_speaker_name(self, speaker: int, name: str) -> str:
        """Name a voice, or clear the name with an empty string."""
        cleaned = (name or "").strip()[:80]
        with self.lock:
            if cleaned:
                self.speaker_names[speaker] = cleaned
            else:
                self.speaker_names.pop(speaker, None)
            return cleaned

    def speaker_roster(self) -> str:
        """Who is who, for the prompts."""
        with self.lock:
            names = dict(self.speaker_names)
            speakers = sorted({s.speaker for s in self.segments if s.speaker is not None})
        if not speakers:
            return ""
        lines = []
        for speaker in speakers:
            tag = f"S{speaker + 1}"
            named = names.get(speaker)
            lines.append(f"- {tag} is {named}" if named else f"- {tag} is not yet identified")
        return "Voices in the transcript:\n" + "\n".join(lines)

    def unnamed_speakers(self) -> list[int]:
        with self.lock:
            names = set(self.speaker_names)
        return [s for s in self.observed_speakers() if s not in names]

    def should_guess_speakers(self, now: float | None = None) -> bool:
        """Worth one cheap call when voices are still anonymous."""
        now = now or time.time()
        with self.lock:
            if now - self.last_speaker_guess_at < config.SPEAKER_GUESS_INTERVAL:
                return False
            if not self.brief.attendees:
                return False  # nothing to map names to
            if len(self.segments) < config.SPEAKER_GUESS_MIN_SEGMENTS:
                return False
        return bool(self.unnamed_speakers())

    # ---------------------------------------------------------------- thinking

    def new_chars_since_think(self) -> int:
        with self.lock:
            return sum(len(s.text) for s in self.segments[self.thought_upto :])

    def should_think(self, urgent: bool = False, now: float | None = None) -> bool:
        """Rate limited by the clock AND by how much new speech there is.

        `urgent` (someone just asked a question) shortens the interval but does
        not remove it, so a rapid back-and-forth cannot spam the model.
        """
        now = now or time.time()
        floor = config.THINK_URGENT_INTERVAL if urgent else config.THINK_MIN_INTERVAL
        with self.lock:
            if now - self.last_think_at < floor:
                return False
            needed = 1 if urgent else config.THINK_MIN_NEW_CHARS
            return self.new_chars_since_think() >= needed

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

    def remember_attendee_turn(self, line: str) -> None:
        with self.lock:
            if line:
                self.attendee_history.append(line)
            del self.attendee_history[:-15]

    def already_said(self, line: str) -> bool:
        """Has the attendee already said this?

        The prompt tells the model not to repeat itself and mostly it obeys, but
        a participant that says the same sentence three times is the worst
        failure this panel has, so it is worth a second line of defence.
        """
        needle = _normalise(line)
        if not needle:
            return True
        with self.lock:
            history = list(self.attendee_history)
        for previous in history:
            other = _normalise(previous)
            if not other:
                continue
            if needle == other:
                return True
            # One containing the other only counts as a repeat when they are
            # close in length; a genuinely longer new point should get through.
            shorter, longer = sorted((needle, other), key=len)
            if shorter in longer and len(shorter) >= 0.8 * len(longer):
                return True
        return False

    def already_answered(self, question: str) -> bool:
        """Crude duplicate guard so one unanswered question is not looked up
        again every think cycle while it stays on the table."""
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
