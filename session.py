"""One live meeting: audio in, transcript and copilot output out.

Single-user local app, so there is exactly one active session at a time. The
browser is a view onto it -- refreshing the page replays the session from
`snapshot()` rather than starting anything new.
"""

import logging
import threading
import time
import wave

import config
from copilot.engine import CopilotEngine
from copilot.llm import OpenRouterClient
from copilot.state import MeetingState
from stt.deepgram_live import DeepgramLiveSTT
from storage import db

log = logging.getLogger(__name__)

MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000


class MeetingSession:
    def __init__(self, title: str, brief: str, sample_rate: int, emit):
        self.sample_rate = _validate_sample_rate(sample_rate)
        self._emit = emit
        self._lock = threading.Lock()

        self.stt_status = {"state": "starting", "detail": ""}
        self.cards: list[dict] = []  # advice + answer cards, newest last
        self.interim = ""
        self.error: str | None = None
        # An event rather than a flag, so the cost ticker can wait on it and exit
        # the moment the meeting ends instead of sleeping through its interval.
        self._stopped = threading.Event()

        self.meeting_id = db.create_meeting(
            title=title,
            brief=brief,
            language=config.DEEPGRAM_LANGUAGE,
            stt_model=config.DEEPGRAM_MODELS[0] if config.DEEPGRAM_MODELS else "",
        )
        self.state = MeetingState(meeting_id=self.meeting_id, title=title, brief=brief)

        # The client owns the usage counter; cost() reads it back rather than
        # keeping a second copy that could drift out of step.
        self.llm = OpenRouterClient()
        self.engine = CopilotEngine(self.state, self.llm, self._engine_emit)

        self.stt = DeepgramLiveSTT(
            api_key=config.DEEPGRAM_API_KEY,
            sample_rate=self.sample_rate,
            language=config.DEEPGRAM_LANGUAGE,
            models=config.DEEPGRAM_MODELS,
            on_interim=self._on_interim,
            on_utterance=self._on_utterance,
            on_status=self._on_stt_status,
            on_error=self._on_stt_error,
        )

        self._wav: wave.Wave_write | None = None
        if config.SAVE_AUDIO:
            self._open_wav()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    @stopped.setter
    def stopped(self, value: bool) -> None:
        if value:
            self._stopped.set()
        else:
            self._stopped.clear()

    @property
    def stopped_event(self) -> threading.Event:
        return self._stopped

    # ----------------------------------------------------------------- control

    def start(self) -> None:
        self.stt.start()

    def feed_audio(self, chunk: bytes) -> None:
        if self.stopped:
            return
        self.stt.send_audio(chunk)
        if self._wav is not None:
            try:
                self._wav.writeframes(chunk)
            except Exception:
                log.exception("failed to write audio to disk; disabling")
                self._close_wav()

    def stop(self) -> None:
        """Stop capture, take one last pass over the notes, then persist."""
        with self._lock:
            if self._stopped.is_set():
                return
            self._stopped.set()

        self.stt.stop()
        self._close_wav()
        self._emit("status", {"state": "wrapping_up", "detail": "writing final notes"})

        # Give any utterance still in flight a moment to land before the last
        # note-taking pass, so the closing remarks make it into the notes.
        time.sleep(1.5)
        self.engine.run_notes_now()
        self.engine.close(wait=False)

        with self.state.lock:
            notes = dict(self.state.notes)
            summary = self.state.rolling_summary
        db.save_notes(self.meeting_id, notes)
        db.save_summary(self.meeting_id, summary)
        db.finish_meeting(
            self.meeting_id, self.stt.audio_seconds, self.llm.usage.snapshot(), self.stt.model
        )

        self._emit("cost", self.cost())
        self._emit("meeting_stopped", {"meeting_id": self.meeting_id, "notes": notes})

    def ask(self, question: str) -> None:
        self.engine.ask(question)

    def set_user_notes(self, text: str) -> None:
        with self.state.lock:
            self.state.user_notes = text
        db.save_user_notes(self.meeting_id, text)

    # ------------------------------------------------------------- STT handlers

    def _on_interim(self, text: str) -> None:
        self.interim = text
        self._emit("interim", {"text": text})

    def _on_utterance(self, utt) -> None:
        seg = self.state.add_utterance(utt.text, utt.speaker)
        db.add_segment(self.meeting_id, seg.index, seg.at, seg.speaker, seg.text)
        self.interim = ""
        self._emit("segment", seg.as_dict())
        self._emit("cost", self.cost())
        self.engine.on_utterance(seg)

    def _on_stt_status(self, **kw) -> None:
        state = kw.pop("state", "")
        detail = kw.pop("detail", "")
        if kw:
            extra = ", ".join(f"{k}={v}" for k, v in kw.items())
            detail = f"{detail} ({extra})".strip() if detail else extra
        self.stt_status = {"state": state, "detail": detail}
        log.info("stt status: %s %s", state, detail)
        self._emit("status", self.stt_status)

    def _on_stt_error(self, message: str) -> None:
        self.error = message
        log.error("stt error: %s", message)
        self._emit("error", {"message": message})

    # ---------------------------------------------------------- copilot output

    def _engine_emit(self, event: str, payload: dict) -> None:
        if event in ("advice", "answer"):
            card = {"kind": event, **payload}
            self.cards.append(card)
            del self.cards[:-60]
            db.add_event(self.meeting_id, event, payload)
        elif event == "notes":
            db.save_notes(self.meeting_id, payload.get("notes") or {})
        elif event == "summary":
            db.save_summary(self.meeting_id, payload.get("summary") or "")
        self._emit(event, payload)
        if event in ("advice", "answer", "notes", "usage"):
            self._emit("cost", self.cost())

    # -------------------------------------------------------------------- cost

    def cost(self) -> dict:
        audio_minutes = self.stt.audio_seconds / 60.0
        stt_usd = audio_minutes * config.DEEPGRAM_USD_PER_MINUTE
        llm = self.llm.usage.snapshot()
        return {
            "elapsed_seconds": round(self.state.elapsed(), 1),
            "audio_minutes": round(audio_minutes, 2),
            "stt_usd": round(stt_usd, 4),
            "llm_usd": llm["cost_usd"],
            "total_usd": round(stt_usd + llm["cost_usd"], 4),
            "llm_calls": llm["calls"],
            "prompt_tokens": llm["prompt_tokens"],
            "completion_tokens": llm["completion_tokens"],
            "stt_model": self.stt.model,
        }

    def snapshot(self) -> dict:
        """Everything a freshly loaded page needs to render the meeting."""
        with self.state.lock:
            return {
                "meeting_id": self.meeting_id,
                "title": self.state.title,
                "brief": self.state.brief,
                "running": not self.stopped,
                "status": self.stt_status,
                "error": self.error,
                "interim": self.interim,
                "segments": [s.as_dict() for s in self.state.segments],
                "cards": list(self.cards),
                "notes": dict(self.state.notes),
                "user_notes": self.state.user_notes,
                "summary": self.state.rolling_summary,
                "cost": self.cost(),
                "web_search": bool(config.TAVILY_API_KEY),
            }

    # ------------------------------------------------------------ audio to disk

    def _open_wav(self) -> None:
        try:
            config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            path = config.AUDIO_DIR / f"meeting-{self.meeting_id}.wav"
            wav = wave.open(str(path), "wb")
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            self._wav = wav
            log.info("saving audio to %s", path)
        except Exception:
            log.exception("could not open audio file; continuing without saving")
            self._wav = None

    def _close_wav(self) -> None:
        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:
                log.exception("failed to close audio file")
            self._wav = None


def _validate_sample_rate(value) -> int:
    try:
        rate = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid sample_rate: {value!r}") from None
    if not MIN_SAMPLE_RATE <= rate <= MAX_SAMPLE_RATE:
        raise ValueError(
            f"sample_rate {rate} outside supported range "
            f"{MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}"
        )
    return rate
