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
from copilot.brief import Brief
from copilot.engine import CopilotEngine
from copilot.llm import OpenRouterClient
from copilot.state import MeetingState
from stt import create_engine
from storage import db

log = logging.getLogger(__name__)

MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000


class MeetingSession:
    def __init__(self, brief: Brief, sample_rate: int, emit,
                 language: str | None = None, model: str | None = None,
                 provider: str | None = None, resume: dict | None = None):
        """`resume` is a stored meeting (from db.get_meeting) that was never
        stopped -- the process died, the laptop slept. The session continues it:
        same meeting id, same transcript, the copilot's memory restored, and
        the clock and the cost carrying on from where they were."""
        self.sample_rate = _validate_sample_rate(sample_rate)
        self._emit = emit
        self._lock = threading.Lock()
        self.language = (language or (resume or {}).get("language") or config.DEEPGRAM_LANGUAGE).strip()
        self.provider = (provider or (resume or {}).get("provider") or config.STT_PROVIDER).strip().lower()
        chosen_model = (model or "").strip()
        self.resumed = resume is not None
        # Speech already billed before an interruption; cost() adds it back on.
        self._audio_offset = float((resume or {}).get("audio_seconds") or 0.0)
        # Paused means the microphone is muted, not that the meeting ended: the
        # Deepgram socket stays open on keepalives, so no audio is billed and
        # resuming needs no reconnect and no second microphone prompt.
        self._paused = threading.Event()

        self.stt_status = {"state": "starting", "detail": ""}
        self.cards: list[dict] = []  # copilot panel: advice + answers
        self.attendee_turns: list[dict] = []  # what the AI attendee has said
        # Pauses and resumptions, so a reloaded page can draw the seams in the
        # transcript where they belong instead of losing them.
        self.markers: list[dict] = []
        self.interim = ""
        self.error: str | None = None
        # An event rather than a flag, so the cost ticker can wait on it and exit
        # the moment the meeting ends instead of sleeping through its interval.
        self._stopped = threading.Event()

        if resume is None:
            self.meeting_id = db.create_meeting(
                title=brief.title,
                brief=brief.as_dict(),
                language=self.language,
                stt_model=chosen_model,
                provider=self.provider,
            )
            self.state = MeetingState(meeting_id=self.meeting_id, brief=brief)
        else:
            self.meeting_id = int(resume["id"])
            # Keep the original start so line timestamps stay continuous: a
            # meeting interrupted at 1:20:00 and resumed resumes at 1:20:xx plus
            # the gap, which is the truth of what happened.
            self.state = MeetingState(
                meeting_id=self.meeting_id,
                brief=brief,
                started_at=float(resume.get("started_at") or time.time()),
            )
            self.state.load_stored(resume)
            # Anything the copilot said before the break is still on screen.
            for event in resume.get("events") or []:
                kind, payload = event.get("kind"), event.get("payload") or {}
                if kind in ("advice", "answer"):
                    self.cards.append({"kind": kind, **payload})
                elif kind == "attendee":
                    self.attendee_turns.append(payload)
                elif kind in ("pause", "resume") and "at" in payload:
                    self.markers.append(payload)
            del self.cards[:-60]
            del self.attendee_turns[:-40]

        # The client owns the usage counter; cost() reads it back rather than
        # keeping a second copy that could drift out of step.
        self.llm = OpenRouterClient()
        if resume is not None:
            self.llm.usage.seed(resume.get("usage_json") or {})
        self.engine = CopilotEngine(self.state, self.llm, self._engine_emit)

        self.stt = create_engine(
            provider=self.provider,
            sample_rate=self.sample_rate,
            language=self.language,
            model=chosen_model,
            keyterms=brief.keyterms(),
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

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def set_paused(self, paused: bool) -> None:
        """Mute or unmute the microphone without ending the meeting."""
        if self.stopped or paused == self.paused:
            return
        if paused:
            self._paused.set()
        else:
            self._paused.clear()
            # Anything Deepgram was still holding belongs to the old segment.
            self.interim = ""
            self._emit("interim", {"text": ""})

        payload = {"paused": paused, "at": round(self.state.elapsed(), 1)}
        self.markers.append(payload)
        db.add_event(self.meeting_id, "pause", payload)
        self._emit("paused", payload)

    # ----------------------------------------------------------------- control

    def start(self) -> None:
        self.stt.start()
        if self.resumed:
            # A visible seam in the transcript, so nobody later mistakes the gap
            # for a silence. Rendered by the page the same way a pause is.
            payload = {"paused": False, "resumed": True, "at": round(self.state.elapsed(), 1)}
            self.markers.append(payload)
            db.add_event(self.meeting_id, "resume", payload)
            self._emit("paused", payload)

    def feed_audio(self, chunk: bytes) -> None:
        # Dropping the audio here rather than in the browser means the cost
        # readout stops too: audio_seconds counts bytes actually sent onward.
        if self.stopped or self.paused:
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
            summarised_upto = self.state.summarised_upto
            speakers = dict(self.state.speaker_names)
        db.save_notes(self.meeting_id, notes)
        db.save_summary(self.meeting_id, summary, summarised_upto)
        db.save_speakers(self.meeting_id, speakers)
        db.finish_meeting(
            self.meeting_id,
            self._audio_offset + self.stt.audio_seconds,
            self.llm.usage.snapshot(),
            self.stt.model,
        )

        self._emit("cost", self.cost())
        self._emit("meeting_stopped", {"meeting_id": self.meeting_id, "notes": notes})

    def ask(self, question: str, web: bool = False) -> None:
        self.engine.ask(question, web=web)

    def set_user_notes(self, text: str) -> None:
        with self.state.lock:
            self.state.user_notes = text
        db.save_user_notes(self.meeting_id, text)

    # ---------------------------------------------------------------- speakers

    def set_speaker_name(self, speaker: int, name: str) -> None:
        """Attach a name to a diarised voice. Relabels the whole transcript."""
        self.state.set_speaker_name(speaker, name)
        with self.state.lock:
            names = dict(self.state.speaker_names)
        db.save_speakers(self.meeting_id, names)
        self._emit("speakers", {"speaker_names": _stringify(names)})

    def set_segment_speaker(self, index: int, name: str) -> None:
        """Correct the speaker on one line. Beats the voice-level name, because
        it is the user telling us something the diarisation got wrong."""
        seg = self.state.set_segment_speaker(index, name)
        if seg is None:
            return
        db.set_segment_speaker(self.meeting_id, index, seg.speaker_name)
        with self.state.lock:
            names = dict(self.state.speaker_names)
        self._emit("segment_speaker", seg.as_dict(names))

    def apply_speaker_suggestion(self, accept: bool) -> None:
        with self.state.lock:
            suggestion = self.state.speaker_suggestion
            self.state.speaker_suggestion = None
        if not suggestion:
            return
        if accept:
            for proposal in suggestion.get("proposals") or []:
                self.state.set_speaker_name(proposal["speaker"], proposal["name"])
            with self.state.lock:
                names = dict(self.state.speaker_names)
            db.save_speakers(self.meeting_id, names)
            self._emit("speakers", {"speaker_names": _stringify(names)})
        self._emit("speaker_suggestion_cleared", {})

    # ------------------------------------------------------------- STT handlers

    def _on_interim(self, text: str) -> None:
        self.interim = text
        self._emit("interim", {"text": text})

    def _on_utterance(self, utt) -> None:
        seg = self.state.add_utterance(utt.text, utt.speaker)
        try:
            db.add_segment(self.meeting_id, seg.index, seg.at, seg.speaker, seg.text)
        except Exception as exc:  # noqa: BLE001 - the meeting must go on, loudly
            # A line that reaches the screen but not the database is exactly the
            # failure that only shows up days later as an empty recording. Say
            # so now, in the log and on the page, instead of carrying on quietly.
            log.exception("could not save line %s to the database", seg.index)
            self._save_failures = getattr(self, "_save_failures", 0) + 1
            if self._save_failures <= 3:
                self._emit("error", {
                    "message": f"Could not save a transcript line to the database ({exc}). "
                    "Check the console window; if the data folder is in OneDrive, move it out."
                })
        self.interim = ""
        with self.state.lock:
            names = dict(self.state.speaker_names)
        self._emit("segment", seg.as_dict(names))
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
            self.cards.append({"kind": event, **payload})
            del self.cards[:-60]
            db.add_event(self.meeting_id, event, payload)
        elif event == "attendee":
            self.attendee_turns.append(payload)
            del self.attendee_turns[:-40]
            db.add_event(self.meeting_id, event, payload)
        elif event == "notes":
            db.save_notes(self.meeting_id, payload.get("notes") or {})
        elif event == "summary":
            db.save_summary(self.meeting_id, payload.get("summary") or "", payload.get("upto"))
        self._emit(event, payload)
        if event in ("advice", "answer", "attendee", "notes", "usage"):
            self._emit("cost", self.cost())

    # -------------------------------------------------------------------- cost

    def cost(self) -> dict:
        audio_minutes = (self._audio_offset + self.stt.audio_seconds) / 60.0
        # Each engine prices its own minute -- the local one is free, and saying
        # otherwise would make the readout a lie.
        rate = self.stt.usd_per_minute
        if self.provider == "deepgram":
            rate = config.DEEPGRAM_USD_PER_MINUTE
        stt_usd = audio_minutes * rate
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
            "provider": self.provider,
            "stt_usd_per_minute": rate,
        }

    def snapshot(self) -> dict:
        """Everything a freshly loaded page needs to render the meeting."""
        with self.state.lock:
            names = dict(self.state.speaker_names)
            return {
                "meeting_id": self.meeting_id,
                "brief": self.state.brief.as_dict(),
                "title": self.state.brief.title,
                "running": not self.stopped,
                "resumed": self.resumed,
                "paused": self.paused,
                "language": self.language,
                "provider": self.provider,
                "status": self.stt_status,
                "error": self.error,
                "interim": self.interim,
                "segments": [s.as_dict(names) for s in self.state.segments],
                "markers": list(self.markers),
                "speaker_names": _stringify(names),
                "speaker_suggestion": self.state.speaker_suggestion,
                "cards": list(self.cards),
                "attendee_turns": list(self.attendee_turns),
                "notes": dict(self.state.notes),
                "user_notes": self.state.user_notes,
                "summary": self.state.rolling_summary,
                "cost": self.cost(),
                "web_search": bool(config.TAVILY_API_KEY),
                "attendee_enabled": config.ATTENDEE_ENABLED,
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


def _stringify(names: dict) -> dict:
    """JSON object keys must be strings; the browser converts back."""
    return {str(k): v for k, v in names.items()}


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
