"""Speechmatics real-time speech-to-text.

A second cloud provider, mainly so the same meeting style can be judged against
two engines instead of taken on trust. Speechmatics supports Cantonese (`yue`)
with real-time speaker diarization, which is what this app needs.

Their protocol differs from Deepgram's but carries the same information:

    -> StartRecognition {audio_format, transcription_config}
    <- RecognitionStarted
    -> binary audio frames
    <- AddPartialTranscript  (interim)
    <- AddTranscript         (final)
    -> EndOfStream {last_seq_no}
    <- EndOfTranscript

Speaker labels arrive per word as "S1", "S2"..., which are mapped onto the
integer indices the rest of the app already uses for Deepgram's diarisation, so
naming, filtering and the copilot's roster work unchanged.
"""

import json
import logging
import queue
import threading

import websocket

from .base import STTEngine, Utterance

log = logging.getLogger(__name__)

DEFAULT_URL = "wss://eu2.rt.speechmatics.com/v2"
MAX_QUEUED_CHUNKS = 400
KEEPALIVE_IDLE_SECONDS = 5.0
MAX_CONSECUTIVE_FAILURES = 6


class SpeechmaticsLiveSTT(STTEngine):
    def __init__(
        self,
        api_key: str,
        sample_rate: int,
        language: str = "yue",
        url: str = DEFAULT_URL,
        operating_point: str = "enhanced",
        diarize: bool = True,
        keyterms: list[str] | None = None,
        usd_per_minute: float = 0.0,
        on_interim=None,
        on_utterance=None,
        on_status=None,
        on_error=None,
    ):
        self.api_key = api_key
        self.sample_rate = int(sample_rate)
        self.language = language
        self.url = url or DEFAULT_URL
        self.operating_point = operating_point
        self.diarize = diarize
        # Their custom dictionary takes plain words; the brief's glossary maps
        # onto it directly.
        self.keyterms = [t for t in (keyterms or []) if t.strip()][:1000]
        self._usd_per_minute = usd_per_minute

        self._on_interim = on_interim or (lambda text: None)
        self._on_utterance = on_utterance or (lambda utt: None)
        self._on_status = on_status or (lambda **kw: None)
        self._on_error = on_error or (lambda msg: None)

        self._audio: queue.Queue[bytes | None] = queue.Queue(maxsize=MAX_QUEUED_CHUNKS)
        self._stopping = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._ws = None

        self._bytes_sent = 0
        self._dropped_chunks = 0
        self._seq_sent = 0
        self._proven = False
        self._speaker_ids: dict[str, int] = {}
        self._speaker_lock = threading.Lock()

    # ------------------------------------------------------------------ public

    @property
    def provider(self) -> str:
        return "speechmatics"

    @property
    def model(self) -> str:
        return f"speechmatics-{self.operating_point}"

    @property
    def usd_per_minute(self) -> float:
        return self._usd_per_minute

    @property
    def audio_seconds(self) -> float:
        per_second = self.sample_rate * 2
        return self._bytes_sent / per_second if per_second else 0.0

    def start(self) -> None:
        if self._supervisor and self._supervisor.is_alive():
            return
        self._stopping.clear()
        self._supervisor = threading.Thread(
            target=self._run_supervisor, name="speechmatics-supervisor", daemon=True
        )
        self._supervisor.start()

    def send_audio(self, chunk: bytes) -> None:
        if self._stopping.is_set() or not chunk:
            return
        try:
            self._audio.put_nowait(chunk)
        except queue.Full:
            try:
                self._audio.get_nowait()
                self._dropped_chunks += 1
                self._audio.put_nowait(chunk)
            except (queue.Empty, queue.Full):
                self._dropped_chunks += 1

    def stop(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        try:
            self._audio.put_nowait(None)
        except queue.Full:
            pass
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def join(self, timeout: float = 5.0) -> None:
        if self._supervisor is not None:
            self._supervisor.join(timeout)

    # ------------------------------------------------------------- connection

    def _start_message(self) -> dict:
        transcription: dict = {
            "language": self.language,
            "operating_point": self.operating_point,
            "enable_partials": True,
            "max_delay": 2.0,
        }
        if self.diarize:
            transcription["diarization"] = "speaker"
        if self.keyterms:
            transcription["additional_vocab"] = [
                {"content": term} for term in self.keyterms
            ]
        return {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "transcription_config": transcription,
        }

    def _run_supervisor(self) -> None:
        failures = 0
        while not self._stopping.is_set():
            self._proven = False
            close_info = self._run_one_connection()
            if self._stopping.is_set():
                break

            failures += 1
            if failures > MAX_CONSECUTIVE_FAILURES:
                self._on_error(
                    f"Speechmatics kept dropping the connection ({close_info}). "
                    "Check the API key and the network, then start again."
                )
                break
            delay = min(2 ** (failures - 1), 16)
            self._on_status(
                state="reconnecting",
                detail=f"connection dropped ({close_info}); retrying in {delay}s",
            )
            if self._stopping.wait(delay):
                break
        self._on_status(state="closed")

    def _run_one_connection(self) -> str:
        connection_closed = threading.Event()
        close_reason: list[str] = []

        def on_open(ws):
            ws.send(json.dumps(self._start_message()))
            threading.Thread(
                target=self._sender_loop,
                args=(ws, connection_closed),
                name="speechmatics-sender",
                daemon=True,
            ).start()

        def on_message(ws, message):
            try:
                self._handle_message(json.loads(message))
            except Exception:
                log.exception("failed to handle a Speechmatics message")

        def on_error(ws, err):
            close_reason.append(str(err) or err.__class__.__name__)

        def on_close(ws, status_code, msg):
            if status_code or msg:
                close_reason.append(f"code={status_code} {msg or ''}".strip())
            connection_closed.set()

        ws = websocket.WebSocketApp(
            self.url,
            header={"Authorization": f"Bearer {self.api_key}"},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._ws = ws
        try:
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:  # noqa: BLE001 - surfaced through close_reason
            close_reason.append(str(exc))
        finally:
            connection_closed.set()
            self._ws = None
        return "; ".join(r for r in close_reason if r) or "closed without a reason"

    def _sender_loop(self, ws, connection_closed: threading.Event) -> None:
        while not connection_closed.is_set():
            try:
                chunk = self._audio.get(timeout=KEEPALIVE_IDLE_SECONDS)
            except queue.Empty:
                continue  # Speechmatics tolerates gaps; nothing to send
            if chunk is None:
                try:
                    ws.send(json.dumps(
                        {"message": "EndOfStream", "last_seq_no": self._seq_sent}
                    ))
                except Exception:
                    pass
                return
            try:
                ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                self._bytes_sent += len(chunk)
                self._seq_sent += 1
            except Exception:
                try:
                    self._audio.put_nowait(chunk)
                except queue.Full:
                    pass
                return

    # ---------------------------------------------------------------- parsing

    def _handle_message(self, msg: dict) -> None:
        kind = msg.get("message")

        if kind == "RecognitionStarted":
            self._proven = True
            self._on_status(state="listening", model=self.model, language=self.language)
        elif kind == "AddPartialTranscript":
            text = self._text_of(msg)
            if text:
                self._on_interim(text)
        elif kind == "AddTranscript":
            self._proven = True
            self._handle_final(msg)
        elif kind == "EndOfTranscript":
            self._on_interim("")
        elif kind in ("Error", "Warning"):
            detail = msg.get("reason") or msg.get("type") or json.dumps(msg)
            if kind == "Error":
                self._on_error(f"Speechmatics: {detail}")
            else:
                log.warning("Speechmatics warning: %s", detail)

    @staticmethod
    def _text_of(msg: dict) -> str:
        # Prefer the assembled transcript; fall back to joining the tokens.
        text = (msg.get("metadata") or {}).get("transcript")
        if text:
            return text.strip()
        pieces = []
        for result in msg.get("results") or []:
            alternatives = result.get("alternatives") or []
            if alternatives:
                pieces.append(alternatives[0].get("content") or "")
        return " ".join(p for p in pieces if p).strip()

    def _handle_final(self, msg: dict) -> None:
        text = self._text_of(msg)
        if not text:
            return

        results = msg.get("results") or []
        speaker = self._speaker_index(results)
        starts = [r.get("start_time") for r in results if r.get("start_time") is not None]
        ends = [r.get("end_time") for r in results if r.get("end_time") is not None]

        self._on_interim("")
        self._on_utterance(
            Utterance(
                text=text,
                speaker=speaker,
                start=min(starts) if starts else None,
                end=max(ends) if ends else None,
            )
        )

    def _speaker_index(self, results: list[dict]) -> int | None:
        """Map Speechmatics' "S1"/"S2" labels onto the integer indices the rest
        of the app uses, keeping first-seen order stable for the whole meeting."""
        label = None
        for result in results:
            for alternative in result.get("alternatives") or []:
                label = alternative.get("speaker")
                if label:
                    break
            if label:
                break
        if not label or label in ("UU", "unknown"):
            return None

        with self._speaker_lock:
            if label not in self._speaker_ids:
                self._speaker_ids[label] = len(self._speaker_ids)
            return self._speaker_ids[label]
