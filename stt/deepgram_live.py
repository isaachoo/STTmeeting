"""Deepgram streaming (live) speech-to-text.

Uses the raw WebSocket API via websocket-client rather than the Deepgram SDK:
one dependency, no SDK version churn, and full control over reconnection --
which matters for a five-hour meeting where the socket will drop at least once.

Two behaviours worth knowing about:

* Model fallback. Cantonese (`zh-HK`) is not available on every Deepgram model.
  We try each model in `models` in order; if a connection is rejected before it
  ever produces a result, we move to the next one. So `nova-3,nova-2` means
  "prefer nova-3, quietly fall back to nova-2 if it won't take zh-HK".
* Reconnect. Once a model has proven itself, an unexpected close is treated as a
  network blip and retried with backoff, keeping the same model.
"""

import json
import logging
import queue
import threading
import time
import urllib.parse

import websocket

from .base import STTEngine, Utterance

log = logging.getLogger(__name__)

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

# Bounded so a long reconnect cannot grow memory without limit. Each chunk is
# ~4096 samples; 400 chunks is roughly 100 seconds at 16 kHz.
MAX_QUEUED_CHUNKS = 400
KEEPALIVE_IDLE_SECONDS = 5.0
MAX_CONSECUTIVE_FAILURES = 6
# Deepgram caps boosted terms; well beyond what a meeting glossary needs.
MAX_KEYTERMS = 100


class DeepgramLiveSTT(STTEngine):
    def __init__(
        self,
        api_key: str,
        sample_rate: int,
        language: str = "zh-HK",
        models: list[str] | None = None,
        channels: int = 1,
        diarize: bool = True,
        keyterms: list[str] | None = None,
        on_interim=None,
        on_utterance=None,
        on_status=None,
        on_error=None,
    ):
        self.api_key = api_key
        self.sample_rate = int(sample_rate)
        self.language = language
        self.models = list(models or ["nova-3", "nova-2"])
        self.channels = channels
        self.diarize = diarize
        # Jargon and names from the brief. Boosting is not supported on every
        # model/language pair, so it is the first thing dropped when a
        # connection is rejected -- see _run_supervisor.
        self.keyterms = [t for t in (keyterms or []) if t.strip()][:MAX_KEYTERMS]
        self._use_keyterms = bool(self.keyterms)

        self._on_interim = on_interim or (lambda text: None)
        self._on_utterance = on_utterance or (lambda utt: None)
        self._on_status = on_status or (lambda **kw: None)
        self._on_error = on_error or (lambda msg: None)

        self._audio: queue.Queue[bytes | None] = queue.Queue(maxsize=MAX_QUEUED_CHUNKS)
        self._stopping = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._ws = None  # the live connection, so stop() can shut it immediately

        self._model_index = 0
        self._proven = False  # current model has returned at least one result
        self._bytes_sent = 0
        self._dropped_chunks = 0

        # Final-but-not-yet-flushed fragments of the current utterance.
        self._pending: list[str] = []
        self._pending_speaker: int | None = None
        self._pending_start: float | None = None
        self._pending_end: float | None = None
        self._pending_words: list[dict] = []
        self._pending_lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        if self._supervisor and self._supervisor.is_alive():
            return
        self._stopping.clear()
        self._supervisor = threading.Thread(
            target=self._run_supervisor, name="deepgram-supervisor", daemon=True
        )
        self._supervisor.start()

    def send_audio(self, chunk: bytes) -> None:
        if self._stopping.is_set() or not chunk:
            return
        try:
            self._audio.put_nowait(chunk)
        except queue.Full:
            # Drop the oldest chunk rather than the newest: during a reconnect
            # the freshest audio is the audio worth keeping.
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
            self._audio.put_nowait(None)  # wakes the sender loop, sends CloseStream
        except queue.Full:
            pass
        self._flush_pending(reason="stop")
        # Close from this side too, so the supervisor thread winds down promptly
        # instead of waiting on a server that may never close first.
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def join(self, timeout: float = 5.0) -> None:
        """Wait for the connection thread to finish. Used by tests."""
        if self._supervisor is not None:
            self._supervisor.join(timeout)

    @property
    def audio_seconds(self) -> float:
        bytes_per_second = self.sample_rate * self.channels * 2
        return self._bytes_sent / bytes_per_second if bytes_per_second else 0.0

    @property
    def model(self) -> str:
        return self.models[min(self._model_index, len(self.models) - 1)]

    # ------------------------------------------------------------- connection

    def _url(self) -> str:
        params: list[tuple[str, str]] = list(self._base_params().items())
        if self._use_keyterms and self.keyterms:
            # nova-3 calls it keyterm, earlier models call it keywords. Both take
            # the parameter repeated once per term.
            name = "keyterm" if self.model.startswith("nova-3") else "keywords"
            params.extend((name, term) for term in self.keyterms)
        return f"{DEEPGRAM_URL}?{urllib.parse.urlencode(params)}"

    def _base_params(self) -> dict:
        params = {
            "model": self.model,
            "language": self.language,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "channels": str(self.channels),
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
            "vad_events": "true",
            # Endpointing closes an utterance on a short pause; utterance_end is
            # the safety net when no endpoint is detected at all.
            "endpointing": "400",
            "utterance_end_ms": "1200",
        }
        if self.diarize:
            params["diarize"] = "true"
        return params

    def _run_supervisor(self) -> None:
        failures = 0
        while not self._stopping.is_set():
            self._proven = False
            close_info = self._run_one_connection()
            if self._stopping.is_set():
                break

            if self._proven:
                # Worked, then dropped: network blip, same model, back off.
                failures += 1
                if failures > MAX_CONSECUTIVE_FAILURES:
                    self._on_error(
                        "Lost the Deepgram connection repeatedly and gave up. "
                        "Check your network, then stop and start the meeting again."
                    )
                    break
                delay = min(2 ** (failures - 1), 16)
                self._on_status(
                    state="reconnecting",
                    detail=f"connection dropped ({close_info}); retrying in {delay}s",
                )
                if self._stopping.wait(delay):
                    break
                continue

            # Never produced a result, so the connection itself was refused.
            # Drop the optional extras before giving up on the model: term
            # boosting is not supported on every model/language pair, and
            # losing it is far better than losing transcription.
            if self._use_keyterms and self.keyterms:
                self._use_keyterms = False
                self._on_status(
                    state="degraded",
                    detail=(
                        f"{self.model} would not accept term boosting "
                        f"({close_info}); retrying without it"
                    ),
                )
                failures = 0
                continue

            if self._model_index + 1 < len(self.models):
                rejected = self.model
                self._model_index += 1
                # A different model may well accept boosting, so offer it again.
                self._use_keyterms = bool(self.keyterms)
                self._on_status(
                    state="fallback",
                    detail=(
                        f"{rejected} did not accept language {self.language} "
                        f"({close_info}); trying {self.model}"
                    ),
                )
                failures = 0
                continue

            self._on_error(
                f"Deepgram rejected every configured model "
                f"({', '.join(self.models)}) for language {self.language}: {close_info}"
            )
            break

        self._on_status(state="closed")

    def _run_one_connection(self) -> str:
        """Run one WebSocket session to completion. Returns a close description."""
        connection_closed = threading.Event()
        close_reason: list[str] = []
        sender_started = threading.Event()

        def on_open(ws):
            self._on_status(state="listening", model=self.model, language=self.language)
            sender = threading.Thread(
                target=self._sender_loop,
                args=(ws, connection_closed),
                name="deepgram-sender",
                daemon=True,
            )
            sender.start()
            sender_started.set()

        def on_message(ws, message):
            try:
                self._handle_message(json.loads(message))
            except Exception:  # a malformed frame must not kill the session
                log.exception("failed to handle Deepgram message")

        def on_error(ws, err):
            close_reason.append(str(err) or err.__class__.__name__)

        def on_close(ws, status_code, msg):
            if status_code or msg:
                close_reason.append(f"code={status_code} {msg or ''}".strip())
            connection_closed.set()

        ws = websocket.WebSocketApp(
            self._url(),
            header={"Authorization": f"Token {self.api_key}"},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._ws = ws
        try:
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:  # noqa: BLE001 - surfaced via close_reason
            close_reason.append(str(exc))
        finally:
            connection_closed.set()
            self._ws = None

        return "; ".join(r for r in close_reason if r) or "closed without a reason"

    def _sender_loop(self, ws, connection_closed: threading.Event) -> None:
        """Single writer for this connection: audio frames plus keepalives."""
        while not connection_closed.is_set():
            try:
                chunk = self._audio.get(timeout=KEEPALIVE_IDLE_SECONDS)
            except queue.Empty:
                # Deepgram closes an idle socket after ~10s of silence.
                try:
                    ws.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    return
                continue

            if chunk is None:  # stop() sentinel
                try:
                    ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass
                return

            try:
                ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                self._bytes_sent += len(chunk)
            except Exception:
                # Put it back so the next connection picks up where we left off.
                try:
                    self._audio.put_nowait(chunk)
                except queue.Full:
                    pass
                return

    # ---------------------------------------------------------------- parsing

    def _handle_message(self, msg: dict) -> None:
        kind = msg.get("type")

        if kind == "Results":
            self._proven = True
            self._handle_results(msg)
        elif kind == "UtteranceEnd":
            self._proven = True
            self._flush_pending(reason="utterance_end")
        elif kind == "Metadata":
            self._proven = True
        elif kind == "SpeechStarted":
            self._proven = True
        elif kind in ("Error", "Warning"):
            detail = msg.get("description") or msg.get("message") or json.dumps(msg)
            if kind == "Error":
                self._on_error(f"Deepgram: {detail}")
            else:
                log.warning("Deepgram warning: %s", detail)

    def _handle_results(self, msg: dict) -> None:
        alternatives = (msg.get("channel") or {}).get("alternatives") or []
        if not alternatives:
            return
        alt = alternatives[0]
        text = (alt.get("transcript") or "").strip()
        words = alt.get("words") or []
        is_final = bool(msg.get("is_final"))
        speech_final = bool(msg.get("speech_final"))

        if not is_final:
            if text:
                self._on_interim(self._preview(text))
            return

        if text:
            with self._pending_lock:
                self._pending.append(text)
                self._pending_words.extend(words)
                if self._pending_speaker is None and words:
                    self._pending_speaker = words[0].get("speaker")
                if self._pending_start is None:
                    self._pending_start = msg.get("start")
                start = msg.get("start") or 0.0
                self._pending_end = start + (msg.get("duration") or 0.0)
            # The interim line now shows the accumulated utterance so the user
            # does not see text vanish between a final and the next interim.
            self._on_interim(self._preview(""))

        if speech_final:
            self._flush_pending(reason="speech_final")

    def _preview(self, interim_text: str) -> str:
        with self._pending_lock:
            settled = " ".join(self._pending)
        return " ".join(p for p in (settled, interim_text) if p).strip()

    def _flush_pending(self, reason: str) -> None:
        with self._pending_lock:
            if not self._pending:
                return
            utt = Utterance(
                text=" ".join(self._pending).strip(),
                speaker=self._pending_speaker,
                start=self._pending_start,
                end=self._pending_end,
                words=self._pending_words,
            )
            self._pending = []
            self._pending_speaker = None
            self._pending_start = None
            self._pending_end = None
            self._pending_words = []

        if utt.text:
            self._on_interim("")
            self._on_utterance(utt)
