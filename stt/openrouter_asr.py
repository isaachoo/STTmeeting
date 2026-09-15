"""Qwen3-ASR through OpenRouter, in near-real time.

Qwen3-ASR (Alibaba) is the one model reachable with the OpenRouter key this app
already has that was trained on Cantonese *and* on Chinese-English mixing inside
a single sentence, which is how a Hong Kong meeting actually sounds. Deepgram's
zh-HK model is stronger on pure Cantonese but tends to mangle the English words.

The catch: OpenRouter's transcription endpoint takes a finished audio file and
returns text. There is no streaming socket. So this engine builds its own
stream out of short files:

* Microphone audio accumulates in a buffer. A simple energy detector watches
  for a pause. At a pause after enough speech -- or at a hard ceiling, so one
  long monologue cannot delay everything -- the buffer is closed off as one
  segment, wrapped in a WAV header and posted to OpenRouter.
* Requests run on a small pool so a slow answer does not hold up the next
  segment, but results are emitted in the order the audio was spoken.
* While a segment is buffering or in flight, the interim line shows that the
  engine is listening, so the page does not look dead during the few seconds of
  lag this design costs.

There is no speaker separation; every line arrives unattributed and the UI
lets you name lines by hand, exactly as with the local engine.
"""

import array
import base64
import io
import logging
import math
import queue
import threading
import time
import wave
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import requests

from .base import STTEngine, Utterance

log = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen/qwen3-asr-flash-2026-02-10"
QUEUE_TIMEOUT = 0.2
MAX_QUEUED_CHUNKS = 400
WORKERS = 2
REQUEST_TIMEOUT = (10, 60)
RETRIES = 2  # on top of the first attempt, for 429/5xx/network only

# Segmentation. Seconds.
PAUSE_SECONDS = 0.7  # this much quiet after speech closes a segment
MIN_SEGMENT_SECONDS = 1.0  # shorter than this is not worth a request
MAX_SEGMENT_SECONDS = 12.0  # hard ceiling; cut here even mid-sentence
PRE_ROLL_SECONDS = 0.4  # quiet kept in front of speech so word onsets survive

# Energy detector. RMS of 16-bit samples (0..32768).
ABSOLUTE_FLOOR = 120.0  # below this is always silence, whatever the room
SPEECH_RATIO = 3.0  # speech must be this many times louder than the noise floor
NOISE_ADAPT = 0.05  # how fast the noise floor follows quiet frames
# If this much audio goes by without a single frame counted as speech, say so
# on the page with the levels seen, so a quiet microphone is not mistaken for a
# broken transcriber.
QUIET_WARNING_SECONDS = 15.0
PROBE_SECONDS = 1.0  # silence sent at start to prove key, model and endpoint


class OpenRouterASR(STTEngine):
    def __init__(
        self,
        api_key: str,
        sample_rate: int,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
        usd_per_minute: float = 0.0,
        max_segment_seconds: float = MAX_SEGMENT_SECONDS,
        to_traditional: bool = True,
        speech_floor: float = ABSOLUTE_FLOOR,
        probe: bool = True,
        on_interim=None,
        on_utterance=None,
        on_status=None,
        on_error=None,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.sample_rate = int(sample_rate)
        self.model_name = (model or DEFAULT_MODEL).strip()
        self.base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self._usd_per_minute = float(usd_per_minute)
        self.max_segment_seconds = max(MIN_SEGMENT_SECONDS + 1.0, float(max_segment_seconds))
        self.to_traditional = to_traditional
        self.speech_floor = max(20.0, float(speech_floor or ABSOLUTE_FLOOR))
        self.probe = probe

        self._on_interim = on_interim or (lambda text: None)
        self._on_utterance = on_utterance or (lambda utt: None)
        self._on_status = on_status or (lambda **kw: None)
        self._on_error = on_error or (lambda msg: None)

        self._http = session or requests.Session()
        self._audio: queue.Queue[bytes | None] = queue.Queue(maxsize=MAX_QUEUED_CHUNKS)
        self._stopping = threading.Event()
        self._worker: threading.Thread | None = None
        self._pool: ThreadPoolExecutor | None = None

        self._bytes_sent = 0
        self._dropped_chunks = 0
        self._converter = None

        # Segmenter state.
        self._segment = bytearray()
        self._segment_start: float | None = None  # seconds into the meeting
        self._speech_seen = False
        self._silence_run = 0.0
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_bytes = 0
        # Starts so that the threshold is exactly the configured floor; it only
        # rises from there as the room's own noise is learnt.
        self._noise_floor = self.speech_floor / SPEECH_RATIO
        self._last_interim_at = 0.0
        # Diagnostics for the "nothing happens" case.
        self._last_rms = 0.0
        self._peak_rms = 0.0
        self._speech_frames = 0
        self._quiet_warned = False
        self._shape_warned = False

        # Results are emitted in speech order even when a later request is
        # answered first.
        self._pending: deque[tuple[int, float, float, Future]] = deque()
        self._seq = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.requests_made = 0
        self.reported_cost_usd = 0.0

    # ------------------------------------------------------------------ public

    @property
    def provider(self) -> str:
        return "qwen"

    @property
    def model(self) -> str:
        return f"openrouter:{self.model_name}"

    @property
    def usd_per_minute(self) -> float:
        return self._usd_per_minute

    @property
    def audio_seconds(self) -> float:
        per_second = self.sample_rate * 2
        return self._bytes_sent / per_second if per_second else 0.0

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stopping.clear()
        self._worker = threading.Thread(target=self._run, name="openrouter-asr", daemon=True)
        self._worker.start()

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

    def join(self, timeout: float = 30.0) -> None:
        if self._worker is not None:
            self._worker.join(timeout)

    # -------------------------------------------------------------------- loop

    def _run(self) -> None:
        if not self.api_key:
            self._on_error("OpenRouter API key is missing, so Qwen3-ASR cannot run.")
            self._on_status(state="closed")
            return
        self._load_converter()
        self._pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="openrouter-asr")
        self._on_status(state="connecting", detail="checking OpenRouter and the Qwen3-ASR model")
        if self.probe and not self._probe():
            # The error has already been reported. Keep listening anyway: a
            # rate limit at start is not a reason to lose the meeting.
            self._on_status(state="listening", model=self.model,
                            detail="the start-up check failed; still trying each segment")
        else:
            self._on_status(
                state="listening",
                model=self.model,
                detail="near-real time: lines arrive a few seconds after a pause",
            )
        try:
            while True:
                chunk = self._take_chunk()
                if chunk is None:
                    break
                self._feed(chunk)
                self._emit_ready(block=False)
            self._close_segment(reason="stop")
            self._emit_ready(block=True)
        except Exception as exc:  # noqa: BLE001 - a crash here must be visible
            log.exception("OpenRouter transcription failed")
            self._on_error(f"Qwen3-ASR transcription stopped: {exc}")
        finally:
            self._pool.shutdown(wait=True)
            self._on_interim("")
            self._on_status(state="closed")

    def _take_chunk(self) -> bytes | None:
        while True:
            try:
                chunk = self._audio.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                if self._stopping.is_set():
                    return None
                # Quiet microphone, but a segment may be waiting on the pause
                # timer, and finished requests should not wait for more audio.
                self._feed(b"", seconds=QUEUE_TIMEOUT)
                self._emit_ready(block=False)
                continue
            if chunk is None:
                return None
            self._bytes_sent += len(chunk)
            return chunk

    # --------------------------------------------------------------- segmenter

    def _feed(self, chunk: bytes, seconds: float | None = None) -> None:
        """Push one chunk (or a stretch of nothing) through the pause detector."""
        if seconds is None:
            seconds = len(chunk) / (2 * self.sample_rate)
        loud = self._is_speech(chunk) if chunk else False

        if not self._speech_seen:
            if loud:
                self._speech_seen = True
                self._silence_run = 0.0
                pre = b"".join(self._pre_roll)
                self._pre_roll.clear()
                self._pre_roll_bytes = 0
                self._segment_start = self.audio_seconds - (len(pre) + len(chunk)) / (2 * self.sample_rate)
                self._segment.extend(pre)
                self._segment.extend(chunk)
                self._show_listening(force=True)
            elif chunk:
                self._pre_roll.append(chunk)
                self._pre_roll_bytes += len(chunk)
                limit = int(PRE_ROLL_SECONDS * self.sample_rate * 2)
                while self._pre_roll_bytes > limit and len(self._pre_roll) > 1:
                    self._pre_roll_bytes -= len(self._pre_roll.popleft())
            self._show_listening()  # idle: the level readout
            return

        self._segment.extend(chunk)
        self._silence_run = 0.0 if loud else self._silence_run + seconds
        length = len(self._segment) / (2 * self.sample_rate)
        self._show_listening()

        if length >= self.max_segment_seconds:
            self._close_segment(reason="ceiling")
        elif self._silence_run >= PAUSE_SECONDS and length >= MIN_SEGMENT_SECONDS:
            self._close_segment(reason="pause")

    @property
    def threshold(self) -> float:
        return max(self.speech_floor, self._noise_floor * SPEECH_RATIO)

    def _is_speech(self, chunk: bytes) -> bool:
        rms = _rms(chunk)
        self._last_rms = rms
        self._peak_rms = max(self._peak_rms, rms)
        if rms < self.threshold:
            # Quiet: let the noise floor drift towards it. Fast when the room
            # gets quieter, slow when it gets louder so speech is not learnt
            # as noise.
            rate = NOISE_ADAPT if rms < self._noise_floor else NOISE_ADAPT / 4
            self._noise_floor += (rms - self._noise_floor) * rate
            self._noise_floor = max(self._noise_floor, self.speech_floor / SPEECH_RATIO)
            self._check_quiet()
            return False
        self._speech_frames += 1
        return True

    def _check_quiet(self) -> None:
        """A microphone that never crosses the threshold looks exactly like a
        transcriber that does nothing. Say which it is, once."""
        if self._quiet_warned or self._speech_frames or self.audio_seconds < QUIET_WARNING_SECONDS:
            return
        self._quiet_warned = True
        self._on_error(
            f"Qwen3-ASR has heard {self.audio_seconds:.0f}s of audio but nothing loud enough "
            f"to count as speech: microphone level peaks at {self._peak_rms:.0f}, the speech "
            f"threshold is {self.threshold:.0f}. Move closer to the microphone, raise its "
            "input level in Windows sound settings, or lower OPENROUTER_ASR_SPEECH_FLOOR."
        )

    def _close_segment(self, reason: str) -> None:
        pcm = bytes(self._segment)
        start = self._segment_start
        self._segment = bytearray()
        self._segment_start = None
        had_speech = self._speech_seen
        self._speech_seen = False
        self._silence_run = 0.0
        if not had_speech:
            return
        length = len(pcm) / (2 * self.sample_rate)
        if length < MIN_SEGMENT_SECONDS and reason != "stop":
            return
        if length < 0.3:
            return
        end = (start or 0.0) + length
        seq = self._seq
        self._seq += 1
        log.debug("segment %d: %.1fs (%s)", seq, length, reason)
        future = self._pool.submit(self._transcribe, pcm)
        self._pending.append((seq, start or 0.0, end, future))
        self._show_listening(force=True)

    def _show_listening(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_interim_at < 1.0:
            return
        self._last_interim_at = now
        parts = []
        if self._speech_seen:
            parts.append(f"聽到 {len(self._segment) / (2 * self.sample_rate):.0f}s…")
        if self._pending:
            parts.append(f"轉寫中 ({len(self._pending)})")
        if not parts and self.audio_seconds > 0:
            # Idle: show the level so a too-quiet microphone is visible at a
            # glance rather than looking like a dead transcriber.
            parts.append(f"音量 {self._last_rms:.0f} / 門檻 {self.threshold:.0f}")
        self._on_interim(" ".join(parts))

    # ----------------------------------------------------------------- results

    def _emit_ready(self, block: bool) -> None:
        """Hand finished segments up, oldest first, never out of order."""
        while self._pending:
            seq, start, end, future = self._pending[0]
            if not block and not future.done():
                return
            self._pending.popleft()
            try:
                text = future.result(timeout=REQUEST_TIMEOUT[1] * (RETRIES + 1) + 5)
            except Exception as exc:  # noqa: BLE001 - reported below
                log.warning("segment %d failed: %s", seq, exc)
                text = ""
            if text:
                self._on_utterance(Utterance(text=text, speaker=None, start=start, end=end))
            self._show_listening(force=True)

    def _transcribe(self, pcm: bytes) -> str:
        """One request for one segment. Returns the text, or "" after reporting."""
        body = {
            "model": self.model_name,
            "input_audio": {"data": _wav_base64(pcm, self.sample_rate), "format": "wav"},
        }
        attempt = 0
        while True:
            attempt += 1
            started = time.monotonic()
            try:
                resp = self._http.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-Title": "Cantonese Meeting Copilot",
                    },
                    json=body,
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.RequestException as exc:
                if attempt <= RETRIES:
                    time.sleep(attempt)
                    continue
                return self._fail(f"could not reach OpenRouter ({exc})")

            self.requests_made += 1
            if resp.status_code in (429, 500, 502, 503, 504) and attempt <= RETRIES:
                time.sleep(attempt)
                continue
            if resp.status_code >= 400:
                return self._fail(_describe_http_error(resp))
            try:
                payload = resp.json()
            except ValueError:
                return self._fail("OpenRouter returned something that was not JSON")
            if isinstance(payload, dict) and payload.get("error"):
                err = payload["error"]
                message = err.get("message") if isinstance(err, dict) else str(err)
                return self._fail(f"OpenRouter error: {message}")

            usage = payload.get("usage") if isinstance(payload, dict) else None
            if isinstance(usage, dict):
                try:
                    self.reported_cost_usd += float(usage.get("cost") or 0.0)
                except (TypeError, ValueError):
                    pass
            text = _extract_text(payload)
            if text is None:
                # Answered, but not in a shape this code knows. Silence here
                # would look like a broken microphone; say what came back.
                shape = _describe_shape(payload)
                log.warning("asr %s: unrecognised response: %s", self.model_name, shape)
                if not self._shape_warned:
                    self._shape_warned = True
                    self._on_error(
                        "OpenRouter answered, but not with a transcript this app "
                        f"understands: {shape}. Please report this."
                    )
                return ""
            log.info(
                "asr %s: %.1fs audio -> %d chars in %.1fs",
                self.model_name, len(pcm) / (2 * self.sample_rate),
                len(text), time.monotonic() - started,
            )
            self.consecutive_failures = 0
            return self._finish_text(text)

    def _probe(self) -> bool:
        """One second of silence, sent before the meeting starts.

        Costs a few thousandths of a cent and turns 'nothing happens' into a
        message on the page within seconds when the key, the model name or the
        endpoint is wrong. Silence, so the model has nothing to hallucinate.
        """
        started = time.monotonic()
        try:
            resp = self._http.post(
                f"{self.base_url}/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "Cantonese Meeting Copilot",
                },
                json={
                    "model": self.model_name,
                    "input_audio": {
                        "data": _wav_base64(b"\x00\x00" * int(PROBE_SECONDS * self.sample_rate), self.sample_rate),
                        "format": "wav",
                    },
                },
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            self._on_error(f"Qwen3-ASR start-up check: could not reach OpenRouter ({exc}).")
            return False
        self.requests_made += 1
        if resp.status_code >= 400:
            self._on_error(f"Qwen3-ASR start-up check failed: {_describe_http_error(resp)}.")
            return False
        try:
            payload = resp.json()
        except ValueError:
            self._on_error("Qwen3-ASR start-up check: OpenRouter returned something that was not JSON.")
            return False
        if isinstance(payload, dict) and payload.get("error"):
            err = payload["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            self._on_error(f"Qwen3-ASR start-up check failed: {message}.")
            return False
        if _extract_text(payload) is None:
            shape = _describe_shape(payload)
            self._shape_warned = True
            self._on_error(
                "Qwen3-ASR start-up check: OpenRouter answered, but not with a transcript "
                f"this app understands: {shape}. Please report this."
            )
            return False
        log.info("asr %s: start-up check passed in %.1fs", self.model_name, time.monotonic() - started)
        return True

    def _fail(self, reason: str) -> str:
        self.failures += 1
        self.consecutive_failures += 1
        log.warning("Qwen3-ASR segment dropped: %s", reason)
        # The first failure is worth interrupting the user for; after that,
        # only every tenth, or a bad key would paint the page red once a
        # segment for the rest of the meeting.
        if self.failures == 1 or self.failures % 10 == 0:
            self._on_error(
                f"Qwen3-ASR (OpenRouter) could not transcribe a segment: {reason}. "
                f"{self.failures} segment(s) lost so far."
            )
        return ""

    # -------------------------------------------------------------------- text

    def _load_converter(self) -> None:
        """Qwen writes simplified characters for Cantonese speech, like the
        local model does. OpenCC maps them to Hong Kong traditional."""
        if not self.to_traditional:
            return
        try:
            from opencc import OpenCC  # noqa: PLC0415 - optional dependency

            self._converter = OpenCC("s2hk")
        except Exception:
            log.info("opencc not available; leaving Qwen's characters as they are")

    def _finish_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if self._converter is not None:
            try:
                text = self._converter.convert(text)
            except Exception:
                log.exception("traditional conversion failed; using the original")
        return text


# ----------------------------------------------------------------- helpers


def _rms(chunk: bytes) -> float:
    if len(chunk) < 4:
        return 0.0
    if len(chunk) % 2:
        chunk = chunk[:-1]
    samples = array.array("h")
    samples.frombytes(chunk)
    total = 0
    for s in samples:
        total += s * s
    return math.sqrt(total / len(samples))


def _wav_base64(pcm: bytes, sample_rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_text(payload) -> str | None:
    """The transcript out of whatever shape OpenRouter used; None if unknown.

    The documented shape is {"text": ...}. Some providers behind OpenRouter
    have answered in chat-completion form instead, and an empty transcript is
    a legitimate answer for silence, so "" and None mean different things.
    """
    if not isinstance(payload, dict):
        return None
    for key in ("text", "transcript", "transcription"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
    segments = payload.get("segments")
    if isinstance(segments, list) and segments and all(isinstance(s, dict) for s in segments):
        return " ".join(str(s.get("text", "")) for s in segments).strip()
    return None


def _describe_shape(payload) -> str:
    if isinstance(payload, dict):
        return "object with keys " + ", ".join(sorted(map(str, payload.keys()))[:12])
    return f"{type(payload).__name__}: {str(payload)[:120]}"


def _describe_http_error(resp) -> str:
    detail = ""
    try:
        payload = resp.json()
        err = payload.get("error") if isinstance(payload, dict) else None
        detail = (err.get("message") if isinstance(err, dict) else str(err or "")) or ""
    except ValueError:
        detail = (resp.text or "")[:200]
    hints = {
        401: "OpenRouter rejected the API key",
        402: "OpenRouter says the account has no credit left",
        404: "OpenRouter does not know this model -- check OPENROUTER_ASR_MODEL",
        429: "OpenRouter is rate limiting this key",
    }
    hint = hints.get(resp.status_code, f"OpenRouter returned HTTP {resp.status_code}")
    return f"{hint}{': ' + detail if detail else ''}"
