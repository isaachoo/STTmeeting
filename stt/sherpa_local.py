"""Local, offline speech-to-text with sherpa-onnx.

Runs a streaming Paraformer on the CPU, so audio never leaves the machine and
transcription costs nothing. The model
`sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en` covers Mandarin,
Cantonese and English in one pass, which is the mix actually spoken in a Hong
Kong office.

Measured on 2 CPU threads: real-time factor around 0.09, i.e. roughly eleven
times faster than the audio arrives, so keeping up is not a concern.

Three things this model does not do, and how they are handled here:

* No punctuation. An optional CT-Transformer punctuation model is applied per
  utterance when its directory is configured.
* Simplified characters only, even for Cantonese speech (it writes 有無 as 有无).
  OpenCC converts to Hong Kong traditional when available, which is a
  deterministic mapping rather than something to ask an LLM to guess at.
* No diarisation, so every utterance comes back with `speaker=None`. The UI
  falls back to an unlabelled tag; there is nothing to guess wrong.

sherpa-onnx and its friends are optional dependencies, imported lazily, so the
app still runs with only the cloud providers installed.
"""

import logging
import queue
import threading
import time

from .base import STTEngine, Utterance

log = logging.getLogger(__name__)

MODEL_SAMPLE_RATE = 16000
FEATURE_DIM = 80
QUEUE_TIMEOUT = 0.2
MAX_QUEUED_CHUNKS = 400


class SherpaLocalSTT(STTEngine):
    def __init__(
        self,
        model_dir,
        sample_rate: int,
        num_threads: int = 2,
        to_traditional: bool = True,
        punctuation_dir=None,
        on_interim=None,
        on_utterance=None,
        on_status=None,
        on_error=None,
    ):
        self.model_dir = model_dir
        self.sample_rate = int(sample_rate)
        self.num_threads = max(1, int(num_threads))
        self.to_traditional = to_traditional
        self.punctuation_dir = punctuation_dir

        self._on_interim = on_interim or (lambda text: None)
        self._on_utterance = on_utterance or (lambda utt: None)
        self._on_status = on_status or (lambda **kw: None)
        self._on_error = on_error or (lambda msg: None)

        self._audio: queue.Queue[bytes | None] = queue.Queue(maxsize=MAX_QUEUED_CHUNKS)
        self._stopping = threading.Event()
        self._worker: threading.Thread | None = None

        self._bytes_sent = 0
        self._dropped_chunks = 0
        self._samples_fed = 0  # this model has no timestamps; count them ourselves
        self._utterance_start: float | None = None

        self._recognizer = None
        self._punctuation = None
        self._converter = None

    # ------------------------------------------------------------------ public

    @property
    def provider(self) -> str:
        return "sherpa-local"

    @property
    def model(self) -> str:
        name = getattr(self.model_dir, "name", str(self.model_dir))
        return f"local:{name}"

    @property
    def usd_per_minute(self) -> float:
        return 0.0  # the entire point

    @property
    def audio_seconds(self) -> float:
        return self._samples_fed / MODEL_SAMPLE_RATE if self._samples_fed else 0.0

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stopping.clear()
        self._worker = threading.Thread(
            target=self._run, name="sherpa-local", daemon=True
        )
        self._worker.start()

    def send_audio(self, chunk: bytes) -> None:
        if self._stopping.is_set() or not chunk:
            return
        try:
            self._audio.put_nowait(chunk)
        except queue.Full:
            # Drop the oldest: if decoding ever falls behind, the freshest audio
            # is the audio worth keeping.
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

    def join(self, timeout: float = 10.0) -> None:
        if self._worker is not None:
            self._worker.join(timeout)

    # ------------------------------------------------------------------ loading

    def _load(self) -> bool:
        """Build the recognizer. Returns False after reporting why it could not."""
        try:
            import sherpa_onnx  # noqa: PLC0415 - optional dependency, loaded on use
        except ImportError:
            self._on_error(
                "Local transcription needs the sherpa-onnx package. Install it with "
                "`pip install -r requirements-local.txt`, or set STT_PROVIDER back "
                "to deepgram."
            )
            return False

        encoder = self.model_dir / "encoder.int8.onnx"
        decoder = self.model_dir / "decoder.int8.onnx"
        tokens = self.model_dir / "tokens.txt"
        missing = [p.name for p in (encoder, decoder, tokens) if not p.exists()]
        if missing:
            self._on_error(
                f"The local model is missing {', '.join(missing)} in {self.model_dir}. "
                "Run `python scripts/download_models.py` to fetch it."
            )
            return False

        self._on_status(state="loading", detail="loading the local model (a few seconds)")
        started = time.time()
        try:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
                tokens=str(tokens),
                encoder=str(encoder),
                decoder=str(decoder),
                num_threads=self.num_threads,
                sample_rate=MODEL_SAMPLE_RATE,
                feature_dim=FEATURE_DIM,
                # Endpointing decides where one utterance ends and the next
                # begins, which is what the copilot triggers on.
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=0.8,
                rule3_min_utterance_length=20.0,
                provider="cpu",
            )
        except Exception as exc:  # noqa: BLE001 - reported to the user as-is
            self._on_error(f"Could not load the local model: {exc}")
            return False
        log.info("local model loaded in %.1fs", time.time() - started)

        self._load_punctuation(sherpa_onnx)
        self._load_converter()
        return True

    def _load_punctuation(self, sherpa_onnx) -> None:
        """Optional. Streaming Paraformer emits no punctuation, which hurts both
        readability and the LLM's parsing of long unbroken text."""
        if not self.punctuation_dir:
            return
        model = self.punctuation_dir / "model.onnx"
        if not model.exists():
            log.info("no punctuation model at %s; continuing without", model)
            return
        try:
            self._punctuation = sherpa_onnx.OfflinePunctuation(
                sherpa_onnx.OfflinePunctuationConfig(
                    model=sherpa_onnx.OfflinePunctuationModelConfig(
                        ct_transformer=str(model), num_threads=1, provider="cpu"
                    )
                )
            )
            log.info("punctuation model loaded")
        except Exception:
            log.exception("could not load the punctuation model; continuing without")

    def _load_converter(self) -> None:
        """Optional. The model writes simplified characters even for Cantonese."""
        if not self.to_traditional:
            return
        try:
            from opencc import OpenCC  # noqa: PLC0415 - optional dependency

            self._converter = OpenCC("s2hk")
        except Exception:
            log.info("opencc not available; leaving simplified characters as they are")

    # ------------------------------------------------------------- decode loop

    def _run(self) -> None:
        if not self._load():
            self._on_status(state="closed")
            return

        recognizer = self._recognizer
        stream = recognizer.create_stream()
        self._on_status(state="listening", model=self.model, detail="offline, on this machine")
        last_interim = ""

        try:
            while True:
                chunk = self._take_chunk()
                if chunk is None:  # stop requested
                    break

                samples = self._to_float32(chunk)
                if samples is None:
                    continue
                if self._utterance_start is None:
                    self._utterance_start = self.audio_seconds
                self._samples_fed += len(samples)
                stream.accept_waveform(MODEL_SAMPLE_RATE, samples)

                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                text = recognizer.get_result(stream)
                if text != last_interim:
                    last_interim = text
                    self._on_interim(self._finish_text(text, punctuate=False))

                if recognizer.is_endpoint(stream):
                    self._flush(text)
                    recognizer.reset(stream)
                    last_interim = ""

            # Stopping: decode whatever is left so the last sentence is not lost.
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            self._flush(recognizer.get_result(stream))
        except Exception as exc:  # noqa: BLE001 - a crash here must be visible
            log.exception("local transcription failed")
            self._on_error(f"Local transcription stopped: {exc}")
        finally:
            self._on_status(state="closed")

    def _take_chunk(self) -> bytes | None:
        """Next audio chunk, or None when it is time to stop."""
        while True:
            try:
                chunk = self._audio.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                if self._stopping.is_set():
                    return None
                continue
            if chunk is None:
                return None
            self._bytes_sent += len(chunk)
            return chunk

    def _to_float32(self, chunk: bytes):
        """16-bit PCM bytes to the float32 the model expects, resampled if needed."""
        import numpy as np  # noqa: PLC0415 - arrives with sherpa-onnx

        if len(chunk) < 2:
            return None
        if len(chunk) % 2:
            chunk = chunk[:-1]  # a split sample would shift every one after it
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

        if self.sample_rate != MODEL_SAMPLE_RATE and len(samples) > 1:
            # Linear resampling is plenty for speech, and the browser normally
            # hands us 16 kHz anyway so this rarely runs.
            target_len = max(1, round(len(samples) * MODEL_SAMPLE_RATE / self.sample_rate))
            samples = np.interp(
                np.linspace(0, len(samples) - 1, target_len, dtype=np.float32),
                np.arange(len(samples), dtype=np.float32),
                samples,
            ).astype(np.float32)
        return samples

    def _flush(self, text: str) -> None:
        text = (text or "").strip()
        start = self._utterance_start
        self._utterance_start = None
        if not text:
            return
        self._on_interim("")
        self._on_utterance(
            Utterance(
                text=self._finish_text(text, punctuate=True),
                speaker=None,  # this model does not diarise
                start=start,
                end=self.audio_seconds,
            )
        )

    def _finish_text(self, text: str, punctuate: bool) -> str:
        """Punctuate, then convert to Hong Kong traditional.

        That order matters: the punctuation model was trained on simplified
        text, so it must see the raw output rather than a converted version.
        Interim text skips punctuation -- it changes on every update and the
        extra pass would be wasted work.
        """
        text = (text or "").strip()
        if not text:
            return ""
        if punctuate and self._punctuation is not None:
            try:
                text = self._punctuation.add_punctuation(text)
            except Exception:
                log.exception("punctuation failed; using the unpunctuated text")
        if self._converter is not None:
            try:
                text = self._converter.convert(text)
            except Exception:
                log.exception("traditional conversion failed; using the original")
        return text
