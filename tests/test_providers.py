"""Tests for the three transcription providers and per-line speaker correction.

The local engine is exercised against the REAL sherpa-onnx model when it is
present, using the Cantonese and code-switching samples that ship with it. Those
tests skip themselves when the model has not been downloaded, so a fresh clone
still runs green -- but when the model is there, this is genuine end-to-end
proof rather than a mock agreeing with itself.

Run with:  python -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

_TMP = tempfile.TemporaryDirectory()
config.DATA_DIR = Path(_TMP.name)
config.AUDIO_DIR = config.DATA_DIR / "audio"
config.DB_PATH = config.DATA_DIR / "providers.sqlite3"

import stt  # noqa: E402
from copilot.brief import Attendee, Brief  # noqa: E402
from copilot.state import MeetingState  # noqa: E402
from stt.base import Utterance  # noqa: E402
from stt.speechmatics_live import SpeechmaticsLiveSTT  # noqa: E402
from storage import db, export  # noqa: E402


def wait_for(predicate, timeout=30.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# The spike downloaded the model here; SHERPA_MODEL_DIR overrides it in real use.
_SCRATCH = Path(
    "/tmp/claude-0/-home-user-STTmeeting/fca95b25-e06d-56a0-8fec-a64b78cb788c/scratchpad"
)
MODEL_DIR = Path(
    getattr(config, "SHERPA_MODEL_DIR", "")
) if Path(getattr(config, "SHERPA_MODEL_DIR", "/nonexistent")).exists() else (
    _SCRATCH / "sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en"
)
PUNCT_DIR = _SCRATCH / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"

HAVE_MODEL = (MODEL_DIR / "encoder.int8.onnx").exists()


def read_wav_bytes(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2
        return w.readframes(w.getnframes()), w.getframerate()


# --------------------------------------------------------------- the factory


class TestProviderFactory(unittest.TestCase):
    def _make(self, provider):
        return stt.create_engine(
            provider=provider, sample_rate=16000, language="zh-HK", model="nova-3",
            keyterms=["Falcon"], on_interim=None, on_utterance=None,
            on_status=None, on_error=None,
        )

    def test_each_name_builds_its_own_engine(self):
        self.assertEqual(self._make("deepgram").provider, "deepgram")
        self.assertEqual(self._make("speechmatics").provider, "speechmatics")

    def test_an_unknown_provider_falls_back_to_deepgram(self):
        self.assertEqual(self._make("nonsense").provider, "deepgram")
        self.assertEqual(self._make("").provider, "deepgram")

    def test_every_offered_provider_is_buildable(self):
        for choice in config.STT_PROVIDER_CHOICES:
            code = choice["code"]
            if code == "local" and not HAVE_MODEL:
                continue  # constructing it is fine, but keep this test honest
            engine = self._make(code)
            self.assertTrue(engine.provider)
            self.assertIsInstance(engine.usd_per_minute, float)

    def test_a_local_engine_is_free_and_the_cloud_ones_are_not(self):
        self.assertEqual(config.STT_PROVIDER_CHOICES[2]["code"], "local")
        self.assertGreater(self._make("speechmatics").usd_per_minute, 0)
        self.assertGreater(config.DEEPGRAM_USD_PER_MINUTE, 0)


class TestProviderKeys(unittest.TestCase):
    def setUp(self):
        self.saved = (config.DEEPGRAM_API_KEY, config.SPEECHMATICS_API_KEY,
                      config.OPENROUTER_API_KEY)

        def restore():
            (config.DEEPGRAM_API_KEY, config.SPEECHMATICS_API_KEY,
             config.OPENROUTER_API_KEY) = self.saved

        self.addCleanup(restore)
        config.OPENROUTER_API_KEY = "or"

    def test_local_needs_no_speech_key(self):
        config.DEEPGRAM_API_KEY = ""
        config.SPEECHMATICS_API_KEY = ""
        self.assertEqual(config.missing_keys("local"), [])

    def test_each_cloud_provider_asks_only_for_its_own_key(self):
        config.DEEPGRAM_API_KEY = ""
        config.SPEECHMATICS_API_KEY = "sm"
        self.assertEqual(config.missing_keys("deepgram"), ["DEEPGRAM_API_KEY"])
        self.assertEqual(config.missing_keys("speechmatics"), [])

    def test_the_llm_key_is_always_required(self):
        config.OPENROUTER_API_KEY = ""
        self.assertIn("OPENROUTER_API_KEY", config.missing_keys("local"))


# ------------------------------------------------------------- Speechmatics


class TestSpeechmaticsProtocol(unittest.TestCase):
    def setUp(self):
        self.interims: list[str] = []
        self.utterances: list[Utterance] = []
        self.errors: list[str] = []
        self.stt = SpeechmaticsLiveSTT(
            api_key="sm-test", sample_rate=16000, language="yue",
            keyterms=["Falcon", "NocolyHAP"],
            on_interim=self.interims.append,
            on_utterance=self.utterances.append,
            on_error=self.errors.append,
        )

    @staticmethod
    def transcript(text, speaker="S1", start=1.0, end=2.0, partial=False):
        return {
            "message": "AddPartialTranscript" if partial else "AddTranscript",
            "metadata": {"transcript": text, "start_time": start, "end_time": end},
            "results": [
                {
                    "type": "word",
                    "start_time": start,
                    "end_time": end,
                    "alternatives": [{"content": text, "speaker": speaker}],
                }
            ],
        }

    def test_start_message_carries_language_diarization_and_vocabulary(self):
        msg = self.stt._start_message()
        self.assertEqual(msg["message"], "StartRecognition")
        self.assertEqual(msg["transcription_config"]["language"], "yue")
        self.assertEqual(msg["transcription_config"]["diarization"], "speaker")
        self.assertTrue(msg["transcription_config"]["enable_partials"])
        vocab = [v["content"] for v in msg["transcription_config"]["additional_vocab"]]
        self.assertEqual(vocab, ["Falcon", "NocolyHAP"])
        self.assertEqual(msg["audio_format"]["encoding"], "pcm_s16le")
        self.assertEqual(msg["audio_format"]["sample_rate"], 16000)

    def test_partials_become_interims_and_finals_become_utterances(self):
        self.stt._handle_message(self.transcript("我想爭取", partial=True))
        self.assertEqual(self.interims[-1], "我想爭取")
        self.assertEqual(self.utterances, [])

        self.stt._handle_message(self.transcript("我想爭取多兩個 headcount"))
        self.assertEqual(len(self.utterances), 1)
        self.assertEqual(self.utterances[0].text, "我想爭取多兩個 headcount")

    def test_speaker_labels_map_to_stable_integers(self):
        self.stt._handle_message(self.transcript("first", speaker="S1"))
        self.stt._handle_message(self.transcript("second", speaker="S2"))
        self.stt._handle_message(self.transcript("third", speaker="S1"))
        self.assertEqual([u.speaker for u in self.utterances], [0, 1, 0])

    def test_an_unknown_speaker_is_left_unattributed(self):
        self.stt._handle_message(self.transcript("mystery", speaker="UU"))
        self.assertIsNone(self.utterances[0].speaker)

    def test_transcript_falls_back_to_joining_words(self):
        msg = {
            "message": "AddTranscript",
            "results": [
                {"alternatives": [{"content": "hello", "speaker": "S1"}]},
                {"alternatives": [{"content": "world", "speaker": "S1"}]},
            ],
        }
        self.stt._handle_message(msg)
        self.assertEqual(self.utterances[0].text, "hello world")

    def test_recognition_started_proves_the_connection(self):
        statuses = []
        self.stt._on_status = lambda **kw: statuses.append(kw)
        self.stt._handle_message({"message": "RecognitionStarted"})
        self.assertTrue(self.stt._proven)
        self.assertEqual(statuses[0]["state"], "listening")

    def test_errors_are_surfaced(self):
        self.stt._handle_message({"message": "Error", "reason": "bad language"})
        self.assertEqual(self.errors, ["Speechmatics: bad language"])

    def test_empty_transcripts_are_ignored(self):
        self.stt._handle_message(self.transcript(""))
        self.assertEqual(self.utterances, [])

    def test_cantonese_maps_to_yue_and_mandarin_to_cmn(self):
        self.assertEqual(config.speechmatics_language("zh-HK"), "yue")
        self.assertEqual(config.speechmatics_language("zh-CN"), "cmn")
        self.assertEqual(config.speechmatics_language("en"), "en")


# ------------------------------------------------- the real local model


@unittest.skipUnless(HAVE_MODEL, f"local model not downloaded to {MODEL_DIR}")
class TestLocalModelForReal(unittest.TestCase):
    """Runs the actual streaming Paraformer over the actual test audio."""

    @classmethod
    def setUpClass(cls):
        from stt.sherpa_local import SherpaLocalSTT

        cls.engine_class = SherpaLocalSTT

    def _transcribe(self, wav_name, punctuate=False, to_traditional=True,
                    trailing_silence=0.0, stop_first=True):
        """Transcribe a test clip.

        By default it stops the engine before collecting results, because these
        clips end abruptly: an utterance is only flushed at an endpoint (trailing
        silence) or at stop, so a clip with no silence at the end legitimately
        produces nothing until the meeting ends. Pass `trailing_silence` to
        exercise the mid-stream endpoint path instead.
        """
        utterances: list[Utterance] = []
        interims: list[str] = []
        statuses: list[dict] = []
        errors: list[str] = []

        engine = self.engine_class(
            model_dir=MODEL_DIR,
            sample_rate=16000,
            num_threads=2,
            to_traditional=to_traditional,
            punctuation_dir=PUNCT_DIR if punctuate else None,
            on_interim=interims.append,
            on_utterance=utterances.append,
            on_status=lambda **kw: statuses.append(kw),
            on_error=errors.append,
        )
        engine.start()
        self.assertTrue(
            wait_for(lambda: any(s.get("state") == "listening" for s in statuses), 90),
            f"model never became ready: {statuses} {errors}",
        )

        raw, rate = read_wav_bytes(MODEL_DIR / "test_wavs" / wav_name)
        self.assertEqual(rate, 16000)
        if trailing_silence:
            raw += b"\x00\x00" * int(16000 * trailing_silence)
        step = 16000 * 2 // 10  # 100 ms of 16-bit mono, as the browser sends
        for i in range(0, len(raw), step):
            engine.send_audio(raw[i:i + step])

        if stop_first:
            self.assertTrue(wait_for(lambda: engine._audio.empty(), 60), "audio not consumed")
            engine.stop()
            engine.join(30)
        self.assertTrue(wait_for(lambda: utterances, 60), f"nothing transcribed: {errors}")
        if not stop_first:
            engine.stop()
            engine.join(30)
        self.assertEqual(errors, [])
        return utterances, interims, engine

    def test_transcribes_cantonese(self):
        utterances, interims, engine = self._transcribe("1.wav")
        text = " ".join(u.text for u in utterances)
        # "有冇人知道灣仔活道係點去" -- the place names are the checkable part.
        self.assertIn("灣仔", text, text)
        self.assertIn("活道", text, text)
        self.assertTrue(interims, "there should be interim updates while decoding")

    def test_converts_simplified_output_to_hong_kong_traditional(self):
        utterances, _, _ = self._transcribe("2.wav", to_traditional=True)
        text = " ".join(u.text for u in utterances)
        self.assertIn("黃大仙", text, text)   # 黄 -> 黃
        self.assertIn("九龍塘", text, text)   # 龙 -> 龍
        self.assertNotIn("黄大仙", text, "should not still be simplified")

    def test_leaves_simplified_alone_when_conversion_is_off(self):
        utterances, _, _ = self._transcribe("2.wav", to_traditional=False)
        text = " ".join(u.text for u in utterances)
        self.assertIn("黄大仙", text, text)

    def test_handles_chinese_english_code_switching(self):
        utterances, _, _ = self._transcribe("6-zh-en.wav")
        text = " ".join(u.text for u in utterances).lower()
        self.assertIn("yesterday", text, text)
        self.assertTrue(any("星期" in u.text for u in utterances), text)

    def test_punctuation_model_adds_marks(self):
        if not (PUNCT_DIR / "model.onnx").exists():
            self.skipTest("punctuation model not downloaded")
        plain, _, _ = self._transcribe("1.wav", punctuate=False)
        marked, _, _ = self._transcribe("1.wav", punctuate=True)
        plain_text = " ".join(u.text for u in plain)
        marked_text = " ".join(u.text for u in marked)
        self.assertFalse(any(c in plain_text for c in "，。？"), plain_text)
        self.assertTrue(any(c in marked_text for c in "，。？"), marked_text)

    def test_emits_an_utterance_at_a_pause_without_being_stopped(self):
        """The live path: a pause in the middle of a meeting must close an
        utterance, or the copilot would never be triggered until the end."""
        utterances, _, _ = self._transcribe(
            "1.wav", trailing_silence=1.5, stop_first=False
        )
        self.assertTrue(utterances)
        self.assertIn("灣仔", " ".join(u.text for u in utterances))

    def test_reports_no_speaker_because_it_does_not_diarise(self):
        utterances, _, _ = self._transcribe("1.wav")
        self.assertTrue(all(u.speaker is None for u in utterances))

    def test_costs_nothing_and_counts_the_audio_it_heard(self):
        utterances, _, engine = self._transcribe("1.wav")
        self.assertEqual(engine.usd_per_minute, 0.0)
        # 1.wav is about six seconds long.
        self.assertGreater(engine.audio_seconds, 4)
        self.assertLess(engine.audio_seconds, 8)
        self.assertTrue(engine.model.startswith("local:"))

    def test_keeps_up_comfortably_with_real_time(self):
        raw, _ = read_wav_bytes(MODEL_DIR / "test_wavs" / "3-sichuan.wav")
        duration = len(raw) / (16000 * 2)
        started = time.time()
        self._transcribe("3-sichuan.wav")
        elapsed = time.time() - started
        # Loading the model dominates a single short clip, so this is a loose
        # bound; the point is that decoding is not the bottleneck.
        self.assertLess(elapsed, duration + 60, f"{elapsed:.1f}s for {duration:.1f}s audio")


@unittest.skipUnless(HAVE_MODEL, "local model not downloaded")
class TestLocalModelFailsClearly(unittest.TestCase):
    def test_a_missing_model_directory_says_what_to_do(self):
        from stt.sherpa_local import SherpaLocalSTT

        errors: list[str] = []
        engine = SherpaLocalSTT(
            model_dir=Path("/nonexistent/model"), sample_rate=16000,
            on_error=errors.append,
        )
        engine.start()
        self.assertTrue(wait_for(lambda: errors, 20))
        self.assertIn("download_models.py", errors[0])


# --------------------------------------------- per-line speaker correction


class TestPerLineSpeaker(unittest.TestCase):
    def setUp(self):
        db.init()
        self.state = MeetingState(
            meeting_id=1,
            brief=Brief(attendees=[Attendee(name="Alan"), Attendee(name="Wing")]),
        )

    def test_a_line_name_beats_the_voice_name(self):
        self.state.add_utterance("first", 0)
        self.state.add_utterance("second", 0)
        self.state.set_speaker_name(0, "Alan")
        self.assertEqual(self.state.segments[0].label(self.state.speaker_names), "Alan")

        self.state.set_segment_speaker(1, "Wing")
        names = self.state.speaker_names
        self.assertEqual(self.state.segments[0].label(names), "Alan")
        self.assertEqual(self.state.segments[1].label(names), "Wing")

    def test_the_override_reaches_the_prompt(self):
        self.state.add_utterance("hello", 0)
        self.state.set_segment_speaker(0, "Kelvin")
        self.assertIn("Kelvin: hello", self.state.text_from(0))

    def test_clearing_the_override_returns_to_the_voice_name(self):
        self.state.add_utterance("hello", 0)
        self.state.set_speaker_name(0, "Alan")
        self.state.set_segment_speaker(0, "Wing")
        self.state.set_segment_speaker(0, "")
        self.assertEqual(self.state.segments[0].label(self.state.speaker_names), "Alan")

    def test_an_unknown_line_is_refused_rather_than_creating_one(self):
        self.assertIsNone(self.state.set_segment_speaker(5, "Alan"))
        self.assertIsNone(self.state.set_segment_speaker(-1, "Alan"))
        self.assertEqual(self.state.segments, [])

    def test_an_unattributed_line_can_still_be_named(self):
        """The local engine reports no speaker at all, so this is the only way
        to attribute its lines."""
        self.state.add_utterance("from nowhere", None)
        self.assertEqual(self.state.segments[0].label(), "?")
        self.state.set_segment_speaker(0, "Alan")
        self.assertEqual(self.state.segments[0].label(), "Alan")

    def test_overrides_survive_to_the_exports(self):
        mid = db.create_meeting("Override", {}, "zh-HK", "nova-3")
        db.add_segment(mid, 0, 0.0, 0, "line one")
        db.add_segment(mid, 1, 1.0, 0, "line two")
        db.save_speakers(mid, {0: "Alan"})
        db.set_segment_speaker(mid, 1, "Wing")

        meeting = db.get_meeting(mid)
        markdown = export.to_markdown(meeting)
        self.assertIn("**Alan**: line one", markdown)
        self.assertIn("**Wing**: line two", markdown)

        payload = json.loads(export.to_json(meeting))
        self.assertEqual(
            [t["speaker_label"] for t in payload["transcript"]], ["Alan", "Wing"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
