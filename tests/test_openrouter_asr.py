"""Qwen3-ASR through OpenRouter: the segmenter, the request, the ordering.

No network. A fake `requests.Session` records what would have been posted and
answers with canned transcripts, so these tests prove the audio is cut where it
should be, wrapped as a valid WAV, and that lines come back in speech order even
when the API answers out of order.

Run with:  python -m unittest tests.test_openrouter_asr -v
"""

import base64
import io
import math
import struct
import sys
import threading
import time
import unittest
import wave
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import stt  # noqa: E402
from stt import openrouter_asr  # noqa: E402
from stt.base import Utterance  # noqa: E402
from stt.openrouter_asr import OpenRouterASR  # noqa: E402

RATE = 16000


def wait_for(predicate, timeout=10.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def tone(seconds: float, amplitude: int = 8000, freq: float = 220.0) -> bytes:
    n = int(seconds * RATE)
    return struct.pack(
        f"<{n}h", *(int(amplitude * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n))
    )


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * RATE)


def in_chunks(pcm: bytes, ms: int = 100):
    step = RATE * 2 * ms // 1000
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Answers each POST with the next scripted reply; records every body."""

    def __init__(self, replies=None, delays=None):
        self.calls: list[dict] = []
        self.replies = list(replies or [])
        self.delays = list(delays or [])
        self.lock = threading.Lock()

    def post(self, url, headers=None, json=None, timeout=None):
        with self.lock:
            index = len(self.calls)
            self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            reply = self.replies[index] if index < len(self.replies) else None
            delay = self.delays[index] if index < len(self.delays) else 0
        if delay:
            time.sleep(delay)
        if isinstance(reply, Exception):
            raise reply
        if reply is None:
            return FakeResponse(200, {"text": f"line {index + 1}", "usage": {"cost": 0.0001}})
        return reply


def make_engine(session, **kw):
    interims, utterances, statuses, errors = [], [], [], []
    engine = OpenRouterASR(
        api_key="or-test", sample_rate=RATE, model="qwen/test-asr",
        base_url="https://example.test/api/v1", usd_per_minute=0.0021,
        to_traditional=kw.pop("to_traditional", False),
        on_interim=interims.append, on_utterance=utterances.append,
        on_status=lambda **s: statuses.append(s), on_error=errors.append,
        session=session, **kw,
    )
    return engine, interims, utterances, statuses, errors


def run_audio(engine, pcm, trailing=0.0):
    engine.start()
    for chunk in in_chunks(pcm + silence(trailing)):
        engine.send_audio(chunk)
    assert wait_for(lambda: engine._audio.empty(), 10), "audio not consumed"


def finish(engine):
    engine.stop()
    engine.join(20)


# --------------------------------------------------------------- the factory


class TestFactoryAndConfig(unittest.TestCase):
    def test_qwen_is_offered_and_builds_the_openrouter_engine(self):
        codes = [c["code"] for c in config.STT_PROVIDER_CHOICES]
        self.assertIn("qwen", codes)
        engine = stt.create_engine(
            provider="qwen", sample_rate=16000, language="zh-HK", model="",
            keyterms=[], on_interim=None, on_utterance=None, on_status=None, on_error=None,
        )
        self.assertIsInstance(engine, OpenRouterASR)
        self.assertEqual(engine.provider, "qwen")
        self.assertTrue(engine.model.startswith("openrouter:qwen/"))
        self.assertAlmostEqual(engine.usd_per_minute, config.OPENROUTER_ASR_USD_PER_MINUTE)

    def test_qwen_needs_only_the_openrouter_key(self):
        saved = config.DEEPGRAM_API_KEY, config.SPEECHMATICS_API_KEY, config.OPENROUTER_API_KEY
        try:
            config.DEEPGRAM_API_KEY = config.SPEECHMATICS_API_KEY = ""
            config.OPENROUTER_API_KEY = "or"
            self.assertEqual(config.missing_keys("qwen"), [])
            config.OPENROUTER_API_KEY = ""
            self.assertEqual(config.missing_keys("qwen"), ["OPENROUTER_API_KEY"])
        finally:
            config.DEEPGRAM_API_KEY, config.SPEECHMATICS_API_KEY, config.OPENROUTER_API_KEY = saved


# ------------------------------------------------------------- the request


class TestRequest(unittest.TestCase):
    def test_a_segment_is_posted_as_base64_wav_to_the_transcription_endpoint(self):
        session = FakeSession(replies=[FakeResponse(200, {"text": "我想爭取多兩個 headcount"})])
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, tone(2.0), trailing=1.0)
        self.assertTrue(wait_for(lambda: utterances, 10), errors)
        finish(engine)

        self.assertEqual(len(session.calls), 1)
        call = session.calls[0]
        self.assertEqual(call["url"], "https://example.test/api/v1/audio/transcriptions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer or-test")
        body = call["json"]
        self.assertEqual(body["model"], "qwen/test-asr")
        self.assertEqual(body["input_audio"]["format"], "wav")

        raw = base64.b64decode(body["input_audio"]["data"])
        with wave.open(io.BytesIO(raw)) as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), RATE)
            seconds = w.getnframes() / RATE
        # The two seconds of speech plus the pause that closed it, give or take
        # the pre-roll; nowhere near the whole three seconds of silence.
        self.assertGreater(seconds, 2.0)
        self.assertLess(seconds, 3.5)

        self.assertEqual(utterances[0].text, "我想爭取多兩個 headcount")
        self.assertIsNone(utterances[0].speaker, "Qwen does not diarise")
        self.assertAlmostEqual(utterances[0].start, 0.0, delta=0.6)
        self.assertGreater(utterances[0].end, utterances[0].start)
        self.assertEqual(errors, [])
        self.assertEqual(statuses[0]["state"], "listening")
        self.assertEqual(statuses[-1]["state"], "closed")
        self.assertAlmostEqual(engine.audio_seconds, 3.0, delta=0.15)

    def test_silence_alone_costs_no_request(self):
        session = FakeSession()
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, silence(4.0))
        finish(engine)
        self.assertEqual(session.calls, [])
        self.assertEqual(utterances, [])

    def test_a_pause_splits_two_sentences_into_two_requests(self):
        session = FakeSession(replies=[
            FakeResponse(200, {"text": "第一句"}), FakeResponse(200, {"text": "第二句"}),
        ])
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, tone(1.5) + silence(1.2) + tone(1.5), trailing=1.2)
        self.assertTrue(wait_for(lambda: len(utterances) == 2, 10), (utterances, errors))
        finish(engine)
        self.assertEqual([u.text for u in utterances], ["第一句", "第二句"])
        self.assertGreater(utterances[1].start, utterances[0].end - 0.5)

    def test_a_long_monologue_is_cut_at_the_ceiling(self):
        session = FakeSession()
        engine, interims, utterances, statuses, errors = make_engine(
            session, max_segment_seconds=3.0
        )
        run_audio(engine, tone(7.0), trailing=1.0)
        self.assertTrue(wait_for(lambda: len(utterances) == 3, 10), (utterances, errors))
        finish(engine)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual([u.text for u in utterances], ["line 1", "line 2", "line 3"])

    def test_the_last_words_are_sent_on_stop_even_without_a_pause(self):
        session = FakeSession(replies=[FakeResponse(200, {"text": "散會"})])
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, tone(1.2))  # ends abruptly, mid-speech
        finish(engine)
        self.assertEqual([u.text for u in utterances], ["散會"])

    def test_the_interim_line_says_it_is_listening_and_clears_at_the_end(self):
        session = FakeSession()
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, tone(2.0), trailing=1.0)
        self.assertTrue(wait_for(lambda: utterances, 10))
        finish(engine)
        self.assertTrue(any("聽到" in t for t in interims), interims)
        self.assertTrue(any("轉寫中" in t for t in interims), interims)
        self.assertEqual(interims[-1], "")


class TestOrderingAndFailures(unittest.TestCase):
    def test_results_come_out_in_speech_order_even_if_the_api_answers_backwards(self):
        # First segment is slow, second is instant: the second must still wait.
        session = FakeSession(
            replies=[FakeResponse(200, {"text": "first"}), FakeResponse(200, {"text": "second"})],
            delays=[0.8, 0.0],
        )
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, tone(1.5) + silence(1.2) + tone(1.5), trailing=1.2)
        self.assertTrue(wait_for(lambda: len(utterances) == 2, 10), (utterances, errors))
        finish(engine)
        self.assertEqual([u.text for u in utterances], ["first", "second"])

    def test_a_server_error_is_retried_then_succeeds(self):
        session = FakeSession(replies=[
            FakeResponse(503, {"error": {"message": "busy"}}),
            FakeResponse(200, {"text": "after retry"}),
        ])
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, tone(1.5), trailing=1.0)
        self.assertTrue(wait_for(lambda: utterances, 15), errors)
        finish(engine)
        self.assertEqual(utterances[0].text, "after retry")
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(errors, [])

    def test_a_bad_key_is_reported_once_and_the_meeting_goes_on(self):
        session = FakeSession(replies=[
            FakeResponse(401, {"error": {"message": "No auth credentials found"}}),
            FakeResponse(200, {"text": "still alive"}),
        ])
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, tone(1.5) + silence(1.2) + tone(1.5), trailing=1.2)
        self.assertTrue(wait_for(lambda: utterances, 10), errors)
        finish(engine)
        self.assertEqual(len(errors), 1)
        self.assertIn("rejected the API key", errors[0])
        self.assertIn("No auth credentials", errors[0])
        self.assertEqual([u.text for u in utterances], ["still alive"])
        self.assertEqual(engine.failures, 1)

    def test_a_network_failure_is_retried_and_then_dropped_with_a_message(self):
        boom = requests.ConnectionError("dns")
        session = FakeSession(replies=[boom, boom, boom])
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, tone(1.5), trailing=1.0)
        self.assertTrue(wait_for(lambda: errors, 20))
        finish(engine)
        self.assertEqual(len(session.calls), 1 + openrouter_asr.RETRIES)
        self.assertIn("could not reach OpenRouter", errors[0])
        self.assertEqual(utterances, [])

    def test_an_empty_transcript_is_not_a_line(self):
        session = FakeSession(replies=[FakeResponse(200, {"text": "   "})])
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, tone(1.5), trailing=1.0)
        finish(engine)
        self.assertEqual(utterances, [])
        self.assertEqual(errors, [])

    def test_reported_cost_is_accumulated(self):
        session = FakeSession(replies=[
            FakeResponse(200, {"text": "a", "usage": {"cost": 0.0003}}),
            FakeResponse(200, {"text": "b", "usage": {"cost": 0.0004}}),
        ])
        engine, interims, utterances, statuses, errors = make_engine(session)
        run_audio(engine, tone(1.5) + silence(1.2) + tone(1.5), trailing=1.2)
        self.assertTrue(wait_for(lambda: len(utterances) == 2, 10))
        finish(engine)
        self.assertAlmostEqual(engine.reported_cost_usd, 0.0007)

    def test_a_missing_key_fails_before_any_audio(self):
        session = FakeSession()
        errors, statuses = [], []
        engine = OpenRouterASR(
            api_key="", sample_rate=RATE, on_error=errors.append,
            on_status=lambda **s: statuses.append(s), session=session,
        )
        engine.start()
        engine.join(5)
        self.assertTrue(errors and "missing" in errors[0])
        self.assertEqual(session.calls, [])


class TestTraditionalConversion(unittest.TestCase):
    def test_simplified_output_becomes_hong_kong_traditional(self):
        try:
            import opencc  # noqa: F401
        except ImportError:
            self.skipTest("opencc not installed")
        session = FakeSession(replies=[FakeResponse(200, {"text": "有无人知道湾仔活道 yesterday"})])
        engine, interims, utterances, statuses, errors = make_engine(session, to_traditional=True)
        run_audio(engine, tone(1.5), trailing=1.0)
        self.assertTrue(wait_for(lambda: utterances, 10), errors)
        finish(engine)
        self.assertEqual(utterances[0].text, "有無人知道灣仔活道 yesterday")


class TestEnergyDetector(unittest.TestCase):
    def test_quiet_room_noise_is_not_speech_but_a_voice_is(self):
        engine, *_ = make_engine(FakeSession())
        # Low hiss well under the absolute floor.
        self.assertFalse(engine._is_speech(tone(0.1, amplitude=100)))
        self.assertTrue(engine._is_speech(tone(0.1, amplitude=6000)))

    def test_the_noise_floor_learns_a_loud_room(self):
        engine, *_ = make_engine(FakeSession())
        # A steady hum below the speech ratio is learnt as background...
        for _ in range(60):
            engine._is_speech(tone(0.1, amplitude=400))
        self.assertGreater(engine._noise_floor, 200)
        # ...so a voice must clear it by the ratio to count.
        self.assertFalse(engine._is_speech(tone(0.1, amplitude=500)))
        self.assertTrue(engine._is_speech(tone(0.1, amplitude=3000)))

    def test_wav_helper_produces_a_readable_file(self):
        raw = tone(0.5)
        decoded = base64.b64decode(openrouter_asr._wav_base64(raw, RATE))
        with wave.open(io.BytesIO(decoded)) as w:
            self.assertEqual(w.readframes(w.getnframes()), raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
