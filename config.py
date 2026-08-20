"""Configuration, loaded from environment / .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"

load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# --- Speech to text: which engine ---
# deepgram | speechmatics | local
STT_PROVIDER = (os.getenv("STT_PROVIDER") or "deepgram").strip().lower()

STT_PROVIDER_CHOICES = [
    {"code": "deepgram", "label": "Deepgram — cloud, fast, ~$0.0077/min"},
    {"code": "speechmatics", "label": "Speechmatics — cloud, Cantonese + diarization"},
    {"code": "local", "label": "Local (sherpa-onnx) — offline, free, no diarization"},
]

# --- Deepgram ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "zh-HK").strip()
DEEPGRAM_MODELS = [
    m.strip()
    for m in (os.getenv("DEEPGRAM_MODELS") or "nova-3,nova-2").split(",")
    if m.strip()
]
DEEPGRAM_USD_PER_MINUTE = _float("DEEPGRAM_USD_PER_MINUTE", 0.0077)

# Offered in the pre-meeting form. `multi` is Deepgram's code-switching model,
# which covers English, Spanish, French, German, Hindi, Russian, Portuguese,
# Japanese, Italian and Dutch -- Cantonese is NOT among them, so it is labelled
# plainly rather than left to look like a better option for Cantonese.
DEEPGRAM_LANGUAGE_CHOICES = [
    {"code": "zh-HK", "label": "Cantonese (zh-HK)"},
    {"code": "zh-CN", "label": "Mandarin, Simplified (zh-CN)"},
    {"code": "zh-TW", "label": "Mandarin, Traditional (zh-TW)"},
    {"code": "en", "label": "English (en)"},
    {"code": "multi", "label": "Code-switching — English + 9 others, no Cantonese"},
]

DEEPGRAM_MODEL_CHOICES = [
    {"code": "nova-3", "label": "nova-3 — newest, supports Cantonese"},
    {"code": "nova-2", "label": "nova-2 — older fallback"},
]


def models_from(preferred: str) -> list[str]:
    """Preferred model first, the others after it as automatic fallbacks."""
    known = [m["code"] for m in DEEPGRAM_MODEL_CHOICES]
    if preferred not in known:
        return list(DEEPGRAM_MODELS)
    return [preferred] + [m for m in known if m != preferred]

# --- Speechmatics ---
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY", "").strip()
SPEECHMATICS_URL = os.getenv(
    "SPEECHMATICS_URL", "wss://eu2.rt.speechmatics.com/v2"
).strip()
SPEECHMATICS_OPERATING_POINT = (
    os.getenv("SPEECHMATICS_OPERATING_POINT") or "enhanced"
).strip()
SPEECHMATICS_USD_PER_MINUTE = _float("SPEECHMATICS_USD_PER_MINUTE", 0.0173)

# Speechmatics uses ISO codes rather than Deepgram's locale strings, so the
# language chosen in the form has to be translated on the way through.
_SPEECHMATICS_LANGUAGES = {
    "zh-HK": "yue",
    "zh-CN": "cmn",
    "zh-TW": "cmn",
    "zh": "cmn",
    "en": "en",
    "multi": "en",
}


def speechmatics_language(code: str) -> str:
    return _SPEECHMATICS_LANGUAGES.get((code or "").strip(), (code or "en").strip())


# --- Local model (sherpa-onnx) ---
SHERPA_MODEL_DIR = Path(
    os.getenv("SHERPA_MODEL_DIR")
    or DATA_DIR / "models" / "sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en"
)
SHERPA_PUNCTUATION_DIR = Path(
    os.getenv("SHERPA_PUNCTUATION_DIR")
    or DATA_DIR / "models" / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
)
SHERPA_NUM_THREADS = _int("SHERPA_NUM_THREADS", 2)
SHERPA_PUNCTUATE = _bool("SHERPA_PUNCTUATE", True)
# The model writes simplified characters even for Cantonese speech; OpenCC maps
# them to Hong Kong traditional.
SHERPA_TO_TRADITIONAL = _bool("SHERPA_TO_TRADITIONAL", True)

# --- LLM ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v3.2").strip()
OPENROUTER_NOTES_MODEL = (
    os.getenv("OPENROUTER_NOTES_MODEL", "").strip() or OPENROUTER_MODEL
)

# --- Web search ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# --- Copilot behaviour ---
COPILOT_OUTPUT_LANGUAGE = os.getenv(
    "COPILOT_OUTPUT_LANGUAGE",
    "Traditional Chinese (Hong Kong), keeping English technical terms in English",
).strip()
# One "think" cycle produces both the private coaching and the AI attendee's
# turn, so the two panels never contradict each other and it costs one call.
THINK_MIN_INTERVAL = _float("THINK_MIN_INTERVAL", _float("ADVICE_MIN_INTERVAL", 15))
THINK_MIN_NEW_CHARS = _int("THINK_MIN_NEW_CHARS", _int("ADVICE_MIN_NEW_CHARS", 120))
# When someone in the room just asked a question, think sooner than the normal
# cadence -- but not instantly, or a fast back-and-forth would spam the model.
THINK_URGENT_INTERVAL = _float("THINK_URGENT_INTERVAL", 5)
NOTES_INTERVAL = _float("NOTES_INTERVAL", 90)

# How the AI attendee behaves. "quiet" only answers direct questions, "normal"
# also raises questions it thinks matter, "active" contributes more freely.
ATTENDEE_MODE = (os.getenv("ATTENDEE_MODE") or "normal").strip().lower()
ATTENDEE_ENABLED = _bool("ATTENDEE_ENABLED", True)

# Speaker-name inference: map diarised voices to the attendee roster.
SPEAKER_GUESS_INTERVAL = _float("SPEAKER_GUESS_INTERVAL", 120)
SPEAKER_GUESS_MIN_SEGMENTS = _int("SPEAKER_GUESS_MIN_SEGMENTS", 8)

# Boost jargon and names from the brief in the transcriber. Deepgram rejects
# this parameter on some model/language pairs; the app retries without it.
DEEPGRAM_KEYTERMS = _bool("DEEPGRAM_KEYTERMS", True)

# How much verbatim transcript the copilot sees; older material is folded into
# a rolling summary so the prompt stays a predictable size (and cheap).
RECENT_WINDOW_CHARS = _int("RECENT_WINDOW_CHARS", 4000)
SUMMARY_TRIGGER_CHARS = _int("SUMMARY_TRIGGER_CHARS", 6000)

# --- Review workspace (after the meeting) ---
# A five-hour meeting is far too long to put in one prompt, so the review side
# works from a digest built once by reading the transcript in chunks, and from
# retrieved passages for anything that needs the actual words.
REVIEW_CHUNK_CHARS = _int("REVIEW_CHUNK_CHARS", 4500)
REVIEW_WINDOW_CHARS = _int("REVIEW_WINDOW_CHARS", 700)  # one retrievable passage
REVIEW_PASSAGES = _int("REVIEW_PASSAGES", 8)  # passages sent with a question
REVIEW_MODEL = (os.getenv("REVIEW_MODEL") or "").strip()  # falls back to OPENROUTER_MODEL

REPORT_KINDS = [
    {
        "code": "minutes",
        "label": "Minutes",
        "hint": "Formal record: attendees, what was discussed, decisions, actions.",
    },
    {
        "code": "actions",
        "label": "Action items",
        "hint": "Owner, task, date — the list you send round afterwards.",
    },
    {
        "code": "summary",
        "label": "Executive summary",
        "hint": "One page for someone who was not there and has two minutes.",
    },
    {
        "code": "email",
        "label": "Follow-up email",
        "hint": "Ready to paste and send to the people who were in the room.",
    },
]


def review_model() -> str:
    return REVIEW_MODEL or OPENROUTER_MODEL

# --- Server ---
HOST = os.getenv("HOST", "127.0.0.1")
PORT = _int("PORT", 5000)
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
SAVE_AUDIO = _bool("SAVE_AUDIO", False)

DB_PATH = DATA_DIR / "meetings.sqlite3"


def missing_keys(provider: str | None = None) -> list[str]:
    """Keys the app cannot run without, for the chosen transcriber.

    The local engine needs no speech key at all, so demanding one would block a
    setup that is perfectly able to run.
    """
    provider = (provider or STT_PROVIDER or "deepgram").strip().lower()
    missing = []
    if provider == "deepgram" and not DEEPGRAM_API_KEY:
        missing.append("DEEPGRAM_API_KEY")
    if provider == "speechmatics" and not SPEECHMATICS_API_KEY:
        missing.append("SPEECHMATICS_API_KEY")
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    return missing
