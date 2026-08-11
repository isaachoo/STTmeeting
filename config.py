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


# --- Speech to text ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "zh-HK").strip()
DEEPGRAM_MODELS = [
    m.strip()
    for m in (os.getenv("DEEPGRAM_MODELS") or "nova-3,nova-2").split(",")
    if m.strip()
]
DEEPGRAM_USD_PER_MINUTE = _float("DEEPGRAM_USD_PER_MINUTE", 0.0077)

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

# --- Server ---
HOST = os.getenv("HOST", "127.0.0.1")
PORT = _int("PORT", 5000)
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
SAVE_AUDIO = _bool("SAVE_AUDIO", False)

DB_PATH = DATA_DIR / "meetings.sqlite3"


def missing_keys() -> list[str]:
    """Keys the app cannot run without."""
    missing = []
    if not DEEPGRAM_API_KEY:
        missing.append("DEEPGRAM_API_KEY")
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    return missing
