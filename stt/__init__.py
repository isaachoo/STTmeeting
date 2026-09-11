"""Speech-to-text engines.

Everything above this package talks in `Utterance` objects and interim strings,
so swapping providers is a matter of choosing a different class here.
"""

import logging

import config

from .base import STTEngine, Utterance

log = logging.getLogger(__name__)

PROVIDERS = ("deepgram", "speechmatics", "local", "qwen")


def create_engine(
    provider: str,
    sample_rate: int,
    language: str,
    model: str,
    keyterms: list[str],
    on_interim,
    on_utterance,
    on_status,
    on_error,
) -> STTEngine:
    """Build the transcriber for one meeting.

    Imports are local so that installing only the cloud dependencies, or only
    the local ones, still leaves a working app.
    """
    provider = (provider or "deepgram").strip().lower()

    if provider == "local":
        from .sherpa_local import SherpaLocalSTT

        return SherpaLocalSTT(
            model_dir=config.SHERPA_MODEL_DIR,
            sample_rate=sample_rate,
            num_threads=config.SHERPA_NUM_THREADS,
            to_traditional=config.SHERPA_TO_TRADITIONAL,
            punctuation_dir=(
                config.SHERPA_PUNCTUATION_DIR if config.SHERPA_PUNCTUATE else None
            ),
            on_interim=on_interim,
            on_utterance=on_utterance,
            on_status=on_status,
            on_error=on_error,
        )

    if provider == "qwen":
        from .openrouter_asr import OpenRouterASR

        return OpenRouterASR(
            api_key=config.OPENROUTER_API_KEY,
            sample_rate=sample_rate,
            model=config.OPENROUTER_ASR_MODEL,
            base_url=config.OPENROUTER_BASE_URL,
            usd_per_minute=config.OPENROUTER_ASR_USD_PER_MINUTE,
            max_segment_seconds=config.OPENROUTER_ASR_MAX_SEGMENT_SECONDS,
            to_traditional=config.OPENROUTER_ASR_TO_TRADITIONAL,
            on_interim=on_interim,
            on_utterance=on_utterance,
            on_status=on_status,
            on_error=on_error,
        )

    if provider == "speechmatics":
        from .speechmatics_live import SpeechmaticsLiveSTT

        return SpeechmaticsLiveSTT(
            api_key=config.SPEECHMATICS_API_KEY,
            sample_rate=sample_rate,
            language=config.speechmatics_language(language),
            url=config.SPEECHMATICS_URL,
            operating_point=config.SPEECHMATICS_OPERATING_POINT,
            keyterms=keyterms,
            usd_per_minute=config.SPEECHMATICS_USD_PER_MINUTE,
            on_interim=on_interim,
            on_utterance=on_utterance,
            on_status=on_status,
            on_error=on_error,
        )

    from .deepgram_live import DeepgramLiveSTT

    return DeepgramLiveSTT(
        api_key=config.DEEPGRAM_API_KEY,
        sample_rate=sample_rate,
        language=language,
        models=config.models_from(model),
        keyterms=keyterms if config.DEEPGRAM_KEYTERMS else [],
        on_interim=on_interim,
        on_utterance=on_utterance,
        on_status=on_status,
        on_error=on_error,
    )


__all__ = ["STTEngine", "Utterance", "PROVIDERS", "create_engine"]
