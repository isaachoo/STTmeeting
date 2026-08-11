"""Speech-to-text engine interface.

Everything above this layer talks in terms of `Utterance` objects and interim
strings, so swapping Deepgram out for a local Whisper later touches only this
package.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Utterance:
    """A finalised chunk of speech, roughly one turn or sentence."""

    text: str
    speaker: int | None = None
    start: float | None = None
    end: float | None = None
    words: list[dict] = field(default_factory=list)


class STTEngine(ABC):
    """Streaming transcriber fed raw PCM from the browser microphone."""

    @abstractmethod
    def start(self) -> None:
        """Open the connection and begin consuming audio."""

    @abstractmethod
    def send_audio(self, chunk: bytes) -> None:
        """Queue a chunk of little-endian 16-bit mono PCM."""

    @abstractmethod
    def stop(self) -> None:
        """Flush and close. Safe to call more than once."""

    @property
    @abstractmethod
    def audio_seconds(self) -> float:
        """Seconds of audio submitted so far, for the cost readout."""
