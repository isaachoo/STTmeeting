"""Finding the parts of a transcript that answer a question.

A five-hour meeting is roughly 60,000 characters of Cantonese. That would fit in
a long-context model, but sending it for every question costs real money and
buries the relevant three lines under everything else that was said. So the
transcript is cut into short passages and only the ones that match the question
are sent.

The matching is deliberately dependency-free: no embeddings, no vector database,
no second API to hold a key for. Scoring is BM25-style lexical overlap over
features chosen to work for how Hong Kong meetings are actually spoken --
character bigrams for the Chinese (there are no spaces to split on, and no word
segmenter here) and whole words for the English terms mixed into it. That is a
good fit for the questions people ask about a meeting, which almost always
contain a name, a number, or a project word that was said out loud.

Every passage carries the segment indices it came from, so an answer can cite
`[#42]` and the UI can scroll to that exact line.
"""

import math
import re
from dataclasses import dataclass, field

import config
from copilot.state import label_for

# A run of CJK characters, and a Latin/digit word. Everything else (punctuation,
# spaces) is a separator and contributes nothing.
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")
_WORD = re.compile(r"[a-z0-9][a-z0-9'&.-]*")
_STOP = {
    # English words too common to help, plus the ones that show up in every
    # question about a meeting ("what did we say about X").
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "was",
    "were", "are", "be", "did", "do", "does", "we", "i", "he", "she", "they",
    "it", "that", "this", "what", "when", "who", "how", "why", "about", "said",
    "say", "any", "all", "with", "from", "at", "by", "as", "but", "if", "not",
    "have", "has", "had", "there", "their", "our", "you", "me", "my",
}


# Single Chinese characters are kept as features but discounted. A bigram like
# 預算 is strong evidence; 加 on its own is weak. Dropping single characters
# entirely, though, loses the common case where a question and the transcript
# say the same thing in a different order -- someone asks "budget 加幾多" about a
# line that says "最多加 50 萬", and the two share no bigram at all. Weak evidence
# from 加 and 多 is what bridges that.
UNIGRAM = "1:"
UNIGRAM_WEIGHT = 0.35


def features(text: str) -> list[str]:
    """The units matching works on: CJK bigrams, single CJK characters, and
    Latin words. Single characters are tagged so they can be weighted down."""
    out: list[str] = []
    lowered = (text or "").lower()
    for run in _CJK.findall(lowered):
        for character in run:
            out.append(UNIGRAM + character)
        out.extend(run[i : i + 2] for i in range(len(run) - 1))
    for word in _WORD.findall(lowered):
        if len(word) > 1 and word not in _STOP:
            out.append(word)
    return out


@dataclass
class Passage:
    """A few consecutive transcript lines, with the indices they came from."""

    start: int
    end: int
    lines: list[str] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    text: str = ""

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "indices": self.indices,
            "speakers": self.speakers,
            "text": self.text,
        }


def build_passages(
    segments: list[dict],
    speaker_names: dict | None = None,
    max_chars: int | None = None,
) -> list[Passage]:
    """Cut the transcript into passages of roughly `max_chars`.

    Passages break on size alone. Breaking on speaker changes sounds tidier but
    produces one-line passages in a fast conversation, and a single line out of
    context is exactly what makes a retrieved answer wrong.
    """
    limit = max_chars or config.REVIEW_WINDOW_CHARS
    names = speaker_names or {}
    passages: list[Passage] = []
    current: Passage | None = None
    size = 0

    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        index = int(segment.get("idx", segment.get("index", 0)))
        label = (segment.get("speaker_name") or "").strip() or label_for(
            segment.get("speaker"), names
        )
        line = f"[#{index}] {label}: {text}"

        if current is None or size + len(line) > limit:
            current = Passage(start=index, end=index)
            passages.append(current)
            size = 0
        current.end = index
        current.lines.append(line)
        current.indices.append(index)
        if label not in current.speakers:
            current.speakers.append(label)
        size += len(line)

    for passage in passages:
        passage.text = "\n".join(passage.lines)
    return passages


class Index:
    """A tiny BM25 index over one meeting's passages.

    Built fresh per request. For a five-hour meeting that is a few hundred
    passages and a few thousand features -- microseconds -- so caching it would
    add a staleness problem to solve no measurable cost.
    """

    K1 = 1.4  # term-frequency saturation
    B = 0.72  # length normalisation

    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.postings: list[dict[str, int]] = []
        self.lengths: list[int] = []
        document_frequency: dict[str, int] = {}

        for passage in passages:
            counts: dict[str, int] = {}
            for feature in features(passage.text):
                counts[feature] = counts.get(feature, 0) + 1
            self.postings.append(counts)
            self.lengths.append(sum(counts.values()) or 1)
            for feature in counts:
                document_frequency[feature] = document_frequency.get(feature, 0) + 1

        self.document_frequency = document_frequency
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0

    def _idf(self, feature: str) -> float:
        total = len(self.passages)
        df = self.document_frequency.get(feature, 0)
        if not df:
            return 0.0
        # Standard BM25 idf, floored at zero: a feature in nearly every passage
        # (為, 我們) should count for nothing rather than pushing scores negative.
        idf = max(0.0, math.log(1 + (total - df + 0.5) / (df + 0.5)))
        return idf * UNIGRAM_WEIGHT if feature.startswith(UNIGRAM) else idf

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Passage, float]]:
        wanted = top_k or config.REVIEW_PASSAGES
        query_features = set(features(query))
        if not query_features or not self.passages:
            return []

        scored: list[tuple[int, float]] = []
        for i, counts in enumerate(self.postings):
            score = 0.0
            length = self.lengths[i]
            for feature in query_features:
                tf = counts.get(feature)
                if not tf:
                    continue
                norm = self.K1 * (1 - self.B + self.B * length / self.average_length)
                score += self._idf(feature) * (tf * (self.K1 + 1)) / (tf + norm)
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        chosen = scored[:wanted]
        # Back into transcript order: an answer reads better when its evidence
        # runs forwards through the meeting rather than by descending score.
        chosen.sort(key=lambda pair: pair[0])
        return [(self.passages[i], score) for i, score in chosen]


def find(
    segments: list[dict],
    speaker_names: dict | None,
    query: str,
    top_k: int | None = None,
) -> list[Passage]:
    index = Index(build_passages(segments, speaker_names))
    return [passage for passage, _ in index.search(query, top_k)]


_CITATION = re.compile(r"\[#(\d+)\]")


def cited_indices(answer: str, valid: set[int] | None = None) -> list[int]:
    """The line numbers an answer cites, in the order they appear, deduplicated.

    Filtered against the lines actually supplied when `valid` is given -- a model
    that invents `[#9999]` should not produce a citation the user can click and
    land nowhere.
    """
    seen: list[int] = []
    for match in _CITATION.finditer(answer or ""):
        index = int(match.group(1))
        if valid is not None and index not in valid:
            continue
        if index not in seen:
            seen.append(index)
    return seen


def indices_in(passages: list[Passage]) -> set[int]:
    """Exactly the lines that were sent -- not the range they span, which would
    include lines that were dropped for being empty."""
    covered: set[int] = set()
    for passage in passages:
        covered.update(passage.indices)
    return covered
