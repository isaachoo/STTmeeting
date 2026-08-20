"""Reading a whole meeting once, so everything else can be cheap.

Reports have to cover the entire meeting, and questions often do too. Feeding a
five-hour transcript into every one of those calls is slow, expensive, and worse
than the alternative -- models lose detail in the middle of very long inputs.

So the transcript is read once, in chunks, and each chunk is condensed into
structured facts that keep their transcript line numbers. The result is the
digest: a few thousand characters that stand in for the whole meeting, cached in
the database so the reading is paid for once no matter how many reports and
questions follow.

The line numbers are the point. Without them the digest is a summary you have to
take on faith; with them, every claim in a report can be traced back to the words
that were actually spoken.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import config
from copilot import prompts
from copilot.llm import LLMError, OpenRouterClient
from storage import db

from . import retrieval

log = logging.getLogger(__name__)

# Chunks are read concurrently. Small on purpose: this shares a rate limit with
# nothing else in a finished meeting, but a user with a slow model should not
# have twenty requests in flight against their OpenRouter account at once.
WORKERS = 4

_LIST_KEYS = ("topics", "decisions", "actions", "questions", "facts", "quotes")


def empty() -> dict:
    return {key: [] for key in _LIST_KEYS}


def chunk_transcript(
    segments: list[dict], speaker_names: dict | None, max_chars: int | None = None
) -> list[str]:
    """The transcript as numbered lines, cut into readable chunks.

    Built out of the retrieval passages so the two see exactly the same text and
    the same line numbers -- a citation from a report and a citation from a
    question then mean the same thing.
    """
    limit = max_chars or config.REVIEW_CHUNK_CHARS
    # Chunks are whole passages, so a chunk can never be smaller than one
    # passage. Cap the passage size too, or asking for small chunks silently
    # produces one enormous one.
    passages = retrieval.build_passages(
        segments, speaker_names, max_chars=min(config.REVIEW_WINDOW_CHARS, limit)
    )
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for passage in passages:
        if current and size + len(passage.text) > limit:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(passage.text)
        size += len(passage.text)
    if current:
        chunks.append("\n".join(current))
    return chunks


def build(
    meeting: dict,
    client: OpenRouterClient | None = None,
    on_progress=None,
) -> dict:
    """Read the whole transcript and return the digest. Does not cache; see `ensure`."""
    segments = meeting.get("segments") or []
    chunks = chunk_transcript(segments, meeting.get("speaker_names") or {})
    if not chunks:
        return empty()

    client = client or OpenRouterClient()
    context = brief_text(meeting)
    voices = roster(meeting)
    total = len(chunks)
    done = [0]

    def read(numbered: tuple[int, str]) -> dict:
        i, chunk = numbered
        position = f"section {i + 1} of {total}"
        try:
            result = client.chat_json(
                prompts.digest_messages(context, voices, chunk, position),
                model=config.review_model(),
                max_tokens=1600,
                temperature=0.2,
            )
        except LLMError as exc:
            # One unreadable section must not throw away the other fourteen. The
            # gap is recorded so the digest can say what it is missing rather
            # than quietly presenting itself as complete.
            log.warning("digest section %d failed: %s", i + 1, exc)
            result = {"_failed": position}
        done[0] += 1
        if on_progress:
            on_progress(done[0], total)
        return result

    with ThreadPoolExecutor(max_workers=min(WORKERS, total)) as pool:
        parts = list(pool.map(read, enumerate(chunks)))

    return merge(parts, total)


def merge(parts: list[dict], sections: int) -> dict:
    """Concatenate the per-section results, keeping meeting order."""
    digest = empty()
    failed = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("_failed"):
            failed.append(part["_failed"])
            continue
        for key in _LIST_KEYS:
            value = part.get(key)
            if isinstance(value, list):
                digest[key].extend(item for item in value if isinstance(item, dict))
    digest["sections"] = sections
    digest["failed_sections"] = failed
    return digest


def ensure(meeting: dict, client: OpenRouterClient | None = None, on_progress=None):
    """The digest for this meeting, built and cached if it is not there yet.

    Returns (digest, built) so a caller can tell the user whether it just paid
    for a full read of the transcript or reused one.
    """
    meeting_id = meeting["id"]
    segments = meeting.get("segments") or []
    cached, upto = db.get_digest(meeting_id)
    if cached and upto == len(segments) and not cached.get("failed_sections"):
        return cached, False

    digest = build(meeting, client=client, on_progress=on_progress)
    db.save_digest(meeting_id, digest, len(segments))
    return digest, True


# ---------------------------------------------------------------- rendering


def to_text(digest: dict, max_chars: int = 24000) -> str:
    """The digest as prose for a prompt, in the order a reader would want it.

    Truncated at the end rather than sampled: a very long meeting's digest runs
    to the end of the meeting, and the discussion detail is the part that can be
    dropped with the least loss, since decisions, actions and facts are listed
    separately above it.
    """
    if not digest:
        return ""
    out: list[str] = []

    def lines_of(entry: dict) -> str:
        numbers = entry.get("lines") or []
        refs = " ".join(f"[#{n}]" for n in numbers if isinstance(n, int))
        return f" {refs}" if refs else ""

    for entry in digest.get("decisions") or []:
        if not out:
            out.append("## Decisions")
        by = f" (by {entry['by']})" if entry.get("by") else ""
        out.append(f"- {entry.get('decision', '')}{by}{lines_of(entry)}")

    actions = digest.get("actions") or []
    if actions:
        out += ["", "## Actions people took on"]
        for entry in actions:
            due = f", due {entry['due']}" if entry.get("due") else ""
            out.append(
                f"- {entry.get('who') or 'unassigned'}: {entry.get('what', '')}"
                f"{due}{lines_of(entry)}"
            )

    facts = digest.get("facts") or []
    if facts:
        out += ["", "## Numbers, dates and commitments stated"]
        for entry in facts:
            out.append(f"- {entry.get('fact', '')}{lines_of(entry)}")

    questions = digest.get("questions") or []
    if questions:
        out += ["", "## Questions raised"]
        for entry in questions:
            state = "answered" if entry.get("answered") else "NOT answered"
            asked = f" (asked by {entry['asked_by']})" if entry.get("asked_by") else ""
            out.append(f"- {entry.get('question', '')}{asked} — {state}{lines_of(entry)}")

    quotes = digest.get("quotes") or []
    if quotes:
        out += ["", "## Worth quoting"]
        for entry in quotes:
            line = entry.get("line")
            ref = f" [#{line}]" if isinstance(line, int) else ""
            out.append(f'- {entry.get("who", "")}: "{entry.get("said", "")}"{ref}')

    topics = digest.get("topics") or []
    if topics:
        out += ["", "## How the discussion went, in order"]
        for entry in topics:
            out.append(
                f"### {entry.get('topic', '')}{lines_of(entry)}\n"
                f"{entry.get('what_happened', '')}"
            )

    if digest.get("failed_sections"):
        out += [
            "",
            "## Gaps",
            "The following sections of the transcript could not be read: "
            + ", ".join(digest["failed_sections"])
            + ". Anything from those parts of the meeting is missing here.",
        ]

    text = "\n".join(out)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n[digest truncated]"
    return text


def brief_text(meeting: dict) -> str:
    from copilot.brief import Brief

    try:
        return Brief.from_payload(meeting.get("brief_json") or {}).render()
    except Exception:  # a stored brief from an older version should not break review
        log.debug("could not render stored brief for meeting %s", meeting.get("id"))
        return (meeting.get("brief") or "").strip()


def roster(meeting: dict) -> str:
    names = meeting.get("speaker_names") or {}
    if not names:
        return ""
    return "Voices in the transcript:\n" + "\n".join(
        f"- S{int(k) + 1} is {v}" for k, v in sorted(names.items())
    )
