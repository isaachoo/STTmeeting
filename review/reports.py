"""The documents a meeting turns into: minutes, actions, summary, email.

All four are written from the digest rather than the transcript, so a five-hour
meeting produces a document that reflects all five hours instead of whichever
part happened to fit in the prompt. The digest is built on demand and cached, so
generating all four costs one full read of the transcript, not four.

Reports are stored, not streamed and forgotten. Each generation is a new row: a
regenerated report never destroys the wording someone already edited and sent.
"""

import logging

import config
from copilot import prompts
from copilot.llm import LLMError, OpenRouterClient
from storage import db, export

from . import digest as digest_module

log = logging.getLogger(__name__)

KINDS = {kind["code"]: kind for kind in config.REPORT_KINDS}

# Minutes of a long meeting genuinely need the length; an email does not.
_MAX_TOKENS = {"minutes": 3500, "actions": 2000, "summary": 1200, "email": 900}


def generate(
    meeting: dict,
    kind: str,
    client: OpenRouterClient | None = None,
    on_progress=None,
) -> dict:
    """Write one report and save it. Returns the stored row."""
    if kind not in KINDS:
        raise ValueError(f"unknown report kind: {kind}")

    client = client or OpenRouterClient()
    if on_progress:
        on_progress("reading the meeting", 0, 0)

    def digest_progress(done: int, total: int) -> None:
        if on_progress:
            on_progress("reading the meeting", done, total)

    digest, built = digest_module.ensure(
        meeting, client=client, on_progress=digest_progress
    )
    if on_progress:
        on_progress(f"writing the {KINDS[kind]['label'].lower()}", 0, 0)

    actions = db.list_actions(meeting["id"])
    body = client.chat(
        prompts.report_messages(
            kind=kind,
            brief_text=digest_module.brief_text(meeting),
            roster=digest_module.roster(meeting),
            digest_text=digest_module.to_text(digest),
            notes=meeting.get("notes_json") or {},
            actions=actions,
            user_notes=meeting.get("user_notes") or "",
            meta=_meta(meeting),
        ),
        model=config.review_model(),
        max_tokens=_MAX_TOKENS.get(kind, 2000),
        temperature=0.3,
    ).strip()

    if not body:
        raise LLMError("the model returned an empty document")

    row = db.save_report(
        meeting_id=meeting["id"],
        kind=kind,
        title=_title(meeting, kind),
        body=body,
        model=config.review_model(),
    )
    row["digest_built"] = built
    row["cost_usd"] = client.usage.snapshot().get("cost_usd", 0.0)
    return row


def draft_actions(
    meeting: dict, client: OpenRouterClient | None = None, on_progress=None
) -> dict:
    """Propose action items from the meeting. Saves nothing.

    Deliberately returns proposals for the user to accept one by one: this list
    gets sent to colleagues, and an owner or a date the model inferred rather
    than heard is exactly the kind of mistake nobody catches until it matters.
    """
    client = client or OpenRouterClient()

    def digest_progress(done: int, total: int) -> None:
        if on_progress:
            on_progress("reading the meeting", done, total)

    digest, built = digest_module.ensure(
        meeting, client=client, on_progress=digest_progress
    )
    if on_progress:
        on_progress("drafting action items", 0, 0)

    existing = db.list_actions(meeting["id"])
    result = client.chat_json(
        prompts.action_draft_messages(
            brief_text=digest_module.brief_text(meeting),
            roster=digest_module.roster(meeting),
            digest_text=digest_module.to_text(digest),
            existing=existing,
        ),
        model=config.review_model(),
        max_tokens=1500,
        temperature=0.2,
    )

    proposals = []
    for raw in result.get("actions") or []:
        if not isinstance(raw, dict):
            continue
        what = str(raw.get("what") or "").strip()
        if not what:
            continue
        proposals.append(
            {
                "who": str(raw.get("who") or "").strip()[:80] or "unassigned",
                "what": what[:500],
                "due": str(raw.get("due") or "").strip()[:80],
                "lines": [n for n in (raw.get("lines") or []) if isinstance(n, int)],
                "confidence": "low" if raw.get("confidence") == "low" else "high",
            }
        )

    return {
        "proposals": proposals,
        "digest_built": built,
        "cost_usd": client.usage.snapshot().get("cost_usd", 0.0),
    }


def filename_for(report: dict, meeting: dict) -> str:
    kind = report.get("kind", "report")
    return f"{export.filename_stem(meeting)}-{kind}.md"


def _title(meeting: dict, kind: str) -> str:
    label = KINDS[kind]["label"]
    name = (meeting.get("title") or "").strip()
    return f"{label} — {name}" if name else label


def _meta(meeting: dict) -> str:
    """The facts a document needs that are not in the transcript: when, how long,
    who was in the room. A model cannot infer a date from what people said."""
    lines = []
    started = meeting.get("started_at")
    if started:
        import datetime

        stamp = datetime.datetime.fromtimestamp(started)
        lines.append(f"Date: {stamp.strftime('%Y-%m-%d (%A)')}, started {stamp:%H:%M}")
    minutes = (meeting.get("audio_seconds") or 0) / 60
    if minutes:
        lines.append(f"Length: {minutes:.0f} minutes of speech")
    names = meeting.get("speaker_names") or {}
    if names:
        lines.append("Voices identified: " + ", ".join(sorted(names.values())))
    return "\n".join(lines)
