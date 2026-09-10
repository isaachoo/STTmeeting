"""Asking questions about a finished meeting.

One question, one LLM call. The transcript is not sent -- the passages that match
the question are, plus the digest if one has been built, so the model has both
the specific words and the shape of the whole meeting.

The digest is used when it exists but is never built here. Building it reads the
entire transcript, which costs money and takes a minute; doing that silently
because someone typed a question would be a nasty surprise. The UI offers it
instead, and until then questions are answered from the retrieved passages and
the rolling summary the copilot was already keeping during the meeting.
"""

import logging

import config
from copilot import prompts
from copilot.llm import OpenRouterClient
from storage import db

from . import digest as digest_module
from . import retrieval

log = logging.getLogger(__name__)

MAX_QUESTION = 1000


def ask(
    meeting: dict,
    question: str,
    history: list[dict] | None = None,
    client: OpenRouterClient | None = None,
    save: bool = True,
) -> dict:
    question = (question or "").strip()[:MAX_QUESTION]
    if not question:
        raise ValueError("empty question")

    segments = meeting.get("segments") or []
    names = meeting.get("speaker_names") or {}
    passages = retrieval.find(segments, names, question)
    covered = retrieval.indices_in(passages)

    cached, upto = db.get_digest(meeting["id"])
    if cached:
        digest_text = digest_module.to_text(cached, max_chars=12000)
    else:
        # No digest yet: the rolling summary the copilot kept during the meeting
        # is a weaker substitute, but it is free and already paid for.
        digest_text = _fallback_context(meeting)

    client = client or OpenRouterClient()
    messages = prompts.review_qa_messages(
        question=question,
        brief_text=digest_module.brief_text(meeting),
        roster=digest_module.roster(meeting),
        digest_text=digest_text,
        passages="\n\n".join(p.text for p in passages),
        history=history or [],
    )
    answer = client.chat(
        messages, model=config.review_model(), max_tokens=900, temperature=0.2,
        label="ask the meeting",
    )

    cited = retrieval.cited_indices(answer, valid=covered)
    usage = client.usage.snapshot()
    result = {
        "question": question,
        "answer": answer,
        "cited": cited,
        "passages": [p.as_dict() for p in passages],
        "cost_usd": usage.get("cost_usd", 0.0),
        "used_digest": bool(cached),
        "digest_stale": bool(cached) and upto != len(segments),
    }
    if save:
        turn = db.add_chat_turn(
            meeting["id"], question, answer, cited, result["cost_usd"]
        )
        result["id"] = turn["id"]
        result["at"] = turn["at"]
    return result


def _fallback_context(meeting: dict) -> str:
    parts = []
    summary = (meeting.get("summary") or "").strip()
    if summary:
        parts.append("## Rolling summary kept during the meeting\n" + summary)
    notes = meeting.get("notes_json") or {}
    if notes.get("summary"):
        parts.append("## Notes summary\n" + notes["summary"])
    for label, key in (("Decisions", "decisions"), ("Open questions", "open_questions")):
        items = notes.get(key) or []
        if items:
            parts.append(f"## {label}\n" + "\n".join(f"- {i}" for i in items))
    if not parts:
        return ""
    return (
        "(No full-meeting digest has been built, so this is only what was noted "
        "live. Parts of the meeting may not be represented here at all.)\n\n"
        + "\n\n".join(parts)
    )
