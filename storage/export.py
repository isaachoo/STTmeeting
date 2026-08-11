"""Turn a stored meeting into something you can keep.

Two shapes, both from the same row: Markdown for reading and pasting into an
email, and JSON for everything exactly as recorded. Both are built from the
database rather than the live session, so they work just as well on a meeting
from last week.
"""

import datetime
import json

from copilot.state import label_for


def _clock(seconds) -> str:
    total = int(seconds or 0)
    return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _when(timestamp) -> str:
    if not timestamp:
        return ""
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def filename_stem(meeting: dict) -> str:
    started = meeting.get("started_at")
    stamp = (
        datetime.datetime.fromtimestamp(started).strftime("%Y%m%d-%H%M")
        if started
        else "meeting"
    )
    title = "".join(
        ch if (ch.isalnum() or ch in " -_") else "" for ch in (meeting.get("title") or "")
    ).strip().replace(" ", "-")
    return f"{stamp}-{title}" if title else f"{stamp}-meeting-{meeting.get('id', '')}"


def to_json(meeting: dict) -> str:
    """Everything recorded, verbatim."""
    payload = {
        "meeting": {
            "id": meeting.get("id"),
            "title": meeting.get("title"),
            "started_at": meeting.get("started_at"),
            "started_at_local": _when(meeting.get("started_at")),
            "ended_at": meeting.get("ended_at"),
            "ended_at_local": _when(meeting.get("ended_at")),
            "duration_seconds": (meeting.get("ended_at") or 0)
            - (meeting.get("started_at") or 0)
            if meeting.get("ended_at")
            else None,
            "language": meeting.get("language"),
            "stt_model": meeting.get("stt_model"),
            "audio_seconds": meeting.get("audio_seconds"),
        },
        "brief": meeting.get("brief_json") or {},
        "speaker_names": meeting.get("speaker_names") or {},
        "transcript": [
            {
                "index": s["idx"],
                "at": s["at"],
                "at_clock": _clock(s["at"]),
                "speaker": s["speaker"],
                "speaker_label": label_for(s["speaker"], meeting.get("speaker_names")),
                "text": s["text"],
            }
            for s in meeting.get("segments") or []
        ],
        "notes": meeting.get("notes_json") or {},
        "my_notes": meeting.get("user_notes") or "",
        "rolling_summary": meeting.get("summary") or "",
        "copilot_log": meeting.get("events") or [],
        "usage": meeting.get("usage_json") or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def to_markdown(meeting: dict) -> str:
    names = meeting.get("speaker_names") or {}
    brief = meeting.get("brief_json") or {}
    notes = meeting.get("notes_json") or {}
    usage = meeting.get("usage_json") or {}
    out: list[str] = []

    title = meeting.get("title") or f"Meeting {meeting.get('id', '')}"
    out += [f"# {title}", ""]

    started = _when(meeting.get("started_at"))
    ended = _when(meeting.get("ended_at"))
    when = f"{started} – {ended}" if ended else started
    audio_minutes = (meeting.get("audio_seconds") or 0) / 60
    out += [
        f"*{when}*  ·  {audio_minutes:.0f} min of audio  ·  "
        f"{meeting.get('stt_model', '')} / {meeting.get('language', '')}",
        "",
    ]

    # --- brief ---
    if brief.get("attendees"):
        out += ["## In the room", ""]
        for attendee in brief["attendees"]:
            role = f" — {attendee['role']}" if attendee.get("role") else ""
            me = "  *(me)*" if attendee.get("is_me") else ""
            out.append(f"- **{attendee.get('name', '')}**{role}{me}")
        out.append("")
    for heading, key in (
        ("Agenda", "agenda"),
        ("What I wanted out of it", "my_goal"),
        ("Background", "context"),
    ):
        if brief.get(key):
            out += [f"## {heading}", "", brief[key], ""]
    if brief.get("glossary"):
        out += ["## Glossary", "", ", ".join(brief["glossary"]), ""]

    # --- notes first: it is what people actually read ---
    if notes.get("summary"):
        out += ["## Summary", "", notes["summary"], ""]
    if notes.get("decisions"):
        out += ["## Decisions", ""] + [f"- {d}" for d in notes["decisions"]] + [""]
    if notes.get("action_items"):
        out += ["## Action items", ""]
        for item in notes["action_items"]:
            due = f" — due {item.get('due')}" if item.get("due") else ""
            out.append(f"- **{item.get('who', 'unassigned')}**: {item.get('what', '')}{due}")
        out.append("")
    if notes.get("open_questions"):
        out += ["## Open questions", ""] + [f"- {q}" for q in notes["open_questions"]] + [""]
    if notes.get("topics"):
        out += ["## Topics", "", ", ".join(notes["topics"]), ""]
    if meeting.get("user_notes", "").strip():
        out += ["## My own notes", "", meeting["user_notes"].strip(), ""]

    # --- what the copilot did ---
    events = meeting.get("events") or []
    attendee_turns = [e for e in events if e.get("kind") == "attendee"]
    if attendee_turns:
        out += ["## What the AI attendee said", ""]
        for event in attendee_turns:
            payload = event.get("payload") or {}
            kind = payload.get("kind", "")
            out.append(f"- *({kind})* {payload.get('say', '')}")
            if payload.get("why"):
                out.append(f"  - why: {payload['why']}")
            for source in payload.get("sources") or []:
                out.append(f"  - source: [{source.get('title', '')}]({source.get('url', '')})")
        out.append("")

    advice = [e for e in events if e.get("kind") == "advice"]
    if advice:
        out += ["## Copilot suggestions", ""]
        for event in advice:
            payload = event.get("payload") or {}
            if payload.get("key_point"):
                out.append(f"- {payload['key_point']}")
            if payload.get("watch_out"):
                out.append(f"  - ⚠ {payload['watch_out']}")
            for question in payload.get("questions") or []:
                out.append(f"  - ask: {question}")
        out.append("")

    answers = [e for e in events if e.get("kind") == "answer"]
    if answers:
        out += ["## Questions I asked the copilot", ""]
        for event in answers:
            payload = event.get("payload") or {}
            out += [f"**{payload.get('question', '')}**", "", payload.get("answer", ""), ""]
            for i, source in enumerate(payload.get("sources") or [], start=1):
                out.append(f"[{i}] [{source.get('title', '')}]({source.get('url', '')})")
            if payload.get("sources"):
                out.append("")

    # --- the raw record ---
    if names:
        out += [
            "## Voices",
            "",
            ", ".join(f"S{int(k) + 1} = {v}" for k, v in sorted(names.items())),
            "",
        ]
    out += ["## Full transcript", ""]
    for segment in meeting.get("segments") or []:
        label = label_for(segment["speaker"], names)
        out.append(f"`{_clock(segment['at'])}` **{label}**: {segment['text']}")
    out.append("")

    if meeting.get("summary"):
        out += ["## Rolling summary the copilot was working from", "", meeting["summary"], ""]

    if usage:
        out += [
            "## Cost",
            "",
            f"- LLM: ${usage.get('cost_usd', 0):.4f} across {usage.get('calls', 0)} calls "
            f"({usage.get('prompt_tokens', 0)} in / {usage.get('completion_tokens', 0)} out)",
            f"- Audio transcribed: {audio_minutes:.1f} minutes",
            "",
        ]

    return "\n".join(out)
