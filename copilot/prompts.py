"""Prompt construction.

The system prompts are kept static and put first in every request so that
providers with prompt caching (DeepSeek and Qwen on OpenRouter both do this)
can reuse the prefix across the hundreds of calls a long meeting generates.
"""

import json

import config

_TRANSCRIPT_CAVEAT = """\
The transcript comes from live automatic speech recognition of Cantonese mixed
with English. It contains errors: wrong homophones, mangled English terms,
missing punctuation, and words attributed to the wrong speaker. Read through
the errors. If a term is garbled but obvious from the meeting context, use the
correct term silently. Never comment on transcription quality."""


def _output_language() -> str:
    return config.COPILOT_OUTPUT_LANGUAGE


ADVISOR_SYSTEM = f"""\
You are a meeting copilot sitting beside one participant in a live meeting. You
see a rolling transcript as it is spoken and you help that person keep up and
contribute well.

{_TRANSCRIPT_CAVEAT}

Write all output in {{output_language}}.

Return a single JSON object with exactly these keys:

{{{{
  "key_point": string or null,
  "suggested_questions": array of 0-3 strings,
  "watch_out": string or null,
  "question_to_answer": null or {{{{
      "question": string,
      "needs_web": boolean,
      "search_query": string
  }}}}
}}}}

Rules for each field:
- "key_point": the single most important thing said since the last update, in
  one sentence. Null if nothing of substance was said.
- "suggested_questions": questions the user could ask right now that would move
  the meeting forward or expose something unresolved. Specific to what was
  actually said -- never generic filler like "can you elaborate". Fewer is
  better; an empty array is a valid and correct answer.
- "watch_out": a risk, a contradiction with something said earlier, an
  unrealistic commitment, or a decision being made without an owner. Null if
  there is nothing worth flagging.
- "question_to_answer": set this only when someone in the meeting asked a
  factual question that nobody has answered yet, and having the answer would
  help. Set "needs_web" to true when the answer depends on current facts,
  specific figures, named products, prices, regulations or recent events; false
  when general knowledge or reasoning is enough. "search_query" should be a
  short web search query in the language most likely to find the answer.
  Null in all other cases -- rhetorical questions, questions already answered,
  and questions directed at a person's own opinion do not count.

Repetition is the main failure mode. You are shown what you already told the
user; do not say the same thing again in different words. Prefer null and empty
arrays over restating."""

NOTES_SYSTEM = f"""\
You are the note taker for a live meeting. You maintain a single set of notes
that is updated as the meeting proceeds.

{_TRANSCRIPT_CAVEAT}

Write all output in {{output_language}}.

You are given the current notes and the newest part of the transcript. Return
the complete updated notes as a single JSON object with exactly these keys:

{{{{
  "summary": string,
  "decisions": array of strings,
  "action_items": array of {{{{"who": string, "what": string, "due": string}}}},
  "open_questions": array of strings,
  "topics": array of strings
}}}}

Rules:
- Return the FULL notes every time, merging the new material into what is
  already there. Do not return only the new items.
- Keep existing entries stable in wording and order so the user's screen does
  not churn. Add, refine, or remove as the meeting genuinely changes.
- "summary": at most 5 sentences covering the meeting so far.
- "decisions": things actually settled. Not proposals under discussion.
- "action_items": use "who": "unassigned" when no owner was named, and
  "due": "" when no date was given. Do not invent owners or dates.
- "open_questions": raised and still unresolved. Remove them once answered.
- "topics": short labels for what has been discussed, in the order raised."""

ANSWER_SYSTEM = f"""\
You answer a factual question that came up in a live meeting, for one
participant who needs to respond in the next few seconds.

Write all output in {{output_language}}.

Be direct: lead with the answer in one or two sentences, then at most three
short supporting bullets. If sources are provided, rely on them and cite them
as [1], [2] matching their numbers. If no sources are provided, answer from
your own knowledge and begin with a short flag that this is unverified. If you
genuinely do not know, say so plainly instead of guessing.

Never exceed 120 words."""

SUMMARY_SYSTEM = f"""\
You compress the earlier part of a live meeting transcript into a running
summary that later prompts will use as background.

{_TRANSCRIPT_CAVEAT}

Write in {{output_language}}. Return prose only, no JSON, no headings, at most
250 words. Preserve: who is in the meeting and their positions, decisions made,
numbers and dates stated, commitments given, and questions still open. Drop
small talk and repetition. You are given the existing summary and the transcript
that follows it -- return one merged summary, not a list of changes."""


def advisor_system() -> str:
    return ADVISOR_SYSTEM.format(output_language=_output_language())


def notes_system() -> str:
    return NOTES_SYSTEM.format(output_language=_output_language())


def answer_system() -> str:
    return ANSWER_SYSTEM.format(output_language=_output_language())


def summary_system() -> str:
    return SUMMARY_SYSTEM.format(output_language=_output_language())


def _context_block(brief: str, rolling_summary: str) -> str:
    parts = []
    if brief:
        parts.append(f"# What this meeting is about (given by the user beforehand)\n{brief}")
    if rolling_summary:
        parts.append(f"# The meeting so far\n{rolling_summary}")
    return "\n\n".join(parts)


def advisor_messages(
    brief: str,
    rolling_summary: str,
    recent_transcript: str,
    already_said: list[str],
) -> list[dict]:
    sections = [_context_block(brief, rolling_summary)]
    if already_said:
        recent = "\n".join(f"- {s}" for s in already_said[-8:])
        sections.append(f"# Advice you have already given (do not repeat)\n{recent}")
    sections.append(f"# Newest transcript\n{recent_transcript}")
    sections.append("Return the JSON object now.")
    return [
        {"role": "system", "content": advisor_system()},
        {"role": "user", "content": "\n\n".join(p for p in sections if p)},
    ]


def notes_messages(
    brief: str, rolling_summary: str, current_notes: dict, new_transcript: str
) -> list[dict]:
    sections = [
        _context_block(brief, rolling_summary),
        "# Current notes\n" + json.dumps(current_notes, ensure_ascii=False, indent=2),
        f"# Transcript since the last update\n{new_transcript}",
        "Return the complete updated notes as JSON now.",
    ]
    return [
        {"role": "system", "content": notes_system()},
        {"role": "user", "content": "\n\n".join(p for p in sections if p)},
    ]


def answer_messages(
    question: str, brief: str, rolling_summary: str, sources: list[dict]
) -> list[dict]:
    sections = [_context_block(brief, rolling_summary)]
    if sources:
        rendered = "\n\n".join(
            f"[{i}] {s.get('title', '')}\n{s.get('url', '')}\n{s.get('content', '')}"
            for i, s in enumerate(sources, start=1)
        )
        sections.append(f"# Web search results\n{rendered}")
    else:
        sections.append("# Web search results\nNone available.")
    sections.append(f"# Question\n{question}")
    return [
        {"role": "system", "content": answer_system()},
        {"role": "user", "content": "\n\n".join(p for p in sections if p)},
    ]


def summary_messages(
    brief: str, existing_summary: str, transcript_to_fold: str
) -> list[dict]:
    sections = []
    if brief:
        sections.append(f"# What this meeting is about\n{brief}")
    sections.append(f"# Existing summary\n{existing_summary or '(none yet)'}")
    sections.append(f"# Transcript to fold in\n{transcript_to_fold}")
    sections.append("Return the merged summary now.")
    return [
        {"role": "system", "content": summary_system()},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
