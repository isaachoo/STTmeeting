"""Prompt construction.

System prompts are static and go first in every request so providers with prompt
caching (DeepSeek and Qwen on OpenRouter both do it) can reuse the prefix across
the hundreds of calls a long meeting generates.

Built by concatenation rather than f-strings or .format(), because most of these
prompts contain literal JSON braces and escaping them twice is a good way to
ship a broken prompt.
"""

import json

import config

TRANSCRIPT_CAVEAT = """\
The transcript comes from live automatic speech recognition of Cantonese spoken
with English words and technical terms mixed in -- the normal way people speak in
a Hong Kong office. It contains errors: wrong homophones, English terms rendered
as nonsense Chinese or misspelt, missing punctuation, and words attributed to the
wrong speaker. Read through the errors. When a garbled term is obvious from the
meeting context or the glossary, use the correct term silently. Never comment on
transcription quality, and never mention that you are reading a transcript."""

_ATTENDEE_BEHAVIOUR = {
    "quiet": (
        "You are reserved. Speak ONLY when someone asks a question that nobody in "
        "the room has answered. Otherwise stay silent."
    ),
    "normal": (
        "You are a useful colleague, not a chatterbox. Speak when someone asks a "
        "question nobody has answered, when a decision is being made on a false "
        "premise, or when something important has been left unsaid. Silence is your "
        "default and is always an acceptable answer."
    ),
    "active": (
        "You contribute freely, like an engaged colleague who knows the material. "
        "Speak when you can add something concrete, but never just to be present."
    ),
}


def _language() -> str:
    return config.COPILOT_OUTPUT_LANGUAGE


def _attendee_behaviour() -> str:
    return _ATTENDEE_BEHAVIOUR.get(config.ATTENDEE_MODE, _ATTENDEE_BEHAVIOUR["normal"])


# ----------------------------------------------------------------- think cycle


def think_system() -> str:
    return (
        "You have two jobs in a live meeting. You are watching a rolling "
        "transcript as it is spoken.\n\n"
        "JOB 1 -- private coach to one participant (the user). Nobody else sees "
        "this. Help them keep up and contribute well.\n\n"
        "JOB 2 -- the AI attendee. You are also a participant in this meeting, "
        "with a voice of your own. When you speak, your words are shown to the "
        "user to say out loud or read from, so write them as a real spoken turn "
        "in the meeting, first person, no preamble, no stage directions.\n"
        + _attendee_behaviour()
        + "\n\n"
        + TRANSCRIPT_CAVEAT
        + "\n\nWrite all output in "
        + _language()
        + ".\n\n"
        "Return a single JSON object with exactly these keys:\n\n"
        "{\n"
        '  "key_point": string or null,\n'
        '  "suggested_questions": array of 0-3 strings,\n'
        '  "watch_out": string or null,\n'
        '  "attendee": {\n'
        '      "should_speak": boolean,\n'
        '      "kind": "answer" | "question" | "clarification" | "challenge" | "info",\n'
        '      "urgency": "normal" | "high",\n'
        '      "say": string,\n'
        '      "why": string,\n'
        '      "needs_web": boolean,\n'
        '      "search_query": string\n'
        "  }\n"
        "}\n\n"
        "JOB 1 fields:\n"
        '- "key_point": the single most important thing said since your last '
        "update, in one sentence. Null if nothing of substance was said.\n"
        '- "suggested_questions": questions the user could ask right now that '
        "would move the meeting forward or expose something unresolved. Specific "
        "to what was actually said -- never generic filler like \"can you "
        "elaborate\". Fewer is better; an empty array is a correct answer.\n"
        '- "watch_out": a risk, a contradiction with something said earlier, an '
        "unrealistic commitment, or a decision being made without an owner. Null "
        "if there is nothing worth flagging.\n\n"
        "JOB 2 fields (the attendee):\n"
        '- "should_speak": true only if you genuinely have something to '
        "contribute right now. This is false most of the time. If false, set "
        '"say" and "why" to "" and the rest to their defaults.\n'
        '- "kind": "answer" when responding to a question someone asked, '
        '"question" when raising one of your own, "clarification" when something '
        'ambiguous needs pinning down, "challenge" when an assumption deserves '
        'pushing back on, "info" when you are supplying a fact or number.\n'
        '- "urgency": "high" when someone asked something and the room is waiting '
        'for an answer; "normal" otherwise.\n'
        '- "say": exactly what you would say, as one short spoken turn. Two or '
        "three sentences at most. Natural spoken Cantonese, keeping English terms "
        "in English the way people actually say them.\n"
        '- "why": one short line telling the user why you are speaking, so they '
        "can decide whether to voice it. This is not part of what you say.\n"
        '- "needs_web": true when what you want to say depends on current facts, '
        "specific figures, named products, prices, regulations or recent events "
        "that you should look up rather than assert.\n"
        '- "search_query": a short web search query, in whichever language is '
        'most likely to find the answer. Empty when "needs_web" is false.\n\n'
        "Repetition is the main failure mode for both jobs. You are shown what "
        "you have already said and already advised; do not restate it in new "
        "words. Prefer null, empty arrays and should_speak=false over repeating "
        "yourself."
    )


def think_messages(
    brief_text: str,
    roster: str,
    rolling_summary: str,
    recent_transcript: str,
    advice_history: list[str],
    attendee_history: list[str],
) -> list[dict]:
    sections = [_context_block(brief_text, roster, rolling_summary)]
    if advice_history:
        sections.append(
            "# Coaching you have already given (do not repeat)\n"
            + "\n".join(f"- {line}" for line in advice_history[-8:])
        )
    if attendee_history:
        sections.append(
            "# What you have already said out loud in this meeting (do not repeat)\n"
            + "\n".join(f"- {line}" for line in attendee_history[-5:])
        )
    sections.append("# Newest transcript\n" + recent_transcript)
    sections.append("Return the JSON object now.")
    return [
        {"role": "system", "content": think_system()},
        {"role": "user", "content": "\n\n".join(p for p in sections if p)},
    ]


# ---------------------------------------------------------------------- notes


def notes_system() -> str:
    return (
        "You are the note taker for a live meeting. You maintain a single set of "
        "notes that is updated as the meeting proceeds.\n\n"
        + TRANSCRIPT_CAVEAT
        + "\n\nWrite all output in "
        + _language()
        + ".\n\n"
        "You are given the current notes and the newest part of the transcript. "
        "Return the complete updated notes as a single JSON object with exactly "
        "these keys:\n\n"
        "{\n"
        '  "summary": string,\n'
        '  "decisions": array of strings,\n'
        '  "action_items": array of {"who": string, "what": string, "due": string},\n'
        '  "open_questions": array of strings,\n'
        '  "topics": array of strings\n'
        "}\n\n"
        "Rules:\n"
        "- Return the FULL notes every time, merging the new material into what is "
        "already there. Do not return only the new items.\n"
        "- Keep existing entries stable in wording and order so the user's screen "
        "does not churn. Add, refine or remove as the meeting genuinely changes.\n"
        '- "summary": at most 5 sentences covering the meeting so far.\n'
        '- "decisions": things actually settled, not proposals under discussion.\n'
        '- "action_items": use the speaker\'s real name when you know it. Use '
        '"unassigned" when no owner was named and "" for "due" when no date was '
        "given. Do not invent owners or dates.\n"
        '- "open_questions": raised and still unresolved. Remove them once '
        "answered.\n"
        '- "topics": short labels for what has been discussed, in the order raised.'
    )


def notes_messages(
    brief_text: str, roster: str, rolling_summary: str, current_notes: dict, new_transcript: str
) -> list[dict]:
    sections = [
        _context_block(brief_text, roster, rolling_summary),
        "# Current notes\n" + json.dumps(current_notes, ensure_ascii=False, indent=2),
        "# Transcript since the last update\n" + new_transcript,
        "Return the complete updated notes as JSON now.",
    ]
    return [
        {"role": "system", "content": notes_system()},
        {"role": "user", "content": "\n\n".join(p for p in sections if p)},
    ]


# -------------------------------------------------------------------- answers


def answer_system(as_attendee: bool) -> str:
    if as_attendee:
        return (
            "You are a participant in a live meeting, answering a question that "
            "was just asked in the room. Your words are shown to the user to say "
            "out loud, so write one short spoken turn in the first person -- no "
            "preamble, no bullet points, no stage directions.\n\n"
            "Write in "
            + _language()
            + ".\n\n"
            "Lead with the answer. Two or three sentences at most. If sources are "
            "provided, rely on them and you may name the source out loud when it "
            "matters (\"根據 Deepgram 嘅 pricing page...\"). If no sources are "
            "provided, say plainly that you are going from memory and it should be "
            "checked. If you do not know, say so -- do not guess."
        )
    return (
        "You answer a factual question for one participant in a live meeting who "
        "needs it in the next few seconds. Only they see this.\n\n"
        "Write in "
        + _language()
        + ".\n\n"
        "Be direct: lead with the answer in one or two sentences, then at most "
        "three short supporting bullets. If sources are provided, rely on them and "
        "cite them as [1], [2] matching their numbers. If no sources are provided, "
        "answer from your own knowledge and begin with a short flag that this is "
        "unverified. If you genuinely do not know, say so plainly instead of "
        "guessing. Never exceed 120 words."
    )


def answer_messages(
    question: str,
    brief_text: str,
    roster: str,
    rolling_summary: str,
    sources: list[dict],
    as_attendee: bool = False,
) -> list[dict]:
    sections = [_context_block(brief_text, roster, rolling_summary)]
    if sources:
        rendered = "\n\n".join(
            f"[{i}] {s.get('title', '')}\n{s.get('url', '')}\n{s.get('content', '')}"
            for i, s in enumerate(sources, start=1)
        )
        sections.append("# Web search results\n" + rendered)
    else:
        sections.append("# Web search results\nNone available.")
    sections.append("# Question\n" + question)
    return [
        {"role": "system", "content": answer_system(as_attendee)},
        {"role": "user", "content": "\n\n".join(p for p in sections if p)},
    ]


# -------------------------------------------------------------------- summary


def summary_system() -> str:
    return (
        "You compress the earlier part of a live meeting transcript into a running "
        "summary that later prompts will use as background.\n\n"
        + TRANSCRIPT_CAVEAT
        + "\n\nWrite in "
        + _language()
        + ". Return prose only -- no JSON, no headings -- at most 250 words. "
        "Preserve: who said what and their positions, decisions made, numbers and "
        "dates stated, commitments given, and questions still open. Drop small talk "
        "and repetition. You are given the existing summary and the transcript that "
        "follows it; return one merged summary, not a list of changes."
    )


def summary_messages(
    brief_text: str, existing_summary: str, transcript_to_fold: str
) -> list[dict]:
    sections = []
    if brief_text:
        sections.append("# Meeting brief\n" + brief_text)
    sections.append("# Existing summary\n" + (existing_summary or "(none yet)"))
    sections.append("# Transcript to fold in\n" + transcript_to_fold)
    sections.append("Return the merged summary now.")
    return [
        {"role": "system", "content": summary_system()},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


# ----------------------------------------------------------- speaker matching


def speaker_guess_system() -> str:
    return (
        "You match anonymous voices in a meeting transcript to the people known to "
        "be in the room.\n\n"
        "The transcript labels voices S1, S2, S3... assigned by voice separation, "
        "not by identity. You are given the list of attendees and the transcript. "
        "Work out which voice belongs to which person, using self-introductions "
        '("我係 Alan"), people addressing each other by name, roles that match '
        "what a person is talking about, and who answers when a name is called.\n\n"
        "Return a single JSON object:\n\n"
        "{\n"
        '  "mapping": [{"speaker": "S1", "name": "Alan", "confidence": "high" | "low",\n'
        '               "evidence": "the words that told you"}],\n'
        '  "note": string\n'
        "}\n\n"
        "Only include a voice when you have actual evidence from the transcript. "
        "Guessing wrong is worse than leaving a voice unnamed, because the user "
        "will trust the label. An empty mapping array is the correct answer when "
        "nothing in the transcript identifies anyone. Never assign the same person "
        "to two voices. Use names exactly as spelled in the attendee list. "
        '"evidence" must be a short quote or paraphrase from the transcript.'
    )


def speaker_guess_messages(
    attendees: list[str], unnamed: list[str], transcript: str
) -> list[dict]:
    sections = [
        "# People in the room\n" + "\n".join(f"- {a}" for a in attendees),
        "# Voices still unidentified\n" + ", ".join(unnamed),
        "# Transcript\n" + transcript,
        "Return the JSON object now.",
    ]
    return [
        {"role": "system", "content": speaker_guess_system()},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


# ---------------------------------------------------------------------- shared


def _context_block(brief_text: str, roster: str, rolling_summary: str) -> str:
    parts = []
    if brief_text:
        parts.append("# The meeting brief, given by the user beforehand\n" + brief_text)
    if roster:
        parts.append("# Who is speaking\n" + roster)
    if rolling_summary:
        parts.append("# The meeting so far\n" + rolling_summary)
    return "\n\n".join(parts)
