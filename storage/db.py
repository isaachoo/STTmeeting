"""SQLite persistence.

Written to as the meeting happens, so a browser refresh or a crash at hour four
does not lose the transcript. A fresh connection per call keeps this safe across
the several threads that touch it, and at meeting speed the cost is irrelevant.
"""

import json
import sqlite3
import time

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL DEFAULT '',
    brief         TEXT NOT NULL DEFAULT '',
    started_at    REAL NOT NULL,
    ended_at      REAL,
    audio_seconds REAL NOT NULL DEFAULT 0,
    stt_model     TEXT NOT NULL DEFAULT '',
    language      TEXT NOT NULL DEFAULT '',
    usage_json    TEXT NOT NULL DEFAULT '{}',
    notes_json    TEXT NOT NULL DEFAULT '{}',
    summary       TEXT NOT NULL DEFAULT '',
    user_notes    TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS segments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id),
    idx        INTEGER NOT NULL,
    at         REAL NOT NULL,
    speaker    INTEGER,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS segments_meeting ON segments(meeting_id, idx);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id),
    at         REAL NOT NULL,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_meeting ON events(meeting_id, id);
"""

# Columns added after the first release. Applied on every init so an existing
# database from an earlier version keeps working instead of erroring.
MIGRATIONS = {
    "meetings": {
        "brief_json": "ALTER TABLE meetings ADD COLUMN brief_json TEXT NOT NULL DEFAULT '{}'",
        "speakers_json": "ALTER TABLE meetings ADD COLUMN speakers_json TEXT NOT NULL DEFAULT '{}'",
        "provider": "ALTER TABLE meetings ADD COLUMN provider TEXT NOT NULL DEFAULT ''",
    },
    # A name attached to one line, overriding whatever the diarisation said.
    # Needed because engines split one person across two voices, or merge two
    # people into one, and no amount of voice-level naming fixes that.
    "segments": {
        "speaker_name": "ALTER TABLE segments ADD COLUMN speaker_name TEXT NOT NULL DEFAULT ''",
    },
}


def _connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)
        for table, columns in MIGRATIONS.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column, statement in columns.items():
                if column not in existing:
                    conn.execute(statement)


def create_meeting(
    title: str, brief: dict, language: str, stt_model: str
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO meetings (title, brief, brief_json, started_at, language,"
            " stt_model) VALUES (?, ?, ?, ?, ?, ?)",
            (
                title,
                # A plain-text copy so the row is readable without unpacking JSON.
                (brief or {}).get("context", ""),
                json.dumps(brief or {}, ensure_ascii=False),
                time.time(),
                language,
                stt_model,
            ),
        )
        return int(cur.lastrowid)


def add_segment(meeting_id: int, idx: int, at: float, speaker, text: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO segments (meeting_id, idx, at, speaker, text)"
            " VALUES (?, ?, ?, ?, ?)",
            (meeting_id, idx, at, speaker, text),
        )


def add_event(meeting_id: int, kind: str, payload: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events (meeting_id, at, kind, payload) VALUES (?, ?, ?, ?)",
            (meeting_id, time.time(), kind, json.dumps(payload, ensure_ascii=False)),
        )


def _set(meeting_id: int, column: str, value) -> None:
    with _connect() as conn:
        conn.execute(f"UPDATE meetings SET {column} = ? WHERE id = ?", (value, meeting_id))


def save_notes(meeting_id: int, notes: dict) -> None:
    _set(meeting_id, "notes_json", json.dumps(notes, ensure_ascii=False))


def save_user_notes(meeting_id: int, text: str) -> None:
    _set(meeting_id, "user_notes", text)


def save_summary(meeting_id: int, summary: str) -> None:
    _set(meeting_id, "summary", summary)


def set_segment_speaker(meeting_id: int, index: int, name: str) -> None:
    """Override the speaker on one line. An empty name clears the override."""
    with _connect() as conn:
        conn.execute(
            "UPDATE segments SET speaker_name = ? WHERE meeting_id = ? AND idx = ?",
            (name, meeting_id, index),
        )


def save_speakers(meeting_id: int, speaker_names: dict) -> None:
    _set(
        meeting_id,
        "speakers_json",
        json.dumps({str(k): v for k, v in speaker_names.items()}, ensure_ascii=False),
    )


def finish_meeting(
    meeting_id: int, audio_seconds: float, usage: dict, stt_model: str
) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE meetings SET ended_at = ?, audio_seconds = ?, usage_json = ?,"
            " stt_model = ? WHERE id = ?",
            (
                time.time(),
                audio_seconds,
                json.dumps(usage, ensure_ascii=False),
                stt_model,
                meeting_id,
            ),
        )


def list_meetings(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT m.id, m.title, m.started_at, m.ended_at, m.audio_seconds,"
            " m.usage_json, m.brief_json,"
            " (SELECT COUNT(*) FROM segments s WHERE s.meeting_id = m.id) AS segments"
            " FROM meetings m ORDER BY m.id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    meetings = []
    for row in rows:
        meeting = dict(row)
        meeting["usage_json"] = _json(meeting.get("usage_json"))
        meeting["brief_json"] = _json(meeting.get("brief_json"))
        meetings.append(meeting)
    return meetings


def get_meeting(meeting_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        if not row:
            return None
        meeting = dict(row)
        meeting["segments"] = [
            dict(r)
            for r in conn.execute(
                "SELECT idx, at, speaker, text, speaker_name FROM segments"
                " WHERE meeting_id = ? ORDER BY idx",
                (meeting_id,),
            ).fetchall()
        ]
        meeting["events"] = [
            {"at": r["at"], "kind": r["kind"], "payload": _json(r["payload"])}
            for r in conn.execute(
                "SELECT at, kind, payload FROM events WHERE meeting_id = ? ORDER BY id",
                (meeting_id,),
            ).fetchall()
        ]

    for key in ("notes_json", "usage_json", "brief_json", "speakers_json"):
        meeting[key] = _json(meeting.get(key))
    # JSON object keys are strings; diarisation speakers are ints.
    meeting["speaker_names"] = {
        int(k): v for k, v in (meeting.get("speakers_json") or {}).items() if str(k).isdigit()
    }
    return meeting


def _json(raw):
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {}
