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


def create_meeting(title: str, brief: str, language: str, stt_model: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO meetings (title, brief, started_at, language, stt_model)"
            " VALUES (?, ?, ?, ?, ?)",
            (title, brief, time.time(), language, stt_model),
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


def save_notes(meeting_id: int, notes: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE meetings SET notes_json = ? WHERE id = ?",
            (json.dumps(notes, ensure_ascii=False), meeting_id),
        )


def save_user_notes(meeting_id: int, text: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE meetings SET user_notes = ? WHERE id = ?", (text, meeting_id)
        )


def save_summary(meeting_id: int, summary: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE meetings SET summary = ? WHERE id = ?", (summary, meeting_id)
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
            "SELECT id, title, brief, started_at, ended_at, audio_seconds, usage_json"
            " FROM meetings ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_meeting(meeting_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
        ).fetchone()
        if not row:
            return None
        meeting = dict(row)
        meeting["segments"] = [
            dict(r)
            for r in conn.execute(
                "SELECT idx, at, speaker, text FROM segments"
                " WHERE meeting_id = ? ORDER BY idx",
                (meeting_id,),
            ).fetchall()
        ]
        meeting["events"] = [
            {**dict(r), "payload": json.loads(r["payload"])}
            for r in conn.execute(
                "SELECT at, kind, payload FROM events WHERE meeting_id = ? ORDER BY id",
                (meeting_id,),
            ).fetchall()
        ]
    for key in ("notes_json", "usage_json"):
        try:
            meeting[key] = json.loads(meeting.get(key) or "{}")
        except ValueError:
            meeting[key] = {}
    return meeting
