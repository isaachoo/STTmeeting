"""SQLite persistence.

Written to as the meeting happens, so a browser refresh or a crash at hour four
does not lose the transcript. A fresh connection per call keeps this safe across
the several threads that touch it, and at meeting speed the cost is irrelevant.
"""

import json
import logging
import sqlite3
import time

import config

log = logging.getLogger(__name__)

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

-- ------------------------------------------------ the review workspace
-- Action items live in their own table rather than inside notes_json, because
-- after the meeting they stop being a model's opinion and become the user's
-- list: editable, tickable, and safe from the next regeneration.
CREATE TABLE IF NOT EXISTS action_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id),
    who        TEXT NOT NULL DEFAULT '',
    what       TEXT NOT NULL DEFAULT '',
    due        TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'open',
    source     TEXT NOT NULL DEFAULT 'user',
    position   INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS action_items_meeting
    ON action_items(meeting_id, position, id);

CREATE TABLE IF NOT EXISTS reports (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id),
    kind       TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL DEFAULT '',
    model      TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS reports_meeting ON reports(meeting_id, id);

CREATE TABLE IF NOT EXISTS review_chat (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id),
    at         REAL NOT NULL,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL DEFAULT '',
    cited      TEXT NOT NULL DEFAULT '[]',
    cost_usd   REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS review_chat_meeting ON review_chat(meeting_id, id);

-- Meeting backgrounds typed in advance and kept by name, so a recurring
-- meeting's brief is picked from a list instead of retyped at the door.
CREATE TABLE IF NOT EXISTS briefs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    brief_json TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL
);
"""

# Columns added after the first release. Applied on every init so an existing
# database from an earlier version keeps working instead of erroring.
MIGRATIONS = {
    "meetings": {
        "brief_json": "ALTER TABLE meetings ADD COLUMN brief_json TEXT NOT NULL DEFAULT '{}'",
        "speakers_json": "ALTER TABLE meetings ADD COLUMN speakers_json TEXT NOT NULL DEFAULT '{}'",
        "provider": "ALTER TABLE meetings ADD COLUMN provider TEXT NOT NULL DEFAULT ''",
        # The review digest: a condensed reading of the whole transcript, built
        # once and reused by every report and question. `digest_upto` records how
        # many segments went into it, so it can be rebuilt if that ever changes.
        "digest_json": "ALTER TABLE meetings ADD COLUMN digest_json TEXT NOT NULL DEFAULT '{}'",
        "digest_upto": "ALTER TABLE meetings ADD COLUMN digest_upto INTEGER NOT NULL DEFAULT 0",
        # How far the rolling summary reaches, so an interrupted meeting can be
        # resumed with the copilot's memory intact instead of re-reading hours.
        "summarised_upto": "ALTER TABLE meetings ADD COLUMN summarised_upto INTEGER NOT NULL DEFAULT 0",
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
    title: str, brief: dict, language: str, stt_model: str, provider: str = ""
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO meetings (title, brief, brief_json, started_at, language,"
            " stt_model, provider) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                title,
                # A plain-text copy so the row is readable without unpacking JSON.
                (brief or {}).get("context", ""),
                json.dumps(brief or {}, ensure_ascii=False),
                time.time(),
                language,
                stt_model,
                # Which transcriber heard this meeting -- resuming an interrupted
                # one picks the same engine again, and the history says which
                # engine a transcript came from when comparing them.
                (provider or "").strip().lower(),
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


def save_summary(meeting_id: int, summary: str, upto: int | None = None) -> None:
    if upto is None:
        _set(meeting_id, "summary", summary)
        return
    with _connect() as conn:
        conn.execute(
            "UPDATE meetings SET summary = ?, summarised_upto = ? WHERE id = ?",
            (summary, int(upto), meeting_id),
        )


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
    checkpoint()


def checkpoint() -> None:
    """Fold the write-ahead log into the main database file.

    In WAL mode recent writes live in `meetings.sqlite3-wal` until SQLite gets
    round to a checkpoint. That is fine on a plain disk, but a folder watched by
    OneDrive, Dropbox or Google Drive can upload, lock or replace that side file
    under us, and a meeting's lines vanish with it. Checkpointing at the end of
    every meeting puts the whole record in the one file the user can see.
    """
    try:
        with _connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error as exc:  # not worth failing the meeting over
        log.warning("could not checkpoint the database: %s", exc)


_SYNC_FOLDERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud", "box sync")


def storage_warning() -> str:
    """A plain-language warning when the data folder lives in a cloud-sync
    folder, or '' when it does not. Sync clients and SQLite do not mix."""
    lowered = str(config.DATA_DIR).lower()
    for marker in _SYNC_FOLDERS:
        if marker in lowered:
            return (
                f"The data folder is inside a {marker.title()} folder ({config.DATA_DIR}). "
                "Cloud sync tools can lock or replace the database while a meeting is "
                "being written, and lines go missing. Move the project out of the synced "
                "folder (for example to C:\\Users\\<you>\\STTmeeting) or pause syncing "
                "while the app runs."
            )
    return ""


def summary() -> dict:
    """What is on disk, for the startup log and the health check."""
    with _connect() as conn:
        meetings = conn.execute("SELECT COUNT(*) AS n FROM meetings").fetchone()["n"]
        lines = conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"]
        unfinished = conn.execute(
            "SELECT COUNT(*) AS n FROM meetings WHERE ended_at IS NULL"
        ).fetchone()["n"]
    return {
        "path": str(config.DB_PATH),
        "meetings": int(meetings),
        "lines": int(lines),
        "unfinished": int(unfinished),
        "warning": storage_warning(),
    }


def list_meetings(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT m.id, m.title, m.started_at, m.ended_at, m.audio_seconds,"
            " m.usage_json, m.brief_json, m.language, m.provider,"
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


def unfinished_meetings() -> list[dict]:
    """Meetings that were started and never stopped -- the process died, the
    laptop slept, the browser was closed. Candidates for resuming."""
    return [m for m in list_meetings(limit=200) if m.get("ended_at") is None]


def close_interrupted(meeting_id: int) -> bool:
    """Mark an interrupted meeting finished without resuming it. The end time is
    the last thing said, not now -- the meeting did not run until today."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT started_at, ended_at FROM meetings WHERE id = ?", (meeting_id,)
        ).fetchone()
        if row is None or row["ended_at"] is not None:
            return False
        last = conn.execute(
            "SELECT MAX(at) AS at FROM segments WHERE meeting_id = ?", (meeting_id,)
        ).fetchone()["at"]
        ended = row["started_at"] + float(last or 0)
        conn.execute("UPDATE meetings SET ended_at = ? WHERE id = ?", (ended, meeting_id))
        return True


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


# ------------------------------------------------------------ review workspace


def save_digest(meeting_id: int, digest: dict, upto: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE meetings SET digest_json = ?, digest_upto = ? WHERE id = ?",
            (json.dumps(digest, ensure_ascii=False), int(upto), meeting_id),
        )


def get_digest(meeting_id: int) -> tuple[dict, int]:
    """The cached digest and how many segments it covers. ({}, 0) if there is none."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT digest_json, digest_upto FROM meetings WHERE id = ?", (meeting_id,)
        ).fetchone()
    if not row:
        return {}, 0
    return _json(row["digest_json"]), int(row["digest_upto"] or 0)


def list_actions(meeting_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM action_items WHERE meeting_id = ?"
            " ORDER BY position, id",
            (meeting_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_action(
    meeting_id: int, who: str, what: str, due: str = "", source: str = "user"
) -> dict:
    with _connect() as conn:
        nxt = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 AS p FROM action_items"
            " WHERE meeting_id = ?",
            (meeting_id,),
        ).fetchone()["p"]
        cur = conn.execute(
            "INSERT INTO action_items (meeting_id, who, what, due, source, position,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (meeting_id, who, what, due, source, nxt, time.time()),
        )
        row = conn.execute(
            "SELECT * FROM action_items WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def update_action(meeting_id: int, action_id: int, fields: dict) -> dict | None:
    """Patch one item. Only the columns a user is allowed to change."""
    allowed = {"who", "what", "due", "status", "position"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_action(meeting_id, action_id)
    assignments = ", ".join(f"{name} = ?" for name in updates)
    with _connect() as conn:
        conn.execute(
            f"UPDATE action_items SET {assignments} WHERE id = ? AND meeting_id = ?",
            (*updates.values(), action_id, meeting_id),
        )
    return get_action(meeting_id, action_id)


def get_action(meeting_id: int, action_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM action_items WHERE id = ? AND meeting_id = ?",
            (action_id, meeting_id),
        ).fetchone()
    return dict(row) if row else None


def delete_action(meeting_id: int, action_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM action_items WHERE id = ? AND meeting_id = ?",
            (action_id, meeting_id),
        )
        return cur.rowcount > 0


def save_report(meeting_id: int, kind: str, title: str, body: str, model: str) -> dict:
    """One row per generation, so regenerating keeps the previous version."""
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO reports (meeting_id, kind, title, body, model, created_at,"
            " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (meeting_id, kind, title, body, model, now, now),
        )
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def update_report(report_id: int, body: str, title: str | None = None) -> dict | None:
    with _connect() as conn:
        if title is None:
            conn.execute(
                "UPDATE reports SET body = ?, updated_at = ? WHERE id = ?",
                (body, time.time(), report_id),
            )
        else:
            conn.execute(
                "UPDATE reports SET body = ?, title = ?, updated_at = ? WHERE id = ?",
                (body, title, time.time(), report_id),
            )
    return get_report(report_id)


def get_report(report_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    return dict(row) if row else None


def delete_report(report_id: int) -> bool:
    with _connect() as conn:
        return conn.execute("DELETE FROM reports WHERE id = ?", (report_id,)).rowcount > 0


def list_reports(meeting_id: int, with_body: bool = True) -> list[dict]:
    columns = "*" if with_body else "id, meeting_id, kind, title, model, created_at, updated_at"
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {columns} FROM reports WHERE meeting_id = ? ORDER BY id DESC",
            (meeting_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_chat_turn(
    meeting_id: int, question: str, answer: str, cited: list, cost_usd: float
) -> dict:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO review_chat (meeting_id, at, question, answer, cited, cost_usd)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                meeting_id,
                time.time(),
                question,
                answer,
                json.dumps(cited, ensure_ascii=False),
                float(cost_usd or 0),
            ),
        )
        row = conn.execute(
            "SELECT * FROM review_chat WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    turn = dict(row)
    turn["cited"] = _json_list(turn.get("cited"))
    return turn


def list_chat(meeting_id: int, limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM review_chat WHERE meeting_id = ? ORDER BY id LIMIT ?",
            (meeting_id, limit),
        ).fetchall()
    turns = []
    for row in rows:
        turn = dict(row)
        turn["cited"] = _json_list(turn.get("cited"))
        turns.append(turn)
    return turns


def clear_chat(meeting_id: int) -> int:
    with _connect() as conn:
        return conn.execute(
            "DELETE FROM review_chat WHERE meeting_id = ?", (meeting_id,)
        ).rowcount


# ------------------------------------------------------------- saved briefs


def list_briefs() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM briefs ORDER BY updated_at DESC").fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["brief"] = _json(item.pop("brief_json"))
        out.append(item)
    return out


def save_brief(name: str, brief: dict) -> dict:
    """Save under a name, replacing whatever that name held before."""
    name = (name or "").strip()[:120]
    if not name:
        raise ValueError("a saved brief needs a name")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO briefs (name, brief_json, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET brief_json = excluded.brief_json,"
            " updated_at = excluded.updated_at",
            (name, json.dumps(brief or {}, ensure_ascii=False), time.time()),
        )
        row = conn.execute("SELECT * FROM briefs WHERE name = ?", (name,)).fetchone()
    item = dict(row)
    item["brief"] = _json(item.pop("brief_json"))
    return item


def delete_brief(brief_id: int) -> bool:
    with _connect() as conn:
        return conn.execute("DELETE FROM briefs WHERE id = ?", (brief_id,)).rowcount > 0


def _json_list(raw) -> list:
    if isinstance(raw, list):
        return raw
    try:
        value = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return []
    return value if isinstance(value, list) else []


def _json(raw):
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {}
