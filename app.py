"""Flask server for the Cantonese meeting copilot.

Audio path: browser microphone -> AudioWorklet -> 16-bit PCM over a plain
WebSocket -> this server -> Deepgram streaming. The API keys stay here; the
browser never sees them, and the server owns the transcript so the copilot can
work on it.

Transport is a bare WebSocket rather than Socket.IO so the page has no external
script dependency -- a local, private tool should not need a CDN to start. One
connection carries both directions: JSON text frames for events, binary frames
for audio.

Each connection is served by one thread that owns its socket: it polls for
inbound frames and drains an outbound queue. Background threads (Deepgram
callbacks, copilot workers) only ever put messages on that queue, never touch
the socket, which keeps sends single-threaded without any locking.
"""

import json
import logging
import os
import queue
import threading

from flask import Flask, Response, jsonify, render_template, request
from flask_sock import Sock
from simple_websocket import ConnectionClosed

import config
import settings
from copilot import prompts, search
from copilot.brief import Brief
from copilot.llm import LLMError, OpenRouterClient
from review import digest as review_digest
from review import jobs as review_jobs
from review import qa as review_qa
from review import reports as review_reports
from session import MeetingSession
from storage import db, export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("app")

# Flask finds templates/ and static/ next to this file. Inside the Android app
# the Python code is unpacked somewhere else from the web files, so the two
# folders can be pointed at explicitly. Unset on a PC, and nothing changes.
app = Flask(
    __name__,
    template_folder=os.getenv("MEETING_TEMPLATE_DIR") or "templates",
    static_folder=os.getenv("MEETING_STATIC_DIR") or "static",
)
app.config["SECRET_KEY"] = config.SECRET_KEY
# No ping_interval on purpose. simple_websocket sends its keepalive PING from
# its own background reader thread, while this app sends events from the
# connection's handler thread. Both go through one wsproto connection and one
# stateful permessage-deflate compressor, so two writers corrupt the frame
# stream and the browser's socket dies -- once every ping interval, at random.
# Leaving pings off makes the handler thread the only writer. Nothing is lost:
# the meeting generates constant traffic, and the client reconnects by itself.
sock = Sock(app)

# Keys saved in the app take effect for every entry point, not just main().
settings.apply_to_config()

POLL_SECONDS = 0.02  # inbound poll interval; also the outbound flush cadence
OUTBOUND_MAX = 500  # per client; a wedged browser must not grow memory here

# One meeting at a time. The lock guards start/stop transitions only -- audio
# frames touch the session without it, keeping the hot path free of contention.
_session: MeetingSession | None = None
_session_lock = threading.Lock()


class _CompleteWrites:
    """Socket wrapper that guarantees every write finishes.

    simple_websocket writes with `sock.send(data)` and ignores the return value,
    but `send` is allowed to write fewer bytes than it was given. That would
    truncate a WebSocket frame on the wire and leave the browser waiting forever
    for the rest of a message it will never get. Small frames on a blocking
    socket rarely under-write, so this is hardening rather than a fix for an
    observed failure -- but a five-hour meeting sends a lot of frames, and
    `sendall` loops until the buffer is gone.
    """

    __slots__ = ("_sock",)

    def __init__(self, sock):
        self._sock = sock

    def send(self, data, *args, **kwargs):
        self._sock.sendall(data, *args, **kwargs)
        return len(data)

    def __getattr__(self, name):
        return getattr(self._sock, name)


class Client:
    """A connected browser tab and its outbound queue."""

    def __init__(self, ws):
        self.ws = ws
        self.out: queue.Queue[str] = queue.Queue(maxsize=OUTBOUND_MAX)
        self.dropped = 0

    def send(self, event: str, payload: dict) -> None:
        message = json.dumps({"event": event, "data": payload}, ensure_ascii=False)
        try:
            self.out.put_nowait(message)
        except queue.Full:
            # Shed the oldest message: on a stalled tab the newest state is the
            # only state worth showing.
            try:
                self.out.get_nowait()
                self.out.put_nowait(message)
            except (queue.Empty, queue.Full):
                pass
            self.dropped += 1

    def flush(self) -> None:
        """Write queued messages. Only ever called from this client's own
        thread -- see the note on ping_interval above; a second writer on one
        WebSocket corrupts the stream."""
        while True:
            try:
                message = self.out.get_nowait()
            except queue.Empty:
                return
            self.ws.send(message)


_clients: set[Client] = set()
_clients_lock = threading.Lock()


def _broadcast(event: str, payload: dict) -> None:
    """Queue an event for every connected tab. Safe from any thread."""
    with _clients_lock:
        clients = list(_clients)
    for client in clients:
        client.send(event, payload)


# ----------------------------------------------------------------------- routes


@app.get("/")
def index():
    return render_template(
        "index.html",
        missing_keys=config.missing_keys(),
        providers=config.STT_PROVIDER_CHOICES,
        default_provider=config.STT_PROVIDER,
        language=config.DEEPGRAM_LANGUAGE,
        models=", ".join(config.DEEPGRAM_MODELS),
        web_search=bool(config.TAVILY_API_KEY),
        attendee_enabled=config.ATTENDEE_ENABLED,
        attendee_mode=config.ATTENDEE_MODE,
        languages=config.DEEPGRAM_LANGUAGE_CHOICES,
        stt_models=config.DEEPGRAM_MODEL_CHOICES,
        default_language=config.DEEPGRAM_LANGUAGE,
        default_model=config.DEEPGRAM_MODELS[0] if config.DEEPGRAM_MODELS else "nova-3",
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": not config.missing_keys(),
            "missing_keys": config.missing_keys(),
            "language": config.DEEPGRAM_LANGUAGE,
            "stt_models": config.DEEPGRAM_MODELS,
            "advisor_model": config.OPENROUTER_MODEL,
            "notes_model": config.OPENROUTER_NOTES_MODEL,
            "web_search": bool(config.TAVILY_API_KEY),
            "meeting_running": bool(_session and not _session.stopped),
        }
    )


@app.get("/api/settings")
def get_settings():
    """Masked values only -- this never returns a usable key."""
    return jsonify(settings.describe())


@app.post("/api/settings")
def post_settings():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "expected an object"}), 400
    try:
        changed = settings.save(payload)
    except OSError as exc:
        return jsonify({"error": f"could not save: {exc}"}), 500
    return jsonify(
        {
            "changed": changed,
            "settings": settings.describe(),
            "missing_keys": config.missing_keys(),
            "providers_ready": {
                p["code"]: not config.missing_keys(p["code"])
                for p in config.STT_PROVIDER_CHOICES
            },
        }
    )


@app.get("/api/meetings")
def meetings():
    live = _session.meeting_id if (_session and not _session.stopped) else None
    rows = db.list_meetings()
    for row in rows:
        # Never stopped and not the one running now: it was interrupted, and
        # can be resumed or closed off.
        row["interrupted"] = row.get("ended_at") is None and row["id"] != live
        row["running"] = row["id"] == live
    return jsonify(rows)


@app.post("/api/meetings/<int:meeting_id>/finish")
def finish_interrupted(meeting_id: int):
    """Close an interrupted meeting without resuming it."""
    if _is_live(meeting_id):
        return jsonify({"error": "That meeting is running; press Stop instead."}), 409
    if not db.close_interrupted(meeting_id):
        return jsonify({"error": "not an unfinished meeting"}), 404
    return jsonify({"finished": meeting_id})


# ------------------------------------------------------------ general chat
#
# The side-panel assistant. Not about the meeting -- that is the Copilot panel's
# job -- so it does not need a running session and has nothing to persist on
# the server: the browser keeps the conversation and sends the recent turns.


@app.post("/api/chat")
def general_chat():
    if not config.OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY is not set"}), 400
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or "").strip()[:2000]
    if not question:
        return jsonify({"error": "no question"}), 400
    web = bool(payload.get("web"))
    history = payload.get("history")
    history = [t for t in history if isinstance(t, dict)] if isinstance(history, list) else []

    sources: list[dict] = []
    if web and search.available():
        sources = search.search(question)

    # A running meeting's brief helps the model understand the person's day; it
    # is context, not the subject.
    meeting_context = ""
    session = _session
    if session is not None and not session.stopped:
        with session.state.lock:
            meeting_context = session.state.brief.render()

    client = OpenRouterClient()
    try:
        answer = client.chat(
            prompts.general_chat_messages(question, sources, history, meeting_context),
            model=config.OPENROUTER_MODEL,
            temperature=0.3,
            max_tokens=900,
        ).strip()
    except LLMError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify(
        {
            "question": question,
            "answer": answer,
            "sources": [{"title": s.get("title", ""), "url": s.get("url", "")} for s in sources],
            "searched": bool(sources),
            "web_requested": web,
            "web_enabled": search.available(),
            "cost_usd": client.usage.snapshot().get("cost_usd", 0.0),
        }
    )


# --------------------------------------------------------------- saved briefs


@app.get("/api/briefs")
def briefs_list():
    return jsonify({"briefs": db.list_briefs()})


@app.post("/api/briefs")
def briefs_save():
    payload = request.get_json(silent=True) or {}
    brief = Brief.from_payload(payload.get("brief"))
    name = str(payload.get("name") or brief.title or "").strip()
    if not name:
        return jsonify({"error": "give the brief a name"}), 400
    saved = db.save_brief(name, brief.as_dict())
    return jsonify({"saved": saved, "briefs": db.list_briefs()})


@app.delete("/api/briefs/<int:brief_id>")
def briefs_delete(brief_id: int):
    if not db.delete_brief(brief_id):
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": brief_id, "briefs": db.list_briefs()})


@app.get("/api/meetings/<int:meeting_id>")
def meeting_detail(meeting_id: int):
    meeting = db.get_meeting(meeting_id)
    if meeting is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(meeting)


@app.get("/api/meetings/<int:meeting_id>/export.md")
def meeting_markdown(meeting_id: int):
    return _download(meeting_id, export.to_markdown, "md", "text/markdown")


@app.get("/api/meetings/<int:meeting_id>/export.json")
def meeting_json(meeting_id: int):
    return _download(meeting_id, export.to_json, "json", "application/json")


def _download(meeting_id: int, render, suffix: str, mimetype: str):
    """Save-everything download. Works for a meeting that is still running --
    the transcript so far is already in the database."""
    meeting = db.get_meeting(meeting_id)
    if meeting is None:
        return jsonify({"error": "not found"}), 404

    # A running meeting has not written its final notes yet, so take the live
    # ones; otherwise the download would be missing the last few minutes.
    session = _session
    if session is not None and session.meeting_id == meeting_id and not session.stopped:
        with session.state.lock:
            meeting["notes_json"] = dict(session.state.notes)
            meeting["user_notes"] = session.state.user_notes
            meeting["summary"] = session.state.rolling_summary
            meeting["speaker_names"] = dict(session.state.speaker_names)
        meeting["audio_seconds"] = session.stt.audio_seconds
        meeting["usage_json"] = session.llm.usage.snapshot()
        meeting["stt_model"] = session.stt.model

    body = render(meeting)
    name = f"{export.filename_stem(meeting)}.{suffix}"
    return Response(
        body,
        mimetype=mimetype,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# ------------------------------------------------------------ review workspace
#
# Everything below works on a stored meeting, over plain HTTP. The live meeting
# uses a WebSocket because it pushes; review is request/response, and a page you
# can reload and bookmark is worth more here than a persistent connection.


@app.get("/review/<int:meeting_id>")
def review_page(meeting_id: int):
    meeting = db.get_meeting(meeting_id)
    if meeting is None:
        return render_template("review.html", meeting=None, meeting_id=meeting_id), 404
    return render_template(
        "review.html",
        meeting=meeting,
        meeting_id=meeting_id,
        report_kinds=config.REPORT_KINDS,
        review_model=config.review_model(),
        # Only the LLM key matters here. Nothing on this page transcribes
        # anything, so complaining about a missing speech key would be noise.
        missing_keys=[] if config.OPENROUTER_API_KEY else ["OPENROUTER_API_KEY"],
    )


@app.get("/api/meetings/<int:meeting_id>/review")
def review_data(meeting_id: int):
    """One request for everything the review page needs to draw itself."""
    meeting = db.get_meeting(meeting_id)
    if meeting is None:
        return jsonify({"error": "not found"}), 404

    digest, upto = db.get_digest(meeting_id)
    segments = meeting.get("segments") or []
    running = bool(
        _session and _session.meeting_id == meeting_id and not _session.stopped
    )
    job = review_jobs.running_for(meeting_id)
    return jsonify(
        {
            "meeting": {
                key: meeting.get(key)
                for key in (
                    "id", "title", "started_at", "ended_at", "audio_seconds",
                    "language", "stt_model", "provider", "summary", "user_notes",
                    "notes_json", "brief_json", "usage_json",
                )
            },
            "segments": segments,
            "speaker_names": {str(k): v for k, v in (meeting.get("speaker_names") or {}).items()},
            "running": running,
            "actions": db.list_actions(meeting_id),
            "reports": db.list_reports(meeting_id, with_body=False),
            "chat": db.list_chat(meeting_id),
            "digest": {
                "built": bool(digest),
                "stale": bool(digest) and upto != len(segments),
                "sections": (digest or {}).get("sections", 0),
                "failed_sections": (digest or {}).get("failed_sections", []),
                # An estimate so the page can say what building it will involve
                # before the user commits to paying for it.
                "estimated_sections": len(
                    review_digest.chunk_transcript(
                        segments, meeting.get("speaker_names") or {}
                    )
                ),
            },
            "job": job.as_dict() if job else None,
        }
    )


@app.post("/api/meetings/<int:meeting_id>/segments/<int:index>/speaker")
def review_name_segment(meeting_id: int, index: int):
    """Fix the speaker on one line, after the meeting."""
    if _is_live(meeting_id):
        return jsonify({"error": "This meeting is still running — rename in the live view."}), 409
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name") or "").strip()[:80]
    db.set_segment_speaker(meeting_id, index, name)
    return jsonify({"index": index, "speaker_name": name})


@app.post("/api/meetings/<int:meeting_id>/speakers")
def review_name_speaker(meeting_id: int):
    """Rename a whole voice, after the meeting."""
    if _is_live(meeting_id):
        return jsonify({"error": "This meeting is still running — rename in the live view."}), 409
    meeting = db.get_meeting(meeting_id)
    if meeting is None:
        return jsonify({"error": "not found"}), 404
    payload = request.get_json(silent=True) or {}
    try:
        speaker = int(payload.get("speaker"))
    except (TypeError, ValueError):
        return jsonify({"error": "speaker must be a number"}), 400

    names = dict(meeting.get("speaker_names") or {})
    name = str(payload.get("name") or "").strip()[:80]
    if name:
        names[speaker] = name
    else:
        names.pop(speaker, None)
    db.save_speakers(meeting_id, names)
    return jsonify({"speaker_names": {str(k): v for k, v in names.items()}})


@app.post("/api/meetings/<int:meeting_id>/ask")
def review_ask(meeting_id: int):
    """One question about the meeting. Synchronous: it is a single LLM call."""
    meeting = db.get_meeting(meeting_id)
    if meeting is None:
        return jsonify({"error": "not found"}), 404
    if not config.OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY is not set"}), 400

    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "no question"}), 400

    try:
        result = review_qa.ask(meeting, question, history=db.list_chat(meeting_id))
    except LLMError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify(result)


@app.delete("/api/meetings/<int:meeting_id>/chat")
def review_clear_chat(meeting_id: int):
    return jsonify({"removed": db.clear_chat(meeting_id)})


# ------------------------------------------------------------- action items


@app.get("/api/meetings/<int:meeting_id>/actions")
def review_actions(meeting_id: int):
    return jsonify({"actions": db.list_actions(meeting_id)})


@app.post("/api/meetings/<int:meeting_id>/actions")
def review_add_action(meeting_id: int):
    """Add one item, or several at once when accepting a batch of proposals."""
    if db.get_meeting(meeting_id) is None:
        return jsonify({"error": "not found"}), 404
    payload = request.get_json(silent=True) or {}
    items = payload.get("items")
    raws = items if isinstance(items, list) else [payload]

    added = []
    for raw in raws[:100]:
        if not isinstance(raw, dict):
            continue
        what = str(raw.get("what") or "").strip()[:500]
        if not what:
            continue
        added.append(
            db.add_action(
                meeting_id,
                who=str(raw.get("who") or "").strip()[:80],
                what=what,
                due=str(raw.get("due") or "").strip()[:80],
                source=("copilot" if raw.get("source") == "copilot" else "user"),
            )
        )
    if not added:
        return jsonify({"error": "nothing to add"}), 400
    return jsonify({"added": added, "actions": db.list_actions(meeting_id)})


@app.patch("/api/meetings/<int:meeting_id>/actions/<int:action_id>")
def review_update_action(meeting_id: int, action_id: int):
    payload = request.get_json(silent=True) or {}
    fields = {}
    for name, limit in (("who", 80), ("what", 500), ("due", 80)):
        if name in payload:
            fields[name] = str(payload[name] or "").strip()[:limit]
    if "status" in payload:
        fields["status"] = "done" if payload["status"] == "done" else "open"
    if "position" in payload:
        try:
            fields["position"] = int(payload["position"])
        except (TypeError, ValueError):
            pass

    updated = db.update_action(meeting_id, action_id, fields)
    if updated is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"action": updated})


@app.delete("/api/meetings/<int:meeting_id>/actions/<int:action_id>")
def review_delete_action(meeting_id: int, action_id: int):
    if not db.delete_action(meeting_id, action_id):
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": action_id})


@app.post("/api/meetings/<int:meeting_id>/actions/draft")
def review_draft_actions(meeting_id: int):
    """Ask the copilot what it thinks the action items are. Saves nothing."""
    return _start_review_job(
        meeting_id,
        kind="draft_actions",
        label="Drafting action items",
        work=lambda meeting, job: review_reports.draft_actions(
            meeting, on_progress=_progress(job)
        ),
    )


# ----------------------------------------------------------------- reports


@app.get("/api/meetings/<int:meeting_id>/reports")
def review_reports_list(meeting_id: int):
    return jsonify({"reports": db.list_reports(meeting_id)})


@app.post("/api/meetings/<int:meeting_id>/reports")
def review_generate_report(meeting_id: int):
    payload = request.get_json(silent=True) or {}
    kind = str(payload.get("kind") or "").strip()
    if kind not in review_reports.KINDS:
        return jsonify({"error": f"unknown report kind: {kind}"}), 400
    label = review_reports.KINDS[kind]["label"]
    return _start_review_job(
        meeting_id,
        kind=f"report:{kind}",
        label=f"Writing the {label.lower()}",
        work=lambda meeting, job: review_reports.generate(
            meeting, kind, on_progress=_progress(job)
        ),
    )


@app.post("/api/meetings/<int:meeting_id>/digest")
def review_build_digest(meeting_id: int):
    """Read the whole transcript up front, so later work is fast and cheap."""
    return _start_review_job(
        meeting_id,
        kind="digest",
        label="Reading the meeting",
        work=lambda meeting, job: _digest_result(
            *review_digest.ensure(meeting, on_progress=_progress(job, "reading"))
        ),
    )


@app.get("/api/reports/<int:report_id>")
def review_get_report(report_id: int):
    report = db.get_report(report_id)
    if report is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"report": report})


@app.put("/api/reports/<int:report_id>")
def review_save_report(report_id: int):
    """Save the user's edits. A report is a draft until they have fixed it."""
    payload = request.get_json(silent=True) or {}
    if "body" not in payload:
        return jsonify({"error": "no body"}), 400
    title = payload.get("title")
    updated = db.update_report(
        report_id,
        str(payload["body"])[:200_000],
        None if title is None else str(title)[:200],
    )
    if updated is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"report": updated})


@app.delete("/api/reports/<int:report_id>")
def review_delete_report(report_id: int):
    if not db.delete_report(report_id):
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": report_id})


@app.get("/api/reports/<int:report_id>/download.md")
def review_download_report(report_id: int):
    report = db.get_report(report_id)
    if report is None:
        return jsonify({"error": "not found"}), 404
    meeting = db.get_meeting(report["meeting_id"]) or {}
    name = review_reports.filename_for(report, meeting)
    return Response(
        report["body"],
        mimetype="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/api/jobs/<job_id>")
def review_job(job_id: str):
    job = review_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"job": job.as_dict()})


def _is_live(meeting_id: int) -> bool:
    session = _session
    return bool(session and session.meeting_id == meeting_id and not session.stopped)


def _progress(job, stage: str = ""):
    """Adapt a job to the two progress shapes the review code reports."""

    def report(*args):
        if len(args) == 3:
            job.progress(args[0], args[1], args[2])
        elif len(args) == 2:
            job.progress(stage or job.stage or "working", args[0], args[1])

    return report


def _digest_result(digest: dict, built: bool) -> dict:
    return {
        "built": built,
        "sections": digest.get("sections", 0),
        "failed_sections": digest.get("failed_sections", []),
        "counts": {
            key: len(digest.get(key) or [])
            for key in ("topics", "decisions", "actions", "questions", "facts")
        },
    }


def _start_review_job(meeting_id: int, kind: str, label: str, work):
    meeting = db.get_meeting(meeting_id)
    if meeting is None:
        return jsonify({"error": "not found"}), 404
    if not config.OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY is not set"}), 400
    try:
        job = review_jobs.start(meeting_id, kind, label, lambda job: work(meeting, job))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"job": job.as_dict()}), 202


# -------------------------------------------------------------------- websocket


@sock.route("/ws")
def ws_route(ws):
    ws.sock = _CompleteWrites(ws.sock)
    client = Client(ws)
    with _clients_lock:
        _clients.add(client)
    log.info("client connected (%d open)", len(_clients))

    try:
        client.send("snapshot", _snapshot())
        while True:
            client.flush()
            frame = ws.receive(timeout=POLL_SECONDS)
            if frame is None:  # poll timeout, not a disconnect
                continue
            if isinstance(frame, (bytes, bytearray)):
                _handle_audio(bytes(frame))
            else:
                _handle_event(client, frame)
    except (ConnectionClosed, OSError):
        # A closed tab shows up as ConnectionClosed, or as a broken pipe / reset
        # if it goes away mid-send. Both are normal, not worth a traceback.
        pass
    except Exception:
        log.exception("websocket handler failed")
    finally:
        with _clients_lock:
            _clients.discard(client)
        log.info("client disconnected (%d open)", len(_clients))


def _snapshot() -> dict:
    if _session is not None:
        return _session.snapshot()
    return {"running": False, "web_search": bool(config.TAVILY_API_KEY)}


def _handle_audio(chunk: bytes) -> None:
    session = _session
    if session is not None and not session.stopped:
        session.feed_audio(chunk)


def _handle_event(client: Client, raw: str) -> None:
    try:
        message = json.loads(raw)
        event = message.get("event")
        data = message.get("data")
    except (ValueError, AttributeError):
        log.warning("ignoring malformed frame: %.120s", raw)
        return

    # `data` must be an object. A client that sends a string or a list gets its
    # frame ignored -- every handler below assumes .get() works.
    if not isinstance(data, dict):
        if data is not None:
            log.warning("ignoring %r frame with non-object data", event)
            return
        data = {}

    if event == "start_meeting":
        _start_meeting(client, data)
    elif event == "stop_meeting":
        _stop_meeting()
    elif event == "pause":
        if _session is not None:
            _session.set_paused(bool(data.get("paused")))
    elif event == "ask":
        session = _session
        question = (data.get("question") or "").strip()
        if session is not None and question:
            session.ask(question[:1000], web=bool(data.get("web")))
    elif event == "user_notes":
        if _session is not None:
            _session.set_user_notes((data.get("text") or "")[:100_000])
    elif event == "name_speaker":
        _name_speaker(data)
    elif event == "name_segment":
        _name_segment(data)
    elif event == "speaker_suggestion":
        if _session is not None:
            _session.apply_speaker_suggestion(bool(data.get("accept")))
    elif event == "resync":
        client.send("snapshot", _snapshot())
    else:
        log.warning("unknown event %r", event)


def _name_segment(data: dict) -> None:
    session = _session
    if session is None:
        return
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError):
        return
    session.set_segment_speaker(index, str(data.get("name") or ""))


def _name_speaker(data: dict) -> None:
    session = _session
    if session is None:
        return
    try:
        speaker = int(data.get("speaker"))
    except (TypeError, ValueError):
        return
    if not 0 <= speaker <= 64:
        return
    session.set_speaker_name(speaker, str(data.get("name") or ""))


def _start_meeting(client: Client, data: dict) -> None:
    global _session

    provider = (data.get("provider") or config.STT_PROVIDER).strip().lower()
    missing = config.missing_keys(provider)
    if missing:
        client.send("error", {"message": f"Missing configuration: {', '.join(missing)}"})
        return

    # Continuing an interrupted meeting: the stored row supplies the brief and
    # everything said so far; the form only supplies the transcriber settings.
    resume = None
    if data.get("resume_meeting_id") is not None:
        try:
            resume = db.get_meeting(int(data["resume_meeting_id"]))
        except (TypeError, ValueError):
            resume = None
        if resume is None:
            client.send("error", {"message": "That meeting no longer exists."})
            return
        if resume.get("ended_at") is not None:
            client.send("error", {"message": "That meeting was finished; start a new one."})
            return

    with _session_lock:
        if _session is not None and not _session.stopped:
            client.send("error", {"message": "A meeting is already running."})
            return
        try:
            brief_payload = data.get("brief")
            if resume is not None and not brief_payload:
                brief_payload = resume.get("brief_json") or {}
            session = MeetingSession(
                brief=Brief.from_payload(brief_payload),
                sample_rate=data.get("sample_rate"),
                emit=_broadcast,
                language=(data.get("language") or "").strip()[:20],
                model=(data.get("model") or "").strip()[:40],
                provider=provider,
                resume=resume,
            )
        except ValueError as exc:
            client.send("error", {"message": str(exc)})
            return
        _session = session

    session.start()
    log.info(
        "meeting %s %s at %s Hz",
        session.meeting_id, "resumed" if resume else "started", session.sample_rate,
    )
    _broadcast("meeting_started", session.snapshot())
    threading.Thread(
        target=_cost_ticker, args=(session,), name="cost-ticker", daemon=True
    ).start()


def _stop_meeting() -> None:
    session = _session
    if session is None or session.stopped:
        return
    # Off the connection thread: the final notes pass makes a blocking LLM call.
    threading.Thread(target=session.stop, name="meeting-stop", daemon=True).start()


def _cost_ticker(session: MeetingSession) -> None:
    """Keep the cost readout moving through long stretches of silence."""
    while not session.stopped:
        if session.stopped_event.wait(5):
            break
        _broadcast("cost", session.cost())


def main() -> None:
    db.init()
    settings.apply_to_config()
    missing = config.missing_keys()
    if missing:
        log.warning(
            "Missing %s -- copy .env.example to .env and fill it in. The UI will "
            "load but will not be able to start a meeting.",
            ", ".join(missing),
        )
    log.info("open http://%s:%s", config.HOST, config.PORT)
    app.run(
        host=config.HOST,
        port=config.PORT,
        threaded=True,  # one thread per WebSocket connection
        debug=False,
    )


if __name__ == "__main__":
    main()
