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
import queue
import threading

from flask import Flask, Response, jsonify, render_template
from flask_sock import Sock
from simple_websocket import ConnectionClosed

import config
from copilot.brief import Brief
from session import MeetingSession
from storage import db, export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("app")

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
# No ping_interval on purpose. simple_websocket sends its keepalive PING from
# its own background reader thread, while this app sends events from the
# connection's handler thread. Both go through one wsproto connection and one
# stateful permessage-deflate compressor, so two writers corrupt the frame
# stream and the browser's socket dies -- once every ping interval, at random.
# Leaving pings off makes the handler thread the only writer. Nothing is lost:
# the meeting generates constant traffic, and the client reconnects by itself.
sock = Sock(app)

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
        language=config.DEEPGRAM_LANGUAGE,
        models=", ".join(config.DEEPGRAM_MODELS),
        web_search=bool(config.TAVILY_API_KEY),
        attendee_enabled=config.ATTENDEE_ENABLED,
        attendee_mode=config.ATTENDEE_MODE,
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


@app.get("/api/meetings")
def meetings():
    return jsonify(db.list_meetings())


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
    elif event == "ask":
        session = _session
        question = (data.get("question") or "").strip()
        if session is not None and question:
            session.ask(question[:1000])
    elif event == "user_notes":
        if _session is not None:
            _session.set_user_notes((data.get("text") or "")[:100_000])
    elif event == "name_speaker":
        _name_speaker(data)
    elif event == "speaker_suggestion":
        if _session is not None:
            _session.apply_speaker_suggestion(bool(data.get("accept")))
    elif event == "resync":
        client.send("snapshot", _snapshot())
    else:
        log.warning("unknown event %r", event)


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

    missing = config.missing_keys()
    if missing:
        client.send("error", {"message": f"Missing configuration: {', '.join(missing)}"})
        return

    with _session_lock:
        if _session is not None and not _session.stopped:
            client.send("error", {"message": "A meeting is already running."})
            return
        try:
            session = MeetingSession(
                brief=Brief.from_payload(data.get("brief")),
                sample_rate=data.get("sample_rate"),
                emit=_broadcast,
            )
        except ValueError as exc:
            client.send("error", {"message": str(exc)})
            return
        _session = session

    session.start()
    log.info("meeting %s started at %s Hz", session.meeting_id, session.sample_rate)
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
