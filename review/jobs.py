"""Background jobs for the slow parts of review.

Reading a five-hour transcript is a dozen or more LLM calls and can take a
minute or two. Doing that inside an HTTP request means a browser spinner with no
progress, a proxy timeout, and no way to tell whether anything is happening. So
these run on a thread and the page polls for progress.

Deliberately small: an in-process dictionary, not a task queue. This is a
single-user local app, and a job that dies with the process is the correct
behaviour -- nothing was half-written, because the digest and the report are
each saved in one step at the end.

One job at a time per meeting. Two report generations running together would
both read the transcript, and the user would pay twice for the same work.
"""

import logging
import threading
import time
import uuid

log = logging.getLogger(__name__)

KEEP = 40  # finished jobs to remember, so a slow poll still finds its result

_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()


class Job:
    def __init__(self, meeting_id: int, kind: str, label: str):
        self.id = uuid.uuid4().hex[:12]
        self.meeting_id = meeting_id
        self.kind = kind
        self.label = label
        self.status = "running"  # running | done | error | cancelled
        self.stage = ""
        self.done = 0
        self.total = 0
        self.result = None
        self.error = ""
        self.started_at = time.time()
        self.finished_at = None

    def progress(self, stage: str, done: int = 0, total: int = 0) -> None:
        self.stage = stage
        self.done = done
        self.total = total

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "stage": self.stage,
            "done": self.done,
            "total": self.total,
            "result": self.result,
            "error": self.error,
            "elapsed": round((self.finished_at or time.time()) - self.started_at, 1),
        }


def running_for(meeting_id: int) -> Job | None:
    with _lock:
        for job in _jobs.values():
            if job.meeting_id == meeting_id and job.status == "running":
                return job
    return None


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def start(meeting_id: int, kind: str, label: str, work) -> Job:
    """Run `work(job)` on a thread. Raises RuntimeError if one is already going.

    `work` receives the job so it can report progress, and whatever it returns
    becomes the job's result.
    """
    with _lock:
        for job in _jobs.values():
            if job.meeting_id == meeting_id and job.status == "running":
                raise RuntimeError(f"already {job.label.lower()} for this meeting")
        job = Job(meeting_id, kind, label)
        _jobs[job.id] = job
        _forget_old()

    def run() -> None:
        started = time.time()
        log.info("review job %s (%s) started for meeting %s", job.id, kind, meeting_id)
        try:
            result = work(job)
            if job.status == "cancelled":
                # The user gave up waiting; whatever came back is not wanted.
                log.info("review job %s (%s) finished after being cancelled; result dropped",
                         job.id, kind)
                return
            job.result = result
            job.status = "done"
            log.info("review job %s (%s) done in %.0fs", job.id, kind, time.time() - started)
        except Exception as exc:  # the browser has to be told, whatever broke
            if job.status == "cancelled":
                return
            log.exception("review job %s (%s) failed", job.id, kind)
            job.error = str(exc) or exc.__class__.__name__
            job.status = "error"
        finally:
            if job.finished_at is None:
                job.finished_at = time.time()

    threading.Thread(target=run, name=f"review-{kind}-{meeting_id}", daemon=True).start()
    return job


def cancel(job_id: str) -> Job | None:
    """Give up on a job. The model call already in flight cannot be interrupted,
    but the meeting is freed for a new job at once and the late result is
    thrown away when it arrives."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        if job.status == "running":
            job.status = "cancelled"
            job.finished_at = time.time()
            job.error = "cancelled"
    log.info("review job %s (%s) cancelled by the user", job.id, job.kind)
    return job


def _forget_old() -> None:
    """Called with the lock held."""
    finished = sorted(
        (j for j in _jobs.values() if j.status != "running"),
        key=lambda j: j.finished_at or 0,
    )
    for job in finished[: max(0, len(finished) - KEEP)]:
        _jobs.pop(job.id, None)
