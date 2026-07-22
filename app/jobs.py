"""In-memory background job manager for the (slow) generation pipeline."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pw_access

from . import config, document, gemini_client, page_images, paper_parser, solver

# Jobs run in a background thread, so an exception here never reaches the request log.
# Log every stage (with timings) and the full traceback, otherwise a slow or crashed
# job is invisible in production and the UI just sits on its last progress message.
log = logging.getLogger("app.jobs")

_STEP_LABELS = (
    ("queued", "Queued"),
    ("access", "Checking access"),
    ("upload", "Preparing upload"),
    ("parse", "Reading paper"),
    ("images", "Preparing pages"),
    ("solve", "Writing answers"),
    ("document", "Building document"),
)
_STEP_ORDER = {key: index for index, (key, _) in enumerate(_STEP_LABELS)}


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | parsing | solving | writing | done | error
    message: str = "Queued..."
    step: str = "queued"
    step_detail: str = "Waiting for the background worker."
    phase_percent: int = 3
    done: int = 0
    total: int = 0
    error: str | None = None
    title: str | None = None
    docx_name: str | None = None
    pdf_name: str | None = None
    pdf_note: str | None = None

    @property
    def percent(self) -> int:
        if self.status == "done":
            return 100
        step = self.step or "queued"
        if step == "solve" and self.total > 0:
            return min(94, 58 + int(self.done / self.total * 36))
        return max(1, min(99, int(self.phase_percent or 3)))

    @property
    def steps(self) -> list[dict[str, str]]:
        current = _STEP_ORDER.get(self.step or "queued", 0)
        rows = []
        for key, label in _STEP_LABELS:
            order = _STEP_ORDER[key]
            if self.status == "done" or order < current:
                state = "done"
            elif order == current:
                state = "error" if self.status == "error" else "active"
            else:
                state = "pending"
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "state": state,
                    "detail": self.step_detail if order == current else "",
                }
            )
        return rows


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_JOB_ID_RE = re.compile(r"^[a-fA-F0-9]{32}$")


def get_job(job_id: str) -> Job | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            return job
    return _load_job(job_id)


def _job_path(job_id: str) -> Path | None:
    if not _JOB_ID_RE.fullmatch(job_id):
        return None
    return config.JOB_DIR / f"{job_id}.json"


def _job_data(job: Job) -> dict:
    return {name: getattr(job, name) for name in Job.__dataclass_fields__}


def _save_job(job: Job) -> None:
    path = _job_path(job.id)
    if path is None:
        return
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_job_data(job), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_job(job_id: str) -> Job | None:
    path = _job_path(job_id)
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {
            name: data[name]
            for name in Job.__dataclass_fields__
            if name in data
        }
        job = Job(**allowed)
    except Exception:  # noqa: BLE001 - a bad cache file should behave like a missing job
        return None
    with _LOCK:
        _JOBS[job.id] = job
    return job


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^\w\-]+", "_", Path(name).stem).strip("_")
    return stem or "solutions"


def _run(
    job: Job,
    paper_path: Path,
    language: str,
    class_level: str,
    subject: str,
    board: str,
    google_token: str,
) -> None:
    usage = pw_access.UsageSession(
        google_token,
        filename=paper_path.name,
        input_unit="No. of questions",
        count=None,
    )
    started = time.monotonic()

    def _stage(msg: str) -> None:
        log.info("job %s [%6.1fs] %s", job.id[:8], time.monotonic() - started, msg)

    def _set_step(
        step: str,
        message: str,
        detail: str = "",
        percent: int | None = None,
    ) -> None:
        job.step = step
        job.message = message
        job.step_detail = detail
        if percent is not None:
            job.phase_percent = percent

    try:
        _stage(f"start: {paper_path.name} lang={language} subject={subject}")
        _set_step(
            "access",
            "Checking access",
            "Verifying your PW access before generation.",
            5,
        )
        if not pw_access.check_allowed(google_token):
            raise PermissionError("Not authorized for this app.")
        gemini_client.set_proxy_context(google_token, usage)
        job.status = "parsing"
        _set_step(
            "upload",
            "Preparing the question paper",
            "Uploading visual files when needed.",
            10,
        )
        # Upload PDFs/images once and reuse for parsing AND for figure questions.
        uploaded = paper_parser.upload_if_multimodal(paper_path)
        _stage("uploaded; parsing paper (OCR + structure)…")

        def parse_progress(detail: str, percent: int | None = None) -> None:
            _set_step("parse", "Reading the question paper", detail, percent)

        parse_progress("Starting OCR and question-structure extraction.", 15)
        paper = paper_parser.parse_paper(
            paper_path,
            uploaded=uploaded,
            progress=parse_progress,
        )
        _stage(f"parsed: {paper.n_questions} questions / {paper.n_units} units")
        job.title = paper.title
        job.total = paper.n_units
        usage.count = paper.n_units
        _set_step(
            "parse",
            "Question paper read",
            f"Found {paper.n_questions} questions / {paper.n_units} answer units.",
            50,
        )

        # High-res page rasters let figure questions read the actual drawn structures
        # (substituents, subscripts, MCQ structures) instead of guessing. Best-effort:
        # an empty list just means figure questions fall back to the uploaded PDF.
        _set_step(
            "images",
            "Preparing page images",
            "Rendering pages for figure and math questions.",
            52,
        )
        paper_pages = page_images.render_pages(paper_path)
        _stage(f"rendered {len(paper_pages)} page image(s)")
        _set_step(
            "images",
            "Preparing page images",
            f"Rendered {len(paper_pages)} page image(s).",
            54,
        )
        # Routing a figure question to its own page needs per-page text aligned with
        # those images. Prefer the markers already in the transcription; if they're
        # absent or miscounted, re-derive from the images so targeting still works.
        if paper_pages and len(paper.page_texts) != len(paper_pages):
            _set_step(
                "images",
                "Preparing page images",
                "Aligning page text for figure routing.",
                55,
            )
            paper.page_texts = paper_parser.page_texts_from_images(paper_pages)
            _stage("re-derived per-page text for figure routing")

        job.status = "solving"
        _set_step(
            "solve",
            "Writing answers",
            f"Preparing to answer {paper.n_units} unit(s).",
            58,
        )
        _stage(f"solving {paper.n_units} unit(s)…")

        def progress(done: int, total: int, label: str) -> None:
            job.done = done
            job.total = total
            if label != "done":
                _set_step(
                    "solve",
                    f"Answering {label}",
                    f"Answer unit {done + 1} of {total}.",
                )
            else:
                _set_step(
                    "solve",
                    "Answers drafted",
                    f"Completed {total} answer units.",
                    94,
                )

        solved = solver.solve_paper(
            paper,
            language,
            class_level=class_level or None,
            subject=subject or None,
            board=board or None,
            paper_file=uploaded,
            paper_pages=paper_pages,
            progress=progress,
        )

        job.status = "writing"
        _set_step(
            "document",
            "Building the solution document",
            "Creating DOCX and PDF output.",
            96,
        )
        _stage("solved; building the document…")
        base = config.OUTPUT_DIR / f"{_safe_stem(paper_path.name)}_{language}_{job.id[:8]}"
        docx_path, pdf_path = document.build_documents(solved, base)
        job.docx_name = docx_path.name
        if pdf_path is not None:
            job.pdf_name = pdf_path.name
        else:
            job.pdf_note = (
                "PDF export needs Microsoft Word or LibreOffice installed — "
                "the .docx is ready and opens everywhere."
            )

        job.status = "done"
        job.message = "Done."
        job.step = "document"
        job.step_detail = "The solution document is ready."
        job.phase_percent = 100
        _stage("DONE")
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = str(exc)
        job.message = "Failed."
        job.step_detail = str(exc) or "Generation failed."
        # Full traceback: this thread's exceptions never surface in the request log,
        # and job.error keeps only str(exc), which is often uninformative on its own.
        log.exception(
            "job %s FAILED after %.1fs during %s: %s",
            job.id[:8], time.monotonic() - started, job.status, exc,
        )
    finally:
        gemini_client.clear_proxy_context()
        if job.status in {"done", "error"}:
            try:
                _save_job(job)
            except OSError:
                pass
        usage.flush()


def start_job(
    paper_path: Path,
    language: str,
    class_level: str,
    subject: str,
    board: str,
    google_token: str,
) -> Job:
    job = Job(id=uuid.uuid4().hex)
    with _LOCK:
        _JOBS[job.id] = job
    threading.Thread(
        target=_run,
        args=(
            job,
            paper_path,
            language,
            class_level,
            subject,
            board,
            google_token,
        ),
        daemon=True,
    ).start()
    return job
