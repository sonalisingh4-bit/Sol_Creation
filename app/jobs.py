"""In-memory background job manager for the (slow) generation pipeline."""
from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config, document, page_images, paper_parser, solver


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | parsing | solving | writing | done | error
    message: str = "Queued…"
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
        if self.total <= 0:
            return 5 if self.status in ("parsing", "queued") else 0
        return min(99, int(self.done / self.total * 100))


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def get_job(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(job_id)


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^\w\-]+", "_", Path(name).stem).strip("_")
    return stem or "solutions"


def _run(
    job: Job,
    paper_path: Path,
    language: str,
    class_level: str,
    subject: str,
    include_sources: bool,
) -> None:
    try:
        job.status = "parsing"
        job.message = "Analysing the question paper…"
        # Upload PDFs/images once and reuse for parsing AND for figure questions.
        uploaded = paper_parser.upload_if_multimodal(paper_path)
        paper = paper_parser.parse_paper(paper_path, uploaded=uploaded)
        job.title = paper.title
        job.total = paper.n_units

        # High-res page rasters let figure questions read the actual drawn structures
        # (substituents, subscripts, MCQ structures) instead of guessing. Best-effort:
        # an empty list just means figure questions fall back to the uploaded PDF.
        paper_pages = page_images.render_pages(paper_path)
        # Routing a figure question to its own page needs per-page text aligned with
        # those images. Prefer the markers already in the transcription; if they're
        # absent or miscounted, re-derive from the images so targeting still works.
        if paper_pages and len(paper.page_texts) != len(paper_pages):
            paper.page_texts = paper_parser.page_texts_from_images(paper_pages)

        job.status = "solving"

        def progress(done: int, total: int, label: str) -> None:
            job.done = done
            job.total = total
            if label != "done":
                job.message = f"Answering {label}  ({done + 1}/{total})…"

        solved = solver.solve_paper(
            paper,
            language,
            class_level=class_level or None,
            subject=subject or None,
            paper_file=uploaded,
            paper_pages=paper_pages,
            progress=progress,
        )

        job.status = "writing"
        job.message = "Building the solution document…"
        base = config.OUTPUT_DIR / f"{_safe_stem(paper_path.name)}_{language}_{job.id[:8]}"
        docx_path, pdf_path = document.build_documents(
            solved, base, include_sources=include_sources
        )
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
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = str(exc)
        job.message = "Failed."


def start_job(
    paper_path: Path,
    language: str,
    class_level: str,
    subject: str,
    include_sources: bool,
) -> Job:
    job = Job(id=uuid.uuid4().hex)
    with _LOCK:
        _JOBS[job.id] = job
    threading.Thread(
        target=_run,
        args=(job, paper_path, language, class_level, subject, include_sources),
        daemon=True,
    ).start()
    return job
