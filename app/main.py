"""FastAPI application: knowledge-base management + solution generation."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config, extract, ingest, jobs
from .vectorstore import get_store

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Solution Creation Tool")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    suffix = Path(upload.filename or "file").suffix
    dest = dest_dir / f"{uuid.uuid4().hex[:10]}{suffix}"
    with dest.open("wb") as fh:
        fh.write(upload.file.read())
    return dest


def _sources_context() -> dict:
    store = get_store()
    return {
        "sources": sorted(store.sources(), key=lambda s: s.get("added_at", "")),
        "chunk_count": store.count(),
    }


# --- pages ----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    ctx = {
        "request": request,
        "languages": config.LANGUAGES,
        "subjects": config.SUBJECTS,
        "has_key": bool(config.GEMINI_API_KEY),
        **_sources_context(),
    }
    return templates.TemplateResponse("index.html", ctx)


# --- knowledge base -------------------------------------------------------
@app.post("/sources/upload", response_class=HTMLResponse)
async def upload_sources(
    request: Request,
    files: list[UploadFile],
    subject: str = Form("General"),
    class_level: str = Form(""),
):
    errors: list[str] = []
    added = 0
    subject = subject if subject in config.SUBJECTS else "General"
    class_level = class_level.strip() or None
    for upload in files:
        name = upload.filename or "file"
        if not extract.is_supported(name):
            errors.append(f"{name}: unsupported file type")
            continue
        saved = _save_upload(upload, config.UPLOAD_DIR)
        try:
            ingest.ingest_file(
                saved,
                original_name=name,
                subject=subject,
                class_level=class_level,
            )
            added += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    ctx = {"request": request, "flash_ok": added, "flash_errors": errors, **_sources_context()}
    return templates.TemplateResponse("_sources.html", ctx)


@app.post("/sources/{source_id}/delete", response_class=HTMLResponse)
def delete_source(request: Request, source_id: str):
    get_store().delete_source(source_id)
    ctx = {"request": request, **_sources_context()}
    return templates.TemplateResponse("_sources.html", ctx)


@app.post("/sources/clear", response_class=HTMLResponse)
def clear_sources(request: Request):
    get_store().clear()
    ctx = {"request": request, **_sources_context()}
    return templates.TemplateResponse("_sources.html", ctx)


# --- generation -----------------------------------------------------------
@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    paper: UploadFile,
    language: str = Form(...),
    class_level: str = Form(""),
    subject: str = Form("General"),
    include_sources: str = Form("on"),
):
    name = paper.filename or "paper"
    if not extract.is_supported(name):
        ctx = {"request": request, "error": f"Unsupported question-paper type: {name}"}
        return templates.TemplateResponse("_job_error.html", ctx)
    saved = _save_upload(paper, config.UPLOAD_DIR)
    subject = subject if subject in config.SUBJECTS else "General"
    job = jobs.start_job(saved, language, class_level.strip(), subject, include_sources == "on")
    return templates.TemplateResponse("_job.html", {"request": request, "job": job})


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        return templates.TemplateResponse(
            "_job_error.html", {"request": request, "error": "Unknown job."}
        )
    return templates.TemplateResponse("_job.html", {"request": request, "job": job})


@app.get("/download/{job_id}/{fmt}")
def download(job_id: str, fmt: str):
    job = jobs.get_job(job_id)
    if job is None:
        return HTMLResponse("Unknown job.", status_code=404)
    filename = job.docx_name if fmt == "docx" else job.pdf_name
    if not filename:
        return HTMLResponse("File not available.", status_code=404)
    path = config.OUTPUT_DIR / filename
    if not path.exists():
        return HTMLResponse("File missing.", status_code=404)
    media = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if fmt == "docx"
        else "application/pdf"
    )
    return FileResponse(str(path), media_type=media, filename=filename)
