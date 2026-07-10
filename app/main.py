"""FastAPI application: solution generation over the prebuilt knowledge base.

The knowledge base is read-only here by design: it is built centrally from the
shared Google Drive folder (scripts/build_kb.py) and downloaded automatically
at startup. Faculty only upload question papers.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pw_access

from . import config, drive_sync, extract, jobs

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Fetch the prebuilt index from Drive in the background if it is missing,
    # so the very first launch on a fresh machine needs zero manual steps.
    drive_sync.ensure_index_async()
    yield


app = FastAPI(title="Solution Creation Tool", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    suffix = Path(upload.filename or "file").suffix
    dest = dest_dir / f"{uuid.uuid4().hex[:10]}{suffix}"
    with dest.open("wb") as fh:
        fh.write(upload.file.read())
    return dest


def _request_google_token(request: Request, form_token: str = "") -> str:
    if form_token.strip():
        return form_token.strip()
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return ""


# --- pages ----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    ctx = {
        "request": request,
        "languages": config.LANGUAGES,
        "subjects": config.SUBJECTS,
        "boards": config.BOARDS,
        "classes": config.CLASSES,
        "google_client_id": config.GOOGLE_CLIENT_ID,
        "session_days": config.APP_SESSION_DAYS,
        "kb": drive_sync.kb_summary(),
    }
    return templates.TemplateResponse("index.html", ctx)


# --- knowledge base (read-only status; content is managed by the admin) ----
@app.get("/kb/status", response_class=HTMLResponse)
def kb_status(request: Request):
    return templates.TemplateResponse(
        "_kb_status.html", {"request": request, "kb": drive_sync.kb_summary()}
    )


# --- generation -----------------------------------------------------------
@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    paper: UploadFile,
    language: str = Form(...),
    board: str = Form(""),
    class_level: str = Form(""),
    subject: str = Form("General"),
    include_sources: str = Form("on"),
    google_token: str = Form(""),
):
    name = paper.filename or "paper"
    if not extract.is_supported(name):
        ctx = {"request": request, "error": f"Unsupported question-paper type: {name}"}
        return templates.TemplateResponse("_job_error.html", ctx)
    token = _request_google_token(request, google_token)
    if not pw_access.check_allowed(token):
        ctx = {
            "request": request,
            "error": "Not authorized for this app. Sign in with a whitelisted @pw.live account.",
        }
        return templates.TemplateResponse("_job_error.html", ctx, status_code=403)
    saved = _save_upload(paper, config.UPLOAD_DIR)
    subject = subject if subject in config.SUBJECTS else "General"
    board = board if board in config.BOARDS else ""
    class_level = class_level if class_level in config.CLASSES else ""
    job = jobs.start_job(
        saved, language, class_level, subject, board, include_sources == "on", token
    )
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
