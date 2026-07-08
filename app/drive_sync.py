"""Google Drive knowledge-base sync (free hosting — nothing large in git).

The foundation source files (Class 6-10, CBSE & ICSE) live in a shared Google
Drive folder. Two roles use this module:

- Admin (scripts/build_kb.py): download the whole source folder, tag every file
  by board/class/subject from its folder path, embed and index it, then publish
  the prebuilt index as one zip back on Drive.
- Faculty (app startup): if the local index is empty and KB_INDEX_URL is set,
  download and unpack that prebuilt zip automatically. Faculty never upload or
  embed anything.

Folder download uses the Drive API when DRIVE_API_KEY is set (complete listing,
any folder size); otherwise gdown, which lists at most 50 files per folder.
Both paths only need the folder shared as "Anyone with the link - Viewer".
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from . import config
from .vectorstore import get_store

# ---------------------------------------------------------------------------
# Classify a file's Drive path into (board, class_level, subject)
# ---------------------------------------------------------------------------
_ROMAN = {"vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12}
_CLASS_RE = re.compile(
    r"\b(?:class|grade|std|standard)[\s._-]*(vi{1,3}|xii|xi|ix|x|1[0-2]|[6-9])\b",
    re.IGNORECASE,
)
_ORDINAL_RE = re.compile(r"\b(1[0-2]|[6-9])\s*th\b", re.IGNORECASE)
# "Class 6-10" / "Grade 6 to 10" names a library's range, not one book's class.
_RANGE_RE = re.compile(r"\d\s*(?:-|–|—|to)\s*\d", re.IGNORECASE)
# Branding like "Physics Wallah"/"PW" must never read as subject Physics.
_BRANDING_RE = re.compile(r"physics\s*wallah|\bpw\b", re.IGNORECASE)

# Checked in order — "Social Science" must win before the bare "science" match,
# and Physics/Chemistry/Biology before it too (ICSE 9-10 splits Science).
_SUBJECT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Social Science", re.compile(
        r"social\s*(?:science|studies)|\bsst\b|history|geography|civics|economics|political",
        re.IGNORECASE)),
    ("English", re.compile(r"english|grammar|literature", re.IGNORECASE)),
    ("Mathematics", re.compile(r"math", re.IGNORECASE)),
    ("Physics", re.compile(r"physic", re.IGNORECASE)),
    ("Chemistry", re.compile(r"chem", re.IGNORECASE)),
    ("Biology", re.compile(r"biolog|botany|zoology|\bbio\b", re.IGNORECASE)),
    ("Science", re.compile(r"scien|\bevs\b", re.IGNORECASE)),
]

_ICSE_RE = re.compile(r"\bicse\b|\bcisce\b|\bselina\b", re.IGNORECASE)
_CBSE_RE = re.compile(r"\bcbse\b|\bncert\b", re.IGNORECASE)
_STATE_RE = re.compile(r"state\s*board", re.IGNORECASE)


def _class_from_segment(segment: str) -> str | None:
    if _RANGE_RE.search(segment):
        return None  # a range ("Class 6-10") describes the collection, not the book
    m = _CLASS_RE.search(segment) or _ORDINAL_RE.search(segment)
    if m:
        token = m.group(1).lower()
        return f"Class {_ROMAN.get(token, token)}"
    if segment.strip() in {"6", "7", "8", "9", "10", "11", "12"}:
        return f"Class {segment.strip()}"
    return None


def _subject_from_segment(segment: str) -> str | None:
    segment = _BRANDING_RE.sub(" ", segment)
    for name, pattern in _SUBJECT_PATTERNS:
        if pattern.search(segment):
            return name
    return None


def classify_path(rel_path: str | Path) -> tuple[str | None, str | None, str | None]:
    """(board, class_level, subject) read from a file's folder path.

    Any part of the path may carry the tag ("ICSE/Class 9/Physics/ch1.pdf",
    "Maths/8th CBSE/..."). The most specific wins: the deepest folder that
    names a class beats a "Foundation Class 6-10" wrapper above it, and the
    containing folder's subject beats keywords inside the chapter filename
    ("Science/Chemical Reactions.pdf" is Science, not Chemistry). Missing tags
    come back as None: an untagged board stays visible to every board, an
    untagged class/subject is a fallback pool.
    """
    text = str(rel_path).replace("\\", "/")
    segments = [s for s in text.split("/") if s]
    joined = " / ".join(segments)

    board: str | None = None
    if _ICSE_RE.search(joined):
        board = "ICSE"
    elif _CBSE_RE.search(joined):
        board = "CBSE"
    elif _STATE_RE.search(joined):
        board = "State Board"

    class_level: str | None = None
    for seg in reversed(segments):  # deepest (most specific) segment first
        class_level = _class_from_segment(seg)
        if class_level:
            break
    if class_level is not None and class_level not in config.CLASSES:
        class_level = None

    # Subject: deepest DIRECTORY first, chapter filename only as a last resort —
    # chapter titles ("Physical and Chemical Changes.pdf") routinely name other
    # subjects than the book they belong to.
    subject: str | None = None
    for seg in [*reversed(segments[:-1]), segments[-1]]:
        subject = _subject_from_segment(seg)
        if subject:
            break

    return board, class_level, subject


# ---------------------------------------------------------------------------
# Drive downloads
# ---------------------------------------------------------------------------
_FOLDER_ID_RE = re.compile(r"folders/([A-Za-z0-9_-]{10,})")
_FILE_ID_RE = re.compile(r"(?:/d/|id=)([A-Za-z0-9_-]{10,})")


def _folder_id(url_or_id: str) -> str:
    m = _FOLDER_ID_RE.search(url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", url_or_id):
        return url_or_id
    raise ValueError(f"Not a Drive folder link or id: {url_or_id!r}")


def download_drive_file(url: str, dest: Path) -> Path:
    """Download one shared Drive file (handles the large-file confirm step)."""
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    out = gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)
    if not out:
        raise RuntimeError(
            "Could not download from Google Drive. Is the file shared as "
            "'Anyone with the link - Viewer'?"
        )
    return Path(out)


_FOLDER_MIME = "application/vnd.google-apps.folder"
_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"


def _api_list_children(folder_id: str, key: str) -> list[dict]:
    files: list[dict] = []
    page_token: str | None = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken, files(id, name, mimeType, shortcutDetails)",
            "pageSize": "1000",
            "key": key,
        }
        if page_token:
            params["pageToken"] = page_token
        url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url) as resp:
            data = json.load(resp)
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            return files


def _api_download(file_id: str, key: str, dest: Path) -> None:
    import requests  # gdown dependency, always present

    # Stream to a .part file and swap in only when complete, so an interrupted
    # download is never mistaken for a finished (cached) file on the next run.
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={key}"
    tmp = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _dedup_name(name: str, file_id: str) -> str:
    p = Path(name)
    return f"{p.stem}__{file_id[:8]}{p.suffix}"


def _api_download_folder(folder_id: str, key: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    # Resolve Drive shortcuts to their targets (they carry no content of their
    # own); a shortcut to a folder is mirrored like a real subfolder.
    children: list[dict] = []
    for item in _api_list_children(folder_id, key):
        if item["mimeType"] == _SHORTCUT_MIME:
            details = item.get("shortcutDetails") or {}
            target_id = details.get("targetId")
            if not target_id:
                print(f"SKIP (shortcut without target): {dest / item['name']}")
                continue
            children.append({
                "id": target_id,
                "name": item["name"],
                "mimeType": details.get("targetMimeType", "application/octet-stream"),
            })
        else:
            children.append(item)

    # Drive allows same-named siblings; give duplicates a deterministic
    # id-based suffix instead of silently keeping only the first one.
    sanitised = Counter(
        item["name"].replace("/", "_").replace("\\", "_") for item in children
    )
    expected: set[str] = set()
    for item in children:
        name = item["name"].replace("/", "_").replace("\\", "_")
        if sanitised[name] > 1:
            name = _dedup_name(name, item["id"])
        target = dest / name
        if item["mimeType"] == _FOLDER_MIME:
            expected.add(name)
            _api_download_folder(item["id"], key, target)
        elif item["mimeType"].startswith("application/vnd.google-apps"):
            print(f"SKIP (Google Workspace doc, not downloadable): {target}")
        else:
            if target.is_dir():  # a file sharing its name with a sibling folder
                name = _dedup_name(name, item["id"])
                target = dest / name
            expected.add(name)
            if target.exists() and target.stat().st_size > 0:
                print(f"cached: {target.name}")
                continue
            print(f"downloading: {target}")
            _api_download(item["id"], key, target)

    # True mirror: drop local files/folders that were deleted or renamed on
    # Drive, so stale books never linger in the published knowledge base.
    for existing in dest.iterdir():
        if existing.name in expected or existing.name.endswith(".part"):
            continue
        print(f"pruned (no longer on Drive): {existing}")
        if existing.is_dir():
            shutil.rmtree(existing, ignore_errors=True)
        else:
            existing.unlink(missing_ok=True)


def download_drive_folder(url_or_id: str, dest: Path) -> Path:
    """Mirror a shared Drive folder locally; returns the local root."""
    folder_id = _folder_id(url_or_id)
    dest.mkdir(parents=True, exist_ok=True)
    if config.DRIVE_API_KEY:
        _api_download_folder(folder_id, config.DRIVE_API_KEY, dest)
        return dest

    import gdown

    # Keyless path: gdown scrapes the public folder page. Google caps that
    # listing at 50 files per folder — fine for chapterwise subfolders; set
    # DRIVE_API_KEY for a complete listing on bigger folders.
    got = gdown.download_folder(
        id=folder_id,
        output=str(dest),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    if got is None:
        raise RuntimeError(
            "Could not download the Drive folder. Share it as "
            "'Anyone with the link - Viewer' and try again."
        )
    return dest


# ---------------------------------------------------------------------------
# Prebuilt-index publish/bootstrap (what faculty machines actually fetch)
# ---------------------------------------------------------------------------
_INDEX_FILES = ("vectors.npy", "meta.json", "sources.json")


def pack_index(zip_path: Path) -> Path:
    """Zip the built index so the admin can upload ONE file to Drive."""
    vectors = config.INDEX_DIR / "vectors.npy"
    meta = config.INDEX_DIR / "meta.json"
    if not (vectors.exists() and vectors.stat().st_size > 0 and meta.exists()):
        raise RuntimeError(
            "data/index is incomplete (vectors.npy/meta.json missing) — "
            "nothing was indexed, so there is no knowledge base to publish."
        )
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in _INDEX_FILES:
            path = config.INDEX_DIR / name
            if path.exists():
                zf.write(path, arcname=name)
    return zip_path


def unpack_index(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        if "meta.json" not in names or "vectors.npy" not in names:
            raise RuntimeError(
                "The downloaded zip is not a complete knowledge-base index."
            )
        # Extract everything to .tmp names first, then swap all files in — a
        # crash mid-extract must never leave a half-written live index behind.
        staged: list[tuple[Path, Path]] = []
        for name in _INDEX_FILES:
            if name in names:
                final = config.INDEX_DIR / name
                tmp = config.INDEX_DIR / (name + ".tmp")
                with zf.open(names[name]) as src, tmp.open("wb") as out:
                    shutil.copyfileobj(src, out)
                staged.append((tmp, final))
        for tmp, final in staged:
            os.replace(tmp, final)


_status_lock = threading.Lock()
_status: dict = {"state": "idle", "detail": ""}
_bootstrap_started = False


def _set_status(state: str, detail: str = "") -> None:
    with _status_lock:
        _status["state"] = state
        _status["detail"] = detail


def bootstrap_status() -> dict:
    with _status_lock:
        return dict(_status)


def _bootstrap() -> None:
    try:
        _set_status("downloading", "Downloading the prebuilt knowledge base from Google Drive…")
        zip_path = config.KB_CACHE_DIR / "kb_index.zip"
        download_drive_file(config.KB_INDEX_URL, zip_path)
        _set_status("extracting", "Unpacking the knowledge base…")
        unpack_index(zip_path)
        get_store().reload()
        if get_store().count() > 0:
            _set_status("ready")
        else:
            _set_status("error", "The downloaded index is empty.")
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI, never crashes startup
        _set_status("error", str(exc))


def ensure_index_async() -> None:
    """At startup: fetch the prebuilt index in the background if it is missing."""
    global _bootstrap_started
    if get_store().count() > 0:
        _set_status("ready")
        return
    if not config.KB_INDEX_URL:
        _set_status("empty")
        return
    with _status_lock:
        if _bootstrap_started:
            return
        _bootstrap_started = True
    threading.Thread(target=_bootstrap, daemon=True).start()


# ---------------------------------------------------------------------------
# UI summary
# ---------------------------------------------------------------------------
def _class_sort_key(class_level: str | None) -> tuple[int, str]:
    if class_level and class_level.startswith("Class "):
        try:
            return (int(class_level.split()[1]), "")
        except ValueError:
            pass
    return (99, class_level or "")


def kb_summary() -> dict:
    """Status + per board/class breakdown for the read-only KB panel."""
    store = get_store()
    status = bootstrap_status()
    groups: dict[tuple[str | None, str | None], dict] = {}
    for src in store.sources():
        key = (src.get("board"), src.get("class_level"))
        g = groups.setdefault(
            key,
            {
                "board": src.get("board") or "Any board",
                "class_level": src.get("class_level") or "All classes",
                "n_sources": 0,
                "n_chunks": 0,
                "subjects": set(),
            },
        )
        g["n_sources"] += 1
        g["n_chunks"] += src.get("n_chunks", 0)
        if src.get("subject"):
            g["subjects"].add(src["subject"])
    ordered = sorted(
        groups.values(),
        key=lambda g: (g["board"], _class_sort_key(g["class_level"])),
    )
    for g in ordered:
        g["subjects"] = ", ".join(sorted(g["subjects"]))
    state = status["state"]
    if state in ("idle", "ready", "empty"):
        state = "ready" if store.count() > 0 else "empty"
    return {
        "state": state,
        "detail": status.get("detail", ""),
        "chunk_count": store.count(),
        "source_count": len(store.sources()),
        "groups": ordered,
    }
