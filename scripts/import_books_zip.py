"""Import the chapterwise books zip into the vector knowledge base.

Usage:
    python scripts/import_books_zip.py "C:/path/to/All Book Chapterwise.zip" --clear

The importer expects the current zip layout:
    All Book Chapterwise/<subject folder>/<Grade 11|Grade 12>/<chapter>.pdf
    All Book Chapterwise/Maths 11/<chapter>.pdf
    All Book Chapterwise/Maths 12 Part-1/<chapter>.pdf
    All Book Chapterwise/Maths Part-2/<chapter>.pdf
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import extract  # noqa: E402
from app.ingest import ingest_file  # noqa: E402
from app.vectorstore import get_store  # noqa: E402


def _class_from_text(text: str) -> str | None:
    lower = text.lower()
    if "grade 11" in lower or "class 11" in lower or "maths 11" in lower:
        return "Class 11"
    if (
        "grade 12" in lower
        or "class 12" in lower
        or "maths 12" in lower
        or "maths part-2" in lower
    ):
        return "Class 12"
    return None


def _subject_from_text(text: str) -> str | None:
    lower = text.lower()
    if "chemistry" in lower:
        return "Chemistry"
    if "physics" in lower:
        return "Physics"
    if "botany" in lower or "zoology" in lower or "biology" in lower:
        return "Biology"
    if "maths" in lower or "mathematics" in lower:
        return "Mathematics"
    return None


def classify(zip_name: str) -> tuple[str | None, str | None]:
    parts = [p for p in zip_name.replace("\\", "/").split("/") if p]
    joined = " / ".join(parts)
    return _subject_from_text(joined), _class_from_text(joined)


def _safe_extract_one(zf: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path) -> Path:
    name = Path(info.filename.replace("\\", "/")).name
    out = dest / name
    with zf.open(info) as src, out.open("wb") as fh:
        shutil.copyfileobj(src, fh)
    return out


def import_zip(zip_path: Path, *, clear: bool = False, limit: int | None = None) -> None:
    if clear:
        print("Clearing existing knowledge base...")
        get_store().clear()

    added = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        entries = [
            info
            for info in zf.infolist()
            if not info.is_dir() and extract.is_supported(info.filename)
        ]
        for info in entries:
            if limit is not None and added >= limit:
                break
            subject, class_level = classify(info.filename)
            if not subject:
                skipped += 1
                print(f"SKIP no subject: {info.filename}")
                continue
            local = _safe_extract_one(zf, info, tmpdir)
            display_name = "/".join(
                p for p in info.filename.replace("\\", "/").split("/") if p
            )
            print(f"Importing [{subject} / {class_level or 'All classes'}] {display_name}")
            try:
                ingest_file(
                    local,
                    original_name=display_name,
                    subject=subject,
                    class_level=class_level,
                )
                added += 1
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"ERROR {display_name}: {exc}")
            finally:
                try:
                    local.unlink()
                except OSError:
                    pass
    print(f"Done. Imported {added} files; skipped/failed {skipped}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--clear", action="store_true", help="Clear the index before import")
    parser.add_argument("--limit", type=int, default=None, help="Import only N files for testing")
    args = parser.parse_args()
    import_zip(args.zip_path, clear=args.clear, limit=args.limit)


if __name__ == "__main__":
    main()
