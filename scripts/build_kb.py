"""ADMIN: build the foundation knowledge base from the shared Google Drive folder.

One command does everything — download, auto-tag, embed, index, and pack:

    python scripts/build_kb.py                      # uses KB_DRIVE_FOLDER from .env
    python scripts/build_kb.py --folder "https://drive.google.com/drive/folders/..."

Steps:
  1. Mirror the Drive folder into data/kb_cache/sources (already-downloaded
     files are kept, so re-runs only fetch new material).
  2. Tag every file by Board / Class / Subject from its folder path, e.g.
     "CBSE/Class 8/Science/ch04.pdf" -> CBSE, Class 8, Science.
  3. Extract -> chunk -> embed (Gemini) -> index into data/index/.
  4. Zip the index to data/kb_foundation_index.zip.

Then upload that ONE zip to Google Drive, share it as
"Anyone with the link - Viewer", and put the share link in .env as
KB_INDEX_URL. Every faculty machine downloads it automatically on first
start — faculty never add files themselves, and nothing big lands in git.

Options:
    --skip-download   re-index what is already in data/kb_cache/sources
    --keep            add to the existing index instead of rebuilding it
    --limit N         only ingest N files (for a quick test)
    --dry-run         show how every file would be tagged, ingest nothing
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Book/chapter names contain characters (thin spaces, dashes) that Windows'
# default cp1252 console cannot print; never let a progress line kill a build.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, drive_sync, extract  # noqa: E402
from app.ingest import ingest_file  # noqa: E402
from app.vectorstore import get_store  # noqa: E402

SOURCES_DIR = config.KB_CACHE_DIR / "sources"
INDEX_ZIP = config.DATA_DIR / "kb_foundation_index.zip"


def build(
    folder: str,
    *,
    skip_download: bool = False,
    keep: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    fresh: bool = False,
) -> None:
    if fresh and SOURCES_DIR.exists():
        # With DRIVE_API_KEY the mirror prunes Drive deletions by itself; the
        # keyless gdown path cannot, so --fresh forces a clean re-download.
        print(f"--fresh: clearing {SOURCES_DIR} …")
        shutil.rmtree(SOURCES_DIR, ignore_errors=True)
    if not skip_download:
        if not folder:
            sys.exit(
                "No Drive folder given. Set KB_DRIVE_FOLDER in .env or pass --folder URL."
            )
        print(f"Downloading Drive folder into {SOURCES_DIR} …")
        drive_sync.download_drive_folder(folder, SOURCES_DIR)

    files = sorted(
        p for p in SOURCES_DIR.rglob("*")
        if p.is_file() and extract.is_supported(p.name)
    )
    if not files:
        sys.exit(f"No supported files found under {SOURCES_DIR}.")
    print(f"{len(files)} supported files found.")

    if not dry_run and not keep:
        print("Rebuilding index from scratch…")
        get_store().clear()

    added = failed = 0
    for path in files:
        if limit is not None and added >= limit:
            break
        rel = path.relative_to(SOURCES_DIR)
        board, class_level, subject = drive_sync.classify_path(rel)
        tag = f"{board or 'Any board'} / {class_level or 'All classes'} / {subject or 'General'}"
        if dry_run:
            print(f"[{tag}] {rel}")
            continue
        print(f"Indexing [{tag}] {rel}")
        try:
            ingest_file(
                path,
                original_name=str(rel).replace("\\", "/"),
                subject=subject or "General",
                class_level=class_level,
                board=board,
            )
            added += 1
        except Exception as exc:  # noqa: BLE001 - one bad scan must not kill the build
            failed += 1
            print(f"  ERROR {rel}: {exc}")

    if dry_run:
        print("Dry run — nothing ingested.")
        return

    print(f"Indexed {added} files; {failed} failed. Total chunks: {get_store().count()}")
    drive_sync.pack_index(INDEX_ZIP)
    size_mb = INDEX_ZIP.stat().st_size / (1024 * 1024)
    print(
        f"\nIndex packed: {INDEX_ZIP} ({size_mb:.1f} MB)\n\n"
        "Next steps:\n"
        "  1. Upload this zip to Google Drive.\n"
        "  2. Share it: right-click -> Share -> 'Anyone with the link' (Viewer).\n"
        "  3. Copy the link into .env on every machine as KB_INDEX_URL=<link>.\n"
        "Faculty machines will download it automatically on the next start."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", default=config.KB_DRIVE_FOLDER,
                        help="Shared Drive folder link (default: KB_DRIVE_FOLDER from .env)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Re-index the already-downloaded cache")
    parser.add_argument("--keep", action="store_true",
                        help="Add to the existing index instead of rebuilding")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only ingest N files (quick test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the Board/Class/Subject tag for every file, ingest nothing")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete the local cache first so Drive deletions/renames "
                             "propagate (automatic when DRIVE_API_KEY is set)")
    args = parser.parse_args()
    build(
        args.folder,
        skip_download=args.skip_download,
        keep=args.keep,
        limit=args.limit,
        dry_run=args.dry_run,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
