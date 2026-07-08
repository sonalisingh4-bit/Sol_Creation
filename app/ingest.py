"""Ingest an uploaded source file into the knowledge base (vector store)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import chunking, extract, gemini_client
from .vectorstore import get_store


class IngestResult:
    def __init__(
        self,
        source_id: str,
        filename: str,
        n_chunks: int,
        chars: int,
        subject: str | None = None,
        class_level: str | None = None,
        board: str | None = None,
    ):
        self.source_id = source_id
        self.filename = filename
        self.n_chunks = n_chunks
        self.chars = chars
        self.subject = subject
        self.class_level = class_level
        self.board = board


def ingest_file(
    path: str | Path,
    original_name: str | None = None,
    *,
    subject: str | None = None,
    class_level: str | None = None,
    board: str | None = None,
) -> IngestResult:
    """Extract -> chunk -> embed -> store a single source file."""
    path = Path(path)
    filename = original_name or path.name

    text = extract.extract_text(path)
    if not text.strip():
        raise ValueError(f"No text could be extracted from '{filename}'.")

    chunks = chunking.chunk_text(text)
    if not chunks:
        raise ValueError(f"'{filename}' produced no usable chunks.")

    embeddings = gemini_client.embed_texts(chunks, is_query=False)

    source_id = uuid.uuid4().hex[:12]
    get_store().add(
        source_id=source_id,
        filename=filename,
        chunks=chunks,
        embeddings=embeddings,
        added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        subject=subject,
        class_level=class_level,
        board=board,
    )
    return IngestResult(
        source_id, filename, len(chunks), len(text), subject, class_level, board
    )
