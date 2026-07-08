"""Split extracted text into overlapping chunks on natural boundaries."""
from __future__ import annotations

import re

from . import config


def _split_paragraphs(text: str) -> list[str]:
    # Normalise whitespace, split on blank lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Greedily pack paragraphs into ~chunk_size character windows with overlap."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        # A single huge paragraph is hard-split.
        if len(para) > chunk_size:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(para), chunk_size - overlap):
                chunks.append(para[i : i + chunk_size])
            continue

        if not buf:
            buf = para
        elif len(buf) + len(para) + 2 <= chunk_size:
            buf += "\n\n" + para
        else:
            chunks.append(buf)
            # Carry an overlapping tail of the previous chunk for context.
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + "\n\n" + para).strip()

    if buf:
        chunks.append(buf)

    return [c.strip() for c in chunks if c.strip()]
