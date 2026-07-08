"""Render PDF pages to high-resolution PNGs.

Gemini's native PDF pipeline can blur the fine detail of a chemistry paper — small
subscripts, ring substituents (–OH, –NO₂), the exact structure drawn as an MCQ
option. Handing it a sharp, per-page raster instead lets it read what is actually
drawn rather than guess. Rendering is cached per (path, mtime) so a paper is
rasterised once per run, not once per question.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# 180 DPI (A4 ≈ 1488x2105 px) keeps subscripts and ring substituents legible while
# sitting in Gemini's effective image resolution. JPEG, not PNG: exam pages are
# photographic scans that balloon to multi-MB as PNG but compress ~10x as JPEG with
# no loss that matters for reading a structure.
_DPI = 180
_JPEG_QUALITY = 82
MIME = "image/jpeg"


def _render_fitz(path: str, dpi: int) -> list[bytes]:
    import fitz  # PyMuPDF

    out: list[bytes] = []
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)  # no alpha by default -> JPEG-safe
            out.append(pix.tobytes("jpg", jpg_quality=_JPEG_QUALITY))
    return out


def _render_pdfium(path: str, dpi: int) -> list[bytes]:
    import io

    import pypdfium2 as pdfium

    out: list[bytes] = []
    pdf = pdfium.PdfDocument(path)
    try:
        scale = dpi / 72.0
        for i in range(len(pdf)):
            img = pdf[i].render(scale=scale).to_pil().convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            out.append(buf.getvalue())
    finally:
        pdf.close()
    return out


@lru_cache(maxsize=8)
def _cached(path: str, mtime: float, dpi: int) -> tuple[bytes, ...]:
    for renderer in (_render_fitz, _render_pdfium):
        try:
            pages = renderer(path, dpi)
            if pages:
                return tuple(pages)
        except Exception:  # noqa: BLE001 - try the next backend, then give up
            continue
    return ()


def render_pages(path: str | Path, *, dpi: int = _DPI) -> list[bytes]:
    """Return one PNG (bytes) per page, high-resolution. Empty list for non-PDFs
    or if no rasteriser is available — callers must degrade gracefully."""
    p = Path(path)
    if p.suffix.lower() != ".pdf" or not p.exists():
        return []
    try:
        return list(_cached(str(p), p.stat().st_mtime, dpi))
    except Exception:  # noqa: BLE001 - never let rendering sink a job
        return []
