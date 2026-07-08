"""Convert a .docx to .pdf using whatever engine is available on the machine.

Indic scripts (Bangla, Telugu, Kannada ...) need a real shaping engine, so we
rely on MS Word (via docx2pdf) or LibreOffice (soffice) — both shape complex
scripts correctly. If neither is present we return None and the caller keeps the
.docx only.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _try_docx2pdf(docx_path: Path, pdf_path: Path) -> bool:
    try:
        import pythoncom  # noqa: F401  (present with pywin32; needed for COM in a thread)

        pythoncom.CoInitialize()
    except Exception:  # noqa: BLE001
        pass
    try:
        from docx2pdf import convert

        convert(str(docx_path), str(pdf_path))
        return pdf_path.exists()
    except Exception:  # noqa: BLE001 - Word may be absent or COM may fail
        return False


def _find_soffice() -> str | None:
    for name in ("soffice", "soffice.exe", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    for guess in (
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ):
        if Path(guess).exists():
            return guess
    return None


def _try_libreoffice(docx_path: Path, pdf_path: Path) -> bool:
    soffice = _find_soffice()
    if not soffice:
        return False
    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir",
             str(pdf_path.parent), str(docx_path)],
            check=True,
            timeout=180,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        produced = pdf_path.parent / (docx_path.stem + ".pdf")
        if produced.exists() and produced != pdf_path:
            produced.replace(pdf_path)
        return pdf_path.exists()
    except Exception:  # noqa: BLE001
        return False


def docx_to_pdf(docx_path: str | Path) -> Path | None:
    """Return the path to a generated PDF, or None if no converter is available."""
    docx_path = Path(docx_path)
    pdf_path = docx_path.with_suffix(".pdf")
    if _try_docx2pdf(docx_path, pdf_path):
        return pdf_path
    if _try_libreoffice(docx_path, pdf_path):
        return pdf_path
    return None
