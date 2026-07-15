"""Parse an uploaded question paper into structured questions with marks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from . import config, extract, gemini_client, page_images

# Pass 1 (OCR) delimits each page with its own marker line, but the exact wording
# drifts run to run — we've seen "**Page 1**", "=== PAGE 1 ===" and Gemini's native
# "==Start of OCR for page 1==". Match any of them (a START marker, never an "End of
# OCR" one), pulling out the page number. The trailing \W* stops it firing on an
# inline "... on page 1 of the reaction ...". When no marker is found the caller
# rebuilds pages from the rendered images instead, so this staying loose is safe.
_PAGE_MARKER = re.compile(r"(?im)^\W*(?:start\s+of\s+ocr\s+for\s+)?page\s+(\d+)\W*$")


def _split_pages(text: str) -> list[str]:
    """Split an OCR transcription into per-page chunks by its 'Page N' markers.
    Returns [] when there are no usable markers, so page targeting stays optional."""
    matches = list(_PAGE_MARKER.finditer(text))
    if len(matches) < 2:
        return []
    pages: dict[int, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        try:
            num = int(m.group(1))
        except ValueError:
            continue
        pages[num] = (pages.get(num, "") + "\n" + text[start:end]).strip()
    if not pages:
        return []
    return [pages.get(n, "") for n in range(1, max(pages) + 1)]

_SYSTEM = (
    "You are an exam-paper analyst. You extract the structure of a question paper "
    "exactly as written. You never answer the questions."
)

# Pass 1: transcribe the whole paper. A focused "copy everything" task is far more
# reliable on a dense scan than asking one call to read AND structure it (which tends
# to fill in the early questions and leave later ones textless).
_OCR_INSTRUCTION = """\
Transcribe this question paper COMPLETELY and VERBATIM into plain text, in reading order, page by page.

- Include EVERY question and EVERY sub-question in order: main numbers (1, 2, 3), sub-parts ((i), (ii), (a), (b) ...), and every "অথবা" / "OR" alternative.
- For each, copy its full text, its marks (e.g. [1 mark], 2, 3×9=27) and every option label ((a) (b) (c) (d)) with the option text.
- Preserve the original language and script (Bengali) exactly. Do NOT translate, summarise, abbreviate, correct, or skip anything.
- Where a question shows a figure/structure/graph/circuit you cannot put into text, write "[FIGURE]" in its place but KEEP all surrounding words and options.

Output only the transcribed text — no commentary."""

_INSTRUCTION = """\
Analyse this question paper and return its full structure as JSON.

Rules:
- Extract EVERY question and sub-question, in order, preserving the original numbering (1, 2, Q3, i, ii, a, b ...).
- Extract ONLY what is actually printed on the paper. Do NOT invent, add, duplicate, split or merge questions, and do NOT create a question from a heading, section title ("Section A"), general instruction line, marks table, page number, or watermark. Every question and sub-part you output MUST correspond to one a reader can point to on the paper — producing an extra entry with no matching question on the paper is a serious error. When unsure whether a fragment is a real question, leave it out rather than invent one.
- Every question and sub-question MUST have its full "text" copied verbatim from the paper. NEVER leave "text" empty, null, or abbreviated — if a part is present in the paper, its text must be present here.
- This applies to FIGURE-BASED sub-parts too. If a sub-part is mostly a drawn structure/reaction (a "[FIGURE]" appears in the transcription), still put its surrounding words, labels, reagents, conditions and product letters in "text" — e.g. "(x) CH₃CH₂CH₂Cl + NaI --(acetone, heat)--> A" or "(z) [FIGURE] + HCl --> D". A sub-part that exists in the paper must never have empty text.
- Include every "অথবা"/"OR" alternative as its own sub-part (e.g. number "a (OR)"), with its full text.
- Capture the marks for each question/sub-question. Look for patterns like [5], (5 marks), 5M, [05]. Use null if no marks are shown.
- Capture any answering instruction attached to a question (word/line limits, "draw a diagram", "with an example", "explain", "define", "derive"). Use null if none.
- Set "requires_figure" to true if the question can ONLY be answered by looking at a figure, diagram, graph, circuit, table or image (e.g. it says "in the figure", "as shown", the options are structures/graphs, or the transcription contains a "[FIGURE]" marker). Otherwise false.
- Set "page" to the 1-based page number on which the question/sub-question appears, read from the page-marker lines in the transcription (e.g. "==Start of OCR for page 3==", "Page 3") or, when the pages are supplied as images, from the image order (image 1 = page 1). A sub-question inherits its parent's page unless a later marker precedes it. Use null only if the page cannot be determined.
- Copy the question text verbatim. Do NOT answer anything.
- If a question has sub-parts, put them in "subparts"; otherwise leave it an empty list.

Return JSON with this exact shape:
{
  "title": "subject / paper title if visible, else null",
  "total_marks": <number or null>,
  "questions": [
    {
      "number": "1",
      "text": "the question text",
      "marks": <number or null>,
      "instruction": "<string or null>",
      "requires_figure": <true or false>,
      "page": <number or null>,
      "subparts": [
        {"number": "a", "text": "...", "marks": <number or null>, "instruction": "<string or null>", "requires_figure": <true or false>, "page": <number or null>}
      ]
    }
  ]
}
"""


@dataclass
class SubPart:
    number: str
    text: str
    marks: float | None = None
    instruction: str | None = None
    requires_figure: bool = False
    page: int | None = None


@dataclass
class Question:
    number: str
    text: str
    marks: float | None = None
    instruction: str | None = None
    requires_figure: bool = False
    page: int | None = None
    subparts: list[SubPart] = field(default_factory=list)


@dataclass
class ParsedPaper:
    title: str | None
    total_marks: float | None
    questions: list[Question]
    # Per-page OCR text (index 0 == page 1); empty for text inputs or if the
    # transcription had no page markers. Used only to route figure questions to
    # the right page image.
    page_texts: list[str] = field(default_factory=list)

    @property
    def n_questions(self) -> int:
        return len(self.questions)

    @property
    def n_units(self) -> int:
        """Total answerable units (sub-parts counted individually)."""
        return sum(len(q.subparts) or 1 for q in self.questions)


def _coerce_marks(value) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
        return int(num) if num.is_integer() else num
    except (TypeError, ValueError):
        return None


def _coerce_page(value) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def _txt(value) -> str:
    """Null-safe text: JSON null must become '' , never the string 'None'."""
    return "" if value is None else str(value).strip()


def _build(data: dict) -> ParsedPaper:
    questions: list[Question] = []
    for q in data.get("questions", []) or []:
        subparts: list[SubPart] = []
        for sp in q.get("subparts") or []:
            number = _txt(sp.get("number"))
            text = _txt(sp.get("text"))
            # Keep a sub-part that has a number even if its text didn't parse, so a
            # bad scan surfaces as a visible gap instead of being silently dropped.
            if not number and not text:
                continue
            subparts.append(
                SubPart(
                    number=number,
                    text=text,
                    marks=_coerce_marks(sp.get("marks")),
                    instruction=(sp.get("instruction") or None),
                    requires_figure=bool(sp.get("requires_figure")),
                    page=_coerce_page(sp.get("page")),
                )
            )
        text = _txt(q.get("text"))
        if not text and not subparts:
            continue
        questions.append(
            Question(
                number=_txt(q.get("number")),
                text=text,
                marks=_coerce_marks(q.get("marks")),
                instruction=(q.get("instruction") or None),
                requires_figure=bool(q.get("requires_figure")),
                page=_coerce_page(q.get("page")),
                subparts=subparts,
            )
        )
    return ParsedPaper(
        title=(data.get("title") or None),
        total_marks=_coerce_marks(data.get("total_marks")),
        questions=questions,
    )


def _missing_text(paper: ParsedPaper) -> int:
    """Count answerable units whose question text failed to parse."""
    n = 0
    for q in paper.questions:
        if q.subparts:
            n += sum(1 for sp in q.subparts if not sp.text)
        elif not q.text:
            n += 1
    return n


_PAGE_OCR_INSTRUCTION = """These images are the pages of ONE question paper, in order (image 1 = page 1, image 2 = page 2, and so on).

Transcribe each page separately and return JSON in exactly this shape:
{"pages": ["<all text visible on page 1>", "<all text visible on page 2>", ...]}

Rules:
- Output EXACTLY one string per input image, in the SAME order as the images.
- Each string must include that page's question numbers, sub-part labels ((i), (a), (x) ...) and key formulae/reagents so the page's content is identifiable. Preserve the original script (Bengali).
- Do NOT merge, drop, reorder or invent pages. No commentary."""


def page_texts_from_images(page_pngs: list[bytes]) -> list[str]:
    """Per-page text aligned 1:1 with the rendered page images. Used only when the
    main transcription's page markers are missing or don't line up with the page
    count, so a figure question can still be routed to its own page. One extra call;
    returns [] on any problem (targeting then simply stays off)."""
    if not page_pngs:
        return []
    parts = [gemini_client.image_part(p, mime_type=page_images.MIME) for p in page_pngs]
    try:
        data = gemini_client.generate_json(
            [*parts, _PAGE_OCR_INSTRUCTION],
            model=config.GEMINI_PARSE_MODEL,  # plain OCR — flash is enough and cheap
            system="You are a meticulous OCR transcriber.",
        )
    except Exception:  # noqa: BLE001 - targeting is optional
        return []
    pages = data.get("pages") if isinstance(data, dict) else None
    if isinstance(pages, list) and len(pages) == len(page_pngs):
        return [str(p or "") for p in pages]
    return []


def upload_if_multimodal(path: str | Path):
    """Upload PDFs/images so Gemini can read their figures; return None for text files."""
    path = Path(path)
    if path.suffix.lower() in extract.PDF_EXTS | extract.IMAGE_EXTS:
        return gemini_client.upload_file(path)
    return None


_OCR_SYSTEM = (
    "You are a meticulous OCR transcriber. You reproduce every character and never "
    "skip content."
)


# Raw page bytes per OCR request. base64 inflates by ~4/3, so this keeps a batch under
# the proxy's inline cap (see gemini_client._MAX_INLINE_B64) with headroom for the
# prompt. Most papers fit in ONE batch — batching only kicks in for long/heavy scans,
# so transcription stays a single fast call in the common case.
_OCR_BATCH_RAW_BYTES = 2_400_000


def _page_batches(pngs: list[bytes]):
    """Yield (first_page_number, [page_png, ...]) groups that each fit one request."""
    batch: list[bytes] = []
    size = 0
    start = 1
    for i, png in enumerate(pngs, start=1):
        if batch and size + len(png) > _OCR_BATCH_RAW_BYTES:
            yield start, batch
            batch, size, start = [], 0, i
        batch.append(png)
        size += len(png)
    if batch:
        yield start, batch


def _ocr_attachments(attachments: list, instruction: str = _OCR_INSTRUCTION) -> str:
    """Transcribe one request's worth of attachments. Retries a few times because the
    model occasionally returns an empty response on a dense scan — a real transcription
    is far longer than a handful of characters."""
    text = ""
    for _ in range(3):
        text = gemini_client.generate_text(
            instruction,
            model=config.GEMINI_GEN_MODEL,  # the stronger model transcribes most completely
            system=_OCR_SYSTEM,
            temperature=0.0,
            attachments=attachments,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        )
        if len(text.strip()) >= 40:
            return text
    return text


def _ocr_paper(uploaded, path) -> str:
    """Pass 1 — transcribe the whole paper to plain text (reliable on dense scans).

    Send rendered page images (never the whole PDF inline — a scanned PDF can blow past
    the proxy's payload cap and 413), grouped into as FEW requests as fit the cap. A
    normal paper is one request, exactly as before; only a long/heavy scan splits into a
    couple. Pages go at full render resolution and are never downscaled, so transcription
    quality is preserved. Each batch is told its real page numbers so the "=== Page N ==="
    markers stay correct for figure routing. Falls back to a single whole-file request
    only when page rendering is unavailable."""
    pngs = page_images.render_pages(path)
    if not pngs:
        return _ocr_attachment_fallback(uploaded)
    chunks = []
    for start, batch in _page_batches(pngs):
        parts = [gemini_client.image_part(p, mime_type=page_images.MIME) for p in batch]
        last = start + len(batch) - 1
        instruction = (
            f"{_OCR_INSTRUCTION}\n\nThese {len(batch)} images are pages {start} to {last} "
            f"of the paper, in order (the first image is page {start}). Begin each page's "
            f"transcription with a line '=== Page N ===' using its REAL page number from "
            "that range."
        )
        chunks.append(_ocr_attachments(parts, instruction))
    return "\n\n".join(chunks)


def _ocr_attachment_fallback(uploaded) -> str:
    """Whole-file transcription, used only when page rendering is unavailable."""
    return _ocr_attachments([uploaded])


def _embedded_pdf_text(path: Path) -> str:
    """Return embedded PDF text with page markers, or '' when it looks scanned.

    Many generated/digital papers already contain selectable text. For dense maths
    that text is often more faithful than model OCR: integral limits, exponents and
    fractions survive as separate text runs even when OCR normalises them into a
    nearby textbook example. Page markers keep the later structuring/page-routing
    path working exactly like OCR output.
    """
    if path.suffix.lower() not in extract.PDF_EXTS:
        return ""
    try:
        reader = PdfReader(str(path))
    except Exception:  # noqa: BLE001
        return ""
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        pages.append(f"=== Page {i} ===\n{text.strip()}")
    joined = "\n\n".join(pages).strip()
    n_pages = max(len(reader.pages), 1)
    # Below this it is probably a scan; use multimodal OCR instead.
    return joined if len(joined) >= 200 * n_pages else ""


def parse_paper(path: str | Path, *, uploaded=None) -> ParsedPaper:
    path = Path(path)
    ext = path.suffix.lower()

    multimodal = ext in extract.PDF_EXTS | extract.IMAGE_EXTS
    if multimodal:
        if uploaded is None:
            uploaded = gemini_client.upload_file(path)
        paper_text = _embedded_pdf_text(path) or _ocr_paper(uploaded, path)
    else:
        paper_text = extract.extract_text(path)

    def _structure_text(model: str) -> ParsedPaper:
        contents = f"{_INSTRUCTION}\n\n=== QUESTION PAPER (verbatim transcription) ===\n{paper_text}"
        data = gemini_client.generate_json(contents, model=model, system=_SYSTEM)
        return _build(data if isinstance(data, dict) else {})

    def _structure_images(model: str) -> ParsedPaper:
        """Structure straight from the page images (high-res, else the uploaded file).
        The text pass reliably STRUCTURES but sometimes leaves figure-heavy sub-parts
        with empty text; reading the actual page recovers that text — and it also
        covers a transcription that came back empty."""
        atts = None
        if multimodal:
            pngs = page_images.render_pages(path)
            if pngs:
                atts = [gemini_client.image_part(p, mime_type=page_images.MIME) for p in pngs]
            elif uploaded is not None:
                atts = [uploaded]
        if not atts:
            return ParsedPaper(title=None, total_marks=None, questions=[])
        data = gemini_client.generate_json([*atts, _INSTRUCTION], model=model, system=_SYSTEM)
        return _build(data if isinstance(data, dict) else {})

    def _better(candidate: ParsedPaper) -> None:
        nonlocal paper
        if candidate.questions and (
            not paper.questions or _missing_text(candidate) < _missing_text(paper)
        ):
            paper = candidate

    # Structuring clean text is reliable, but a transient empty/short response can
    # drop some questions or return none at all. Escalate only when needed and keep
    # whichever attempt read the most — answering a question whose text didn't parse
    # only produces hallucinations.
    paper = ParsedPaper(title=None, total_marks=None, questions=[])
    try:
        paper = _structure_text(config.GEMINI_PARSE_MODEL)
    except gemini_client.GeminiJSONError:
        # A dense paper can make the model truncate or leave a string open. Treat
        # that as a parse attempt failure and continue to the stronger/image pass.
        pass
    if _missing_text(paper) >= 1 or not paper.questions:
        try:
            _better(_structure_text(config.GEMINI_GEN_MODEL))
        except Exception:  # noqa: BLE001 - keep the best parse so far
            pass

    # If the text pass still left questions textless (common for figure-dense parts
    # like organic reaction schemes) or found nothing at all, re-read from the page
    # images and keep whichever structuring recovered the most text.
    if multimodal and (not paper.questions or _missing_text(paper) >= 1):
        try:
            _better(_structure_images(config.GEMINI_GEN_MODEL))
        except Exception:  # noqa: BLE001
            pass

    if not paper.questions:
        raise ValueError(
            "Could not read any questions from the paper. This can happen on a very "
            "dense or low-quality scan, or from a temporary model error — please try "
            "generating again, or re-upload a clearer copy."
        )
    if multimodal:
        paper.page_texts = _split_pages(paper_text)
    return paper
