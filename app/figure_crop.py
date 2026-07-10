"""Reuse a figure that is printed in the uploaded question paper.

When a question SHOWS a figure the answer must reproduce (a given circuit, graph,
map or labelled diagram), redrawing it risks getting it wrong. Instead the answer
model emits a directive naming the page and a bounding box:

    [[FIG PAPER]]
    {"page": 2, "bbox": [0.10, 0.20, 0.62, 0.55], "caption": "The given circuit"}
    [[/FIG]]

`resolve_paper_directives` crops that region straight out of the rendered page
image and rewrites the directive into a self-contained [[FIG IMG]] block (base64
JPEG) that the document builder embeds verbatim. bbox values are fractions of the
page (0..1) as [left, top, right, bottom]. Unresolvable directives (bad page or
bbox, no page image) collapse to their caption so no raw tag reaches the document.
"""
from __future__ import annotations

import base64
import json
import re
from io import BytesIO

try:
    from PIL import Image

    _PIL = True
except Exception:  # noqa: BLE001 - Pillow missing
    _PIL = False

# [[FIG PAPER]] {json} [[/FIG]] — mirrors the tolerant close of figures.DIRECTIVE_RE.
_PAPER_RE = re.compile(
    r"\[\[\s*FIG\s+PAPER\s*\]\]\s*(.*?)\s*"
    r"(?:\[\[\s*/\s*FIG\s*\]\]|(?=\[\[\s*FIG\b)|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_MAX_W = 1000     # cap crop width (px) so the inlined base64 stays small
_PAD = 0.03       # pad the model's bbox slightly — its boxes tend to run tight


def _parse(body: str):
    s = body.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
        s = s.rsplit("```", 1)[0]
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None


def _clamp01(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def crop_from_page(page_png: bytes, bbox, *, pad: float = _PAD) -> bytes | None:
    """Crop the normalized bbox [left, top, right, bottom] out of a page image and
    return JPEG bytes, or None if the crop is invalid."""
    if not _PIL or not page_png or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    vals = [_clamp01(x) for x in bbox]
    if any(v is None for v in vals):
        return None
    left, top, right, bottom = vals
    left, top = max(0.0, left - pad), max(0.0, top - pad)
    right, bottom = min(1.0, right + pad), min(1.0, bottom + pad)
    if right - left < 0.02 or bottom - top < 0.02:
        return None
    try:
        img = Image.open(BytesIO(page_png)).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    w, h = img.size
    box = (int(left * w), int(top * h), int(right * w), int(bottom * h))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    crop = img.crop(box)
    if crop.width > _MAX_W:
        scale = _MAX_W / crop.width
        crop = crop.resize((_MAX_W, max(1, int(crop.height * scale))))
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def resolve_paper_directives(text: str, paper_pages) -> str:
    """Replace every [[FIG PAPER]] directive with a [[FIG IMG]] block carrying the
    cropped figure, or with its caption text when the crop cannot be made."""
    if not text or "PAPER" not in text.upper() or not paper_pages:
        return text

    def repl(m: "re.Match") -> str:
        spec = _parse(m.group(1)) or {}
        caption = str(spec.get("caption", "")).strip()
        try:
            page = int(spec.get("page"))
        except (TypeError, ValueError):
            page = None
        png = None
        if page and 1 <= page <= len(paper_pages):
            png = crop_from_page(paper_pages[page - 1], spec.get("bbox"))
        if png is None:
            return caption  # drop the tag; keep any caption as plain text
        data = base64.b64encode(png).decode("ascii")
        payload = json.dumps({"data": data, "caption": caption})
        return f"[[FIG IMG]]\n{payload}\n[[/FIG]]"

    return _PAPER_RE.sub(repl, text)
