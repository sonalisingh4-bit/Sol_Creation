"""Render LaTeX math into native Word equations (OMML).

The model writes mathematics as LaTeX ($...$ inline, $$...$$ display). Word does
not read LaTeX, but it *does* read OMML (Office Math Markup Language) — real,
editable, crisply-rendered equations that also survive the .docx -> .pdf step.

Pipeline:  LaTeX --latex2mathml--> MathML --(Office's MML2OMML.XSL)--> OMML.

MML2OMML.XSL is the exact stylesheet Word itself uses; it ships with every
Microsoft Office install. If it (or latex2mathml) is unavailable, `available()`
returns False and callers fall back to readable text — the document still builds,
just without typeset equations.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from docx.oxml import parse_xml
from lxml import etree

# Vertical-bar tokens that already state their side/size — we must NOT touch these
# when normalising bare '|' bars.
_EXPLICIT_BARS = (
    r"\left|", r"\right|", r"\|", r"\lvert", r"\rvert", r"\lVert", r"\rVert",
    r"\vert", r"\Vert", r"\mid", r"\bigl|", r"\bigr|", r"\big|", r"\Big|",
    r"\Bigl|", r"\Bigr|", r"\bigm|", r"\Bigm|",
)


def _fix_abs_bars(latex: str) -> str:
    r"""Turn bare |...| bars into \left|...\right| pairs.

    Word drops the content of a bare '|...|' when it sits inside a \frac (a
    magnitude like |\vec a||\vec b| in a denominator renders as empty bars), but
    \left|...\right| renders correctly everywhere. Bars come in matched pairs
    (magnitude, modulus, norm), so alternate left/right. If the count is odd
    (ambiguous, e.g. a set-builder "such that" bar) leave it untouched rather than
    risk an unbalanced expression."""
    # Protect bars that already carry an explicit side/size.
    protected = latex
    holes: list[str] = []
    for tok in _EXPLICIT_BARS:
        while tok in protected:
            holes.append(tok)
            protected = protected.replace(tok, f"\x00{len(holes) - 1}\x00", 1)
    if protected.count("|") % 2 != 0:
        return latex  # odd number of bare bars — don't guess
    out, left = [], True
    for ch in protected:
        if ch == "|":
            out.append(r"\left|" if left else r"\right|")
            left = not left
        else:
            out.append(ch)
    result = "".join(out)
    for i, tok in enumerate(holes):
        result = result.replace(f"\x00{i}\x00", tok, 1)
    return result

# Office ships MML2OMML.XSL under ...\Microsoft Office\root\OfficeNN\ (and, on
# older installs, directly under ...\OfficeNN\). Search both, newest first.
_XSL_GLOBS = (
    r"C:/Program Files/Microsoft Office/root/Office*/MML2OMML.XSL",
    r"C:/Program Files (x86)/Microsoft Office/root/Office*/MML2OMML.XSL",
    r"C:/Program Files/Microsoft Office/Office*/MML2OMML.XSL",
    r"C:/Program Files (x86)/Microsoft Office/Office*/MML2OMML.XSL",
)


def _find_xsl() -> Path | None:
    # pathlib.glob needs a pattern relative to a base dir, so split off the drive
    # anchor (e.g. "C:/") and glob the remainder under it.
    for pattern in _XSL_GLOBS:
        base = Path(pattern).anchor
        rel = pattern[len(base):]
        try:
            hits = sorted(Path(base).glob(rel), reverse=True)
        except (OSError, ValueError):
            hits = []
        for h in hits:
            if h.is_file():
                return h
    return None


@lru_cache(maxsize=1)
def _transform() -> etree.XSLT | None:
    xsl = _find_xsl()
    if xsl is None:
        return None
    try:
        return etree.XSLT(etree.parse(str(xsl)))
    except Exception:  # noqa: BLE001 - malformed/unreadable stylesheet
        return None


@lru_cache(maxsize=1)
def available() -> bool:
    """True if LaTeX->OMML rendering is possible on this machine."""
    try:
        import latex2mathml.converter  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return _transform() is not None


@lru_cache(maxsize=512)
def _omml_bytes(latex: str) -> bytes | None:
    """Cached LaTeX -> OMML conversion, returned as serialized XML bytes (or None if
    conversion is unavailable or the LaTeX cannot be parsed). Caching the *bytes*
    (not a live element) lets the same formula convert once while each caller still
    gets its own element — see latex_to_omath."""
    transform = _transform()
    if transform is None:
        return None
    try:
        from latex2mathml.converter import convert

        mathml = convert(_fix_abs_bars(latex))
        omml = transform(etree.fromstring(mathml.encode("utf-8")))
        return etree.tostring(omml.getroot())
    except Exception:  # noqa: BLE001 - latex2mathml / XSLT reject some input
        return None


def latex_to_omath(latex: str):
    """Convert a LaTeX fragment to an OMML <m:oMath> element ready to append to a
    paragraph, or None if conversion is unavailable or the LaTeX cannot be parsed.

    Returns a FRESH element on every call. An lxml element can live in only one
    place in a tree, so appending a shared (cached) element to a second paragraph
    would MOVE it out of the first — blanking every occurrence of a repeated formula
    except the last. Parsing fresh from the cached bytes avoids that."""
    latex = (latex or "").strip()
    if not latex:
        return None
    xml = _omml_bytes(latex)
    if xml is None:
        return None
    return parse_xml(xml)
