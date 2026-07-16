"""Render LaTeX math into native Word equations (OMML).

The model writes mathematics as LaTeX ($...$ inline, $$...$$ display). Word does
not read LaTeX, but it *does* read OMML (Office Math Markup Language) — real,
EDITABLE, crisply-rendered equations that also survive the .docx -> .pdf step.

Pipeline:  LaTeX --latex2mathml--> MathML --> OMML, via either
  1. Office's own MML2OMML.XSL, when this machine has Word installed, or
  2. the pure-Python `mathml2omml` package — the path that matters in production.

Route 2 exists because MML2OMML.XSL ships only with Microsoft Office, so a Linux
host (Render) has no stylesheet: every equation used to silently fall back to a
matplotlib PNG, i.e. a picture the faculty could not edit. `mathml2omml` needs no
Office and no pandoc, so hosted builds now get real editable equations too.

Only if BOTH routes fail does a caller drop to `latex_to_png` (a picture) and then
to plain text — the document always builds, it just degrades.
"""
from __future__ import annotations

from io import BytesIO
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


def _has_mathml2omml() -> bool:
    try:
        import mathml2omml  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@lru_cache(maxsize=1)
def available() -> bool:
    """True if LaTeX->OMML (native, editable) rendering is possible here — via either
    Office's stylesheet or the pure-Python converter."""
    try:
        import latex2mathml.converter  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return _transform() is not None or _has_mathml2omml()


# mathml2omml emits bare m:/w: prefixes with no xmlns declarations, so lxml cannot
# parse its output until they are declared on the root element.
_OMML_NS = (
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
)


# mathml2omml 0.0.2 closes <m:groupChrPr> with </m:groupChr> instead of
# </m:groupChrPr>, so EVERY accent (\vec, \hat, \bar — i.e. most of physics) comes out
# as malformed XML that lxml rejects, silently demoting the equation to a picture.
# Repair that one mismatched tag: the first </m:groupChr> after an unclosed
# <m:groupChrPr> belongs to the Pr element.
_BAD_GROUPCHRPR = re.compile(r"(<m:groupChrPr>(?:(?!</m:groupChr>).)*?)</m:groupChr>")


def _omml_from_mathml(mathml: str) -> bytes | None:
    """MathML -> OMML using the pure-Python converter (no Office, no pandoc)."""
    try:
        import mathml2omml

        omml = mathml2omml.convert(mathml)
    except Exception:  # noqa: BLE001 - converter missing or rejects the input
        return None
    if not omml or "<m:oMath" not in omml:
        return None
    omml = _BAD_GROUPCHRPR.sub(r"\1</m:groupChrPr>", omml)
    if "xmlns:m=" not in omml:
        omml = omml.replace("<m:oMath>", f"<m:oMath {_OMML_NS}>", 1)
    try:
        return etree.tostring(etree.fromstring(omml.encode("utf-8")))
    except Exception:  # noqa: BLE001 - malformed OMML
        return None


@lru_cache(maxsize=512)
def _omml_bytes(latex: str) -> bytes | None:
    """Cached LaTeX -> OMML conversion, returned as serialized XML bytes (or None if
    neither route can express it). Caching the *bytes* (not a live element) lets the
    same formula convert once while each caller still gets its own element — see
    latex_to_omath."""
    try:
        from latex2mathml.converter import convert

        mathml = convert(_fix_abs_bars(latex))
    except Exception:  # noqa: BLE001 - latex2mathml rejects some input
        return None
    # 1) Office's own stylesheet, when Word is installed on this machine.
    transform = _transform()
    if transform is not None:
        try:
            omml = transform(etree.fromstring(mathml.encode("utf-8")))
            return etree.tostring(omml.getroot())
        except Exception:  # noqa: BLE001 - XSLT rejects some MathML; try route 2
            pass
    # 2) Pure-Python route — the one that runs on the Linux host.
    return _omml_from_mathml(mathml)


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


def _mathtext_latex(latex: str) -> str:
    """Small compatibility pass for Matplotlib's built-in math renderer."""
    s = (latex or "").strip()
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\operatorname", r"\mathrm")
    return s


@lru_cache(maxsize=512)
def latex_to_png(latex: str) -> bytes | None:
    """Render LaTeX-ish math to a transparent PNG using Matplotlib mathtext.

    LAST RESORT ONLY: a picture is not editable, which faculty need it to be. It is
    reached only when BOTH OMML routes fail for this particular expression (see
    _omml_bytes). It used to be the normal path on Linux hosts, before the
    pure-Python mathml2omml route removed the dependency on Office's stylesheet.
    """
    latex = _mathtext_latex(latex)
    if not latex:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.mathtext import math_to_image

        buf = BytesIO()
        math_to_image(f"${latex}$", buf, dpi=220, format="png")
        return buf.getvalue()
    except Exception:  # noqa: BLE001 - fall back to readable text in caller
        return None
