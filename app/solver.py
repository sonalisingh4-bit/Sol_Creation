"""Generate marks-aware, knowledge-base-grounded answers in the chosen language."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from . import config, gemini_client, page_images
from .paper_parser import ParsedPaper, Question, SubPart
from .vectorstore import Hit, get_store

ProgressFn = Callable[[int, int, str], None]

# Above this page count we stop attaching every page image to a figure question we
# couldn't place, and fall back to the uploaded PDF file — keeps payloads sane.
_MAX_FIG_PAGES = 12
# Shorter than this (stripped) an answer is treated as an empty/failed generation and
# retried, so a hard question is never left silently blank in the document.
_MIN_ANSWER_CHARS = 20
# Words (including Bengali) and integers; used to place a question on its page.
_TOKEN_RE = re.compile(r"[^\W\d_]{2,}|\d+", re.UNICODE)

# Dense math notation (integrals, exponents, fractions, derivatives, roots, vectors)
# is where the plain-text transcription most often garbles the expression — an
# "^(3/2)" becomes stray digits, a numerator gets dropped, etc. We hand these
# questions the page image too (like figure questions) so the model can read the
# real formula rather than solve a mangled one.
_MATH_RE = re.compile(
    r"[∫∑∏√∂∇≤≥≠±]"                         # operators / relations
    r"|\^"                                    # exponent caret
    r"|[⁰¹²³⁴⁵⁶⁷⁸⁹]"                          # superscripts (x², (…)³)
    r"|\d\s*/\s*\d"                           # numeric fraction like 3/2
    r"|d\s*[²2]?\s*[xyz]\s*/\s*d\s*[xyz]"     # derivatives dy/dx, d²y/dx²
    r"|\b(?:lim|log|ln|sin|cos|tan|cot|sec|cosec|det|matrix|dx|dy|dz)\b"
    r"|[îĵ]"                                   # unit vectors
)

_MCQ_KEYWORD_RE = re.compile(
    r"multiple[- ]choice|choose\s+the\s+correct|correct\s+option"
    r"|which\s+of\s+the\s+following|tick\s+the\s+correct",
    re.IGNORECASE,
)
# An option marker: "(a)", "a)", "A.", "(2)", "3)" after start/space/punctuation.
# The (?!\d) guard keeps decimals ("2.5 km", "3.14") from reading as markers.
_OPTION_MARKER_RE = re.compile(r"(?:^|[\s:;,])\(?([A-Da-d1-4])[.)\]](?!\d)")


def _is_math_heavy(text: str | None) -> bool:
    return bool(text) and bool(_MATH_RE.search(text))


def _is_mcq(text: str | None, class_level: str | None = None) -> bool:
    """True only for genuine option-style questions. A real option list shows at
    least three distinct labels in one style ((a)(b)(c) or (1)(2)(3)) — which
    numericals ("2.5 km") and part enumerations "(a) … (b) …" never do."""
    level = (class_level or "").lower()
    if "neet" in level or "jee" in level:
        return True
    if not text:
        return False
    if _MCQ_KEYWORD_RE.search(text):
        return True
    labels = {m.group(1).lower() for m in _OPTION_MARKER_RE.finditer(text)}
    return len(labels & set("abcd")) >= 3 or len(labels & set("1234")) >= 3


def _exam_name(class_level: str | None) -> str:
    level = (class_level or "").lower()
    if "neet" in level:
        return "NEET"
    if "jee" in level:
        return "JEE"
    return "MCQ"


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


def _match_page(query: str, page_texts: list[str]) -> int | None:
    """Return the index of the page this question sits on, or None if unsure.
    Conservative: needs a strong overlap that clearly beats the runner-up, so a
    wrong guess never hides the page that actually holds the figure."""
    q = _tokens(query)
    if len(q) < 4:
        return None
    scored = sorted(
        ((len(q & _tokens(pg)) / len(q), i) for i, pg in enumerate(page_texts)),
        reverse=True,
    )
    if not scored:
        return None
    best, best_i = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best >= 0.5 and (len(scored) == 1 or best - second >= 0.08):
        return best_i
    return None


def _one_page(paper_pages, idx):
    return [gemini_client.image_part(paper_pages[idx], mime_type=page_images.MIME)]


def _figure_attachments(query, paper_file, paper_pages, page_texts, page):
    """Pick the sharpest, most focused view of the paper for a figure question:
    the single page image it lives on when we can place it, else all page images
    (capped), else the uploaded PDF file. A whole-paper or all-pages attachment is a
    last resort — the model reads a drawn structure far more reliably from the one
    page it is on than from the entire paper."""
    if not paper_pages:
        return [paper_file] if paper_file is not None else None
    # 1) The page the parser recorded for this question — reliable even when the
    #    sub-part's text is almost entirely figures (nothing to text-match on).
    if page and 1 <= page <= len(paper_pages):
        return _one_page(paper_pages, page - 1)
    # 2) Otherwise locate it by text overlap against the per-page transcription.
    if len(page_texts) == len(paper_pages):
        idx = _match_page(query, page_texts)
        if idx is not None:
            return _one_page(paper_pages, idx)
    # 3) Couldn't place it: show all pages (small papers) or fall back to the PDF.
    if len(paper_pages) <= _MAX_FIG_PAGES:
        return [gemini_client.image_part(p, mime_type=page_images.MIME) for p in paper_pages]
    return [paper_file] if paper_file is not None else None

# Shown when the paper parser could not read a question's text (e.g. poor scan).
# Better an honest gap than a confident answer to an invented question.
_UNREADABLE = (
    "[This question could not be read from the uploaded paper — the scan may be "
    "unclear. Please re-upload a clearer copy and regenerate.]"
)

_SYSTEM = (
    "You are an expert subject teacher and examiner who writes model answers for "
    "exam question papers. Your answers are accurate, well-structured, and earn "
    "full marks. You ground answers in the supplied reference material."
)

# How the model should request a real, system-rendered figure when one is needed.
_FIGURE_INSTRUCTIONS = '''\
Write prose in plain text: no HTML (no <br>, <img>, &nbsp;), no image URLs, and never use ASCII art or spaces/tabs to position things. Write every mathematical expression as LaTeX (see the MATHEMATICS section).

When the answer GENUINELY needs a drawn figure, emit it as a block that the system renders into a real image. Use this exact form, tags on their own lines:
[[FIG TYPE]]
<body>
[[/FIG]]

Supported TYPE and body:
- MOL — one molecule. Body: a valid SMILES, optionally " | caption".
    e.g.
    [[FIG MOL]]
    CC(=O)C | acetone
    [[/FIG]]
- RXN — a reaction. Body: reaction SMILES "reactants>>products" ("." between species), optional " | caption".
    e.g.
    [[FIG RXN]]
    CC(=O)O.O=S(Cl)Cl>>CC(=O)Cl | acetyl chloride formation
    [[/FIG]]
- PLOT — a graph (kinetics curve, energy profile, titration, line/scatter/bar). Body: JSON
    {"title":"..","xlabel":"..","ylabel":"..","series":[{"label":"..","x":[..],"y":[..],"kind":"line|scatter|bar"}],"annotations":[{"text":"..","x":0,"y":0}],"caption":".."}
- CIRCUIT — an electrical circuit. Body: JSON
    {"elements":[{"type":"battery|cell|resistor|capacitor|inductor|lamp|switch|diode|source|ammeter|voltmeter|line|dot|ground","label":"..","dir":"right|down|left|up"}],"caption":".."}
    List elements in loop order; the directions trace the circuit (e.g. right, right, down, left, up to close a rectangle).
- FLOW — a flowchart / block diagram (extraction steps, process flow). Body: JSON
    {"nodes":[{"id":"a","label":".."}],"edges":[{"from":"a","to":"b","label":".."}],"direction":"TB|LR","caption":".."}
- DIAGRAM — a generic labelled diagram (ray diagram, free-body diagram, vectors, geometry). Body: JSON
    {"shapes":[{"type":"line|arrow|circle|rect|point|label","x1":0,"y1":0,"x2":0,"y2":0,"cx":0,"cy":0,"r":0,"x":0,"y":0,"w":0,"h":0,"text":"..","label":".."}],"caption":".."}
    Use x1,y1,x2,y2 for line/arrow; cx,cy,r for circle; x,y,w,h for rect; x,y(+label) for point; x,y,text for label.

Rules for figures:
- A figure only SUPPLEMENTS the written answer; it NEVER replaces it. Always write the COMPLETE answer in plain text — every definition, step, equation and the FINAL result — so the answer stands on its own even if the figure is removed. Never end an answer on a figure, and never leave the result to be "read off" a diagram (e.g. still write "so i = tan⁻¹(μ₂/μ₁)" or "so the structure of D is 4-hydroxybenzyl chloride" in words). A drawn figure is in addition to that written conclusion, not instead of it.
- A bare SMILES, formula or caption on its own line does NOT become a figure — you MUST wrap it in the [[FIG TYPE]] ... [[/FIG]] tags exactly as shown, or it will appear as raw text. Never write a SMILES string or a figure caption on its own without the tags.
- Give correct numbers, structures and connectivity — these render EXACTLY as specified.
- Keep labels inside figures short and in Latin script/symbols/numbers (axis names, R1, t, [R], A, B) even when the rest of the answer is in another language, so they render cleanly.
- Use a figure only when the question actually calls for one; plain text needs no figure.
- For a figure that none of the TYPEs above can express (e.g. a realistic lab-apparatus sketch), briefly describe it in words instead — do not attempt to draw it.'''


# Every mathematical expression must be LaTeX so the document renders it as a real,
# typeset equation instead of garbled ASCII ("^(3/2)", stray backticks). This is the
# single most important formatting rule for maths/physics/chemistry answers.
_MATH_INSTRUCTIONS = r'''MATHEMATICS — write ALL mathematics as LaTeX:
- Wrap every inline expression in single dollars: $ ... $  (e.g. "যেহেতু $\tan(\pi/2 - x) = \cot x$, তাই ...").
- Put every equation or working step that sits on its own line inside double dollars, on ONE line: $$ ... $$  (e.g. $$|\vec{a}| = \sqrt{(-3)^2 + 6^2 + (-2)^2} = \sqrt{49} = 7$$).
- This covers EVERYTHING mathematical: fractions, powers, roots, integrals, limits, sums, derivatives, vectors, subscripts/superscripts, Greek letters, angles, and even a lone symbol like $\theta$, $x^2$, $\pi$, or a value like $60^\circ$.
- Use standard LaTeX: \frac{a}{b}, \sqrt{...}, x^{2}, a_{1}, \int_{a}^{b}, \lim_{x \to 0}, \sum, \vec{a}, \hat{i}, \sin \cos \tan \cot \sec \log \ln, \sin^{-1}, \cos^{-1}, \theta \alpha \beta \pi \lambda \mu, \times \cdot \pm \leq \geq \neq \Rightarrow \rightarrow \infty, \left( ... \right), and ^\circ for degrees.
- NEVER write maths as plain ASCII or ad-hoc notation: no "^(3/2)", no "∫(a)^(b)" for limits, no bare "sqrt", no "a/b" typed inline for a real fraction. NEVER wrap maths (or anything) in backticks.
- Never put words or prose (in any language) inside $...$ — the delimiters hold ONLY mathematical notation; keep all prose outside them.
- For a magnitude, modulus or absolute value, use \left| ... \right| (e.g. $\left|\vec{a}\right|$, $\cos\theta = \frac{\vec{a}\cdot\vec{b}}{\left|\vec{a}\right|\left|\vec{b}\right|}$), not bare | ... | bars.
- Keep each $$...$$ on a SINGLE line with ONE equation. For a multi-step derivation, put each step on its own line as its own $$...$$.
- Do NOT use \begin{matrix}, \begin{vmatrix}, \begin{array}, \begin{cases} or \begin{aligned}. Write a determinant as its expansion on one line (e.g. $$\vec{b}\times\vec{c} = \hat{i}(b_2 c_3 - b_3 c_2) - \hat{j}(b_1 c_3 - b_3 c_1) + \hat{k}(b_1 c_2 - b_2 c_1)$$) and write systems/cases as separate lines.
- Chemical formulae and reaction equations are NOT LaTeX maths: write them as plain text (H₂SO₄, CH₃COOH, 2KMnO₄ → ...) or as a [[FIG RXN]] figure, never inside $...$.'''


# Subjects whose answers are quantitative and need typeset equations (LaTeX). Other
# subjects (Biology, General) are prose/point-wise and skip the maths instructions.
# Combined Science includes physics/chemistry numericals, so it needs them too.
_QUANTITATIVE = {"Mathematics", "Physics", "Chemistry", "Science"}

# Per-subject "answer-writing style" — how a full-marks answer in that subject is
# actually written. This is the single most important lever for answer quality, and
# it differs sharply by subject (a maths answer IS its working; a biology answer is
# descriptive points). Chosen once per paper from the subject picked at upload.
_SUBJECT_STYLE: dict[str, str] = {
    "Mathematics": (
        "Answer in EXAM ANSWER-WRITING style — the WORKING is the answer. Show every "
        "step of the solution as an equation; never replace a mathematical step with a "
        "sentence that merely describes it, and never skip a step. Keep prose to the "
        "short linking phrases an examiner expects between steps (e.g. 'ধরি', "
        "'সুতরাং', 'উভয় পক্ষে সমাকলন করে পাই'); do not add background exposition or "
        "justify routine algebra at length. Carry the solution all the way to the FINAL "
        "answer the question asks for (the value, closed form, area, angle, general "
        "solution or completed proof) — never stop at an intermediate step — and state "
        "the result clearly at the end. Treat any word-count guidance as a loose upper "
        "bound, not a target."
    ),
    "Physics": (
        "Answer in EXAM ANSWER-WRITING style. State the relevant law or principle and "
        "the formula used, substitute the given values WITH THEIR UNITS, carry units "
        "through the working, and give the final numerical answer with the correct unit "
        "and sensible significant figures. Show each working step as an equation. Add a "
        "brief conceptual justification only where the question asks for it, and use a "
        "labelled diagram (ray diagram, circuit, free-body) where it genuinely aids the "
        "answer. Be concise — do not pad with textbook explanation. Treat any word-count "
        "guidance as a loose upper bound, not a target."
    ),
    "Chemistry": (
        "Answer in EXAM ANSWER-WRITING style suited to the question type. For reactions: "
        "give BALANCED equations with correct formulae, name the reaction where relevant "
        "(e.g. Finkelstein, Cannizzaro, Fries) and write reagents/conditions over the "
        "arrow. For structures or mechanisms: give correct structures and IUPAC names, "
        "drawn as [[FIG MOL]]/[[FIG RXN]] figures with the product ALSO named in words. "
        "For numerical parts: state the formula, substitute with units, and give the "
        "final value with its unit. For descriptive parts: be concise, correct and to "
        "the point. Do not over-explain."
    ),
    "Biology": (
        "Answer in EXAM ANSWER-WRITING style for biology — answers are DESCRIPTIVE and "
        "POINT-WISE, not equations. Use correct biological terminology, organise longer "
        "answers as clearly separated points or short headings, and cover enough "
        "distinct points to earn the marks (roughly one substantive point per mark). "
        "Give definitions, functions, roles, examples or differences exactly as the "
        "question asks. Where a labelled diagram is expected (a cell, a cycle, an organ), "
        "request it as a figure and still describe it in words. Be complete but concise "
        "— do not restate the question or add irrelevant background."
    ),
    "Science": (
        "Answer in EXAM ANSWER-WRITING style for school Science (combined physics, "
        "chemistry and biology). For numericals: state the formula, substitute the "
        "given values WITH UNITS, and give the final value with its unit. For "
        "reactions: write balanced equations with correct formulae. For descriptive "
        "parts: answer point-wise with correct terminology, roughly one substantive "
        "point per mark, and use a labelled diagram where the question expects one. "
        "Keep the depth appropriate for the student's class — no advanced material "
        "beyond the syllabus. Be complete but concise."
    ),
    "Social Science": (
        "Answer in EXAM ANSWER-WRITING style for Social Science (history, geography, "
        "civics, economics). Answers are DESCRIPTIVE and POINT-WISE, never equations. "
        "Use correct names, dates, places and terms; organise longer answers as "
        "clearly separated points or short headings with roughly one substantive "
        "point per mark. For 'explain/describe' give causes, features or consequences "
        "as the question asks; for map/data questions state exactly what is asked. Be "
        "complete but concise — do not restate the question or pad with background."
    ),
    "English": (
        "Answer in EXAM ANSWER-WRITING style for English. For grammar: give the "
        "corrected/transformed sentence directly, with a one-line rule only if asked. "
        "For comprehension: answer from the passage in your own words, brief and "
        "precise. For literature: answer with reference to the text — name the "
        "work/author where relevant and support points with brief evidence or quotes. "
        "For writing tasks (letter, essay, notice): follow the standard exam format "
        "exactly. Match length to the marks; never pad."
    ),
    "General": (
        "Answer in a clear, well-structured, exam-appropriate way: address exactly what "
        "the question asks, at the depth the marks warrant, using correct terminology "
        "and worked steps or points as suits the question. Be complete but concise — do "
        "not restate the question or pad with irrelevant background."
    ),
}


def _subject_style(subject: str | None) -> str:
    return _SUBJECT_STYLE.get(subject or "General", _SUBJECT_STYLE["General"])


def _marks_guidance(marks: float | None) -> str:
    if marks is None:
        return (
            "No marks are specified. Write a complete, well-structured answer "
            "appropriate to the question's depth."
        )
    m = float(marks)
    if m <= 1:
        return "This is a 1-mark question: answer in one or two precise sentences."
    if m <= 3:
        return (
            f"This is a {marks}-mark question: write a focused short answer of about "
            f"{int(m * 40)}-{int(m * 60)} words covering the key points."
        )
    if m <= 6:
        return (
            f"This is a {marks}-mark question: write a detailed answer (~{int(m * 40)}-"
            f"{int(m * 60)} words) with the main points clearly explained, an example "
            "where useful, and correct terminology."
        )
    return (
        f"This is a {marks}-mark long answer: write a comprehensive, well-organised "
        f"response (~{int(m * 35)}-{int(m * 55)} words) using headings or numbered "
        "points, definitions, explanation, steps/derivation or a described diagram, "
        "and examples as appropriate."
    )


def _mcq_guidance(subject: str | None, class_level: str | None) -> str:
    exam = _exam_name(class_level)
    subj = subject or "General"
    ncert = ""
    if exam == "NEET" and subj in {"Physics", "Chemistry", "Biology"}:
        ncert = (
            f"Use NCERT {subj} Class 11 and Class 12 as the main authority. "
            "The retrieved knowledge-base material, when present, is preferred."
        )
    elif exam == "JEE" and subj in {"Physics", "Chemistry", "Mathematics"}:
        ncert = (
            "Use the retrieved reference material first, then standard JEE-level "
            f"{subj} knowledge where needed."
        )
    else:
        ncert = "Use the retrieved reference material first."
    return (
        f"This is a {exam} / multiple-choice style question. Keep the solution brief "
        "and accuracy-focused. Start with the final choice in this format: "
        "'Correct option: (X) ...'. Then give only the essential reasoning/calculation "
        "needed to justify the option. Do not write a long textbook-style answer. "
        f"{ncert} If the question text/options are unclear, incomplete, or you cannot "
        "determine the answer confidently, write 'UNSURE' first and briefly explain why."
    )


def _build_prompt(
    *,
    question_text: str,
    parent_text: str | None,
    marks: float | None,
    instruction: str | None,
    language: str,
    class_level: str | None,
    context: str,
    figure_label: str | None,
    subject: str | None = None,
    board: str | None = None,
    math_page: bool = False,
    is_mcq: bool = False,
) -> str:
    parts: list[str] = []
    if math_page:
        parts.append(
            "=== SOURCE PAGE (read the exact notation) ===\n"
            "The relevant page of the question paper is attached as an image. The "
            "plain-text transcription of this question may have GARBLED its mathematical "
            "notation — an exponent like \"^(3/2)\" turning into stray digits, a "
            "fraction's numerator/denominator swapped or dropped, integral limits or "
            "subscripts mangled. Read the EXACT expression from the attached page image "
            "and solve precisely THAT. If the transcription and the image disagree, the "
            "image is authoritative."
        )
    if figure_label:
        parts.append(
            "=== FIGURE ===\n"
            "The relevant page(s) of the question paper are attached as image(s). This "
            f"question ({figure_label}) depends on something drawn — a structure, "
            "diagram, graph, circuit, table, or answer options drawn as figures. Read it "
            "directly from the attached image and base your answer on what is actually "
            "drawn. Do NOT guess or invent its contents.\n\n"
            "When the question shows reactions or structures as drawings (species "
            "labelled A, B, C … or options (a)-(d)):\n"
            "- Solve each labelled species and each separate reaction INDEPENDENTLY. "
            "Reactions separated by ';', by commas, or on different lines are NOT one "
            "sequence — never feed the product of one into the next unless an arrow "
            "explicitly connects them.\n"
            "- NAME the reaction before giving its product (e.g. Finkelstein, "
            "Hunsdiecker, Fries, Étard, Reimer–Tiemann, Cannizzaro, Clemmensen, "
            "Rosenmund, Kolbe) so the transformation is the correct one.\n"
            "- Copy EVERY substituent shown on each ring or chain — including its "
            "position (–OH, –NO₂, –Cl, –CH₃, ortho/meta/para) — into your product. "
            "Never drop a group the drawing shows, and never add one it does not.\n"
            "- If the answer options are themselves drawn structures, first state "
            "briefly what each option is, then choose; never assume a structure that is "
            "not drawn."
        )
    if context.strip():
        parts.append("=== REFERENCE MATERIAL (knowledge base) ===\n" + context)
        grounding = (
            "Base your answer primarily on the reference material above. If it does not "
            "fully cover the question, supplement it with your own accurate subject "
            "knowledge, but never contradict the references and never invent facts."
        )
    else:
        grounding = (
            "No reference material was retrieved for this question. Answer accurately "
            "from your own subject knowledge."
        )

    parts.append("=== TASK ===")
    if parent_text:
        parts.append(f"Main question context: {parent_text}")
    parts.append(f"Question to answer: {question_text}")
    if instruction:
        parts.append(f"Specific instruction from the paper: {instruction}")
    if is_mcq:
        parts.append(_mcq_guidance(subject, class_level))
    else:
        parts.append(_marks_guidance(marks))
        parts.append(_subject_style(subject))
    if class_level or board:
        level = " ".join(x for x in (board, class_level) if x)
        parts.append(
            f"The student is at this level: {level}. Pitch the depth, vocabulary, "
            "examples and rigour to that level and board's syllabus/answer "
            "conventions — thorough enough to score full marks, but neither "
            "over-advanced nor over-simplified."
        )
    parts.append(grounding)
    parts.append(
        f"Write the ENTIRE answer in {language}, using ONLY its own native script for "
        "the prose. NEVER insert a word written in any other script (e.g. Arabic, "
        "Devanagari, Chinese) — if you don't know a term in the target language, use "
        "the target-language word, not a foreign-script one. The only Latin-script "
        "text allowed is standard formulae, equations, chemical/mathematical symbols, "
        "and universally-used technical terms or proper nouns, as conventionally "
        "written. Do not restate the question, do not add headings like 'Answer:'. "
        "Output only the answer content."
    )
    parts.append(_FIGURE_INSTRUCTIONS)
    # Typeset-maths (LaTeX) instructions only for quantitative subjects; Biology/General
    # answers are prose/point-wise and would only be cluttered by them.
    if (subject or "General") in _QUANTITATIVE:
        parts.append(_MATH_INSTRUCTIONS)
    return "\n\n".join(parts)


@dataclass
class SolvedItem:
    number: str
    text: str
    marks: float | None
    instruction: str | None
    answer: str
    sources: list[str]


@dataclass
class SolvedQuestion:
    number: str
    text: str
    marks: float | None
    instruction: str | None
    answer: str | None = None
    sources: list[str] = field(default_factory=list)
    subparts: list[SolvedItem] = field(default_factory=list)


@dataclass
class SolvedPaper:
    title: str | None
    total_marks: float | None
    language: str
    class_level: str | None
    questions: list[SolvedQuestion]
    board: str | None = None


def _retrieve(
    query: str,
    top_k: int,
    *,
    subject: str | None = None,
    class_level: str | None = None,
    board: str | None = None,
) -> tuple[str, list[str]]:
    store = get_store()
    if store.count() == 0:
        return "", []
    subj = subject if subject and subject != "General" else None
    cls = class_level.strip() if class_level and class_level.strip() else None
    brd = board.strip() if board and board.strip() else None
    hits: list[Hit] = store.query_text(
        query, top_k, subject=subj, class_level=cls, board=brd
    )
    # Board/class are narrowing filters, never a reason to answer with no
    # references: fall back to the whole board's subject material, then to the
    # subject across boards, before giving up.
    if not hits and cls:
        hits = store.query_text(query, top_k, subject=subj, class_level=None, board=brd)
    if not hits and brd:
        hits = store.query_text(query, top_k, subject=subj, class_level=None, board=None)
    context = "\n\n---\n\n".join(f"[Source: {h.source}]\n{h.text}" for h in hits)
    sources: list[str] = []
    for h in hits:
        if h.source not in sources:
            sources.append(h.source)
    return context, sources


def _has_or_alternative(q: Question) -> bool:
    return any(
        "or" in (sp.number or "").lower()
        or "or" in (sp.text or "").lower()
        or "অথবা" in (sp.text or "")
        for sp in q.subparts
    )


def _parent_is_answerable_with_alternatives(q: Question) -> bool:
    """True when q.text is the main question and subparts are OR alternatives.

    The parser can represent a paper's "Q3 ... অথবা ..." as parent text plus an
    "(OR)" subpart. In that shape the parent is itself answerable and must not be
    skipped. Ordinary parents like "Answer the following" still stay un-answered.
    """
    text = (q.text or "").strip()
    if not text or not q.subparts or not _has_or_alternative(q):
        return False
    if q.marks is not None:
        return True
    lowered = text.lower()
    return "?" in text or "what" in lowered or "find" in lowered or "solve" in lowered


def _answer_unit(
    *,
    question_text: str,
    parent_text: str | None,
    marks,
    instruction,
    language: str,
    class_level: str | None,
    subject: str | None,
    board: str | None,
    figure_label: str | None,
    paper_file,
    paper_pages: list[bytes] | None = None,
    page_texts: list[str] | None = None,
    page: int | None = None,
) -> tuple[str, list[str]]:
    query = f"{parent_text + ' ' if parent_text else ''}{question_text}".strip()
    context, sources = _retrieve(
        query,
        config.TOP_K,
        subject=subject,
        class_level=class_level,
        board=board,
    )
    # Attach the page image when the question depends on a figure OR carries dense math
    # notation (so the model reads the real formula, not a garbled transcription).
    math_heavy = _is_math_heavy(question_text) or _is_math_heavy(parent_text)
    # MCQ detection looks at the question's OWN text (its options live there);
    # scanning the concatenated parent text would flag every subpart of a
    # "(a) … (b) …" enumeration. The parent may still carry an explicit
    # "choose the correct option" instruction, so keywords check it too.
    mcq = _is_mcq(question_text, class_level) or bool(
        parent_text and _MCQ_KEYWORD_RE.search(parent_text)
    )
    attachments = None
    if figure_label or math_heavy:
        attachments = _figure_attachments(
            query, paper_file, paper_pages or [], page_texts or [], page
        )
    use_page = attachments is not None
    careful = use_page or mcq
    prompt = _build_prompt(
        question_text=question_text,
        parent_text=parent_text,
        marks=marks,
        instruction=instruction,
        language=language,
        class_level=class_level,
        context=context,
        subject=subject,
        board=board,
        figure_label=figure_label if (use_page and figure_label) else None,
        math_page=bool(use_page and math_heavy and not figure_label),
        is_mcq=mcq,
    )
    def _generate(temperature: float) -> str:
        return gemini_client.generate_text(
            prompt,
            model=config.GEMINI_GEN_MODEL,
            system=_SYSTEM,
            # Figure/math questions read from the page and must not drift/guess — keep
            # them near deterministic; prose keeps a little warmth for fluent writing.
            temperature=temperature,
            attachments=attachments,
        )

    answer = _generate(0.1 if careful else 0.3)
    # A blank or near-empty answer is almost always a transient model miss (it happens
    # on the occasional hard part). It must NEVER leave a question silently unanswered
    # in the document, so retry with a touch more warmth before giving up.
    if len(answer.strip()) < _MIN_ANSWER_CHARS:
        for temp in (0.4, 0.6):
            answer = _generate(temp)
            if len(answer.strip()) >= _MIN_ANSWER_CHARS:
                break
    return answer, sources


def solve_paper(
    paper: ParsedPaper,
    language: str,
    *,
    class_level: str | None = None,
    subject: str | None = None,
    board: str | None = None,
    paper_file=None,
    paper_pages: list[bytes] | None = None,
    progress: ProgressFn | None = None,
) -> SolvedPaper:
    total = paper.n_units + sum(
        1 for q in paper.questions if _parent_is_answerable_with_alternatives(q)
    )
    done = 0
    page_texts = paper.page_texts
    solved_questions: list[SolvedQuestion] = []

    for q in paper.questions:
        sq = SolvedQuestion(
            number=q.number, text=q.text, marks=q.marks, instruction=q.instruction
        )
        if q.subparts:
            if _parent_is_answerable_with_alternatives(q):
                if progress:
                    progress(done, total, f"Q{q.number}")
                answer, sources = _answer_unit(
                    question_text=q.text,
                    parent_text=None,
                    marks=q.marks,
                    instruction=q.instruction,
                    language=language,
                    class_level=class_level,
                    subject=subject,
                    board=board,
                    figure_label=f"Question {q.number}" if q.requires_figure else None,
                    paper_file=paper_file,
                    paper_pages=paper_pages,
                    page_texts=page_texts,
                    page=q.page,
                )
                sq.answer = answer
                sq.sources = sources
                done += 1
            for sp in q.subparts:
                if progress:
                    progress(done, total, f"Q{q.number}({sp.number})")
                if not sp.text.strip():
                    answer, sources = _UNREADABLE, []
                else:
                    needs_fig = sp.requires_figure or q.requires_figure
                    answer, sources = _answer_unit(
                        question_text=sp.text,
                        parent_text=q.text or None,
                        marks=sp.marks,
                        instruction=sp.instruction,
                        language=language,
                        class_level=class_level,
                        subject=subject,
                        board=board,
                        figure_label=f"Question {q.number}({sp.number})" if needs_fig else None,
                        paper_file=paper_file,
                        paper_pages=paper_pages,
                        page_texts=page_texts,
                        page=sp.page if sp.page is not None else q.page,
                    )
                sq.subparts.append(
                    SolvedItem(
                        number=sp.number,
                        text=sp.text,
                        marks=sp.marks,
                        instruction=sp.instruction,
                        answer=answer,
                        sources=sources,
                    )
                )
                done += 1
        else:
            if progress:
                progress(done, total, f"Q{q.number}")
            if not (q.text or "").strip():
                sq.answer, sq.sources = _UNREADABLE, []
            else:
                answer, sources = _answer_unit(
                    question_text=q.text,
                    parent_text=None,
                    marks=q.marks,
                    instruction=q.instruction,
                    language=language,
                    class_level=class_level,
                    subject=subject,
                    board=board,
                    figure_label=f"Question {q.number}" if q.requires_figure else None,
                    paper_file=paper_file,
                    paper_pages=paper_pages,
                    page_texts=page_texts,
                    page=q.page,
                )
                sq.answer = answer
                sq.sources = sources
            done += 1

        solved_questions.append(sq)

    if progress:
        progress(total, total, "done")

    return SolvedPaper(
        title=paper.title,
        total_marks=paper.total_marks,
        language=language,
        class_level=class_level or None,
        questions=solved_questions,
        board=board or None,
    )
