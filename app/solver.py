"""Generate marks-aware, knowledge-base-grounded answers in the chosen language."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from . import config, figure_crop, gemini_client, page_images
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
- OPTICS — a ray diagram for ONE thin lens or spherical mirror. Give the PHYSICS, not coordinates: the system solves the lens/mirror equation and draws the correct, fully labelled ray diagram (object, image, F, 2F/C, principal axis, image nature). Body: JSON
    {"element":"convex_lens|concave_lens|concave_mirror|convex_mirror","focal_length":15,"object_distance":25,"object_height":5,"caption":".."}
    Use POSITIVE magnitudes in cm; the object is taken to be on the left. ALWAYS use OPTICS (never DIAGRAM) for a lens or mirror ray diagram — it computes the rays correctly, whereas hand-placed coordinates do not.
- FBD — a free-body / force diagram. Body: JSON
    {"body":"block|circle|dot","forces":[{"name":"Weight mg","dir":"down","mag":2},{"name":"Normal N","dir":"up","mag":2},{"name":"Friction f","dir":"left","mag":1},{"name":"Applied F","dir":"right","mag":1.5}],"caption":".."}
    Each force has a name, a direction ("up"/"down"/"left"/"right"/"up-right"/… or "angle" in degrees measured CCW from the +x axis) and an optional "mag" (the arrow length is drawn proportional to it). Use FBD — not DIAGRAM — for any force or free-body diagram.
- MAGNET — a magnetic field pattern. Body: JSON
    {"kind":"bar|wire|solenoid","current":"out|in","caption":".."}
    "bar" draws a bar magnet's N–S dipole field; "wire" the field around a straight current-carrying conductor (set "current" to "out" or "in" for the current's direction through the page — the circles are drawn by the right-hand rule); "solenoid" a solenoid's field with labelled N and S ends. Use MAGNET for any magnetic-field question.
- EYE — a human-eye ray diagram for a vision defect and its correction. Body: JSON
    {"defect":"normal|myopia|hypermetropia","corrected":true,"caption":".."}
    Draws the eye lens, retina and the focused rays: myopia focuses in FRONT of the retina (corrected with a concave lens), hypermetropia BEHIND it (corrected with a convex lens). Set "corrected" to true to also draw the correcting lens. Use EYE for eye-defect questions.
- MICROSCOPE — the compound-microscope ray diagram (objective + eyepiece, object AB, real inverted A'B', virtual magnified A''B'', foci labelled). Body: JSON
    {"caption":".."}
    Use MICROSCOPE for "draw a ray diagram showing image formation by a compound microscope".
- TELESCOPE — the reflecting (Cassegrain) telescope ray diagram: parallel rays -> concave primary -> convex secondary -> through the hole -> eyepiece. Body: JSON
    {"caption":".."}
    Use TELESCOPE for "draw a ray diagram of a reflecting telescope".
- EMWAVE — an electromagnetic wave travelling along X, with the oscillating E (along Y) and B (along Z) drawn mutually perpendicular and labelled. Body: JSON
    {"caption":".."}
    Use EMWAVE for "draw a diagram showing the propagation of an electromagnetic wave".
- FIELDLINES — a point charge with its radial electric field lines AND concentric equipotential surfaces. Body: JSON
    {"charge":"negative|positive","caption":".."}
    Lines point inward for a negative charge, outward for a positive one. Use FIELDLINES for "draw the equipotential surfaces and field lines of a point charge".
- PHASOR — the phasor diagram of a series L-C-R circuit (V_R along the current, V_L up, V_C down, resultant V and the angle phi). Body: JSON
    {"vr":3,"vl":4,"vc":1,"caption":".."}
    Give any consistent relative magnitudes. Use PHASOR for L-C-R phasor questions.
- PRISM — refraction of a ray through a prism at minimum deviation, with A, i1, r1, r2, i2, the normals and the deviation marked. Body: JSON
    {"angle":60,"caption":".."}
    Use PRISM for prism/minimum-deviation questions.
- PAPER — REUSE a figure that is already printed in the question paper (a given circuit, graph, map, data table or labelled diagram) rather than redrawing it. Body: JSON
    {"page":2,"bbox":[left,top,right,bottom],"caption":".."}
    page is the 1-based page of the paper; bbox locates the figure as fractions of that page (0 = left/top, 1 = right/bottom). Use PAPER ONLY when the paper actually SHOWS the figure and you can see it on the attached page image, and box just that figure (a little margin is fine). When the answer needs a NEW figure the paper does not contain (e.g. a ray diagram you must draw), use OPTICS/FBD/MAGNET/EYE/DIAGRAM instead — never PAPER.
- DIAGRAM — a generic labelled diagram for cases the types above cannot express (vectors, geometry, a simple labelled sketch). Body: JSON
    {"shapes":[{"type":"line|arrow|circle|rect|point|label","x1":0,"y1":0,"x2":0,"y2":0,"cx":0,"cy":0,"r":0,"x":0,"y":0,"w":0,"h":0,"text":"..","label":".."}],"caption":".."}
    Use x1,y1,x2,y2 for line/arrow; cx,cy,r for circle; x,y,w,h for rect; x,y(+label) for point; x,y,text for label.

Rules for figures:
- YOU CAN DRAW. Emitting a [[FIG ...]] block produces a REAL, rendered image in the finished document. NEVER claim you cannot draw or create an image, and NEVER hand the drawing back to the reader. Sentences like "Since I cannot draw an image directly...", "যেহেতু আমি ছবি আঁকতে পারি না", "Drawing Instructions", "Graph Plotting Instructions", "Steps to Draw the Diagram", "draw it in your notebook", or "see the diagram in your textbook" are FORBIDDEN — they score ZERO for the diagram. Emit the figure itself instead.
- If the question says draw / sketch / plot / trace / "show diagrammatically" / "draw a ray diagram" / "draw the graph" / "draw the circuit", you MUST emit a [[FIG ...]] block. Choose the closest TYPE and commit to real values. A correctly-labelled figure always beats prose describing one; prose describing a figure earns nothing.
- For a GRAPH question (a characteristic curve, "variation of X with Y", an intensity distribution), use PLOT and supply representative numeric x/y data with the CORRECT SHAPE, computed from the physics — e.g. Kmax = hv - phi0 is a straight line starting at v0; a reverse-bias V-I curve is flat then drops at breakdown. A qualitative graph still needs real numbers to render.
- A figure only SUPPLEMENTS the written answer; it NEVER replaces it. Always write the COMPLETE answer in plain text — every definition, step, equation and the FINAL result — so the answer stands on its own even if the figure is removed. Never end an answer on a figure, and never leave the result to be "read off" a diagram (e.g. still write "so i = tan⁻¹(μ₂/μ₁)" or "so the structure of D is 4-hydroxybenzyl chloride" in words). A drawn figure is in addition to that written conclusion, not instead of it.
- A bare SMILES, formula or caption on its own line does NOT become a figure — you MUST wrap it in the [[FIG TYPE]] ... [[/FIG]] tags exactly as shown, or it will appear as raw text. Never write a SMILES string or a figure caption on its own without the tags.
- Give correct numbers, structures and connectivity — these render EXACTLY as specified.
- ALL text inside a figure — every label, axis name, caption, node name and annotation — MUST be Latin letters, symbols and numbers ONLY (e.g. E_C, E_V, conduction band, N, S, object, image, F, i, r, V_R). This is a HARD rule: the figure renderer has NO Bengali/Hindi/other-script font, so ANY native-script character in a figure comes out as empty boxes (□□□). Even when the answer is in Bengali/Hindi, write the figure's labels in English/Latin and use the standard symbol for each part — never the native-script word. Keep them short.
- LABEL every part a textbook diagram would label — a bare or half-labelled figure loses marks. Mark the object/image and rays, normal and angles on an optics diagram; each component and its value on a circuit; every force and its direction on a free-body diagram; each named part on apparatus or a biology sketch; the axes and key points on a graph. Prefer a correctly and fully labelled figure over an elaborate but unlabelled one.
- Use a figure ONLY when the question actually calls for one; plain text needs no figure. Do NOT add a figure to a question that just asks to state / name / define / explain / fill in the blank / choose the correct option, or whose instruction limits the answer to a single sentence, one word or a very short answer — those get NO figure even if the topic could be illustrated. Draw a figure only when the question explicitly asks for one (draw/sketch/plot/diagram) OR the answer genuinely cannot be understood without it. An unnecessary figure is a defect, not a bonus.
- ONLY if no TYPE above can express the figure at all (e.g. a photo-realistic lab-apparatus sketch) may you describe it — and then in ONE short sentence inside the answer, never as a numbered "how to draw it" procedure and never as a note addressed to the reader. If a rough version is expressible with DIAGRAM, draw that instead of describing it.'''


# Every mathematical expression must be LaTeX so the document renders it as a real,
# typeset equation instead of garbled ASCII ("^(3/2)", stray backticks). This is the
# single most important formatting rule for maths/physics/chemistry answers.
_MATH_INSTRUCTIONS = r'''MATHEMATICS — write ALL mathematics as LaTeX:
- Wrap every inline expression in single dollars: $ ... $  (e.g. "যেহেতু $\tan(\pi/2 - x) = \cot x$, তাই ...").
- Put every equation or working step that sits on its own line inside double dollars, on ONE line: $$ ... $$  (e.g. $$|\vec{a}| = \sqrt{(-3)^2 + 6^2 + (-2)^2} = \sqrt{49} = 7$$).
- This covers EVERYTHING mathematical: fractions, powers, roots, integrals, limits, sums, derivatives, vectors, subscripts/superscripts, Greek letters, angles, and even a lone symbol like $\theta$, $x^2$, $\pi$, or a value like $60^\circ$.
- Use standard LaTeX: \frac{a}{b}, \sqrt{...}, x^{2}, a_{1}, \int_{a}^{b}, \lim_{x \to 0}, \sum, \vec{a}, \hat{i}, \sin \cos \tan \cot \sec \log \ln, \sin^{-1}, \cos^{-1}, \theta \alpha \beta \pi \lambda \mu, \times \cdot \pm \leq \geq \neq \Rightarrow \rightarrow \infty, \left( ... \right), and ^\circ for degrees.
- NEVER write maths as plain ASCII or ad-hoc notation: no "^(3/2)", no "∫(a)^(b)" for limits, no bare "sqrt", no "a/b" typed inline for a real fraction. NEVER wrap maths (or anything) in backticks.
- NEVER write bare tokens such as "frac√2937", "int_0^(π/2)", "sin^(-1)√x/(a+x)" or "√x/a+x". Use valid LaTeX instead: $\frac{\sqrt{293}}{7}$, $\int_{0}^{\pi/2}$, $\sin^{-1}\sqrt{\frac{x}{a+x}}$.
- For inverse-trigonometric functions and roots, put the whole intended argument inside braces. For example write $\sin^{-1}\sqrt{\frac{x}{a+x}}$, not $\sin^{-1}\sqrt{x}/(a+x)$; write $\tan^{-1}\sqrt{\frac{x}{a}}$, not $\tan^{-1}\sqrt{x}/a$.
- If the question uses a specific inverse-trigonometric form, keep that form in the final answer when possible. Equivalent substitutions may use another form in the working, but convert the final line back to the question's form, e.g. finalise $\int \sin^{-1}\sqrt{\frac{x}{a+x}}\,dx$ as $(a+x)\sin^{-1}\sqrt{\frac{x}{a+x}}-\sqrt{ax}+C$ rather than an equivalent $\tan^{-1}$ form.
- Never put words or prose (in any language) inside $...$ — the delimiters hold ONLY mathematical notation; keep all prose outside them.
- For a magnitude, modulus or absolute value, use \left| ... \right| (e.g. $\left|\vec{a}\right|$, $\cos\theta = \frac{\vec{a}\cdot\vec{b}}{\left|\vec{a}\right|\left|\vec{b}\right|}$), not bare | ... | bars.
- Keep each $$...$$ on a SINGLE line with ONE equation. For a multi-step derivation, put each step on its own line as its own $$...$$.
- Do NOT use \begin{matrix}, \begin{vmatrix}, \begin{array}, \begin{cases} or \begin{aligned}. Write a determinant as its expansion on one line (e.g. $$\vec{b}\times\vec{c} = \hat{i}(b_2 c_3 - b_3 c_2) - \hat{j}(b_1 c_3 - b_3 c_1) + \hat{k}(b_1 c_2 - b_2 c_1)$$) and write systems/cases as separate lines.
- Chemical formulae and reaction equations are NOT LaTeX maths: write them as plain text (H₂SO₄, CH₃COOH, 2KMnO₄ → ...) or as a [[FIG RXN]] figure, never inside $...$.'''


# The textbook prints certain answers as a GRID, and an examiner expects a grid back.
# The document builder already turns a markdown table into a real bordered Word table
# (cells go through the same rich renderer, so $...$ becomes a native equation), but
# nothing ever ASKED the model for one — so truth tables came back flattened into
# prose bullets ("The four rows give: p = T, q = T: ...").
_TABLE_INSTRUCTIONS = r'''TABLES — when the standard textbook answer IS a table, write a real table:
- Use a GitHub-style markdown table: a header row, then a separator row of "| :--- |" cells, then one row per line. It is converted into a real Word table with borders.
- Use one WHENEVER the expected answer is tabular: a truth table, a comparison ("differences between X and Y", "distinguish between"), tabulated observations or data, or a classification.
- NEVER flatten a table into prose or bullet points. Writing "The four rows give: p = T, q = T: ..." where the textbook prints a grid is a defect, not a style choice.
- TRUTH TABLES: give one column per simple statement, one column per intermediate expression the proof needs, and one column for each side being compared — the same columns the textbook shows. List EVERY combination of truth values, one per row (2 statements → 4 rows, 3 → 8). Below the table, state in one line which columns are identical and the conclusion that follows.
- Keep cells short. Maths inside a cell still uses $...$ (e.g. $\sim(p \vee q)$, $\sim p \wedge \sim q$); keep prose outside the delimiters.
- Do not leave a blank line inside a table, and never wrap a table in backticks.'''


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
        "Answer in EXAM ANSWER-WRITING style, CRISP and sized to the marks. State the "
        "relevant law or principle and the formula used, substitute the given values "
        "WITH THEIR UNITS, carry units through the working, and give the final numerical "
        "answer with the correct unit and sensible significant figures. Show each working "
        "step as an equation. Add a brief conceptual justification only where the question "
        "asks for it. When a diagram helps (ray diagram, circuit, free-body, field lines), "
        "draw it as a figure the way the textbook does and LABEL every relevant part — for "
        "optics: the object, image, incident/refracted/reflected rays, the normal and the "
        "angles i and r, and the pole/focus/centre; for circuits: each component and its "
        "value; for a free-body diagram: every force with its direction; and mark the sign "
        "convention where one applies. Never leave a diagram bare or half-labelled, and add "
        "a one-line note of what it shows. Be concise: do not pad with textbook exposition, "
        "and keep the length within what the marks warrant."
    ),
    "Chemistry": (
        "Answer in EXAM ANSWER-WRITING style suited to the question type, CRISP and sized "
        "to the marks. For reactions: write the correct reactant and product FORMULAE "
        "first, then BALANCE the equation — verify every atom and the charge balance "
        "before moving on — and add the state symbols and the reagents/conditions over "
        "the arrow; name the reaction where relevant (e.g. Finkelstein, Cannizzaro, "
        "Fries). Getting this opening equation right is essential: a wrong or unbalanced "
        "first equation invalidates the whole answer, so double-check it. For structures "
        "or mechanisms: give correct structures and IUPAC names, drawn as "
        "[[FIG MOL]]/[[FIG RXN]] figures with the product ALSO named in words. For "
        "numerical parts: state the formula, substitute with units, and give the final "
        "value with its unit. For descriptive parts: be concise, correct and to the "
        "point. Do not over-explain or pad beyond what the marks warrant."
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
        "exactly. STRICTLY obey any word limit the question states (e.g. 'in about 100 "
        "words', 'not more than 150 words', 'in 100-150 words') — count your words and "
        "NEVER exceed it; if no limit is given, match the length to the marks. English "
        "answers must be concise and to the point — never pad to fill space."
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


# A word limit stated in the question — "in about 120 words", "not more than 150
# words", "in 100-150 words", "(word limit: 200)". English answers in particular
# must obey these; faculty flag answers that overshoot the paper's stated limit.
_WORD_LIMIT_RE = re.compile(
    r"(\d{1,4})\s*(?:-|–|—|to)\s*(\d{1,4})\s*words?"   # a range, e.g. 100-150 words
    r"|(\d{1,4})\s*words?"                              # a single count, e.g. 120 words
    r"|words?\s*limit\s*[:\-]?\s*(\d{1,4})"            # 'word limit: 120' / 'word limit 120'
    r"|words?\s*[:\-]\s*(\d{1,4})",                     # 'words: 120' / 'word - 120'
    re.IGNORECASE,
)


def _stated_word_limit(*texts: str | None) -> int | None:
    """Return the word count the question states as a limit, or None. For a range
    ('100-150 words') the upper bound is the limit; if several are mentioned the
    largest (most lenient) wins so we never under-constrain a genuine answer."""
    best: int | None = None
    for t in texts:
        if not t:
            continue
        for m in _WORD_LIMIT_RE.finditer(t):
            hi = m.group(2) or m.group(1) or m.group(3) or m.group(4) or m.group(5)
            try:
                n = int(hi)
            except (TypeError, ValueError):
                continue
            if 10 <= n <= 2000 and (best is None or n > best):
                best = n
    return best


def _marks_guidance(marks: float | None) -> str:
    # Examiners award marks point-by-point (or step-by-step), so the surest way to be
    # BOTH complete and crisp is ~one distinct creditable point/step per mark: enough
    # to earn every mark, nothing padded beyond them. Word bands are a secondary cap.
    if marks is None:
        return (
            "No marks are shown for this question. Judge the depth from the question "
            "itself and answer completely but concisely — cover what it asks and no more."
        )
    m = float(marks)
    if m <= 1:
        return (
            "This is a 1-mark question: give ONLY the key fact, value, term or final "
            "result, in one or two precise sentences. No explanation or background."
        )
    if m <= 3:
        return (
            f"This is a {marks}-mark question: give about {int(m)} distinct creditable "
            f"points (or working steps) — roughly one per mark — in about {int(m * 30)}-"
            f"{int(m * 50)} words. Cover exactly what earns the marks, then stop; be crisp "
            "and do not pad."
        )
    if m <= 6:
        return (
            f"This is a {marks}-mark question: give about {int(m)} distinct creditable "
            f"points/steps (roughly one per mark), each explained briefly with correct "
            f"terminology and an example only where it earns a mark, in about "
            f"{int(m * 30)}-{int(m * 50)} words. Enough to score full marks — nothing "
            "padded beyond them."
        )
    return (
        f"This is a {marks}-mark long answer: cover about {int(m)} distinct creditable "
        f"points/steps (roughly one per mark) as numbered points or short headings, in "
        f"about {int(m * 28)}-{int(m * 48)} words — definitions, explanation, "
        "steps/derivation or a described diagram as the question needs. Thorough enough "
        "for full marks, but never padded beyond what the marks reward."
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
        # A word limit written in the question is a hard cap and overrides the
        # marks-based length guidance above — faculty penalise answers that run over.
        limit = _stated_word_limit(question_text, instruction, parent_text)
        if limit:
            parts.append(
                f"STRICT WORD LIMIT: the question asks for an answer within {limit} "
                f"words. Keep the ENTIRE answer at or under {limit} words — do NOT "
                "exceed it. This limit overrides the general length guidance above: "
                "write concisely, include only what the answer needs, and stop as soon "
                "as it is complete. Mentally count the words and trim anything over."
            )
    if class_level or board:
        level = " ".join(x for x in (board, class_level) if x)
        if board:
            convention = "that level and board's syllabus/answer conventions"
        elif class_level in {"NEET", "JEE"}:
            convention = f"the {class_level} exam pattern and expected depth"
        else:
            convention = "that level's syllabus and answer conventions"
        parts.append(
            f"The paper is for this level: {level}. Pitch the depth, vocabulary, "
            f"examples and rigour to {convention} - thorough enough to score full marks, but neither "
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
        "written. If the selected language is English, every explanatory sentence must "
        "be English; do not copy Bengali/Hindi/other-language prose from the question "
        "or sources into the answer. Do not restate the question, do not add headings like 'Answer:'. "
        "Output only the answer content."
    )
    parts.append(
        "The question text can be OCR'd or bilingual, so mathematical notation may have "
        "missing fraction bars or spacing. Infer the standard board-level intent from "
        "the visible question/page image and solve it cleanly. Do not add caveats such "
        "as 'the question is not generally correct' or 'the relation is wrong' unless "
        "the question explicitly asks you to identify an error."
    )
    parts.append(_FIGURE_INSTRUCTIONS)
    # Typeset-maths (LaTeX) instructions only for quantitative subjects; Biology/General
    # answers are prose/point-wise and would only be cluttered by them.
    if (subject or "General") in _QUANTITATIVE:
        parts.append(_MATH_INSTRUCTIONS)
    # Every subject can need a table — truth tables in Maths, comparison tables in
    # Biology/Science, "distinguish between" in Social Science — so this is not gated.
    parts.append(_TABLE_INSTRUCTIONS)
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


_COMBINED_CLASS_LEVELS: dict[str, tuple[str, ...]] = {
    "Class 11+12": ("Class 11", "Class 12"),
    "NEET": ("Class 11", "Class 12"),
    "JEE": ("Class 11", "Class 12"),
}


def _retrieval_levels(class_level: str | None) -> tuple[str | None, ...]:
    if not class_level:
        return (None,)
    return _COMBINED_CLASS_LEVELS.get(class_level, (class_level,))


def _strict_level_filter(class_level: str | None) -> bool:
    levels = set(_retrieval_levels(class_level))
    return bool(levels & {"Class 11", "Class 12"})


def _query_level_hits(
    query: str,
    top_k: int,
    *,
    subject: str | None,
    class_level: str | None,
    board: str | None,
) -> list[Hit]:
    store = get_store()
    levels = _retrieval_levels(class_level)
    hits: list[Hit] = []
    seen: set[tuple[str, int | None]] = set()
    for level in levels:
        for hit in store.query_text(
            query, top_k, subject=subject, class_level=level, board=board
        ):
            key = (hit.metadata.get("source_id") or hit.source, hit.metadata.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            hits.append(hit)
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k]


def _display_sources(hits: list[Hit], class_level: str | None, board: str | None) -> list[str]:
    levels = {level for level in _retrieval_levels(class_level) if level}
    sources: list[str] = []
    for hit in hits:
        meta = hit.metadata or {}
        if levels and meta.get("class_level") not in levels:
            continue
        if board and meta.get("board") not in (board, None):
            continue
        if hit.source not in sources:
            sources.append(hit.source)
    return sources


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
    hits = _query_level_hits(
        query, top_k, subject=subj, class_level=cls, board=brd
    )
    if not hits and cls and brd:
        hits = _query_level_hits(
            query, top_k, subject=subj, class_level=cls, board=None
        )
    # Board/class are narrowing filters for lower-school content. For Class 11/12
    # and entrance papers, do not cite lower-class books as references: if no
    # level-appropriate material exists, answer from subject knowledge instead.
    if not hits and cls and not _strict_level_filter(cls):
        hits = store.query_text(query, top_k, subject=subj, class_level=None, board=brd)
    if not hits and brd:
        hits = store.query_text(query, top_k, subject=subj, class_level=None, board=None)
    context = "\n\n---\n\n".join(f"[Source: {h.source}]\n{h.text}" for h in hits)
    return context, _display_sources(hits, cls, brd)


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
    # If the model chose to reuse a figure printed in the paper, crop it out of the
    # page image now (we have the rasters here) and inline it. Only meaningful for
    # figure questions, where the page was actually shown to the model.
    if attachments is not None and paper_pages:
        answer = figure_crop.resolve_paper_directives(answer, paper_pages)
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
