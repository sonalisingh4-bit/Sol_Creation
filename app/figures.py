"""Render figure directives the answer model emits into real PNG images.

The model writes a block directive instead of trying to "draw" in text:

    [[FIG TYPE]]
    <body>
    [[/FIG]]

TYPE is one of MOL, RXN (chemistry, via RDKit), PLOT (matplotlib), CIRCUIT
(schemdraw), FLOW (networkx + matplotlib) or DIAGRAM (matplotlib primitives).
Each renderer returns PNG bytes or None; on None the caller falls back to the
caption text, so a malformed figure never breaks generation.
"""
from __future__ import annotations

import json
import re
from io import BytesIO

from . import chem_render

# Block form: opening tag, body, then a close. The close is tolerant — models
# listing many structures often omit [[/FIG]], so a body also ends at the next
# [[FIG ...]] opener or at end-of-text. The explicit [[/FIG]] terminator still
# keeps JSON bodies (which contain ']' and even ']]') safe when it is present.
DIRECTIVE_RE = re.compile(
    r"\[\[\s*FIG\s+(MOL|RXN|PLOT|CIRCUIT|FLOW|DIAGRAM)\s*\]\]"  # opener
    r"\s*(.*?)\s*"                                              # body (lazy)
    r"(?:\[\[\s*/\s*FIG\s*\]\]|(?=\[\[\s*FIG\b)|\Z)",           # close | next opener | end
    re.IGNORECASE | re.DOTALL,
)

# Embed width (inches) per figure type.
WIDTH = {
    "MOL": 2.6, "RXN": 4.8, "PLOT": 5.0,
    "CIRCUIT": 4.2, "FLOW": 5.3, "DIAGRAM": 4.6,
}

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    _MPL = True
except Exception:  # noqa: BLE001
    _MPL = False

try:
    import networkx as nx

    _NX = True
except Exception:  # noqa: BLE001
    _NX = False


# A whole line that looks like a bare SMILES the model forgot to wrap in [[FIG]].
_SMILES_LINE = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#%./\\>]{4,}$")


def smiles_line_png(line: str) -> bytes | None:
    """Render a standalone line if it is a bare SMILES/reaction SMILES, else None.

    Safety net for when the model emits a SMILES on its own line instead of a
    proper [[FIG MOL]] directive. Conservative: requires SMILES-only characters
    plus a bracket/bond/digit, and only renders if RDKit actually parses it — so
    prose, option labels like "(c)", and condensed formulae (CH3COCH3) won't match.
    """
    s = line.strip()
    if not _SMILES_LINE.match(s):
        return None
    if not (any(c in s for c in "()[]=#") or any(c.isdigit() for c in s)):
        return None
    return chem_render.rxn_png(s) if ">>" in s else chem_render.mol_png(s)


def _parse_json(body: str):
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


def _fig_png(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# --- PLOT (matplotlib) -----------------------------------------------------
def _plot_png(spec: dict) -> bytes | None:
    if not _MPL:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=130)
    have_label = False
    for s in spec.get("series") or []:
        y = s.get("y") or []
        x = s.get("x")
        if not isinstance(x, list) or len(x) != len(y):
            x = list(range(len(y)))
        kind = str(s.get("kind", "line")).lower()
        label = s.get("label")
        have_label = have_label or bool(label)
        if kind == "scatter":
            ax.scatter(x, y, label=label)
        elif kind == "bar":
            ax.bar(x, y, label=label)
        else:
            ax.plot(x, y, marker="o", label=label)
    if spec.get("title"):
        ax.set_title(str(spec["title"]))
    if spec.get("xlabel"):
        ax.set_xlabel(str(spec["xlabel"]))
    if spec.get("ylabel"):
        ax.set_ylabel(str(spec["ylabel"]))
    for a in spec.get("annotations") or []:
        try:
            ax.annotate(str(a.get("text", "")), (a.get("x"), a.get("y")),
                        fontsize=8, color="black")
        except Exception:  # noqa: BLE001
            pass
    if have_label:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    return _fig_png(fig)


# --- CIRCUIT (schemdraw) ---------------------------------------------------
_ELEM = {
    "battery": "Battery", "cell": "Cell", "resistor": "Resistor", "r": "Resistor",
    "capacitor": "Capacitor", "c": "Capacitor", "inductor": "Inductor",
    "lamp": "Lamp", "bulb": "Lamp", "switch": "Switch", "diode": "Diode",
    "led": "LED", "fuse": "Fuse", "source": "SourceV", "source_v": "SourceV",
    "source_i": "SourceI", "ammeter": "MeterA", "voltmeter": "MeterV",
    "line": "Line", "dot": "Dot", "ground": "Ground",
}


def _circuit_png(spec: dict) -> bytes | None:
    try:
        import schemdraw
        import schemdraw.elements as elm
    except Exception:  # noqa: BLE001
        return None
    d = schemdraw.Drawing(show=False)
    for e in spec.get("elements") or []:
        name = _ELEM.get(str(e.get("type", "line")).lower(), "Line")
        cls = getattr(elm, name, elm.Line)
        el = cls()
        direction = str(e.get("dir", "right")).lower()
        getattr(el, direction, el.right)()
        label = e.get("label")
        if label:
            el.label(str(label))
        d += el
    return bytes(d.get_imagedata("png"))


# --- FLOW (networkx + matplotlib) -----------------------------------------
def _layered_pos(G, direction: str):
    try:
        gens = list(nx.topological_generations(G))
    except Exception:  # noqa: BLE001 - cyclic graph
        gens = None
    if not gens:
        return nx.spring_layout(G, seed=1)
    pos = {}
    for i, layer in enumerate(gens):
        m = len(layer)
        for j, node in enumerate(layer):
            off = j - (m - 1) / 2
            pos[node] = (i * 2.4, -off * 1.6) if direction == "LR" else (off * 2.6, -i * 1.7)
    return pos


def _flow_png(spec: dict) -> bytes | None:
    if not (_MPL and _NX):
        return None
    G = nx.DiGraph()
    labels: dict[str, str] = {}
    for n in spec.get("nodes") or []:
        nid = str(n.get("id", n.get("label", "")))
        if nid:
            G.add_node(nid)
            labels[nid] = str(n.get("label", nid))
    for e in spec.get("edges") or []:
        a, b = str(e.get("from", "")), str(e.get("to", ""))
        if a and b:
            G.add_edge(a, b, label=str(e.get("label", "")))
    if G.number_of_nodes() == 0:
        return None
    for nid in G.nodes:
        labels.setdefault(nid, nid)

    direction = str(spec.get("direction", "TB")).upper()
    pos = _layered_pos(G, direction)

    fig, ax = plt.subplots(figsize=(5.8, 3.9), dpi=130)
    for a, b, data in G.edges(data=True):
        ax.annotate("", xy=pos[b], xytext=pos[a],
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3,
                                    shrinkA=20, shrinkB=20))
        lbl = data.get("label")
        if lbl:
            mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
            ax.text(mx, my, lbl, fontsize=7.5, color="black", ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    for nid, (x, y) in pos.items():
        ax.text(x, y, labels[nid], ha="center", va="center", fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.2))
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.6, max(xs) + 1.6)
    ax.set_ylim(min(ys) - 1.3, max(ys) + 1.3)
    ax.axis("off")
    return _fig_png(fig)


# --- DIAGRAM (matplotlib primitives) --------------------------------------
def _diagram_png(spec: dict) -> bytes | None:
    if not _MPL:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=130)
    xs: list[float] = []
    ys: list[float] = []

    def track(*pts):
        for x, y in pts:
            xs.append(x)
            ys.append(y)

    for sh in spec.get("shapes") or []:
        t = str(sh.get("type", "")).lower()
        try:
            if t == "line":
                ax.plot([sh["x1"], sh["x2"]], [sh["y1"], sh["y2"]], color="#222", lw=1.5)
                track((sh["x1"], sh["y1"]), (sh["x2"], sh["y2"]))
            elif t == "arrow":
                ax.annotate("", xy=(sh["x2"], sh["y2"]), xytext=(sh["x1"], sh["y1"]),
                            arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.5))
                track((sh["x1"], sh["y1"]), (sh["x2"], sh["y2"]))
                if sh.get("label"):
                    ax.text((sh["x1"] + sh["x2"]) / 2, (sh["y1"] + sh["y2"]) / 2,
                            str(sh["label"]), fontsize=8, color="black")
            elif t == "circle":
                ax.add_patch(mpatches.Circle((sh["cx"], sh["cy"]), sh["r"],
                                             fill=False, ec="#222", lw=1.5))
                track((sh["cx"] - sh["r"], sh["cy"] - sh["r"]),
                      (sh["cx"] + sh["r"], sh["cy"] + sh["r"]))
            elif t == "rect":
                ax.add_patch(mpatches.Rectangle((sh["x"], sh["y"]), sh["w"], sh["h"],
                                                fill=False, ec="#222", lw=1.5))
                track((sh["x"], sh["y"]), (sh["x"] + sh["w"], sh["y"] + sh["h"]))
            elif t == "point":
                ax.plot([sh["x"]], [sh["y"]], "o", color="#222")
                track((sh["x"], sh["y"]))
                if sh.get("label"):
                    ax.text(sh["x"] + 0.1, sh["y"] + 0.1, str(sh["label"]), fontsize=8)
            elif t == "label":
                ax.text(sh["x"], sh["y"], str(sh.get("text", "")), fontsize=9, ha="center")
                track((sh["x"], sh["y"]))
        except Exception:  # noqa: BLE001 - skip a malformed shape, keep the rest
            continue

    if xs and ys:
        mx = (max(xs) - min(xs)) * 0.1 + 0.5
        my = (max(ys) - min(ys)) * 0.1 + 0.5
        ax.set_xlim(min(xs) - mx, max(xs) + mx)
        ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect("equal")
    ax.axis("off")
    return _fig_png(fig)


def render_match(match: "re.Match") -> tuple[bytes | None, str, str]:
    """Return (png_or_None, kind, caption) for a [[FIG ...]] directive match."""
    kind = match.group(1).upper()
    body = match.group(2).strip()

    if kind in ("MOL", "RXN"):
        # With a missing [[/FIG]] the body can absorb the next "B:" label, so take
        # only the first non-empty line ("<smiles> | <caption>").
        line = next((ln for ln in body.splitlines() if ln.strip()), "")
        smiles, _, caption = line.partition("|")
        smiles, caption = smiles.strip(), caption.strip()
        png = chem_render.mol_png(smiles) if kind == "MOL" else chem_render.rxn_png(smiles)
        # caption is shown only when the model wrote one; fallback (raw SMILES) is
        # used only if rendering failed, so a good structure never gets a SMILES label.
        return png, kind, caption, (caption or smiles)

    spec = _parse_json(body)
    caption = str((spec or {}).get("caption", "")).strip()
    png = None
    if spec is not None:
        try:
            if kind == "PLOT":
                png = _plot_png(spec)
            elif kind == "CIRCUIT":
                png = _circuit_png(spec)
            elif kind == "FLOW":
                png = _flow_png(spec)
            elif kind == "DIAGRAM":
                png = _diagram_png(spec)
        except Exception:  # noqa: BLE001 - never let a bad figure crash generation
            png = None
    # On failure keep only a real caption. A generic "(diagram figure)" placeholder
    # carries no information and reads as a broken document, so emit nothing instead.
    return png, kind, caption, caption
