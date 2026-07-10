"""Deterministic schematic renderers for two more Class 9-10 physics staples:
magnetic field patterns and the human-eye defect ray diagrams.

Like optics_render, the answer model supplies a small set of PHYSICS choices
(which pattern, which defect) rather than coordinates, and Python draws the
correct, labelled figure. Each entry point returns PNG bytes or None so the
caller can fall back to caption text.
"""
from __future__ import annotations

from io import BytesIO

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    _OK = True
except Exception:  # noqa: BLE001 - matplotlib / numpy missing
    _OK = False

_INK = "#1a1a1a"
_RAY = "#0b62c4"

MAGNET_KINDS = {"bar", "wire", "solenoid"}
EYE_DEFECTS = {"normal", "myopia", "hypermetropia"}
_EYE_ALIASES = {
    "nearsighted": "myopia", "near_sighted": "myopia", "short_sight": "myopia",
    "shortsighted": "myopia", "farsighted": "hypermetropia",
    "far_sighted": "hypermetropia", "hyperopia": "hypermetropia",
    "long_sight": "hypermetropia", "normal_eye": "normal",
}


def available() -> bool:
    return _OK


def _png(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ======================================================================
# MAGNET — magnetic field patterns
# ======================================================================
def _bar_magnet(ax) -> None:
    """Bar-magnet dipole field via streamplot of a +pole (N) and -pole (S), so the
    lines flow N->S and look exactly like the textbook pattern."""
    gy, gx = np.mgrid[-2.6:2.6:220j, -3.6:3.6:220j]
    d = 1.05

    def pole(px, q):
        rx, ry = gx - px, gy - 0.0
        r3 = (rx * rx + ry * ry) ** 1.5 + 1e-9
        return q * rx / r3, q * ry / r3

    bx1, by1 = pole(-d, +1.0)   # N pole (source)
    bx2, by2 = pole(+d, -1.0)   # S pole (sink)
    bx, by = bx1 + bx2, by1 + by2
    ax.streamplot(gx, gy, bx, by, density=0.5, color=_INK, linewidth=0.9,
                  arrowsize=1.3, broken_streamlines=False)
    # magnet body on top (masks the near-pole singularity)
    mx, my = 1.05, 0.42
    ax.add_patch(mpatches.Rectangle((-mx, -my), mx, 2 * my, fc="#e9483b",
                                    ec=_INK, lw=1.4, zorder=5))
    ax.add_patch(mpatches.Rectangle((0, -my), mx, 2 * my, fc="#4a6bd6",
                                    ec=_INK, lw=1.4, zorder=5))
    ax.text(-mx / 2, 0, "N", color="white", ha="center", va="center",
            fontsize=15, fontweight="bold", zorder=6)
    ax.text(mx / 2, 0, "S", color="white", ha="center", va="center",
            fontsize=15, fontweight="bold", zorder=6)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-2.6, 2.6)


def _wire(ax, current_out: bool) -> None:
    """Straight current-carrying conductor perpendicular to the page: concentric
    field circles, direction by the right-hand rule (out of page -> anticlockwise)."""
    ax.add_patch(mpatches.Circle((0, 0), 0.16, fc="white", ec=_INK, lw=1.6, zorder=5))
    if current_out:
        ax.plot([0], [0], "o", color=_INK, ms=6, zorder=6)          # dot = out
        cur = "I (out of page)"
    else:
        ax.plot([-0.11, 0.11], [-0.11, 0.11], color=_INK, lw=1.6, zorder=6)
        ax.plot([-0.11, 0.11], [0.11, -0.11], color=_INK, lw=1.6, zorder=6)  # cross = in
        cur = "I (into page)"
    for R in (0.6, 1.1, 1.6, 2.1):
        ax.add_patch(mpatches.Circle((0, 0), R, fill=False, ec=_RAY, lw=1.2))
        # arrowhead on the right of each circle; anticlockwise if current is out.
        th = 0.0
        x, y = R, 0.0
        dy = 0.18 if current_out else -0.18          # tangent direction
        ax.annotate("", xy=(x, y + dy), xytext=(x, y - dy),
                    arrowprops=dict(arrowstyle="-|>", color=_RAY, lw=1.2))
    ax.text(0, -2.55, cur, ha="center", va="top", fontsize=10, color=_INK)
    ax.text(0.25, 2.15, "B", color=_RAY, fontsize=11, fontstyle="italic")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.7, 2.6)


def _solenoid(ax) -> None:
    """Solenoid: coils, straight internal field lines and external return loops,
    behaving like a bar magnet with N and S ends."""
    n, x0, x1 = 6, -1.6, 1.6
    xs = np.linspace(x0, x1, n)
    for x in xs:
        ax.add_patch(mpatches.Ellipse((x, 0), 0.26, 2.0, fill=False, ec=_INK, lw=1.5))
    # internal field lines (point from S to N inside; field exits N end)
    for y in (-0.55, 0.0, 0.55):
        ax.annotate("", xy=(x1 + 0.15, y), xytext=(x0 - 0.15, y),
                    arrowprops=dict(arrowstyle="-|>", color=_RAY, lw=1.3))
    # external return loops: from the N end (right) up-and-over to the S end (left),
    # so each line closes N -> outside -> S like a bar magnet's field.
    end_n, end_s = x1 + 0.1, x0 - 0.1
    cx, rx = (end_n + end_s) / 2, (end_n - end_s) / 2
    for ry in (1.4, 2.05):
        t = np.linspace(0, np.pi, 80)
        ex = cx + rx * np.cos(t)
        ey = ry * np.sin(t)
        ax.plot(ex, ey, color=_RAY, lw=1.1)          # upper loop
        ax.plot(ex, -ey, color=_RAY, lw=1.1)         # lower loop
        # arrowheads at the top/bottom point left (N -> S outside)
        ax.annotate("", xy=(cx - 0.12, ry), xytext=(cx + 0.12, ry),
                    arrowprops=dict(arrowstyle="-|>", color=_RAY, lw=1.1))
        ax.annotate("", xy=(cx - 0.12, -ry), xytext=(cx + 0.12, -ry),
                    arrowprops=dict(arrowstyle="-|>", color=_RAY, lw=1.1))
    ax.text(x1 + 0.35, 0, "N", ha="left", va="center", fontsize=13, fontweight="bold")
    ax.text(x0 - 0.35, 0, "S", ha="right", va="center", fontsize=13, fontweight="bold")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.5, 2.5)


def magnet(spec: dict) -> bytes | None:
    if not _OK or not isinstance(spec, dict):
        return None
    kind = str(spec.get("kind", "bar")).strip().lower()
    if kind not in MAGNET_KINDS:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=140)
    if kind == "bar":
        _bar_magnet(ax)
    elif kind == "wire":
        out = str(spec.get("current", "out")).strip().lower() in ("out", "out_of_page", "outward", "up")
        _wire(ax, out)
    else:
        _solenoid(ax)
    ax.set_aspect("equal")
    ax.axis("off")
    return _png(fig)


# ======================================================================
# EYE — human-eye defects and their correction
# ======================================================================
def _draw_eye(ax, R):
    ax.add_patch(mpatches.Circle((0, 0), R, fill=False, ec=_INK, lw=1.6))
    # eye lens at the front
    ax.add_patch(mpatches.Ellipse((-R + 0.12, 0), 0.16, 1.1, fc="#dfeafb",
                                  ec=_INK, lw=1.4))
    ax.text(-R + 0.12, -1.0, "Eye lens", ha="center", va="top", fontsize=8.5, color=_INK)
    # retina (back inner wall)
    th = np.linspace(-0.8, 0.8, 40)
    ax.plot(R * np.cos(th) * 0.98, R * np.sin(th) * 0.98, color="#c0392b", lw=2.4)
    ax.text(R * 0.72, -R * 0.8, "Retina", ha="center", va="top", fontsize=8.5,
            color="#c0392b")


def eye(spec: dict) -> bytes | None:
    if not _OK or not isinstance(spec, dict):
        return None
    defect = str(spec.get("defect", "normal")).strip().lower().replace(" ", "_")
    defect = _EYE_ALIASES.get(defect, defect)
    if defect not in EYE_DEFECTS:
        return None
    corrected = bool(spec.get("corrected", False))

    R = 1.15
    lens_x = -R + 0.12
    retina_x = R * 0.98
    hh = 0.42                       # incident ray half-height
    fig, ax = plt.subplots(figsize=(5.8, 3.9), dpi=140)
    ax.axhline(0, color=_INK, lw=0.8, xmin=0.02, xmax=0.98)
    _draw_eye(ax, R)

    # Where the (uncorrected) eye focuses, relative to the retina.
    if defect == "normal":
        focus_x, note = retina_x, "Image forms on the retina (normal eye)"
    elif defect == "myopia":
        focus_x, note = retina_x - 0.55, "Image forms in FRONT of the retina (myopia)"
    else:
        focus_x, note = retina_x + 0.7, "Image would form BEHIND the retina (hypermetropia)"

    corr_x = -2.3                    # corrective-lens position
    start_x = -3.2

    def incident(y):
        ax.plot([start_x, lens_x], [y, y], color=_RAY, lw=1.3)

    if corrected and defect in ("myopia", "hypermetropia"):
        # corrective lens bends the parallel rays so the eye lens focuses on retina
        convex = defect == "hypermetropia"
        _corr_lens(ax, corr_x, convex)
        label = "Convex lens" if convex else "Concave lens"
        ax.text(corr_x, -1.35, label, ha="center", va="top", fontsize=9,
                color=_INK, fontweight="bold")
        for s in (+1, -1):
            y = s * hh
            ax.plot([start_x, corr_x], [y, y], color=_RAY, lw=1.3)      # parallel in
            y2 = y * (0.6 if convex else 1.35)                          # bent at corr lens
            ax.plot([corr_x, lens_x], [y, y2], color=_RAY, lw=1.3)      # to eye lens
            ax.plot([lens_x, retina_x], [y2, 0], color=_RAY, lw=1.3)    # to retina
        note = f"Corrected with a {label.lower()}: image now on the retina"
        focus_x = retina_x
    else:
        for s in (+1, -1):
            y = s * hh
            incident(y)
            ax.plot([lens_x, focus_x], [y, 0], color=_RAY, lw=1.3)      # converge to focus
            if defect == "myopia":                                     # diverge on to retina
                yr = -s * 0.18
                ax.plot([focus_x, retina_x], [0, yr], color=_RAY, lw=1.1)
            elif defect == "hypermetropia":                            # not yet converged
                yr = s * 0.16
                ax.plot([lens_x, retina_x], [y, yr], color=_RAY, lw=1.3)
                ax.plot([retina_x, focus_x], [yr, 0], color="#c0392b", lw=1.0,
                        ls=(0, (4, 3)))
        ax.plot([focus_x], [0], "o", color=_INK, ms=4)

    ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=_INK)
    ax.set_xlim(start_x - 0.2, R + 0.6)
    ax.set_ylim(-1.7, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return _png(fig)


def _corr_lens(ax, x, convex: bool):
    H = 0.6
    ax.plot([x, x], [-H, H], color=_INK, lw=1.6)
    dx, dy = 0.05, 0.14
    if convex:
        ax.plot([x - dx, x, x + dx], [H - dy, H, H - dy], color=_INK, lw=1.6)
        ax.plot([x - dx, x, x + dx], [-H + dy, -H, -H + dy], color=_INK, lw=1.6)
    else:
        ax.plot([x - dx, x, x + dx], [H, H - dy, H], color=_INK, lw=1.6)
        ax.plot([x - dx, x, x + dx], [-H, -H + dy, -H], color=_INK, lw=1.6)
