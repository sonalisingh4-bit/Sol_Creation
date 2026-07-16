"""Deterministic schematic renderers for the recurring board-physics figures:
magnetic field patterns, human-eye defects, EM-wave propagation, electrostatic
field lines with equipotentials, the L-C-R phasor diagram and prism refraction.

Like optics_render, the answer model supplies a small set of PHYSICS choices
(which pattern, which defect) rather than coordinates, and Python draws the
correct, labelled figure. Each entry point returns PNG bytes or None so the
caller can fall back to caption text. These exist because board papers ask for
these same diagrams every year, and a model asked to "draw" them otherwise
returns prose instructions instead of a figure.
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
_VIRT = "#c0392b"      # virtual images / dashed construction lines

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


# ======================================================================
# EMWAVE — electromagnetic wave propagating along X (E in X-Y, B in X-Z)
# ======================================================================
def em_wave(spec: dict) -> bytes | None:
    if not _OK or not isinstance(spec, dict):
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=140)
    x = np.linspace(0, 4 * np.pi, 400)
    wave = np.sin(x)
    zdx, zdy = 0.62, 0.46          # skewed Z axis, so X-Z reads as "into the page"

    # axes
    ax.annotate("", xy=(4 * np.pi + 1.3, 0), xytext=(-1.0, 0),
                arrowprops=dict(arrowstyle="-|>", color=_INK, lw=1.4))
    ax.text(4 * np.pi + 1.35, 0.06, "X (direction of propagation)",
            fontsize=8.5, va="bottom", color=_INK)
    ax.annotate("", xy=(0, 1.75), xytext=(0, -1.75),
                arrowprops=dict(arrowstyle="-|>", color=_INK, lw=1.1))
    ax.text(0.10, 1.78, "Y", fontsize=9.5, color=_INK)
    ax.annotate("", xy=(1.5 * zdx, 1.5 * zdy), xytext=(-1.5 * zdx, -1.5 * zdy),
                arrowprops=dict(arrowstyle="-|>", color=_INK, lw=1.1))
    ax.text(1.5 * zdx + 0.06, 1.5 * zdy + 0.04, "Z", fontsize=9.5, color=_INK)

    # E field: oscillates along Y
    ax.plot(x, wave, color=_RAY, lw=1.7)
    for xi in np.linspace(0.35, 4 * np.pi - 0.35, 13):
        ax.annotate("", xy=(xi, np.sin(xi)), xytext=(xi, 0),
                    arrowprops=dict(arrowstyle="-|>", color=_RAY, lw=0.9))
    ax.text(np.pi / 2 + 0.1, 1.22, "E (electric field)", color=_RAY,
            fontsize=10, fontweight="bold")

    # B field: oscillates along Z (drawn skewed), in phase with E
    bx, by = x + wave * zdx, wave * zdy
    ax.plot(bx, by, color="#c0392b", lw=1.7)
    for xi in np.linspace(0.35, 4 * np.pi - 0.35, 13):
        s = np.sin(xi)
        ax.annotate("", xy=(xi + s * zdx, s * zdy), xytext=(xi, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=0.9))
    ax.text(3 * np.pi / 2 + 0.5, -1.05, "B (magnetic field)", color="#c0392b",
            fontsize=10, fontweight="bold")

    ax.text(0.5, -0.02, "E, B and the direction of propagation are mutually perpendicular",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=_INK)
    ax.set_xlim(-1.6, 4 * np.pi + 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.axis("off")
    return _png(fig)


# ======================================================================
# FIELDLINES — point charge: radial E lines + equipotential surfaces
# ======================================================================
def field_lines(spec: dict) -> bytes | None:
    if not _OK or not isinstance(spec, dict):
        return None
    neg = str(spec.get("charge", "negative")).strip().lower().startswith(("neg", "-"))
    fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=140)
    # equipotential surfaces: spacing widens outwards (field weakens)
    for r in (0.85, 1.5, 2.25, 3.1):
        ax.add_patch(mpatches.Circle((0, 0), r, fill=False, ec="#2f6fd6",
                                     lw=1.1, ls=(0, (5, 4))))
    # radial field lines: into a negative charge, out of a positive one
    for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        inner = (0.30 * np.cos(th), 0.30 * np.sin(th))
        outer = (3.5 * np.cos(th), 3.5 * np.sin(th))
        start, end = (outer, inner) if neg else (inner, outer)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=_INK, lw=1.1))
    ax.add_patch(mpatches.Circle((0, 0), 0.24, fc="#4a6bd6" if neg else "#e9483b",
                                 ec=_INK, lw=1.2, zorder=5))
    ax.text(0, 0, "-q" if neg else "+q", ha="center", va="center", color="white",
            fontsize=10, fontweight="bold", zorder=6)
    ax.text(2.25 * 0.72, 2.25 * 0.72 + 0.12, "equipotential\nsurfaces", fontsize=8,
            color="#2f6fd6", ha="center")
    ax.text(0.5, -0.02,
            "Field lines are radial and meet every equipotential surface at 90°",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=_INK)
    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-3.8, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    return _png(fig)


# ======================================================================
# PHASOR — series L-C-R phasor diagram
# ======================================================================
def phasor(spec: dict) -> bytes | None:
    if not _OK or not isinstance(spec, dict):
        return None
    try:
        vr = abs(float(spec.get("vr", 3)))
        vl = abs(float(spec.get("vl", 4)))
        vc = abs(float(spec.get("vc", 1)))
    except (TypeError, ValueError):
        return None
    if vr <= 0:
        return None
    net = vl - vc
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=140)

    def arrow(dx, dy, color, label, lx, ly):
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))
        ax.text(lx, ly, label, color=color, fontsize=10, fontweight="bold")

    arrow(vr, 0, _INK, "$V_R$ (and I)", vr * 0.5, -0.35)
    arrow(0, vl, "#2f6fd6", "$V_L$", 0.12, vl * 0.7)
    arrow(0, -vc, "#c0392b", "$V_C$", 0.12, -vc * 0.7)
    # resultant of VL and VC, then the total V
    ax.plot([0, 0], [0, net], color="#7d3c98", lw=2.4)
    ax.annotate("", xy=(vr, net), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#137a4b", lw=2.2))
    ax.text(vr * 0.55, net * 0.62 + 0.18, "$V$", color="#137a4b",
            fontsize=11, fontweight="bold")
    ax.plot([vr, vr], [0, net], color="#999", lw=1.0, ls=(0, (4, 3)))
    ax.plot([0, vr], [net, net], color="#999", lw=1.0, ls=(0, (4, 3)))
    ax.text(vr * 0.20, net * 0.10 + 0.06,
            r"$\phi$", fontsize=11, color=_INK)
    z = (vr ** 2 + net ** 2) ** 0.5
    ax.text(0.5, -0.02,
            "$V=\\sqrt{V_R^2+(V_L-V_C)^2}$      "
            "$Z=\\sqrt{R^2+(X_L-X_C)^2}$      " + f"$|V|={z:.2f}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=_INK)
    lim = max(vr, vl, vc) * 1.35 + 0.3
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.axvline(0, color="#bbb", lw=0.8)
    ax.set_xlim(-lim * 0.35, lim)
    ax.set_ylim(-lim * 0.8, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    return _png(fig)


# ======================================================================
# PRISM — refraction through a prism (symmetric / minimum-deviation view)
# ======================================================================
def prism(spec: dict) -> bytes | None:
    if not _OK or not isinstance(spec, dict):
        return None
    try:
        A = float(spec.get("angle", 60))
    except (TypeError, ValueError):
        A = 60.0
    if not (10 <= A <= 120):
        A = 60.0
    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=140)

    # triangle with apex angle A at the top
    h = 2.2
    half = h * np.tan(np.radians(A / 2))
    apex = (0.0, h / 2)
    left = (-half, -h / 2)
    right = (half, -h / 2)
    ax.add_patch(mpatches.Polygon([apex, left, right], closed=True, fill=True,
                                  fc="#eef3fb", ec=_INK, lw=1.6))
    ax.text(apex[0], apex[1] + 0.12, f"A = {A:.0f}°", ha="center", fontsize=10,
            fontweight="bold", color=_INK)

    # entry/exit points on the two slant faces, symmetric about the axis
    t = 0.5
    p_in = (apex[0] + (left[0] - apex[0]) * t, apex[1] + (left[1] - apex[1]) * t)
    p_out = (apex[0] + (right[0] - apex[0]) * t, apex[1] + (right[1] - apex[1]) * t)

    # rays: symmetric passage => the inside ray is horizontal (minimum deviation)
    ax.annotate("", xy=p_in, xytext=(p_in[0] - 1.9, p_in[1] + 0.95),
                arrowprops=dict(arrowstyle="-|>", color=_RAY, lw=1.6))
    ax.plot([p_in[0], p_out[0]], [p_in[1], p_out[1]], color=_RAY, lw=1.6)
    ax.annotate("", xy=(p_out[0] + 1.9, p_out[1] - 0.95), xytext=p_out,
                arrowprops=dict(arrowstyle="-|>", color=_RAY, lw=1.6))

    # normals: genuinely perpendicular to each slant face, dashed, labelled N
    for p, face_a, face_b in ((p_in, apex, left), (p_out, apex, right)):
        fx, fy = face_b[0] - face_a[0], face_b[1] - face_a[1]
        n = (fx * fx + fy * fy) ** 0.5 or 1.0
        nx, ny = -fy / n, fx / n            # perpendicular to the face
        ax.plot([p[0] - nx * 1.0, p[0] + nx * 1.0],
                [p[1] - ny * 1.0, p[1] + ny * 1.0],
                color="#888", lw=1.0, ls=(0, (4, 3)))
        ax.text(p[0] + nx * 1.12, p[1] + ny * 1.12, "N", fontsize=8.5,
                color="#888", ha="center", va="center")
    ax.text(p_in[0] - 0.95, p_in[1] + 0.66, "$i_1$", fontsize=11, color=_INK)
    ax.text(p_in[0] + 0.22, p_in[1] - 0.32, "$r_1$", fontsize=11, color=_INK)
    ax.text(p_out[0] - 0.42, p_out[1] - 0.32, "$r_2$", fontsize=11, color=_INK)
    ax.text(p_out[0] + 0.66, p_out[1] - 0.70, "$i_2$", fontsize=11, color=_INK)

    # deviation: dashed continuation of the incident ray, angle marked away from apex
    ax.plot([p_in[0] - 1.9, p_out[0] + 2.0], [p_in[1] + 0.95, p_in[1] - 0.55],
            color="#bbb", lw=0.9, ls=(0, (3, 3)))
    ax.text(p_out[0] + 1.15, p_out[1] + 0.22, r"$\delta$", fontsize=12, color=_INK)
    ax.text(0.5, -0.02,
            r"At minimum deviation $i_1=i_2$, $r_1=r_2$ and the ray passes symmetrically",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=_INK)
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-2.0, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return _png(fig)


# ======================================================================
# MICROSCOPE — compound microscope ray diagram (objective + eyepiece)
# ======================================================================
def _lens_symbol(ax, x, half_h, lw=1.6):
    """A convex-lens symbol: vertical line with outward arrowheads."""
    ax.plot([x, x], [-half_h, half_h], color=_INK, lw=lw)
    dx, dy = 0.055 * half_h, 0.18 * half_h
    ax.plot([x - dx, x, x + dx], [half_h - dy, half_h, half_h - dy], color=_INK, lw=lw)
    ax.plot([x - dx, x, x + dx], [-half_h + dy, -half_h, -half_h + dy], color=_INK, lw=lw)


def microscope(spec: dict) -> bytes | None:
    """Compound microscope. The two-lens construction is COMPUTED from the lens
    formula, so the rays genuinely meet at the intermediate and final image tips:
    the objective forms a real inverted magnified image just inside the eyepiece's
    focus, and the eyepiece then acts as a magnifier forming a far, virtual, much
    larger image. Drawn with an exaggerated vertical scale, as textbooks do."""
    if not _OK or not isinstance(spec, dict):
        return None
    fo, u_o, h = 1.0, 1.5, 0.35          # objective: object just beyond its focus
    fe, u_e = 2.0, 1.6                   # eyepiece: image falls inside its focus
    v_o = 1.0 / (1.0 / fo - 1.0 / u_o)   # = 3.0  (real, inverted)
    h1 = (v_o / -u_o) * h                # = -0.7 (below the axis)
    xe = v_o + u_e                       # eyepiece position
    v_e = 1.0 / (1.0 / fe - 1.0 / u_e)   # = -8.0 (virtual, same side)
    h2 = (v_e / -u_e) * h1               # = -3.5
    xf = xe + v_e                        # final image position (far left)

    fig, ax = plt.subplots(figsize=(7.0, 3.9), dpi=140)
    ax.axhline(0, color=_INK, lw=0.9)
    _lens_symbol(ax, 0.0, 0.60)
    _lens_symbol(ax, xe, 0.78)
    ax.text(0, 0.68, "Objective", ha="center", fontsize=9, fontweight="bold", color=_INK)
    ax.text(xe, 0.86, "Eyepiece", ha="center", fontsize=9, fontweight="bold", color=_INK)

    for x, lbl in ((fo, "$F_o$"), (-fo, "$F_o'$"), (xe + fe, "$F_e$"), (xe - fe, "$F_e'$")):
        ax.plot([x], [0], "o", color=_INK, ms=3)
        ax.text(x, 0.10, lbl, ha="center", fontsize=8.5, color=_INK)

    # object AB (upright), intermediate A'B' (real, inverted), final A''B'' (virtual)
    def arrow(x, y, color, ls="-"):
        ax.annotate("", xy=(x, y), xytext=(x, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, linestyle=ls))

    arrow(-u_o, h, _INK)
    ax.text(-u_o - 0.12, h, "A", fontsize=10, fontweight="bold", ha="right")
    ax.text(-u_o - 0.12, -0.16, "B", fontsize=10, fontweight="bold", ha="right")
    arrow(v_o, h1, "#137a4b")
    ax.text(v_o + 0.10, h1 - 0.08, "A'", fontsize=10, fontweight="bold", color="#137a4b")
    ax.text(v_o + 0.10, 0.10, "B'", fontsize=10, fontweight="bold", color="#137a4b")
    arrow(xf, h2, _VIRT, ls="--")
    ax.text(xf - 0.18, h2, "A''", fontsize=10, fontweight="bold", color=_VIRT, ha="right")
    ax.text(xf - 0.18, -0.16, "B''", fontsize=10, fontweight="bold", color=_VIRT, ha="right")

    # objective rays: parallel-then-through-Fo, and straight through the centre
    ax.plot([-u_o, 0, v_o], [h, h, h1], color=_RAY, lw=1.2)
    ax.plot([-u_o, 0, v_o], [h, 0, h1], color=_RAY, lw=1.2)
    # eyepiece rays: emerge diverging; dashed back-extensions meet at A''
    for y_at_lens in (h1, 0.0):
        # emergent ray leaves the eyepiece heading away from A''
        s = (y_at_lens - h2) / (xe - xf)
        ax.plot([v_o, xe], [h1, y_at_lens], color=_RAY, lw=1.2)     # inside the tube
        ax.plot([xe, xe + 1.9], [y_at_lens, y_at_lens + s * 1.9], color=_RAY, lw=1.2)
        ax.plot([xe, xf], [y_at_lens, h2], color=_VIRT, lw=1.0, ls=(0, (5, 4)))
    ax.text(xe + 2.0, 0.35, "to eye", fontsize=9, color=_RAY)

    ax.text(0.5, -0.02,
            "Objective → real, inverted, magnified A'B'  |  Eyepiece → virtual, "
            f"magnified A''B''   (M = {(v_o / -u_o) * (v_e / -u_e):+.0f}, not to scale)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=_INK)
    ax.set_xlim(xf - 1.2, xe + 2.6)
    ax.set_ylim(h2 - 0.6, 1.25)
    ax.axis("off")
    return _png(fig)


# ======================================================================
# TELESCOPE — reflecting (Cassegrain) telescope ray diagram
# ======================================================================
def telescope(spec: dict) -> bytes | None:
    """Cassegrain reflector: parallel light from a distant object strikes the large
    concave primary, converges back onto a small convex secondary, which sends it
    out through a hole in the primary to the eyepiece."""
    if not _OK or not isinstance(spec, dict):
        return None
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=140)
    x_p, x_s = 3.6, 0.9            # primary and secondary positions
    hole = 0.26
    ax.axhline(0, color="#bbb", lw=0.8, ls=(0, (4, 3)))

    # primary: concave, hollow facing left, with a central hole
    ys = np.linspace(0.32, 2.1, 60)
    for sgn in (1, -1):
        yy = sgn * ys
        xx = x_p + 0.20 * (yy / 2.1) ** 2
        ax.plot(xx, yy, color=_INK, lw=2.0)
    ax.text(x_p + 1.5, 1.95, "Primary mirror\n(concave, holed)", fontsize=8.5,
            color=_INK, ha="center")
    # secondary: small convex, bulging left
    ys2 = np.linspace(-0.5, 0.5, 40)
    ax.plot(x_s - 0.10 * (ys2 / 0.5) ** 2, ys2, color=_INK, lw=2.0)
    ax.text(x_s, -0.62, "Secondary\n(convex)", fontsize=8.5, color=_INK,
            ha="center", va="top")

    # rays: in parallel -> primary -> back to secondary -> through the hole
    for y in (1.05, 1.65, -1.05, -1.65):
        y_s = y * 0.30                      # where it meets the secondary
        ax.annotate("", xy=(x_p, y), xytext=(-2.6, y),
                    arrowprops=dict(arrowstyle="-|>", color=_RAY, lw=1.3))
        ax.plot([x_p, x_s], [y, y_s], color=_RAY, lw=1.3)          # primary -> secondary
        ax.plot([x_s, x_p], [y_s, 0.0], color=_RAY, lw=1.3)        # secondary -> hole
    ax.plot([x_p, 4.9], [0, 0], color=_RAY, lw=1.3)
    ax.text(-2.5, 1.95, "Parallel rays from\na distant object", fontsize=8.5, color=_RAY)
    # hole gap in the primary
    ax.plot([x_p, x_p], [-hole, hole], color="white", lw=3.5, zorder=4)
    _lens_symbol(ax, 5.15, 0.42)
    ax.text(5.15, 0.58, "Eyepiece", fontsize=8.5, ha="center", color=_INK)
    ax.annotate("", xy=(6.1, 0), xytext=(5.15, 0),
                arrowprops=dict(arrowstyle="-|>", color=_RAY, lw=1.3))
    ax.text(6.15, 0.06, "to eye", fontsize=9, color=_RAY, va="bottom")

    ax.text(0.5, -0.02,
            "No chromatic aberration (reflection, not refraction); a large mirror is "
            "easier to make and support than a large lens",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=_INK)
    ax.set_xlim(-3.0, 7.0)
    ax.set_ylim(-2.6, 2.6)
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
