"""Deterministic optics ray-diagram renderer for lenses and mirrors.

The answer model supplies PHYSICS parameters — element type, focal length,
object distance and object height — NEVER pixel coordinates. This module solves
the thin-lens / mirror equation for the image, then draws a correct, fully
labelled principal-ray diagram. Because the geometry is *computed*, the rays
actually obey the physics and every label lands in the right place, unlike a
model-drawn sketch that guesses coordinates.

Entered distances are POSITIVE magnitudes (in cm); the object always sits on the
left. The element TYPE fixes the sign convention internally, so the model cannot
get the convention wrong. Returns PNG bytes, or None on bad input / no image
(e.g. object at the focus) so the caller can fall back to caption text.
"""
from __future__ import annotations

from io import BytesIO

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL = True
except Exception:  # noqa: BLE001 - matplotlib missing
    _MPL = False

ELEMENTS = {"convex_lens", "concave_lens", "concave_mirror", "convex_mirror"}
_ALIASES = {
    "converging_lens": "convex_lens", "diverging_lens": "concave_lens",
    "convex": "convex_lens", "concave": "concave_lens",
    "concave_mirror": "concave_mirror", "convex_mirror": "convex_mirror",
    "converging_mirror": "concave_mirror", "diverging_mirror": "convex_mirror",
}

_INK = "#1a1a1a"
_RAY = "#0b62c4"       # incident/emergent rays
_VIRT = "#c0392b"      # virtual (dashed) construction lines


def available() -> bool:
    return _MPL


def _normalise(element: str) -> str | None:
    e = (element or "").strip().lower().replace(" ", "_").replace("-", "_")
    e = _ALIASES.get(e, e)
    return e if e in ELEMENTS else None


def _solve(element: str, f_mag: float, u_mag: float):
    """Solve for the image in Cartesian convention (light travels +x, element at
    the origin, object on the left so u < 0). Returns (v, m) where v is the signed
    image position (drawing x-coordinate) and m the linear magnification, or
    (None, None) when the image is at infinity."""
    u = -abs(u_mag)
    f = abs(f_mag)
    if element == "convex_lens":
        f = +f
    elif element == "concave_lens":
        f = -f
    elif element == "concave_mirror":
        f = -f  # Cartesian: concave-mirror focus is in front (negative x)
    elif element == "convex_mirror":
        f = +f
    is_mirror = element.endswith("mirror")
    if is_mirror:
        inv_v = 1.0 / f - 1.0 / u          # 1/v + 1/u = 1/f
    else:
        inv_v = 1.0 / f + 1.0 / u          # 1/v - 1/u = 1/f
    if abs(inv_v) < 1e-9:
        return None, None                   # object at focus -> image at infinity
    v = 1.0 / inv_v
    m = (-v / u) if is_mirror else (v / u)
    return v, m


def _draw_lens(ax, H, convex: bool):
    ax.plot([0, 0], [-H, H], color=_INK, lw=1.6)
    dx, dy = 0.045 * H, 0.16 * H
    if convex:  # arrowheads pointing OUT
        ax.plot([-dx, 0, dx], [H - dy, H, H - dy], color=_INK, lw=1.6)
        ax.plot([-dx, 0, dx], [-H + dy, -H, -H + dy], color=_INK, lw=1.6)
    else:       # concave: arrowheads pointing IN
        ax.plot([-dx, 0, dx], [H, H - dy, H], color=_INK, lw=1.6)
        ax.plot([-dx, 0, dx], [-H, -H + dy, -H], color=_INK, lw=1.6)


def _draw_mirror(ax, H, concave: bool):
    ys = [H * (i / 40.0 - 0.5) * 2 for i in range(41)]
    bulge = 0.16 * H
    # Concave mirror: reflecting (hollow) face toward the object on the left, so
    # the surface bulges to the RIGHT; convex mirror bulges left.
    sign = 1.0 if concave else -1.0
    pts = [(sign * bulge * (y / H) ** 2, y) for y in ys]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=_INK, lw=1.8)
    # Short hatch strokes on the back (non-reflecting) side.
    for x, y in pts[2:-2:4]:
        ax.plot([x, x + sign * 0.06 * H], [y, y + 0.05 * H], color=_INK, lw=0.8)


def _ray(ax, p, image_tip, forward_sign, real, span):
    """Draw one principal ray: the emergent segment from the element point p to
    the image tip (solid if real), or the diverging emergent ray plus a dashed
    backward construction line to a virtual image tip."""
    px, py = p
    tx, ty = image_tip
    if real:
        ax.plot([px, tx], [py, ty], color=_RAY, lw=1.3)
        # a little past the image so converging rays visibly cross
        ex = tx + (tx - px) * 0.25
        ey = ty + (ty - py) * 0.25
        ax.plot([tx, ex], [ty, ey], color=_RAY, lw=1.3)
    else:
        # Emergent ray travels forward (away from the virtual tip); its backward
        # extension is what meets at the tip.
        dx, dy = px - tx, py - ty
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        if (dx > 0) != (forward_sign > 0):  # ensure it points the forward way
            dx, dy = -dx, -dy
        ex, ey = px + dx / n * span, py + dy / n * span
        ax.plot([px, ex], [py, ey], color=_RAY, lw=1.3)
        ax.plot([px, tx], [py, ty], color=_VIRT, lw=1.1, ls=(0, (5, 4)))


def _arrow(ax, x, ytip, color, label, above):
    ax.annotate("", xy=(x, ytip), xytext=(x, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))
    va = "bottom" if ytip >= 0 else "top"
    off = 0.06 if ytip >= 0 else -0.06
    ax.text(x, ytip + off * abs(ytip or 1), label, color=color,
            ha="center", va=va, fontsize=10, fontweight="bold")


def _mark(ax, x, text, H):
    ax.plot([x], [0], "o", color=_INK, ms=3.5)
    ax.text(x, -0.11 * H, text, ha="center", va="top", fontsize=9, color=_INK)


def render(spec: dict) -> bytes | None:
    if not _MPL or not isinstance(spec, dict):
        return None
    element = _normalise(str(spec.get("element", "")))
    if element is None:
        return None
    try:
        f = abs(float(spec.get("focal_length")))
        u = abs(float(spec.get("object_distance")))
        h0 = abs(float(spec.get("object_height", 0) or f * 0.5)) or f * 0.5
    except (TypeError, ValueError):
        return None
    if f <= 0 or u <= 0:
        return None

    v, m = _solve(element, f, u)
    if v is None:
        return None
    is_mirror = element.endswith("mirror")
    convex = element.startswith("convex")
    forward_sign = -1.0 if is_mirror else 1.0     # emergent light direction
    real = (v < 0) if is_mirror else (v > 0)

    xo, ho = -u, h0
    xi, hi = v, m * h0

    H = max(h0, abs(hi), f * 0.5) * 1.25          # element half-height
    fig, ax = plt.subplots(figsize=(6.2, 3.9), dpi=140)

    xs = [xo, xi, -2 * f, 2 * f, 0]
    span = (max(xs) - min(xs)) + 4 * f

    # principal axis
    ax.axhline(0, color=_INK, lw=1.0)
    ax.text(min(xs) - 0.06 * span, 0.04 * H, "Principal axis",
            fontsize=8, color=_INK, va="bottom")

    if is_mirror:
        _draw_mirror(ax, H, concave=not convex)
        # F and C: in front (left) for concave, virtual behind (right) for convex.
        s = -1.0 if not convex else 1.0
        _mark(ax, s * f, "F", H)
        _mark(ax, s * 2 * f, "C", H)
        _mark(ax, 0, "P", H)
    else:
        _draw_lens(ax, H, convex=convex)
        _mark(ax, f, "F", H)
        _mark(ax, -f, "F'", H)
        _mark(ax, 2 * f, "2F", H)
        _mark(ax, -2 * f, "2F'", H)
        _mark(ax, 0, "O", H)  # optical centre

    # object
    _arrow(ax, xo, ho, _INK, "Object", above=True)

    # two principal rays, both passing through the image tip
    image_tip = (xi, hi)
    _ray(ax, (0, ho), image_tip, forward_sign, real, span)   # parallel ray
    _ray(ax, (0, 0), image_tip, forward_sign, real, span)    # ray to pole/centre
    # incident segments (object -> element)
    ax.plot([xo, 0], [ho, ho], color=_RAY, lw=1.3)           # parallel incident
    ax.plot([xo, 0], [ho, 0], color=_RAY, lw=1.3)            # central/pole incident

    # image
    img_color = _INK if real else _VIRT
    _arrow(ax, xi, hi, img_color, "Image", above=(hi >= 0))

    nature = ("Real, inverted" if real and hi < 0 else
              "Real, erect" if real else
              "Virtual, erect" if hi >= 0 else "Virtual, inverted")
    size = ("magnified" if abs(m) > 1.001 else
            "diminished" if abs(m) < 0.999 else "same size")
    ax.text(0.5, -0.02,
            f"v = {xi:+.1f} cm   |m| = {abs(m):.2f}   {nature}, {size}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color=_INK)

    pad = 0.12 * span
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(-H * 1.5, H * 1.5)
    ax.set_aspect("auto")
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()
