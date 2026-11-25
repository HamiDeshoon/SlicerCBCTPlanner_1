"""Tangential (ridge-parallel) sections through the dental arch.

Every other section in this module tree answers "how wide and how deep is *this* site".
A tangential section is the plane spanned by the arch **tangent** and the occlusal-superior
axis, i.e. it looks *along* the ridge and shows how the alveolar crest runs between adjacent
teeth. That is the view a review agent needs to judge continuity (crest drop after an
extraction, a fenestration that spans two sites, a knife-edge ridge seen end-on).

Geometry
--------
At arc position ``s`` the arch frame gives ``P`` (curve point), ``T`` (tangent),
``N`` (buccal normal) and ``B`` (superior). The section samples

    Q(u, v, t) = P + u*T + v*B + t*N        u in [-W/2, +W/2], v in [-a*H, (1-a)*H]

and reduces the thin slab along ``N`` (mean / max / min). Row 0 of the returned array is the
most superior row, so it renders upright without an extra flip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .labels import LOWER_JAW_ID, UPPER_JAW_ID
from .volume_loader import CBCTVolume

UPPER_JAW_IDS = (int(UPPER_JAW_ID),)
LOWER_JAW_IDS = (int(LOWER_JAW_ID),)

SLAB_MODES: Tuple[str, ...] = ("mean", "max", "min")

DEFAULT_WIDTH_MM = 42.0
DEFAULT_HEIGHT_MM = 30.0
DEFAULT_PIXEL_MM = 0.3
DEFAULT_THICKNESS_MM = 2.0


@dataclass
class TangentialSection:
    """One sampled ridge-parallel section plus the geometry needed to render it."""

    array: Optional[np.ndarray] = None
    mm_per_px: float = DEFAULT_PIXEL_MM
    arc_mm: float = 0.0
    width_mm: float = DEFAULT_WIDTH_MM
    height_mm: float = DEFAULT_HEIGHT_MM
    thickness_mm: float = DEFAULT_THICKNESS_MM
    thickness_mode: str = "mean"
    u_mm: Tuple[float, float] = (-DEFAULT_WIDTH_MM / 2.0, DEFAULT_WIDTH_MM / 2.0)
    v_mm: Tuple[float, float] = (-DEFAULT_HEIGHT_MM * 0.6, DEFAULT_HEIGHT_MM * 0.4)
    point_ras: Optional[Tuple[float, float, float]] = None
    tangent: Optional[Tuple[float, float, float]] = None
    normal: Optional[Tuple[float, float, float]] = None
    binormal: Optional[Tuple[float, float, float]] = None
    crest_line_mm: List[Tuple[float, float]] = field(default_factory=list)
    available: bool = True
    reason: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, int]:
        if self.array is None:
            return (0, 0)
        return (int(self.array.shape[0]), int(self.array.shape[1]))

    def to_dict(self) -> Dict[str, Any]:
        """Metadata only (no pixels) - used in agent contexts and JSON exports."""
        return {
            "arc_mm": round(float(self.arc_mm), 2),
            "width_mm": round(float(self.width_mm), 2),
            "height_mm": round(float(self.height_mm), 2),
            "thickness_mm": round(float(self.thickness_mm), 2),
            "thickness_mode": self.thickness_mode,
            "mm_per_px": round(float(self.mm_per_px), 4),
            "point_ras": None if self.point_ras is None else [round(float(v), 2) for v in self.point_ras],
            "crest_samples": len(self.crest_line_mm),
            "available": self.available,
            "reason": self.reason,
        }


def _frame_vectors(frame: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read (P, T, N, B) from an ArchFrame-like object, tolerating attribute aliases."""
    point = np.asarray(getattr(frame, "point", getattr(frame, "origin", (0.0, 0.0, 0.0))), dtype=np.float64)
    tangent = np.asarray(getattr(frame, "tangent", (1.0, 0.0, 0.0)), dtype=np.float64)
    normal = np.asarray(getattr(frame, "normal", (0.0, 1.0, 0.0)), dtype=np.float64)
    binormal = np.asarray(getattr(frame, "binormal", (0.0, 0.0, 1.0)), dtype=np.float64)
    for vec in (tangent, normal, binormal):
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec /= norm
    return point, tangent, normal, binormal


def sample_tangential_section(
    volume: CBCTVolume,
    arch: Any,
    arc_mm: float,
    *,
    width_mm: float = DEFAULT_WIDTH_MM,
    height_mm: float = DEFAULT_HEIGHT_MM,
    pixel_mm: float = DEFAULT_PIXEL_MM,
    thickness_mm: float = DEFAULT_THICKNESS_MM,
    mode: str = "mean",
    apical_bias: float = 0.6,
    mask: Optional[np.ndarray] = None,
    is_mandible: Optional[bool] = None,
) -> TangentialSection:
    """Sample one ridge-parallel section at `arc_mm`.

    `apical_bias` is the fraction of the requested height placed below the arch point, so the
    default 0.6 favours bone under the crest. `mask` (optional) is the (Z, Y, X) label map used
    to draw the cortical crest line.
    """
    mode_key = str(mode or "mean").lower()
    if mode_key not in SLAB_MODES:
        mode_key = "mean"
    try:
        width = max(4.0, float(width_mm))
        height = max(4.0, float(height_mm))
        pixel = min(2.0, max(0.1, float(pixel_mm)))
    except (TypeError, ValueError):
        width, height, pixel = DEFAULT_WIDTH_MM, DEFAULT_HEIGHT_MM, DEFAULT_PIXEL_MM
    try:
        thickness = max(pixel, float(thickness_mm))
    except (TypeError, ValueError):
        thickness = DEFAULT_THICKNESS_MM
    try:
        bias = min(0.95, max(0.05, float(apical_bias)))
    except (TypeError, ValueError):
        bias = 0.6

    if volume is None or arch is None:
        return TangentialSection(
            available=False, reason="a CBCT volume and an arch curve are required", arc_mm=float(arc_mm or 0.0)
        )

    try:
        frame = arch.get_frame_at_mm(float(arc_mm))
    except Exception as exc:  # justification: a bad curve must not crash the viewer
        return TangentialSection(
            available=False,
            reason=f"arch frame unavailable at {float(arc_mm or 0.0):.1f} mm: {exc}",
            arc_mm=float(arc_mm or 0.0),
        )

    point, tangent, normal, binormal = _frame_vectors(frame)

    n_u = max(8, int(round(width / pixel)) + 1)
    n_v = max(8, int(round(height / pixel)) + 1)
    u_vals = np.linspace(-width / 2.0, width / 2.0, n_u)
    v_vals = np.linspace((1.0 - bias) * height, -bias * height, n_v)  # row 0 = superior
    n_slab = max(1, int(round(thickness / pixel)))
    offsets = np.linspace(-thickness / 2.0, thickness / 2.0, n_slab) if n_slab > 1 else np.zeros(1)

    grid_u, grid_v, grid_t = np.meshgrid(u_vals, v_vals, offsets, indexing="ij")
    points = (
        point[None, None, None, :]
        + grid_u[..., None] * tangent[None, None, None, :]
        + grid_v[..., None] * binormal[None, None, None, :]
        + grid_t[..., None] * normal[None, None, None, :]
    )
    sampled = volume.sample_world_points(points, order=1, cval=-1000.0)  # (n_u, n_v, n_slab)
    if mode_key == "max":
        slab = np.max(sampled, axis=2)
    elif mode_key == "min":
        slab = np.min(sampled, axis=2)
    else:
        slab = np.mean(sampled, axis=2)
    image = np.ascontiguousarray(np.transpose(slab, (1, 0)))  # (n_v, n_u)
    if not np.isfinite(image).any():
        return TangentialSection(
            available=False, reason="no samples fell inside the volume", arc_mm=float(arc_mm or 0.0)
        )

    crest: List[Tuple[float, float]] = []
    if mask is not None and np.asarray(mask).size:
        jaw_ids = LOWER_JAW_IDS if (is_mandible is not False) else UPPER_JAW_IDS
        if is_mandible is None:
            jaw_ids = tuple(int(v) for v in (LOWER_JAW_ID, UPPER_JAW_ID))
        grid_mask = (
            point[None, None, :]
            + u_vals[:, None, None] * tangent[None, None, :]
            + v_vals[None, :, None] * binormal[None, None, :]
        )
        labels = volume.sample_mask_world_points(mask, grid_mask)  # (n_u, n_v)
        bone = np.isin(labels, jaw_ids)
        for col in range(n_u):
            rows = np.flatnonzero(bone[col, :])
            if rows.size:
                crest.append((float(u_vals[col]), float(v_vals[int(rows[0])])))

    return TangentialSection(
        array=image.astype(np.float32),
        mm_per_px=float(pixel),
        arc_mm=float(arc_mm or 0.0),
        width_mm=width,
        height_mm=height,
        thickness_mm=thickness,
        thickness_mode=mode_key,
        u_mm=(-width / 2.0, width / 2.0),
        v_mm=(float(v_vals[0]), float(v_vals[-1])),
        point_ras=tuple(float(v) for v in point),
        tangent=tuple(float(v) for v in tangent),
        normal=tuple(float(v) for v in normal),
        binormal=tuple(float(v) for v in binormal),
        crest_line_mm=crest,
        available=True,
    )


def crest_drop_mm(section: TangentialSection) -> Optional[float]:
    """Largest superior-to-apical crest excursion inside the section (mm)."""
    if len(section.crest_line_mm) < 4:
        return None
    values = np.array([v for _u, v in section.crest_line_mm], dtype=np.float64)
    return float(np.max(values) - np.min(values))


def render_tangential_section(
    section: TangentialSection,
    *,
    scale: int = 2,
    window: Optional[float] = None,
    level: Optional[float] = None,
    transparent_bg: bool = True,
    monochrome: bool = False,
    footer_text: str = "",
    show_crest: bool = True,
) -> Any:
    """Render a sampled tangential section to a PIL image (RGBA when `transparent_bg`)."""
    from ..core_print_render import (
        CrossSectionRender,
        _compute_foreground_mask,
        _enhance_bone_clarity,
        _foreground_alpha_u8,
        _load_font,
        _robust_window_level,
        _to_monochrome,
        hu_window_to_rgb,
    )

    if section is None or not section.available or section.array is None or section.array.size == 0:
        reason = "no tangential section" if section is None else (section.reason or "unavailable")
        return CrossSectionRender(None, available=False, reason=reason)

    from PIL import Image, ImageDraw

    arr = np.asarray(section.array, dtype=np.float32)
    fg = _compute_foreground_mask(arr) if transparent_bg else None
    if window is None or level is None:
        try:
            win, lev = _robust_window_level(arr, fg)
        except Exception:  # justification: windowing must never block rendering
            win, lev = 4000.0, 1000.0
        window = win if window is None else window
        level = lev if level is None else level
    gray = _enhance_bone_clarity(hu_window_to_rgb(arr, float(window), float(level)))
    rgb = np.dstack([gray, gray, gray])
    alpha = _foreground_alpha_u8(gray, fg) if fg is not None else np.full(gray.shape, 255, np.uint8)
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")

    scale = max(1, int(scale))
    width_px = max(1, int(gray.shape[1]) * scale)
    height_px = max(1, int(gray.shape[0]) * scale)
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((width_px, height_px), getattr(resampling, "BICUBIC", Image.BILINEAR))

    font_sm = _load_font(max(11, int(round(3.4 * scale))), bold=True)
    draw = ImageDraw.Draw(image)

    # Crest line: the topmost cortical bone per column, drawn dashed amber.
    if show_crest and section.crest_line_mm:
        u_lo, u_hi = section.u_mm
        v_hi, v_lo = section.v_mm[0], section.v_mm[1]
        pts = []
        for u_mm, v_mm in section.crest_line_mm:
            x = (u_mm - u_lo) / max(1e-6, u_hi - u_lo) * (width_px - 1)
            y = (v_hi - v_mm) / max(1e-6, v_hi - v_lo) * (height_px - 1)
            pts.append((x, y))
        for index in range(0, len(pts) - 1, 2):
            draw.line([pts[index], pts[index + 1]], fill=(250, 204, 21, 235), width=max(1, scale))

    # Arc ruler + scale bar (10 mm).
    pad = max(4, 2 * scale)
    bar_len = 10.0 / max(1e-6, section.mm_per_px) * scale
    bar_y = height_px - pad - 2
    bar_x0 = pad + 2
    if bar_x0 + bar_len < width_px - pad:
        draw.line([(bar_x0, bar_y), (bar_x0 + bar_len, bar_y)], fill=(226, 232, 240, 235), width=max(1, scale // 2 + 1))
        draw.text((bar_x0, bar_y - max(12, int(3.6 * scale))), "10 mm", fill=(226, 232, 240, 235), font=font_sm)

    hud = f"Tangential | arc {section.arc_mm:.1f} mm | {section.thickness_mm:.1f} mm slab ({section.thickness_mode})"
    draw.text((pad, pad), hud, fill=(148, 163, 184, 235), font=font_sm)
    drop = crest_drop_mm(section)
    if drop is not None:
        draw.text(
            (pad, pad + max(13, int(4.0 * scale))),
            f"crest drop {drop:.1f} mm",
            fill=(250, 204, 21, 235),
            font=font_sm,
        )

    if not transparent_bg:
        # HUD strokes are drawn with partial alpha; an opaque request must stay fully opaque.
        image.putalpha(255)
    if monochrome:
        image = _to_monochrome(image)
    if footer_text:
        bar = max(14, int(4.6 * scale))
        canvas = Image.new("RGBA", (width_px, height_px + bar), (2, 4, 9, 255))
        canvas.paste(image, (0, 0))
        ImageDraw.Draw(canvas).text(
            (4, height_px + 2), footer_text, fill=(226, 232, 240, 255), font=font_sm
        )
        image = canvas
    return CrossSectionRender(
        image=image,
        mm_per_px=section.mm_per_px / float(scale),
        available=True,
    )
