"""Clinical renderers for cross-sections and the panoramic strip.

Pure numpy geometry with Pillow (and automatic Qt fallback in 3D Slicer). Every overlay
is derived from measured millimetre coordinates:
  - Green vertical Bone Height line (`H`) from alveolar crest (`crest_v`) to basal/canal limit (`basal_v`)
  - Yellow Crestal reference & `W0` Crestal Buccolingual Width line
  - Cyan `W3` (3 mm depth) and Orange/Gold `W5` (5 mm depth) Buccolingual Width lines
  - Red/Cyan Inferior Alveolar Nerve Canal circle + clearance indicator
  - Subtle anatomical contours for segmented teeth/roots and jawbone cortical plates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Pillow is optional in Slicer; callers render an "unavailable" tile when missing.
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without Pillow
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

MM_PER_ROW = 0.25
MM_PER_COL = 0.25


@dataclass
class CrossSectionRender:
    image: Any
    overlay_px: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    labels: List[Tuple[float, float, str]] = field(default_factory=list)
    mm_per_px: float = MM_PER_COL
    available: bool = True
    reason: Optional[str] = None


def hu_window_to_rgb(img_cs: np.ndarray, window: float = 4000.0, level: float = 1000.0) -> np.ndarray:
    """Map HU values to 8-bit grayscale using a window/level (with adaptive fallback for non-HU arrays)."""
    arr = np.asarray(img_cs, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    low = float(level) - float(window) / 2.0
    high = float(level) + float(window) / 2.0
    if high <= low:
        high = low + 1.0
    scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
    # If standard HU window leaves image flat (e.g. non-HU gray values), adapt to [p2, p99]
    if float(np.std(scaled)) < 0.01 and float(np.max(arr) - np.min(arr)) > 1.0:
        p2, p99 = np.percentile(arr, [2.0, 99.0])
        denom = max(1.0, float(p99 - p2))
        scaled = np.clip((arr - float(p2)) / denom, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _px(u: float, v: float, u_coords: np.ndarray, v_coords: np.ndarray, scale: int) -> Tuple[float, float]:
    col = (float(u) - float(u_coords[0])) / (float(u_coords[-1]) - float(u_coords[0]))
    row = (float(v_coords[-1]) - float(v)) / (float(v_coords[-1]) - float(v_coords[0]))
    width = (len(u_coords) - 1) * scale
    height = (len(v_coords) - 1) * scale
    return col * width, row * height


def _apply_anatomical_contour_tint(rgb: np.ndarray, mask_cs: Optional[np.ndarray]) -> np.ndarray:
    """Blend subtle vivid SlicerDentalSegmentator boundary contours for teeth/roots, bone, and canal."""
    if mask_cs is None or mask_cs.shape != rgb.shape[:2]:
        return rgb
    out = rgb.astype(np.float32)
    m = np.asarray(mask_cs)
    # Edge detection via 4-neighbor difference
    edge = np.zeros(m.shape, dtype=bool)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]

    # Tooth & root contour: vivid golden-ivory (labels 3, 4, or 11..48)
    tooth_edge = edge & ((m == 3) | (m == 4) | ((m >= 11) & (m <= 48)))
    if np.any(tooth_edge):
        out[tooth_edge] = 0.35 * out[tooth_edge] + 0.65 * np.array([255.0, 215.0, 90.0], dtype=np.float32)

    # Jawbone cortical contour: subtle teal-emerald (labels 1, 2)
    jaw_edge = edge & ((m == 1) | (m == 2)) & (~tooth_edge)
    if np.any(jaw_edge):
        out[jaw_edge] = 0.55 * out[jaw_edge] + 0.45 * np.array([80.0, 220.0, 180.0], dtype=np.float32)

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def measurement_footer_text(meas: Optional[Any]) -> str:
    """Return the tile footer `H … / W0 …` caption, printing `--` for a withheld metric.

    A withheld crest width must never render as a fabricated `W0 0.0 mm`.
    """
    if meas is None:
        return "H --  W0 --"
    height_mm = getattr(meas, "bone_height_mm", None)
    crest_w = getattr(meas, "crest_width_mm", None)
    h_txt = f"{float(height_mm):.1f} mm" if height_mm is not None else "--"
    w_txt = f"{float(crest_w):.1f} mm" if crest_w is not None else "--"
    return f"H {h_txt}  W0 {w_txt}"


def render_cross_section(
    cs: Any,
    meas: Optional[Any],
    *,
    scale: int = 2,
    window: float = 4000.0,
    level: float = 1000.0,
    dark_film: bool = False,
) -> CrossSectionRender:
    """Render one arch cross-section with real millimetre overlays.

    `cs` is a `CrossSectionData` (duck-typed to keep this module Slicer-free).
    When Pillow is unavailable the result is explicitly unavailable.
    """
    if not _PIL_AVAILABLE:
        return CrossSectionRender(None, available=False, reason="Pillow is not installed")
    if cs is None:
        return CrossSectionRender(None, available=False, reason="no cross-section data")

    u_coords = np.asarray(cs.u_coords, dtype=np.float64)
    v_coords = np.asarray(cs.v_coords, dtype=np.float64)
    gray = hu_window_to_rgb(cs.img_cs, window=window, level=level)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if dark_film:
        rgb = 255 - rgb
    rgb = _apply_anatomical_contour_tint(rgb, getattr(cs, "mask_cs", None))

    image = Image.fromarray(rgb, mode="RGB")
    width = (len(u_coords) - 1) * scale
    height = (len(v_coords) - 1) * scale
    image = image.resize((max(1, width), max(1, height)), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    overlay: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    labels: List[Tuple[float, float, str]] = []

    # Buccal (+u, along N) always on the right; lingual on the left.
    draw.text((2, 2), "L", fill=(0, 255, 0), font=font)
    draw.text((max(2, width - 12), 2), "B", fill=(0, 255, 0), font=font)
    labels.extend([(2.0, 2.0, "L"), (float(max(2, width - 12)), 2.0, "B")])

    # Millimetre ruler (left edge) and 10 mm scale bar, drawn on every tile so the image
    # carries a measurement reference even when no metric could be computed.
    ruler_colour = (255, 255, 255) if dark_film else (205, 205, 205)
    for tick_mm in np.arange(np.ceil(v_coords.min() / 5.0) * 5.0, v_coords.max(), 5.0):
        _, tick_y = _px(u_coords[0], float(tick_mm), u_coords, v_coords, scale)
        major = abs(float(tick_mm) % 10.0) < 1e-6
        draw.line([(0, tick_y), (11 if major else 6, tick_y)], fill=ruler_colour, width=1)
        if major:
            draw.text((13, tick_y - 4), f"{float(tick_mm):+.0f}", fill=ruler_colour, font=font)
    bar_px = 10.0 / max(1e-6, abs(u_coords[-1] - u_coords[0])) * width
    draw.line([(8, height - 34), (8 + bar_px, height - 34)], fill=ruler_colour, width=2)
    draw.line([(8, height - 38), (8, height - 30)], fill=ruler_colour, width=1)
    draw.line([(8 + bar_px, height - 38), (8 + bar_px, height - 30)], fill=ruler_colour, width=1)
    draw.text((12 + bar_px, height - 39), "10 mm", fill=ruler_colour, font=font)

    crest_v = getattr(cs, "crest_v", None)
    if crest_v is not None:
        x_left, y_crest = _px(u_coords[0], crest_v, u_coords, v_coords, scale)
        x_right, _ = _px(u_coords[-1], crest_v, u_coords, v_coords, scale)
        draw.line([(x_left, y_crest), (x_right, y_crest)], fill=(255, 220, 0), width=1)
        overlay.append(((x_left, y_crest), (x_right, y_crest)))

    # Draw vertical Green Bone Height (H) line from crest_v to basal_v (or crest_v -/+ bone_height_mm)
    height_mm_val = getattr(meas, "bone_height_mm", None) if meas is not None else None
    if crest_v is not None and height_mm_val is not None and float(height_mm_val) > 0.0:
        is_mand = bool(getattr(cs, "is_mandible", True))
        basal_v = getattr(cs, "basal_v", None)
        if basal_v is None:
            basal_v = float(crest_v) - float(height_mm_val) if is_mand else float(crest_v) + float(height_mm_val)
        h_u = float(getattr(cs, "height_u", 0.0))
        hx0, hy0 = _px(h_u, crest_v, u_coords, v_coords, scale)
        hx1, hy1 = _px(h_u, basal_v, u_coords, v_coords, scale)
        draw.line([(hx0, hy0), (hx1, hy1)], fill=(34, 197, 94), width=2)
        # End-cap ticks on Bone Height line
        draw.line([(hx0 - 3, hy0), (hx0 + 3, hy0)], fill=(34, 197, 94), width=1)
        draw.line([(hx1 - 3, hy1), (hx1 + 3, hy1)], fill=(34, 197, 94), width=1)
        overlay.append(((hx0, hy0), (hx1, hy1)))
        mid_hy = 0.5 * (hy0 + hy1)
        draw.text((min(width - 36, hx0 + 4), mid_hy - 5), f"H:{float(height_mm_val):.1f}", fill=(34, 255, 110), font=font)

    segments: Dict[float, Tuple[float, float]] = getattr(cs, "width_segments", {}) or {}
    is_mand = bool(getattr(cs, "is_mandible", True))
    for depth, (u_left, u_right) in segments.items():
        if crest_v is None:
            break
        v_level = crest_v - float(depth) if is_mand else crest_v + float(depth)
        start = _px(u_left, v_level, u_coords, v_coords, scale)
        end = _px(u_right, v_level, u_coords, v_coords, scale)
        colour = (0, 220, 255) if depth == 0.0 else ((255, 180, 0) if depth <= 3.5 else (255, 120, 40))
        draw.line([start, end], fill=colour, width=2 if depth in (0.0, 3.0, 5.0) else 1)
        overlay.append((start, end))
        mid_x = (start[0] + end[0]) / 2.0
        w_val = abs(float(u_right) - float(u_left))
        lbl_str = f"W{depth:g}:{w_val:.1f}" if width >= 140 else f"W{depth:g}"
        draw.text((max(2, mid_x - 14), max(0, start[1] - 10)), lbl_str, fill=colour, font=font)

    canal = getattr(cs, "canal_centroid_uv", None)
    if canal is not None:
        cu, cv = canal
        cx, cy = _px(cu, cv, u_coords, v_coords, scale)
        radius_mm = 1.5
        radius_px = radius_mm / max(1e-6, (u_coords[-1] - u_coords[0]) / max(1, len(u_coords) - 1)) * scale
        draw.ellipse(
            [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px],
            outline=(255, 60, 60),
            width=2,
        )
        draw.text((min(width - 28, cx + radius_px + 2), cy - 4), "IAN", fill=(255, 80, 80), font=font)

    if meas is None:
        draw.text((6, height - 16), "not measurable", fill=(255, 60, 60), font=font)
        labels.append((6.0, float(height - 16), "not measurable"))
    else:
        text = measurement_footer_text(meas)
        draw.rectangle([(0, max(0, height - 16)), (width, height)], fill=(10, 15, 28))
        draw.text((4, max(0, height - 14)), text, fill=(255, 255, 255), font=font)
        labels.append((4.0, float(max(0, height - 14)), text))

    return CrossSectionRender(
        image=image,
        overlay_px=overlay,
        labels=labels,
        mm_per_px=MM_PER_COL / float(scale),
        available=True,
    )


def render_panoramic_strip(
    pano_array: np.ndarray,
    *,
    column_spacing_mm: float,
    tick_arcs_mm: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    window: Optional[float] = None,
    level: Optional[float] = None,
    strip_height: int = 160,
    scale: int = 1,
) -> CrossSectionRender:
    """Render a panoramic strip from the actual panoramic volume array.

    Column `k` maps to arc length `k * column_spacing_mm`. No screenshot or
    placeholder is used. Returns an explicitly unavailable render on bad input.
    """
    if not _PIL_AVAILABLE:
        return CrossSectionRender(None, available=False, reason="Pillow is not installed")
    arr = np.asarray(pano_array)
    if arr.ndim == 3:
        if arr.shape[0] < 1:
            return CrossSectionRender(None, available=False, reason="empty panoramic volume")
        arr = arr[0]
    if arr.ndim != 2 or arr.size == 0:
        return CrossSectionRender(None, available=False, reason="invalid panoramic array")

    if window is None or level is None:
        lo, hi = np.percentile(arr.astype(np.float32), [1.0, 99.0])
        level = float((lo + hi) / 2.0)
        window = float(max(1.0, hi - lo))
    gray = hu_window_to_rgb(arr, window=window, level=level)
    image = Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB")
    image = image.resize((max(1, arr.shape[1] * scale), max(1, strip_height)), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    ticks: List[Tuple[float, float, str]] = []

    for index, arc_mm in enumerate(tick_arcs_mm):
        col = float(arc_mm) / float(column_spacing_mm)
        if col < 0 or col > arr.shape[1] - 1:
            continue
        x = col * scale
        draw.line([(x, strip_height - 16), (x, strip_height)], fill=(0, 255, 100), width=1)
        draw.line([(x, 0), (x, 10)], fill=(0, 220, 255), width=1)
        label = labels[index] if labels is not None and index < len(labels) else ""
        draw.text((x + 1, strip_height - 14), str(label), fill=(0, 255, 100), font=font)
        ticks.append((x, float(strip_height - 14), str(label)))

    return CrossSectionRender(image=image, labels=ticks, available=True)


def unavailable_tile(text: str, size: Tuple[int, int] = (640, 400), dark_film: bool = False) -> Any:
    """Render an explicit 'unavailable' tile; never fake clinical content."""
    if not _PIL_AVAILABLE:
        return None
    background = (20, 20, 20) if not dark_film else (245, 245, 245)
    foreground = (255, 255, 255) if not dark_film else (10, 10, 10)
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    draw.text((20, size[1] // 2), f"{text} unavailable", fill=foreground, font=ImageFont.load_default())
    return image


__all__ = [
    "CrossSectionRender",
    "hu_window_to_rgb",
    "measurement_footer_text",
    "render_cross_section",
    "render_panoramic_strip",
    "unavailable_tile",
]
