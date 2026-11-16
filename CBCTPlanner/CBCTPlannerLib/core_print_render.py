"""Clinical renderers for cross-sections and the panoramic strip.

Pure numpy geometry with Pillow for rasterisation. Every overlay is derived from
measured millimetre coordinates; nothing is drawn at fixed pixel offsets.
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

    def png_bytes(self) -> bytes:
        if not self.available or self.image is None:
            raise RuntimeError(self.reason or "cross-section renderer unavailable")
        import io

        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        return buffer.getvalue()


def hu_window_to_rgb(img_cs: np.ndarray, window: float = 4000.0, level: float = 1000.0) -> np.ndarray:
    """Map HU values to 8-bit grayscale using a window/level."""
    arr = np.asarray(img_cs, dtype=np.float32)
    low = float(level) - float(window) / 2.0
    high = float(level) + float(window) / 2.0
    if high <= low:
        high = low + 1.0
    scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _px(u: float, v: float, u_coords: np.ndarray, v_coords: np.ndarray, scale: int) -> Tuple[float, float]:
    col = (float(u) - float(u_coords[0])) / (float(u_coords[-1]) - float(u_coords[0]))
    row = (float(v_coords[-1]) - float(v)) / (float(v_coords[-1]) - float(v_coords[0]))
    width = (len(u_coords) - 1) * scale
    height = (len(v_coords) - 1) * scale
    return col * width, row * height


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

    crest_v = getattr(cs, "crest_v", None)
    if crest_v is not None:
        x_left, y_crest = _px(u_coords[0], crest_v, u_coords, v_coords, scale)
        x_right, _ = _px(u_coords[-1], crest_v, u_coords, v_coords, scale)
        draw.line([(x_left, y_crest), (x_right, y_crest)], fill=(255, 220, 0), width=1)
        overlay.append(((x_left, y_crest), (x_right, y_crest)))

    segments: Dict[float, Tuple[float, float]] = getattr(cs, "width_segments", {}) or {}
    for depth, (u_left, u_right) in segments.items():
        v_line = crest_v - float(depth) if crest_v is not None else None
        if v_line is None:
            continue
        start = _px(u_left, v_line, u_coords, v_coords, scale)
        end = _px(u_right, v_line, u_coords, v_coords, scale)
        colour = (0, 220, 255) if depth == 0.0 else (255, 140, 0)
        draw.line([start, end], fill=colour, width=1)
        overlay.append((start, end))
        mid_x = (start[0] + end[0]) / 2.0
        draw.text((mid_x, max(0, start[1] - 10)), f"W{depth:g}", fill=colour, font=font)

    canal = getattr(cs, "canal_centroid_uv", None)
    if canal is not None:
        cu, cv = canal
        cx, cy = _px(cu, cv, u_coords, v_coords, scale)
        radius_mm = 1.5
        radius_px = radius_mm / max(1e-6, (u_coords[-1] - u_coords[0]) / max(1, len(u_coords) - 1)) * scale
        draw.ellipse(
            [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px],
            outline=(0, 160, 255),
            width=1,
        )
        draw.text((cx + radius_px + 2, cy), "canal", fill=(0, 160, 255), font=font)

    if meas is None:
        draw.text((8, height - 18), "not measurable", fill=(255, 60, 60), font=font)
        labels.append((8.0, float(height - 18), "not measurable"))
    else:
        height_mm = getattr(meas, "bone_height_mm", None)
        crest_w = getattr(meas, "crest_width_mm", None)
        text = "H --  W0 --" if height_mm is None else f"H {height_mm:.1f} mm  W0 {crest_w or 0:.1f} mm"
        draw.text((8, height - 18), text, fill=(255, 255, 255), font=font)

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
    image = image.resize((arr.shape[1] * scale, strip_height), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    ticks: List[Tuple[float, float, str]] = []

    for index, arc_mm in enumerate(tick_arcs_mm):
        col = float(arc_mm) / float(column_spacing_mm)
        if col < 0 or col > arr.shape[1] - 1:
            continue
        x = col * scale
        draw.line([(x, strip_height - 14), (x, strip_height)], fill=(0, 255, 0), width=1)
        label = labels[index] if labels is not None and index < len(labels) else ""
        draw.text((x + 1, strip_height - 13), str(label), fill=(0, 255, 0), font=font)
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
    "render_cross_section",
    "render_panoramic_strip",
    "unavailable_tile",
]
