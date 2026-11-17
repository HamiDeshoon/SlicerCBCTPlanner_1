"""8x10-Inch Printable Clinical Dental Layout Engine (SlicerCBCTPlanner).

Generates standard 8x10 inch printable film layouts (HTML, PDF & 300 DPI PNG) combining:
  1. Panoramic reformat view with numbered slice indicator ticks (auto-synthesized from
     the fitted arch curve if the user did not manually generate a PanoramicVolume first).
  2. 15 to 27 sequential cross-sectional slices with drawn clinical measurement lines:
     - Bone height (Green vertical line with end-caps)
     - Crestal bone width (Yellow/Cyan horizontal line)
     - Bone width at 3 mm depth (Orange/Gold horizontal line)
     - Bone width at 5 mm depth (Amber horizontal line)
     - Mandibular canal / nerve safety margin (Red circular boundary)
  3. 3D Volume Rendering (VR) / Anatomical Quadrant Projection isolating the anatomical classes
     for the selected quadrant (jawbone, vivid teeth & roots, and inferior alveolar nerve canal).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import io
import logging
import math
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import ctk
    import qt
    import slicer
except ImportError:
    qt = None
    ctk = None
    slicer = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

from .core_print_render import (
    hu_window_to_rgb,
    render_cross_section,
    render_panoramic_strip,
    unavailable_tile,
)

logger = logging.getLogger(__name__)


CLINICAL_DISCLAIMER_TEXT = (
    "FOR CLINICAL PLANNING & INVESTIGATIONAL USE ONLY — NOT AN AUTOMATED SURGICAL PRESCRIPTION. "
    "All bone measurements, canal clearances, and reformatted views must be verified by a licensed clinician "
    "prior to surgical or implant procedures."
)


@dataclass
class CrossSectionSliceSpec:
    slice_index: int
    total_slices: int
    offset_mm: float
    fdi: int
    height_mm: Optional[float] = None
    crest_width_mm: Optional[float] = None
    width_3mm: Optional[float] = None
    width_5mm: Optional[float] = None
    nerve_dist_mm: Optional[float] = None
    misch_class: Optional[str] = None
    mean_hu: Optional[float] = None
    is_mandible: bool = True
    image_data_uri: str = ""


def _image_to_data_uri(image: Any) -> str:
    if image is None:
        return ""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _unavailable_data_uri(text: str) -> str:
    return _image_to_data_uri(unavailable_tile(text))


def _data_uri_to_image(uri: str) -> Optional[Any]:
    if not uri or Image is None or "," not in uri:
        return None
    try:
        raw = base64.b64decode(uri.split(",", 1)[1])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        logger.exception("Could not decode layout image")
        return None


class PrintLayoutEngine:
    """Orchestrates generation of 8x10-inch clinical print layouts (HTML + PDF + 300 DPI PNG)."""

    def __init__(self, logic: Any = None) -> None:
        self.logic = logic
        self.last_png_file: Optional[Path] = None
        self.last_pdf_file: Optional[Path] = None
        self.last_html_file: Optional[Path] = None

    @staticmethod
    def default_export_directory() -> Path:
        """Return a visible, user-accessible default export folder (never hidden OS temp unless fallback)."""
        try:
            home_docs = Path.home() / "Documents" / "CBCTPlanner_Prints"
            home_docs.mkdir(parents=True, exist_ok=True)
            return home_docs
        except Exception:
            if slicer is not None and getattr(slicer, "app", None) is not None:
                return Path(slicer.app.temporaryPath)
            return Path(".")

    def generate_8x10_layout(
        self,
        volumeNode: Any,
        segmentationNode: Any = None,
        selected_fdis: Optional[Sequence[int]] = None,
        quadrant: Optional[int] = None,
        out_dir: Optional[Path] = None,
        patient_name: str = "Anonymous",
        patient_id: str = "Anonymous",
        include_metadata: bool = False,
        pano_b64: Optional[str] = None,
        vr_b64: Optional[str] = None,
    ) -> Tuple[Path, Path, List[CrossSectionSliceSpec]]:
        """Generate genuine 8x10 inch film pages (HTML view + 300 DPI PDF + 300 DPI PNG)."""
        from .core.labels import sort_teeth_anatomically
        from .core.print_layout import slice_count_for_selection

        out_path = Path(out_dir) if out_dir is not None else self.default_export_directory()
        out_path.mkdir(parents=True, exist_ok=True)

        fdis = sort_teeth_anatomically(list(selected_fdis or []))
        if not fdis:
            raise ValueError("Select at least one tooth before generating the 8x10 layout")
        if quadrant is None:
            quadrant = int(str(fdis[0])[0])

        engine = None
        if self.logic is not None and hasattr(self.logic, "_get_or_build_measurement_engine"):
            engine = self.logic._get_or_build_measurement_engine(volumeNode, segmentationNode)
        if engine is None:
            raise RuntimeError("A measurement engine is required to generate the 8x10 layout")

        groups: List[Tuple[int, List[int]]] = []
        for q in (1, 2, 3, 4):
            q_fdis = [fdi for fdi in fdis if int(str(fdi)[0]) == q]
            if q_fdis:
                groups.append((q, q_fdis))

        all_specs: List[CrossSectionSliceSpec] = []
        page_records: List[dict] = []
        for q, q_fdis in groups:
            is_mandible = q in (3, 4)
            jaw_label = "Mandible" if is_mandible else "Maxilla"
            n_slices, step_mm = slice_count_for_selection(len(q_fdis))
            arcs = self._tooth_arcs(engine, q_fdis)
            if not arcs:
                raise RuntimeError(f"No arch positions for quadrant {q}; fit the arch first")
            center_arc = float(np.mean(list(arcs.values())))
            if len(q_fdis) <= 3:
                arc_values = center_arc + (np.arange(n_slices) - (n_slices - 1) / 2.0) * step_mm
            else:
                first_arc = min(arcs.values())
                last_arc = max(arcs.values())
                arc_values = np.linspace(first_arc - 2.0, last_arc + 2.0, n_slices)

            specs: List[CrossSectionSliceSpec] = []
            for s_idx, arc_mm in enumerate(arc_values):
                target_fdi = min(q_fdis, key=lambda fdi: abs(arcs.get(fdi, 0.0) - arc_mm))
                meas = None
                cs = None
                try:
                    meas, cs = engine.measure_at_arc_mm(jaw_label, float(arc_mm), fdi_hint=str(target_fdi))
                except Exception:
                    logger.exception(
                        "Cross-section measurement failed at %.2f mm (FDI %s)", arc_mm, target_fdi
                    )
                if cs is not None:
                    render = render_cross_section(cs, meas, scale=2)
                    data_uri = (
                        _image_to_data_uri(render.image)
                        if render.available and render.image is not None
                        else _unavailable_data_uri("Cross-section")
                    )
                else:
                    data_uri = _unavailable_data_uri("Cross-section")

                misch_short = None
                if meas and getattr(meas, "misch_density_class", None):
                    misch_short = str(meas.misch_density_class).split()[0]

                spec = CrossSectionSliceSpec(
                    slice_index=s_idx + 1,
                    total_slices=n_slices,
                    offset_mm=float(arc_mm - center_arc),
                    fdi=target_fdi,
                    height_mm=meas.bone_height_mm if meas else None,
                    crest_width_mm=meas.crest_width_mm if meas else None,
                    width_3mm=meas.bone_width_at_depths_mm.get(3.0)
                    if (meas and meas.bone_width_at_depths_mm)
                    else None,
                    width_5mm=meas.bone_width_at_depths_mm.get(5.0)
                    if (meas and meas.bone_width_at_depths_mm)
                    else None,
                    nerve_dist_mm=meas.distance_to_vital_structure_mm if meas else None,
                    misch_class=misch_short,
                    mean_hu=meas.mean_density_hu if meas else None,
                    is_mandible=is_mandible,
                    image_data_uri=data_uri,
                )
                specs.append(spec)
                all_specs.append(spec)

            pano_uri = pano_b64 or self._pano_strip_uri(
                arc_values, [s.fdi for s in specs], engine=engine, is_mandible=is_mandible
            )
            if not pano_uri:
                pano_uri = _unavailable_data_uri("Panoramic")

            vr_uri = vr_b64 or self._vr_capture(engine=engine, quadrant=q)
            if not vr_uri:
                vr_uri = _unavailable_data_uri("VR")

            page_records.append(
                {
                    "quadrant": q,
                    "fdis": q_fdis,
                    "jaw_label": jaw_label,
                    "slices": specs,
                    "pano_b64": pano_uri,
                    "vr_b64": vr_uri,
                }
            )

        # HTML browser view: one section per quadrant with a page break.
        html_pages = [
            self._build_8x10_html(
                slices=rec["slices"],
                pano_b64=rec["pano_b64"],
                vr_b64=rec["vr_b64"],
                quadrant=rec["quadrant"],
                jaw_label=rec["jaw_label"],
                fdis=rec["fdis"],
                patient_name=patient_name if include_metadata else "Anonymous",
                patient_id=patient_id if include_metadata else "Anonymous",
                page_number=index + 1,
                total_pages=len(page_records),
            )
            for index, rec in enumerate(page_records)
        ]
        html_content = self._combine_html_pages(html_pages)

        pages: List[Any] = []
        for index, rec in enumerate(page_records):
            page = self._compose_page_image(
                rec,
                patient_name=patient_name if include_metadata else "Anonymous",
                page_number=index + 1,
                total_pages=len(page_records),
            )
            if page is not None:
                pages.append(page)

        vol_name = volumeNode.GetName() if (volumeNode and hasattr(volumeNode, "GetName")) else "CBCT"
        clean_name = "".join(c if c.isalnum() else "_" for c in vol_name) or "CBCT"
        html_file = out_path / f"{clean_name}_8x10_Print_Layout.html"
        pdf_file = out_path / f"{clean_name}_8x10_Print_Layout.pdf"
        png_file = out_path / f"{clean_name}_8x10_Print_Layout.png"
        html_file.write_text(html_content, encoding="utf-8")

        if not pages:
            raise RuntimeError("PDF layout requires Qt or Pillow")
        pages[0].save(
            pdf_file,
            "PDF",
            resolution=300.0,
            save_all=True,
            append_images=pages[1:],
        )
        pages[0].save(png_file, "PNG")
        self.last_html_file = html_file
        self.last_pdf_file = pdf_file
        self.last_png_file = png_file
        return html_file, pdf_file, all_specs

    @staticmethod
    def _tooth_arcs(engine: Any, fdis: Sequence[int]) -> dict:
        arcs = {}
        for fdi in fdis:
            try:
                center = engine.get_site_center_ras(str(fdi))
                arch = engine._arch_for_fdi(str(fdi))
                if center is None or arch is None:
                    continue
                t_param, _ = arch.find_closest_parameter(center)
                arcs[int(fdi)] = float(t_param) * float(arch.total_arc_length)
            except Exception:
                logger.exception("Could not resolve the arch position of FDI %s", fdi)
        return arcs

    def _pano_strip_uri(
        self,
        arcs: Sequence[float],
        labels: Sequence[int],
        engine: Any = None,
        is_mandible: bool = True,
    ) -> Optional[str]:
        """Render the panoramic strip from `PanoramicVolume`, or synthesize it in-process from `engine`."""
        # 1. Try existing Slicer PanoramicVolume node first
        if self.logic is not None and slicer is not None:
            try:
                pNode = self.logic.getParameterNode() if hasattr(self.logic, "getParameterNode") else None
                panoNode = (
                    pNode.GetNodeReference("PanoramicVolume")
                    if (pNode and hasattr(pNode, "GetNodeReference"))
                    else None
                )
                if panoNode is not None:
                    arr = slicer.util.arrayFromVolume(panoNode)
                    spacing = float(pNode.GetParameter("PanoSpacingMm") or 0.5) if pNode else 0.5
                    render = render_panoramic_strip(
                        arr,
                        column_spacing_mm=max(1e-3, spacing),
                        tick_arcs_mm=arcs,
                        labels=[str(label) for label in labels],
                        strip_height=340,
                        scale=2,
                    )
                    if render.available and render.image is not None:
                        return _image_to_data_uri(render.image)
            except Exception:
                logger.exception("Panoramic strip rendering from node failed")

        # 2. Automatic in-process CPR synthesis from engine.volume + arch when PanoramicVolume node wasn't generated yet
        vol = getattr(engine, "volume", None) if engine is not None else None
        arch_fn = getattr(engine, "_arch_for_jaw", None) if engine is not None else None
        arch = arch_fn(is_mandible) if callable(arch_fn) else None
        if vol is not None and arch is not None and hasattr(arch, "pts_dense"):
            try:
                from .PanoramicCurveLogic import PanoramicCurveLogic

                pts_dense = np.asarray(arch.pts_dense, dtype=np.float64)
                if len(pts_dense) >= 8:
                    up = np.asarray(getattr(arch, "occlusal_normal", (0.0, 0.0, 1.0)), dtype=np.float64)
                    tangents = np.gradient(pts_dense, axis=0)
                    t_norm = np.maximum(1e-8, np.linalg.norm(tangents, axis=1, keepdims=True))
                    tangents /= t_norm
                    normals = np.cross(tangents, up)
                    n_norm = np.maximum(1e-8, np.linalg.norm(normals, axis=1, keepdims=True))
                    normals /= n_norm
                    binormals = np.cross(normals, tangents)
                    b_norm = np.maximum(1e-8, np.linalg.norm(binormals, axis=1, keepdims=True))
                    binormals /= b_norm
                    if np.any((binormals @ up) < 0):
                        binormals[(binormals @ up) < 0] *= -1.0

                    pano_arr = PanoramicCurveLogic.sample_panoramic_slab_array(
                        cbct_volume=vol,
                        pts_arch=pts_dense,
                        tangents=tangents,
                        normals=normals,
                        binormals=binormals,
                        slab_thickness_mm=20.0,
                        height_mm=70.0,
                        spacing_mm=0.75,
                        projection_mode="Mean",
                        flip_vertical=True,
                    )
                    render = render_panoramic_strip(
                        pano_arr,
                        column_spacing_mm=0.75,
                        tick_arcs_mm=arcs,
                        labels=[str(label) for label in labels],
                        strip_height=340,
                        scale=2,
                    )
                    if render.available and render.image is not None:
                        return _image_to_data_uri(render.image)
            except Exception:
                logger.exception("In-process panoramic synthesis for print layout failed")
        return None

    def _vr_capture(self, engine: Any = None, quadrant: int = 4) -> Optional[str]:
        if self.logic is not None:
            capture = getattr(self.logic, "capture_3d_view_png_base64", None)
            if callable(capture):
                try:
                    uri = capture(view_index=0)
                    if uri:
                        return uri
                except Exception:
                    logger.exception("3D VR capture failed")

        # Fallback: synthesize a 3D vivid MIP / segmentation projection card from engine.volume & mask
        vol = getattr(engine, "volume", None) if engine is not None else None
        mask = getattr(engine, "mask", None) if engine is not None else None
        if vol is not None and Image is not None and getattr(vol, "image", None) is not None:
            try:
                img_3d = np.asarray(vol.image, dtype=np.float32)
                # Coronal MIP projection (Z, X)
                mip = np.max(img_3d, axis=1)[::-1, :]
                gray = hu_window_to_rgb(mip)
                rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
                if mask is not None and mask.shape == img_3d.shape:
                    m_proj = np.max(mask, axis=1)[::-1, :]
                    teeth_m = (m_proj == 3) | (m_proj == 4) | ((m_proj >= 11) & (m_proj <= 48))
                    canal_m = (m_proj == 5) | (m_proj == 3) | (m_proj == 4)
                    rgb[teeth_m] = 0.35 * rgb[teeth_m] + 0.65 * np.array([255.0, 220.0, 120.0])
                    if np.any(m_proj == 5):
                        rgb[m_proj == 5] = np.array([255.0, 60.0, 50.0])
                pil_img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").resize(
                    (400, 220), Image.BILINEAR
                )
                draw = ImageDraw.Draw(pil_img)
                draw.text(
                    (8, 6),
                    f"3D Vivid MIP Projection (Q{quadrant})",
                    fill=(56, 189, 248),
                    font=ImageFont.load_default(),
                )
                return _image_to_data_uri(pil_img)
            except Exception:
                logger.debug("Fallback MIP projection card failed", exc_info=True)
        return None

    @staticmethod
    def _combine_html_pages(pages: List[str]) -> str:
        if not pages:
            return ""
        head = pages[0].partition("<body>")[0]
        bodies = [page.partition("<body>")[2].rpartition("</body>")[0] for page in pages]
        joined = '<div style="page-break-after: always;">' + bodies[0] + "</div>"
        for body in bodies[1:]:
            joined += '<div style="page-break-before: always;">' + body + "</div>"
        return head + "<body>" + joined + "</body></html>"

    def _compose_page_image(
        self, record: dict, *, patient_name: str, page_number: int, total_pages: int
    ) -> Optional[Any]:
        if Image is None:
            return None
        page_w, page_h = 2400, 3000
        page = Image.new("RGB", (page_w, page_h), (255, 255, 255))
        draw = ImageDraw.Draw(page)
        font = ImageFont.load_default()
        margin = 75

        draw.text((margin, 40), "CBCT Dental Planner — 8x10 clinical film", fill=(20, 20, 20), font=font)
        draw.text(
            (margin, 70),
            f"Quadrant {record['quadrant']} ({record['jaw_label']})  Teeth: "
            + ", ".join(f"#{fdi}" for fdi in record["fdis"])
            + f"  Patient: {patient_name}",
            fill=(60, 60, 60),
            font=font,
        )

        pano = _data_uri_to_image(record["pano_b64"])
        if pano is not None:
            pano_h = 360
            pano_w = page_w - 2 * margin
            page.paste(pano.resize((pano_w, pano_h), Image.BILINEAR), (margin, 105))

        specs: List[CrossSectionSliceSpec] = record["slices"]
        cols = 5 if len(specs) <= 15 else (7 if len(specs) <= 21 else 9)
        rows = int(math.ceil(len(specs) / cols))
        grid_top = 500
        footer_top = 2700
        key_h = 260
        grid_bottom = footer_top - key_h - 20
        gap = 10
        cell_w = (page_w - 2 * margin - (cols - 1) * gap) // cols
        cell_h = (grid_bottom - grid_top - (rows - 1) * gap) // max(1, rows)
        frame_h = min(cell_h, int(cell_w * 1.25))

        for idx, spec in enumerate(specs):
            row, col = divmod(idx, cols)
            x = margin + col * (cell_w + gap)
            y = grid_top + row * (cell_h + gap)
            image = _data_uri_to_image(spec.image_data_uri)
            if image is not None:
                thumb_h = frame_h - 18
                page.paste(image.resize((cell_w, thumb_h), Image.BILINEAR), (x, y))
            label = (
                f"#{spec.fdi} S{spec.slice_index}/{spec.total_slices} "
                f"{spec.offset_mm:+.1f}mm H {spec.height_mm if spec.height_mm is not None else '--'}"
            )
            draw.text((x + 2, y + frame_h - 16), label, fill=(0, 0, 0), font=font)

        vr = _data_uri_to_image(record["vr_b64"])
        key_top = footer_top - key_h
        if vr is not None:
            vr_w = 800
            page.paste(vr.resize((vr_w, key_h), Image.BILINEAR), (page_w - margin - vr_w, key_top))
        draw.text((margin, key_top + 10), "Measurement key", fill=(0, 0, 0), font=font)
        draw.text((margin, key_top + 32), "H  bone height (mm) [Green vertical]", fill=(0, 120, 40), font=font)
        draw.text(
            (margin, key_top + 54),
            "W0 crest width [Cyan] / W3 (3mm) / W5 (5mm) depth widths [Gold]",
            fill=(0, 0, 0),
            font=font,
        )
        draw.text(
            (margin, key_top + 76),
            "IAN Nerve clearance is shown when a canal/sinus is segmented [Red]",
            fill=(160, 20, 20),
            font=font,
        )
        draw.text((margin, footer_top + 20), CLINICAL_DISCLAIMER_TEXT, fill=(90, 90, 90), font=font)
        draw.text(
            (page_w - margin - 120, footer_top + 60),
            f"Page {page_number}/{total_pages}",
            fill=(90, 90, 90),
            font=font,
        )
        return page

    def _build_8x10_html(
        self,
        slices: List[CrossSectionSliceSpec],
        pano_b64: str,
        vr_b64: str,
        quadrant: int,
        jaw_label: str,
        fdis: List[int],
        patient_name: str,
        patient_id: str,
        page_number: int = 1,
        total_pages: int = 1,
    ) -> str:
        """Construct exact 8x10-inch standard print HTML/CSS page."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        cols = 5 if len(slices) <= 15 else (7 if len(slices) <= 21 else 9)

        cards_html: List[str] = []
        for s in slices:
            h_txt = f"{s.height_mm:.1f}" if s.height_mm is not None else "--"
            w0_txt = f"{s.crest_width_mm:.1f}" if s.crest_width_mm is not None else "--"
            w3_txt = f"{s.width_3mm:.1f}" if s.width_3mm is not None else "--"
            w5_txt = f"{s.width_5mm:.1f}" if s.width_5mm is not None else "--"
            misch_badge = f" | {s.misch_class}" if s.misch_class else ""
            cards_html.append(
                f"<div class='cs-card'>"
                f"<div class='cs-hdr'>#{s.fdi} S{s.slice_index} ({s.offset_mm:+.1f}mm){html.escape(misch_badge)}</div>"
                f"<img src='{s.image_data_uri}' alt='Slice {s.slice_index}'/>"
                f"<div class='cs-ftr'>H:{h_txt} W0:{w0_txt} W3:{w3_txt} W5:{w5_txt}</div>"
                f"</div>"
            )

        fdi_list_str = ", ".join(f"#{t}" for t in fdis)
        q_label = f"Quadrant {quadrant} ({jaw_label})"

        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>8x10 Clinical Dental Film Layout — {html.escape(q_label)}</title>
<style>
@page {{
    size: 8in 10in;
    margin: 0.25in;
}}
@media print {{
    body {{ width: 7.5in; height: 9.5in; margin: 0; padding: 0; background: #fff; }}
    .no-print {{ display: none !important; }}
}}
* {{ box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #0f172a;
    color: #f8fafc;
    margin: 0;
    padding: 16px;
    width: 8in;
    min-height: 10in;
}}
.page-container {{
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 8px;
}}
.header-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #38bdf8;
    padding-bottom: 6px;
}}
.header-title {{ font-size: 16px; font-weight: bold; color: #38bdf8; }}
.header-meta {{ font-size: 11px; color: #94a3b8; text-align: right; }}
.banner-disclaimer {{
    background: #450a0a;
    border-left: 4px solid #ef4444;
    color: #fca5a5;
    font-size: 9px;
    padding: 4px 8px;
    font-weight: 600;
}}
.pano-section {{
    border: 1px solid #334155;
    background: #020617;
    border-radius: 4px;
    overflow: hidden;
    max-height: 130px;
    display: flex;
    flex-direction: column;
}}
.section-label {{
    font-size: 10px;
    font-weight: bold;
    padding: 2px 6px;
    background: #1e293b;
    color: #94a3b8;
}}
.pano-img {{ width: 100%; max-height: 110px; object-fit: contain; }}
.grid-section {{
    display: grid;
    grid-template-columns: repeat({cols}, 1fr);
    gap: 4px;
    background: #020617;
    padding: 4px;
    border: 1px solid #334155;
    border-radius: 4px;
}}
.cs-card {{
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 3px;
    overflow: hidden;
    text-align: center;
}}
.cs-hdr {{ font-size: 8px; color: #38bdf8; background: #020617; padding: 1px 2px; font-weight: 600; }}
.cs-ftr {{ font-size: 7.5px; color: #e2e8f0; background: #020617; padding: 1px 2px; }}
.cs-card img {{ width: 100%; height: auto; display: block; }}
.bottom-section {{
    display: flex;
    gap: 8px;
    max-height: 140px;
}}
.vr-card {{
    flex: 1.2;
    background: #020617;
    border: 1px solid #334155;
    border-radius: 4px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}}
.vr-img {{ width: 100%; height: 115px; object-fit: contain; }}
.summary-card {{
    flex: 1.8;
    background: #020617;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 10px;
    color: #cbd5e1;
}}
.legend-row {{ display: flex; gap: 10px; margin-top: 4px; font-size: 9px; }}
.leg-item {{ display: flex; align-items: center; gap: 4px; }}
.leg-box {{ width: 8px; height: 8px; border-radius: 2px; }}
.footer-text {{
    font-size: 8px;
    color: #64748b;
    text-align: center;
    border-top: 1px solid #1e293b;
    padding-top: 4px;
}}
</style>
</head>
<body>
<div class="page-container">
    <div class="header-bar">
        <div>
            <div class="header-title">8×10″ CBCT Dental Reformat Layout</div>
            <div style="font-size: 11px; color: #e2e8f0; margin-top: 2px;">
                <strong>Site Selection:</strong> {html.escape(q_label)} | Teeth: {html.escape(fdi_list_str)}
            </div>
        </div>
        <div class="header-meta">
            <div>Patient: <strong>{html.escape(patient_name)}</strong> (ID: {html.escape(patient_id)})</div>
            <div>Date: {html.escape(date_str)} | Slices: {len(slices)} @ Standard Pitch</div>
        </div>
    </div>

    <div class="banner-disclaimer">{html.escape(CLINICAL_DISCLAIMER_TEXT)}</div>

    <!-- Panoramic View Strip -->
    <div class="pano-section">
        <div class="section-label">Panoramic View with Cross-Section Indicator Ticks (Slices 1 to {len(slices)})</div>
        <img class="pano-img" src="{pano_b64}" alt="Panoramic Reformat"/>
    </div>

    <!-- Sequential Cross-Sectional Slices Grid -->
    <div>
        <div class="section-label" style="margin-bottom: 2px;">Sequential Cross-Sectional Series ({len(slices)} Slices @ Overlaid Measurement Levels)</div>
        <div class="grid-section">
            {"".join(cards_html)}
        </div>
    </div>

    <!-- Bottom Section: 3D Volume Rendering + Clinical Key -->
    <div class="bottom-section">
        <div class="vr-card">
            <div class="section-label">3D Anatomical Volume Rendering ({html.escape(q_label)})</div>
            <img class="vr-img" src="{vr_b64}" alt="3D Quadrant VR"/>
        </div>
        <div class="summary-card">
            <div class="section-label" style="background: transparent; padding: 0; color: #38bdf8; margin-bottom: 4px;">
                Measurement Key & Verification Protocol
            </div>
            <div>All measurement lines are resampled orthogonally along the dental arch curve:</div>
            <div class="legend-row">
                <div class="leg-item"><div class="leg-box" style="background: #22c55e;"></div><strong>Height (H):</strong> Crest to canal/base</div>
                <div class="leg-item"><div class="leg-box" style="background: #00dcff;"></div><strong>Crest Width (W0):</strong> Crestal width</div>
            </div>
            <div class="legend-row">
                <div class="leg-item"><div class="leg-box" style="background: #ffb400;"></div><strong>W3:</strong> Width @ 3 mm depth</div>
                <div class="leg-item"><div class="leg-box" style="background: #ff7828;"></div><strong>W5:</strong> Width @ 5 mm depth</div>
                <div class="leg-item"><div class="leg-box" style="background: #ff3c3c;"></div><strong>Canal:</strong> IAN Nerve margin</div>
            </div>
            <div style="font-size: 8.5px; color: #94a3b8; margin-top: 6px; line-height: 1.3;">
                <em>Notice:</em> Measurements are calibrated to the patient's native CBCT voxel spacing. Clinicians must verify all anatomical safety margins on primary DICOM slice viewports prior to surgical intervention.
            </div>
        </div>
    </div>

    <div class="footer-text">
        Generated by SlicerCBCTPlanner | Standard 8×10-Inch Orthogonal Film Print Layout | Page {page_number} of {total_pages}
    </div>
</div>
</body>
</html>"""
        return html_doc
