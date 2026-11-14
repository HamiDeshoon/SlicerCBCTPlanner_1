"""Advanced Clinical & Reporting Tools (F4, F5, F6, F7, F8, F9, F10, F11, F13, F18, B21, B22).

Provides:
  - F4: Segmentation tooth outlines and `#FDI` centroid label overlays on Panoramic & Cross-Sections.
  - F6 & F18: Automated Occlusal Plane SVD fit from segmented tooth cusps/incisal edges + optional
    feature-detected integration with `SlicerAutomatedDentalTools` (`ASO`, `AMASSS`, `ALI`, `AReg`,
    `AutoCrop3D`, `BatchDentalSegmentator`, `MRI2CBCT`).
  - F7: Pharyngeal Airway threshold + seeded 3D connected-component region growing with total
    volume ($\text{mm}^3$ / $\text{cm}^3$) and minimum cross-sectional area ($\text{mm}^2$).
  - F8: Bilateral TMJ Sagittal and Coronal slice view presets through the Left and Right condyles.
  - F9: 3D Angle measurement tool and general surface-to-surface distance computation.
  - B21: Reliable `vtkWindowToImageFilter` / `ScreenCapture` slice screenshot capture.
  - B22 & F10: Clinical PDF generator (`QPrinter`/`QTextDocument` with pure-Python PDF 1.4 fallback).
  - F11: Reformatted DICOM series export (guarded lazy `pydicom` / Slicer DICOM exporter).
  - F13: `UndoRedoManager` stack for cursor, curve, manual ruler, and implant edits.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import qt  # type: ignore
    import slicer  # type: ignore
    import vtk  # type: ignore
except ImportError:
    qt = None
    slicer = None
    vtk = None

from .core import ensure_optional_package
from .core.labels import LOWER_JAW_ID
from .core.volume_loader import CBCTVolume


# =====================================================================
# F13: Undo / Redo Manager
# =====================================================================
@dataclass
class UndoableCommand:
    """Single reversible user action for cursor, curve, ruler, or implant state."""

    description: str
    undo_fn: Callable[[], None]
    redo_fn: Callable[[], None]


class UndoRedoManager:
    """F13: Bounded Undo/Redo command stack for cursor, curve, ruler, and implant edits."""

    def __init__(self, max_depth: int = 50) -> None:
        self.max_depth = max(1, int(max_depth))
        self._undo_stack: List[UndoableCommand] = []
        self._redo_stack: List[UndoableCommand] = []
        self._is_replaying: bool = False

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def push(
        self,
        description: str,
        undo_fn: Callable[[], None],
        redo_fn: Callable[[], None],
    ) -> None:
        if self._is_replaying:
            return
        self._undo_stack.append(UndoableCommand(description=description, undo_fn=undo_fn, redo_fn=redo_fn))
        if len(self._undo_stack) > self.max_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> Optional[str]:
        if not self._undo_stack:
            return None
        cmd = self._undo_stack.pop()
        self._is_replaying = True
        try:
            cmd.undo_fn()
        finally:
            self._is_replaying = False
        self._redo_stack.append(cmd)
        return cmd.description

    def redo(self) -> Optional[str]:
        if not self._redo_stack:
            return None
        cmd = self._redo_stack.pop()
        self._is_replaying = True
        try:
            cmd.redo_fn()
        finally:
            self._is_replaying = False
        self._undo_stack.append(cmd)
        return cmd.description

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()


# =====================================================================
# F6 & F18: Occlusal Plane from Teeth Segmentation + AutomatedDentalTools
# =====================================================================
AUTOMATED_DENTAL_TOOLS_MODULES: Tuple[str, ...] = (
    "ASO",
    "AMASSS",
    "ALI",
    "AReg",
    "AutoCrop3D",
    "BatchDentalSegmentator",
    "MRI2CBCT",
)


def detect_automated_dental_tools_modules() -> Dict[str, bool]:
    """F6 & F18: Feature-detect installed SlicerAutomatedDentalTools modules inside 3D Slicer."""
    available: Dict[str, bool] = {}
    mod_container = getattr(slicer, "modules", None) if slicer is not None else None
    for mod_name in AUTOMATED_DENTAL_TOOLS_MODULES:
        has_mod = bool(
            mod_container is not None
            and (hasattr(mod_container, mod_name.lower()) or hasattr(mod_container, mod_name))
        )
        available[mod_name] = has_mod
    return available


def compute_occlusal_plane_from_segmentation(
    crown_or_tooth_points_ras: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """F6: Fit an occlusal plane `(unit_normal_superior, center_ras)` to cusp/incisal tooth points via SVD."""
    pts = np.asarray(crown_or_tooth_points_ras, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError(f"Need at least 3 3D tooth points to fit occlusal plane, got shape {pts.shape}")
    center = np.mean(pts, axis=0)
    centered = pts - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1, :].astype(np.float64)
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-8:
        raise ValueError("Tooth points are degenerate and do not span a plane.")
    normal /= norm_len
    if normal[2] < 0:
        normal = -normal
    return normal, center


# =====================================================================
# F7: Pharyngeal Airway Volume & Minimum Cross-Sectional Area (mCSA)
# =====================================================================
@dataclass
class AirwayAnalysisResult:
    """F7: Quantitative pharyngeal airway analysis results."""

    total_volume_mm3: float
    total_volume_cm3: float
    min_cross_sectional_area_mm2: float
    min_csa_z_ras_mm: float
    threshold_hu: float
    slice_areas_mm2: Dict[float, float] = field(default_factory=dict)


def compute_pharyngeal_airway_metrics(
    volume: CBCTVolume,
    seed_ras: Optional[Sequence[float]] = None,
    threshold_hu: float = -400.0,
    roi_bounds_ras: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> Optional[AirwayAnalysisResult]:
    """F7: Segment pharyngeal airway via intensity thresholding + 3D connected-component seed region growing.

    Computes total airway volume ($\text{mm}^3$ and $\text{cm}^3$) and minimum axial cross-sectional area ($\text{mm}^2$).
    """
    scipy_nd = ensure_optional_package("scipy.ndimage", "scipy", "airway region growing")
    img = volume.image
    air_candidates = img <= float(threshold_hu)

    if roi_bounds_ras is not None:
        b_min = np.asarray(roi_bounds_ras[0], dtype=np.float64)
        b_max = np.asarray(roi_bounds_ras[1], dtype=np.float64)
        v0 = volume.world_to_voxel(b_min)
        v1 = volume.world_to_voxel(b_max)
        ix0, ix1 = max(0, int(min(v0[0], v1[0]))), min(img.shape[2], int(max(v0[0], v1[0])) + 1)
        iy0, iy1 = max(0, int(min(v0[1], v1[1]))), min(img.shape[1], int(max(v0[1], v1[1])) + 1)
        iz0, iz1 = max(0, int(min(v0[2], v1[2]))), min(img.shape[0], int(max(v0[2], v1[2])) + 1)
        roi_mask = np.zeros_like(air_candidates, dtype=bool)
        roi_mask[iz0:iz1, iy0:iy1, ix0:ix1] = True
        air_candidates = air_candidates & roi_mask

    if not np.any(air_candidates):
        return None

    labeled, num_features = scipy_nd.label(air_candidates)
    if num_features == 0:
        return None

    target_label = 0
    if seed_ras is not None:
        sv = volume.world_to_voxel(seed_ras)
        sx_i, sy_i, sz_i = int(round(sv[0])), int(round(sv[1])), int(round(sv[2]))
        if (
            0 <= sz_i < labeled.shape[0]
            and 0 <= sy_i < labeled.shape[1]
            and 0 <= sx_i < labeled.shape[2]
        ):
            target_label = int(labeled[sz_i, sy_i, sx_i])

    if target_label == 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        target_label = int(np.argmax(counts))
        if counts[target_label] == 0:
            return None

    airway_mask = labeled == target_label
    sz, sy, sx = volume.spacing
    vox_vol = float(sx * sy * sz)
    pixel_area = float(sx * sy)
    total_voxels = int(np.count_nonzero(airway_mask))
    if total_voxels == 0:
        return None

    vol_mm3 = round(total_voxels * vox_vol, 2)
    vol_cm3 = round(vol_mm3 / 1000.0, 3)

    slice_counts = np.count_nonzero(airway_mask, axis=(1, 2))
    valid_z = np.where(slice_counts > 0)[0]
    slice_areas: Dict[float, float] = {}
    min_csa = float("inf")
    min_z_ras = 0.0
    for kz in valid_z:
        area_mm2 = float(slice_counts[kz]) * pixel_area
        z_ras = float(volume.voxel_to_world([0, 0, int(kz)])[2])
        slice_areas[round(z_ras, 2)] = round(area_mm2, 2)
        if area_mm2 < min_csa:
            min_csa = area_mm2
            min_z_ras = z_ras

    return AirwayAnalysisResult(
        total_volume_mm3=vol_mm3,
        total_volume_cm3=vol_cm3,
        min_cross_sectional_area_mm2=round(min_csa, 2),
        min_csa_z_ras_mm=round(min_z_ras, 2),
        threshold_hu=float(threshold_hu),
        slice_areas_mm2=slice_areas,
    )


# =====================================================================
# F8: TMJ Condyle Localization & Sagittal / Coronal Slice Presets
# =====================================================================
@dataclass
class TMJCondyleFrame:
    """F8: Condylar center and oblique sagittal/coronal slice axes for Left or Right TMJ."""

    side: str  # "Right" or "Left"
    condyle_center_ras: np.ndarray
    sagittal_normal_ras: np.ndarray
    sagittal_right_ras: np.ndarray
    coronal_normal_ras: np.ndarray
    coronal_right_ras: np.ndarray


def compute_tmj_condyle_frames(
    volume: Optional[CBCTVolume] = None,
    mask: Optional[np.ndarray] = None,
    right_condyle_ras: Optional[Sequence[float]] = None,
    left_condyle_ras: Optional[Sequence[float]] = None,
    condylar_angle_deg: float = 15.0,
) -> Dict[str, TMJCondyleFrame]:
    """F8: Localize Left and Right mandibular condyles (from segmentation or landmarks) and build TMJ slice frames."""
    r_pt: Optional[np.ndarray] = (
        np.asarray(right_condyle_ras, dtype=np.float64) if right_condyle_ras is not None else None
    )
    l_pt: Optional[np.ndarray] = (
        np.asarray(left_condyle_ras, dtype=np.float64) if left_condyle_ras is not None else None
    )

    if (r_pt is None or l_pt is None) and volume is not None and mask is not None:
        # Find posterior-superior extremes of mandible mask (label 2 in 5-class or LOWER_JAW_ID=1)
        for jaw_id in (2, LOWER_JAW_ID):
            zyx = np.where(mask == jaw_id)
            if len(zyx[0]) >= 20:
                step = max(1, len(zyx[0]) // 2000)
                vox = np.column_stack([zyx[2][::step], zyx[1][::step], zyx[0][::step]])
                w_pts = volume.voxel_to_world(vox)
                # Posterior-superior subset: top 25% Z and posterior 30% Y
                z_th = float(np.percentile(w_pts[:, 2], 75))
                y_th = float(np.percentile(w_pts[:, 1], 35))
                post_sup = w_pts[(w_pts[:, 2] >= z_th) & (w_pts[:, 1] <= y_th)]
                if len(post_sup) >= 4:
                    right_sub = post_sup[post_sup[:, 0] >= 0.0]
                    left_sub = post_sup[post_pts_left(post_sup)]
                    if r_pt is None and len(right_sub) > 0:
                        r_pt = np.mean(right_sub, axis=0)
                    if l_pt is None and len(left_sub) > 0:
                        l_pt = np.mean(left_sub, axis=0)
                break

    if r_pt is None:
        r_pt = np.array([48.0, -35.0, 20.0], dtype=np.float64)
    if l_pt is None:
        l_pt = np.array([-48.0, -35.0, 20.0], dtype=np.float64)

    frames: Dict[str, TMJCondyleFrame] = {}
    theta = math.radians(float(condylar_angle_deg))
    for side, pt, sign in (("Right", r_pt, 1.0), ("Left", l_pt, -1.0)):
        # Condylar long axis points medially and slightly posteriorly by condylar_angle_deg
        sag_normal = np.array(
            [sign * math.cos(theta), math.sin(theta), 0.0], dtype=np.float64
        )
        sag_right = np.array(
            [-math.sin(theta), sign * math.cos(theta), 0.0], dtype=np.float64
        )
        cor_normal = sag_right.copy()
        cor_right = -sag_normal.copy()
        frames[side] = TMJCondyleFrame(
            side=side,
            condyle_center_ras=pt,
            sagittal_normal_ras=sag_normal,
            sagittal_right_ras=sag_right,
            coronal_normal_ras=cor_normal,
            coronal_right_ras=cor_right,
        )
    return frames


def post_pts_left(pts: np.ndarray) -> np.ndarray:
    return pts[:, 0] < 0.0


# =====================================================================
# F4: Segmentation Outlines & FDI Centroid Overlays
# =====================================================================
def create_or_update_fdi_segmentation_overlays(
    volumeNode: Any,
    segmentationNode: Any,
    teeth_centroids_ras: Dict[str, np.ndarray],
    parameterNode: Any = None,
    visible: bool = True,
) -> Any:
    """F4: Enable 2D segmentation outline overlays and create `#FDI` centroid fiducial labels."""
    if slicer is None:
        return None
    if segmentationNode is not None and segmentationNode.GetDisplayNode():
        segDisp = segmentationNode.GetDisplayNode()
        was_mod = segDisp.StartModify() if hasattr(segDisp, "StartModify") else None
        try:
            segDisp.SetVisibility2DOutline(bool(visible))
            segDisp.SetVisibility2DFill(False)
            if hasattr(segDisp, "SetSliceIntersectionThickness"):
                segDisp.SetSliceIntersectionThickness(2)
        finally:
            if was_mod is not None and hasattr(segDisp, "EndModify"):
                segDisp.EndModify(was_mod)

    if not teeth_centroids_ras:
        return None

    fidNode = None
    if parameterNode is not None and hasattr(parameterNode, "GetNodeReference"):
        fidNode = parameterNode.GetNodeReference("FDIOverlayFiducials")
    if fidNode is None:
        vol_name = volumeNode.GetName() if volumeNode else "CBCT"
        fid_name = f"{vol_name}_FDI_Overlay_Fiducials"
        fidNode = slicer.mrmlScene.GetFirstNodeByName(fid_name)
        if not fidNode:
            fidNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", fid_name)
            fidNode.CreateDefaultDisplayNodes()
        if parameterNode is not None and hasattr(parameterNode, "SetNodeReferenceID"):
            parameterNode.SetNodeReferenceID("FDIOverlayFiducials", fidNode.GetID())

    was_mod = fidNode.StartModify() if hasattr(fidNode, "StartModify") else None
    try:
        fidNode.RemoveAllControlPoints()
        for fdi_str, pt in sorted(teeth_centroids_ras.items()):
            idx = fidNode.AddControlPoint(float(pt[0]), float(pt[1]), float(pt[2]))
            fidNode.SetNthControlPointLabel(idx, f"#{fdi_str}")
        if hasattr(fidNode, "SetLocked"):
            fidNode.SetLocked(True)
    finally:
        if was_mod is not None and hasattr(fidNode, "EndModify"):
            fidNode.EndModify(was_mod)

    if fidNode.GetDisplayNode():
        fd = fidNode.GetDisplayNode()
        fd.SetVisibility(bool(visible))
        fd.SetVisibility2D(bool(visible))
        fd.SetVisibility3D(bool(visible))
        if hasattr(fd, "SetPointLabelsVisibility"):
            fd.SetPointLabelsVisibility(bool(visible))
    return fidNode


# =====================================================================
# F9: 3D Angle & Surface-to-Surface Distance Tools
# =====================================================================
def compute_angle_between_three_points_deg(
    p0_ras: Sequence[float],
    vertex_ras: Sequence[float],
    p2_ras: Sequence[float],
) -> float:
    """F9: Compute the 3D angle in degrees at `vertex_ras` formed by rays `vertex->p0` and `vertex->p2`."""
    v0 = np.asarray(p0_ras, dtype=np.float64).reshape(3) - np.asarray(vertex_ras, dtype=np.float64).reshape(3)
    v2 = np.asarray(p2_ras, dtype=np.float64).reshape(3) - np.asarray(vertex_ras, dtype=np.float64).reshape(3)
    n0 = float(np.linalg.norm(v0))
    n2 = float(np.linalg.norm(v2))
    if n0 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_val = float(np.clip(np.dot(v0 / n0, v2 / n2), -1.0, 1.0))
    return round(math.degrees(math.acos(cos_val)), 2)


def compute_surface_to_surface_distance_mm(
    points_a_ras: np.ndarray,
    points_b_ras: np.ndarray,
    radius_a_mm: float = 0.0,
    radius_b_mm: float = 0.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """F9: Compute minimum Euclidean surface-to-surface distance and closest point pair between two 3D surfaces."""
    pts_a = np.atleast_2d(np.asarray(points_a_ras, dtype=np.float64))
    pts_b = np.atleast_2d(np.asarray(points_b_ras, dtype=np.float64))
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("inf"), np.zeros(3), np.zeros(3)
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    flat_idx = int(np.argmin(dists))
    ia, ib = np.unravel_index(flat_idx, dists.shape)
    raw_dist = float(dists[ia, ib])
    surf_dist = max(0.0, raw_dist - float(radius_a_mm) - float(radius_b_mm))
    return round(surf_dist, 3), pts_a[ia].copy(), pts_b[ib].copy()


# =====================================================================
# B21: RenderWindow Slice Screenshot Capture
# =====================================================================
def capture_slice_view_png_base64(slice_name: str, slice_widget: Any = None) -> Optional[str]:
    """B21: Capture a slice view using `vtkWindowToImageFilter` on `sliceView().renderWindow()` (with ScreenCapture fallback)."""
    if slice_widget is None and slicer is not None and getattr(slicer, "app", None) is not None:
        lm = slicer.app.layoutManager()
        if lm:
            slice_widget = lm.sliceWidget(slice_name)
    if slice_widget is None:
        return None

    # 1. Preferred: vtkWindowToImageFilter + vtkPNGWriter on renderWindow()
    if vtk is not None and hasattr(slice_widget, "sliceView"):
        sv = slice_widget.sliceView()
        rw = sv.renderWindow() if (sv and hasattr(sv, "renderWindow")) else None
        if rw is not None:
            try:
                if hasattr(sv, "forceRender"):
                    sv.forceRender()
                elif hasattr(rw, "Render"):
                    rw.Render()
                w2i = vtk.vtkWindowToImageFilter()
                w2i.SetInput(rw)
                if hasattr(w2i, "SetInputBufferTypeToRGB"):
                    w2i.SetInputBufferTypeToRGB()
                w2i.ReadFrontBufferOff()
                w2i.Update()

                writer = vtk.vtkPNGWriter()
                writer.WriteToMemoryOn()
                writer.SetInputConnection(w2i.GetOutputPort())
                writer.Write()
                result = writer.GetResult()
                if result is not None and result.GetNumberOfTuples() > 0:
                    try:
                        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

                        raw_bytes = bytes(vtk_to_numpy(result))
                    except Exception:
                        raw_bytes = bytes(
                            result.GetValue(i) for i in range(result.GetNumberOfValues())
                        )
                    if len(raw_bytes) > 16:
                        b64 = base64.b64encode(raw_bytes).decode("ascii")
                        return f"data:image/png;base64,{b64}"
            except Exception:
                pass

    # 2. Fallback: QPixmap grab if renderWindow memory writer is unavailable in mock harness
    if qt is not None and hasattr(slice_widget, "sliceView"):
        try:
            pixmap = slice_widget.sliceView().grab()
            ba = qt.QByteArray()
            buf = qt.QBuffer(ba)
            buf.open(qt.QIODevice.WriteOnly)
            pixmap.save(buf, "PNG")
            raw = bytes(ba.data())
            if len(raw) > 16:
                b64 = base64.b64encode(raw).decode("ascii")
                return f"data:image/png;base64,{b64}"
        except Exception:
            pass
    return None


# =====================================================================
# B22 & F10: Clinical PDF Report Generator (QPrinter + Pure-Python PDF 1.4 Fallback)
# =====================================================================
def _strip_html_to_lines(html_text: str) -> List[str]:
    text = re.sub(r"<style[^>]*>.*?</style>", "", html_text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</(tr|p|h1|h2|h3|div|li)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<(td|th)[^>]*>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    for ent, ch in (("&mdash;", "-"), ("&ndash;", "-"), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"')):
        text = text.replace(ent, ch)
    lines: List[str] = []
    for raw_line in text.splitlines():
        clean = " ".join(raw_line.strip().split())
        if clean:
            # Keep ASCII printable for Type1 Helvetica PDF stream
            ascii_line = clean.encode("ascii", errors="replace").decode("ascii")
            lines.append(ascii_line[:105])
    return lines


def _write_minimal_pdf(lines: Sequence[str], out_pdf_path: Path) -> Path:
    """Write a valid standards-compliant PDF 1.4 document with Helvetica text lines (headless fallback)."""
    escaped_ops: List[str] = ["BT", "/F1 9 Tf", "36 756 Td", "12 TL"]
    for idx, line in enumerate(lines[:58]):
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if idx == 0:
            escaped_ops.append(f"({safe}) Tj")
        else:
            escaped_ops.append(f"T* ({safe}) Tj")
    escaped_ops.append("ET")
    stream_bytes = ("\n".join(escaped_ops) + "\n").encode("ascii")

    objs: List[bytes] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
        + stream_bytes
        + b"endstream\nendobj\n",
    ]

    header = b"%PDF-1.4\n"
    body = bytearray(header)
    offsets: List[int] = []
    for obj in objs:
        offsets.append(len(body))
        body.extend(obj)

    xref_offset = len(body)
    xref_lines = [f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"]
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n \n")
    trailer = (
        "".join(xref_lines)
        + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    body.extend(trailer)
    Path(out_pdf_path).write_bytes(bytes(body))
    return Path(out_pdf_path)


def export_html_to_pdf(html_content: str, out_pdf_path: Path) -> Path:
    """B22 & F10: Render HTML report to PDF using `QPrinter`/`QTextDocument` when Qt is active, or PDF 1.4 fallback."""
    out_path = Path(out_pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if qt is not None and hasattr(qt, "QPrinter") and hasattr(qt, "QTextDocument"):
        try:
            printer = qt.QPrinter(qt.QPrinter.HighResolution)
            printer.setOutputFormat(qt.QPrinter.PdfFormat)
            printer.setOutputFileName(str(out_path))
            doc = qt.QTextDocument()
            doc.setHtml(html_content)
            doc.print_(printer)
            if out_path.exists() and out_path.stat().st_size > 100:
                return out_path
        except Exception:
            pass
    lines = _strip_html_to_lines(html_content)
    return _write_minimal_pdf(lines, out_path)


# =====================================================================
# F11: Reformatted DICOM Series Export
# =====================================================================
def export_reformatted_dicom_series(
    image_array: np.ndarray,
    spacing_mm: Sequence[float],
    out_dir: Path,
    series_description: str = "CBCTPlanner_Panoramic",
    patient_name: str = "Anonymous",
    patient_id: str = "Anonymous",
) -> List[Path]:
    """F11: Export a 2D or 3D reformatted array (e.g. panoramic or cross-section lightbox) as DICOM slices via pydicom."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image_array)
    if arr.ndim == 2:
        arr = arr[None, :, :]

    try:
        pydicom = ensure_optional_package("pydicom", "pydicom", "DICOM export")
        from pydicom.dataset import FileDataset, FileMetaDataset  # type: ignore
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid  # type: ignore

        study_uid = generate_uid()
        series_uid = generate_uid()
        written: List[Path] = []
        for k in range(arr.shape[0]):
            slice_2d = np.rint(arr[k]).astype(np.int16)
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            dcm_file = out_path / f"slice_{k:04d}.dcm"
            ds = FileDataset(str(dcm_file), {}, file_meta=file_meta, preamble=b"\0" * 128)
            ds.PatientName = str(patient_name)
            ds.PatientID = str(patient_id)
            ds.Modality = "OT"
            ds.SeriesDescription = str(series_description)
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.SOPClassUID = SecondaryCaptureImageStorage
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.InstanceNumber = k + 1
            ds.Rows, ds.Columns = int(slice_2d.shape[0]), int(slice_2d.shape[1])
            ds.PixelSpacing = [float(spacing_mm[0]), float(spacing_mm[1])]
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 1
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelData = slice_2d.tobytes()
            ds.save_as(str(dcm_file), write_like_original=False)
            written.append(dcm_file)
        return written
    except Exception:
        # Fallback: write raw metadata + numpy `.npy` when pydicom is not installed
        fallback_path = out_path / f"{series_description}.npy"
        np.save(str(fallback_path), arr)
        return [fallback_path]


def run_nnunet_toothfairy3_inference(
    volumeNode,
    model_dir_or_ckpt: str,
    device_mode: str = "auto",
    fold: str = "0",
    step_size: float = 0.5,
    progress_callback: Optional[Callable[[int, str], None]] = None,
):
    """Run ToothFairy3 nnU-Net inference with CPU INT8 dynamic quantization or GPU FP32."""
    import json
    import shutil
    import tempfile
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from .core.labels import LABELS_BY_ID, match_segment_to_label_and_precedence, remap_toothfairy3_array

    def _report(pct: int, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(pct, msg)
        if slicer is not None and hasattr(slicer, "app"):
            slicer.app.processEvents()

    if volumeNode is None:
        raise ValueError("Input volumeNode is required for nnU-Net inference.")
    if not model_dir_or_ckpt:
        raise ValueError("Model folder or checkpoint path is required.")

    raw_path = Path(model_dir_or_ckpt).expanduser()
    checkpoint_name = "checkpoint_final.pth"
    model_folder = raw_path
    if raw_path.is_file():
        checkpoint_name = raw_path.name
        if raw_path.parent.name.startswith("fold_"):
            model_folder = raw_path.parent.parent
            if not fold:
                fold = raw_path.parent.name.replace("fold_", "")
        else:
            model_folder = raw_path.parent

    if not model_folder.exists():
        raise FileNotFoundError(f"nnU-Net model folder not found: {model_folder}")

    dataset_labels = None
    dataset_json_path = model_folder / "dataset.json"
    if dataset_json_path.exists():
        try:
            with open(dataset_json_path, "r", encoding="utf-8") as fh:
                dataset_labels = json.load(fh).get("labels")
        except Exception:
            dataset_labels = None

    mode = (device_mode or "auto").strip().lower()
    if mode in ("gpu_fp32", "gpu", "cuda", "fp32"):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU FP32 mode requested, but CUDA is not available.")
        use_gpu = True
    elif mode in ("cpu_int8", "cpu", "int8"):
        use_gpu = False
    else:
        use_gpu = bool(torch.cuda.is_available())

    fold_str = str(fold).strip()
    if fold_str.lower() == "all":
        fold_dirs = sorted(
            p.name.replace("fold_", "")
            for p in model_folder.glob("fold_*")
            if p.is_dir()
        )
        use_folds = tuple(int(f) if f.isdigit() else f for f in fold_dirs) if fold_dirs else "all"
    else:
        use_folds = (int(fold_str) if fold_str.isdigit() else fold_str,)

    _report(10, "Initializing ToothFairy3 nnU-Net predictor...")
    if not use_gpu:
        device = torch.device("cpu")
        supported = getattr(torch.backends.quantized, "supported_engines", [])
        if "fbgemm" in supported:
            torch.backends.quantized.engine = "fbgemm"
        elif "qnnpack" in supported:
            torch.backends.quantized.engine = "qnnpack"
        perform_on_device = False
    else:
        device = torch.device("cuda")
        perform_on_device = True

    predictor = nnUNetPredictor(
        tile_step_size=float(step_size),
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=perform_on_device,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=use_folds,
        checkpoint_name=checkpoint_name,
    )

    if not use_gpu:
        _report(25, "Applying CPU INT8 dynamic quantization...")
        predictor.network.eval()
        try:
            predictor.network = torch.ao.quantization.quantize_dynamic(
                predictor.network,
                {torch.nn.Linear, torch.nn.Conv3d, torch.nn.ConvTranspose3d},
                dtype=torch.qint8,
            )
        except Exception:
            predictor.network = torch.ao.quantization.quantize_dynamic(
                predictor.network,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
    else:
        _report(25, "Preparing GPU FP32 full-precision network...")
        predictor.network = predictor.network.float()
        predictor.network.eval()

    tmp_dir = Path(tempfile.mkdtemp(prefix="cbct_tf3_"))
    pred_label_node = None
    try:
        in_dir = tmp_dir / "in"
        out_dir = tmp_dir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        _report(40, "Exporting CBCT volume with NIfTI RAS/LPS header...")
        in_file = in_dir / "case_0000.nii.gz"
        if not slicer.util.saveNode(volumeNode, str(in_file)):
            raise RuntimeError("Failed to export volumeNode to temporary NIfTI file.")

        _report(55, "Running ToothFairy3 nnU-Net inference...")
        if use_gpu:
            with torch.cuda.amp.autocast(enabled=False):
                predictor.predict_from_files(
                    str(in_dir),
                    str(out_dir),
                    save_probabilities=False,
                    overwrite=True,
                    num_processes_preprocessing=1,
                    num_processes_segmentation_export=1,
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0,
                )
        else:
            predictor.predict_from_files(
                str(in_dir),
                str(out_dir),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0,
            )

        pred_files = sorted(out_dir.glob("*.nii.gz")) + sorted(out_dir.glob("*.nii"))
        if not pred_files:
            raise RuntimeError("nnU-Net inference completed but produced no output mask.")

        _report(80, "Remapping ToothFairy3 labels to canonical IDs...")
        pred_label_node = slicer.util.loadLabelVolume(str(pred_files[0]))
        if pred_label_node is None:
            raise RuntimeError(f"Failed to load predicted mask: {pred_files[0]}")

        mask_arr = slicer.util.arrayFromVolume(pred_label_node)
        remapped_arr = remap_toothfairy3_array(mask_arr, dataset_labels=dataset_labels)
        slicer.util.updateVolumeFromArray(pred_label_node, remapped_arr.astype(np.int32))

        _report(90, "Populating ToothFairy3 segmentation node...")
        seg_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            f"{volumeNode.GetName()}_ToothFairy3",
        )
        seg_node.CreateDefaultDisplayNodes()
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(pred_label_node, seg_node)

        segmentation = seg_node.GetSegmentation()
        for idx in range(segmentation.GetNumberOfSegments()):
            segment = segmentation.GetNthSegment(idx)
            seg_id = segmentation.GetNthSegmentID(idx)
            seg_name = segment.GetName()
            label_id, category, _, _ = match_segment_to_label_and_precedence(seg_name, seg_id)
            info = LABELS_BY_ID.get(label_id)
            if info is not None:
                segment.SetName(info.name)
                r, g, b = (
                    float(c) / 255.0 if max(info.color) > 1.0 else float(c)
                    for c in info.color
                )
                segment.SetColor(r, g, b)
                segment.SetTag("CBCTPlanner.LabelID", str(info.id))
                segment.SetTag("CBCTPlanner.Category", info.category)
                if info.category == "tooth" or (11 <= info.id <= 48):
                    segment.SetTag("CBCTPlanner.FDI", str(info.id))

        _report(100, "ToothFairy3 nnU-Net segmentation complete.")
        return seg_node
    finally:
        if pred_label_node is not None and slicer is not None:
            try:
                slicer.mrmlScene.RemoveNode(pred_label_node)
            except Exception:
                pass
        shutil.rmtree(tmp_dir, ignore_errors=True)

