"""Extracted workflow UI helpers for CBCTPlannerWidget & CBCTPlannerLogic.

Provides headless-testable, self-contained state management for:
  - Clinical workspace role layouts (Radiologist, Implant/Surgical Dentist, Ortho, Full Suite)
  - Step 5 cross-section gallery single-active-slice navigation, lazy full-gallery generation,
    and stale-state tracking
  - RISE-MAR model-weights QSettings persistence and validation
  - Robust volume/segmentation/transform cache-token computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .RiseMarEngine import validate_risemar_weights_path

RISEMAR_QSETTINGS_KEY = "CBCTPlanner/RiseMarWeightsPath"


@dataclass(frozen=True)
class WorkspaceRoleProfile:
    """Declarative UI layout profile for one clinical workspace role."""

    key: str
    status_text: str
    visible_boxes: Tuple[str, ...]
    expanded_boxes: Tuple[str, ...]
    collapsed_boxes: Tuple[str, ...]


WORKSPACE_ROLE_PROFILES: Dict[str, WorkspaceRoleProfile] = {
    "full": WorkspaceRoleProfile(
        key="full",
        status_text="Workspace: Full Clinical Suite (All Steps Visible)",
        visible_boxes=(
            "step1Box",
            "step2Box",
            "step3Box",
            "step4Box",
            "step5Box",
            "step6Box",
            "step7Box",
            "risemarBox",
        ),
        expanded_boxes=("step1Box", "step2Box", "step3Box", "step4Box", "step5Box"),
        collapsed_boxes=("step6Box", "step7Box", "advancedStep5Box"),
    ),
    "radiologist": WorkspaceRoleProfile(
        key="radiologist",
        status_text="Workspace: Radiologist (RISE-MAR, Panoramic, Cross-Sections & Film Print)",
        visible_boxes=(
            "step1Box",
            "step2Box",
            "step3Box",
            "step4Box",
            "step5Box",
            "step6Box",
            "step7Box",
            "risemarBox",
        ),
        expanded_boxes=("step1Box", "risemarBox", "step3Box", "step4Box", "step5Box"),
        collapsed_boxes=("step2Box", "step6Box", "step7Box", "advancedStep5Box"),
    ),
    "implant": WorkspaceRoleProfile(
        key="implant",
        status_text="Workspace: Implant & Surgical Dentist (Arch, Measurements, Implants & Guide)",
        visible_boxes=(
            "step1Box",
            "step2Box",
            "step3Box",
            "step4Box",
            "step5Box",
            "step6Box",
            "step7Box",
            "risemarBox",
        ),
        expanded_boxes=("step1Box", "step2Box", "step5Box", "advancedStep5Box", "step6Box"),
        collapsed_boxes=("risemarBox", "step3Box", "step4Box", "step7Box"),
    ),
    "ortho": WorkspaceRoleProfile(
        key="ortho",
        status_text="Workspace: Ortho & Craniofacial (ALI Landmarks, ASO, CLI-C, VFACE, Airway & Ceph)",
        visible_boxes=(
            "step1Box",
            "step2Box",
            "step3Box",
            "step4Box",
            "step5Box",
            "step6Box",
            "step7Box",
            "risemarBox",
        ),
        expanded_boxes=("step1Box", "step6Box", "step7Box"),
        collapsed_boxes=("risemarBox", "step2Box", "step3Box", "step4Box", "step5Box", "advancedStep5Box"),
    ),
}


def resolve_role_key(mode_text: str) -> str:
    """Map a UI combo-box role string to its canonical profile key."""
    low = str(mode_text or "").lower()
    if "radiologist" in low:
        return "radiologist"
    if "implant" in low or "dentist" in low or "surgical" in low:
        return "implant"
    if "ortho" in low or "craniofacial" in low:
        return "ortho"
    return "full"


def apply_workspace_role(widget: Any, mode_text: str) -> WorkspaceRoleProfile:
    """Apply a workspace role profile without erasing selections, measurements, or essential controls."""
    profile = WORKSPACE_ROLE_PROFILES[resolve_role_key(mode_text)]
    for attr in profile.visible_boxes:
        box = getattr(widget, attr, None)
        if box is not None and hasattr(box, "setVisible"):
            box.setVisible(True)
    for attr in profile.expanded_boxes:
        box = getattr(widget, attr, None)
        if box is not None and hasattr(box, "collapsed"):
            box.collapsed = False
    for attr in profile.collapsed_boxes:
        box = getattr(widget, attr, None)
        if box is not None and hasattr(box, "collapsed"):
            box.collapsed = True
    status_lbl = getattr(widget, "globalStatusLabel", None)
    if status_lbl is not None and hasattr(status_lbl, "setText"):
        status_lbl.setText(profile.status_text)
    return profile


@dataclass
class GalleryNavigationState:
    """State controller for the Step 5 single-active-slice viewer and lazy thumbnail gallery."""

    items: List[Any] = field(default_factory=list)
    active_index: int = 0
    full_gallery_loaded: bool = False
    is_stale: bool = False
    stale_reason: str = ""

    def set_items(self, items: Sequence[Any], *, full_gallery: bool = False) -> None:
        self.items = list(items)
        self.full_gallery_loaded = bool(full_gallery)
        self.is_stale = False
        self.stale_reason = ""
        if not self.items:
            self.active_index = 0
            return
        # Default to center slice (offset ~ 0.0 mm) when available
        center_idx = len(self.items) // 2
        best_abs = float("inf")
        for idx, item in enumerate(self.items):
            off = item.get("offset_mm", 0.0) if isinstance(item, dict) else getattr(item, "offset_mm", 0.0)
            if abs(float(off)) < best_abs:
                best_abs = abs(float(off))
                center_idx = idx
        self.active_index = center_idx

    def mark_stale(self, reason: str = "Input data changed") -> None:
        if self.items:
            self.is_stale = True
            self.stale_reason = str(reason)

    def step(self, delta: int) -> int:
        if not self.items:
            self.active_index = 0
            return 0
        self.active_index = max(0, min(len(self.items) - 1, self.active_index + int(delta)))
        return self.active_index

    def select(self, index: int) -> int:
        if not self.items:
            self.active_index = 0
            return 0
        self.active_index = max(0, min(len(self.items) - 1, int(index)))
        return self.active_index

    @property
    def active_item(self) -> Optional[Any]:
        if not self.items or not (0 <= self.active_index < len(self.items)):
            return None
        return self.items[self.active_index]

    @property
    def can_prev(self) -> bool:
        return bool(self.items) and self.active_index > 0

    @property
    def can_next(self) -> bool:
        return bool(self.items) and self.active_index < len(self.items) - 1


def load_risemar_weights_setting(qt_module: Any) -> str:
    """Read persisted RISE-MAR weights path from Slicer QSettings (empty string if unset)."""
    if qt_module is None or not hasattr(qt_module, "QSettings"):
        return ""
    try:
        val = qt_module.QSettings().value(RISEMAR_QSETTINGS_KEY, "")
        return str(val or "").strip()
    except Exception:
        return ""


def save_risemar_weights_setting(qt_module: Any, path: str) -> None:
    """Persist RISE-MAR weights path in Slicer QSettings."""
    if qt_module is None or not hasattr(qt_module, "QSettings"):
        return
    try:
        qt_module.QSettings().setValue(RISEMAR_QSETTINGS_KEY, str(path or "").strip())
    except Exception:
        pass


def apply_risemar_validation_to_widget(widget: Any, weights_path: str) -> Tuple[bool, str]:
    """Validate RISE-MAR weights and update button enabled states and status label gracefully."""
    ok, msg = validate_risemar_weights_path(weights_path)
    for btn_attr in ("btnRunRiseMar", "btnQuickRiseMar"):
        btn = getattr(widget, btn_attr, None)
        if btn is not None and hasattr(btn, "setEnabled"):
            btn.setEnabled(bool(ok))
    lbl = getattr(widget, "risemarStatusLabel", None)
    if lbl is not None and hasattr(lbl, "setText"):
        lbl.setText(msg)
    return ok, msg


def compute_volume_segmentation_cache_token(volumeNode: Any, segmentationNode: Any) -> Tuple[Any, ...]:
    """Compute a modification-sensitive cache token covering volume, segmentation, geometry, and transforms.

    Ensures editing a segment, replacing image data, modifying IJKToRAS geometry, or applying a
    parent transform ALWAYS invalidates cached masks and measurements.
    """
    parts: List[Any] = []
    for node in (volumeNode, segmentationNode):
        if node is None:
            parts.append(None)
            continue
        node_id = node.GetID() if hasattr(node, "GetID") else str(id(node))
        node_mtime = int(node.GetMTime()) if hasattr(node, "GetMTime") else 0
        parts.append((node_id, node_mtime))

        # Track underlying VTK image data mtime (volumes)
        if hasattr(node, "GetImageData"):
            img_data = node.GetImageData()
            parts.append(
                int(img_data.GetMTime()) if (img_data is not None and hasattr(img_data, "GetMTime")) else 0
            )
        else:
            parts.append(0)

        # Track underlying vtkSegmentation mtime & segment count/mtimes (segmentations)
        if hasattr(node, "GetSegmentation"):
            seg = node.GetSegmentation()
            if seg is not None:
                seg_mtime = int(seg.GetMTime()) if hasattr(seg, "GetMTime") else 0
                n_segs = int(seg.GetNumberOfSegments()) if hasattr(seg, "GetNumberOfSegments") else 0
                parts.append((seg_mtime, n_segs))
            else:
                parts.append((0, 0))
        else:
            parts.append((0, 0))

        # Track parent transform node ID and mtime
        if hasattr(node, "GetParentTransformNode"):
            t_node = node.GetParentTransformNode()
            if t_node is not None:
                t_id = t_node.GetID() if hasattr(t_node, "GetID") else str(id(t_node))
                t_mtime = int(t_node.GetMTime()) if hasattr(t_node, "GetMTime") else 0
                parts.append((t_id, t_mtime))
            else:
                parts.append(None)
        else:
            parts.append(None)

        # Track volume geometry (origin/spacing/IJKToRAS) when available
        if hasattr(node, "GetOrigin") and hasattr(node, "GetSpacing"):
            try:
                orig = tuple(round(float(v), 5) for v in node.GetOrigin())
                spac = tuple(round(float(v), 5) for v in node.GetSpacing())
                parts.append((orig, spac))
            except Exception:
                parts.append(None)
        else:
            parts.append(None)

    return tuple(parts)
