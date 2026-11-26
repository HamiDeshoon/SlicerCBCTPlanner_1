"""Shared physical-unit contracts and structured provenance models for CBCTPlanner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

UNIT_MM = "mm"
UNIT_DEG = "deg"
UNIT_HU = "HU"
UNIT_MM3 = "mm3"
UNIT_PERCENT = "%"

# Normalized provenance vocabulary required across clinical & JSON reports
SOURCE_MEASURED = "measured"
SOURCE_MODEL_PREDICTED = "model_predicted"
SOURCE_INTERPOLATED = "interpolated"
SOURCE_ESTIMATED = "estimated"
SOURCE_MANUALLY_CORRECTED = "manually_corrected"
SOURCE_NOT_AVAILABLE = "not_available"

VALID_SOURCE_TYPES = frozenset(
    {
        SOURCE_MEASURED,
        SOURCE_MODEL_PREDICTED,
        SOURCE_INTERPOLATED,
        SOURCE_ESTIMATED,
        SOURCE_MANUALLY_CORRECTED,
        SOURCE_NOT_AVAILABLE,
    }
)


def read_module_version() -> str:
    """Read authoritative software version from CBCTPlanner/Resources/VERSION."""
    version_path = Path(__file__).resolve().parent.parent.parent / "Resources" / "VERSION"
    if version_path.is_file():
        txt = version_path.read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return "0.0.0"


def normalize_provenance(
    raw_provenance: Optional[str],
    *,
    manually_corrected: bool = False,
    value: Any = None,
) -> str:
    """Map internal measurement/model tags onto the canonical provenance taxonomy."""
    if manually_corrected:
        return SOURCE_MANUALLY_CORRECTED
    if value is None:
        return SOURCE_NOT_AVAILABLE
    key = (raw_provenance or "").strip().lower()
    if key in VALID_SOURCE_TYPES:
        return key
    if key in ("label_derived", "label_derived_boundary", "edt_sdf", "graph_skeleton", "exact_capsule"):
        return SOURCE_MEASURED
    if "model" in key or "ai" in key or "checkpoint" in key or "stunet" in key or "nnunet" in key:
        return SOURCE_MODEL_PREDICTED
    if "interpolat" in key:
        return SOURCE_INTERPOLATED
    if "estimat" in key or "approx" in key or "fallback" in key or "intens" in key:
        return SOURCE_ESTIMATED
    if "manual" in key:
        return SOURCE_MANUALLY_CORRECTED
    if "unavail" in key or "withheld" in key or not key:
        return SOURCE_NOT_AVAILABLE if value is None else SOURCE_MEASURED
    return SOURCE_ESTIMATED


@dataclass
class StructuredMeasurementEntry:
    """Explicit unit-bearing, provenance-tracked clinical measurement record."""

    name: str
    value: Optional[float]
    unit: str
    source_type: str
    method: str
    structure: str = ""
    fdi: Optional[str] = None
    location_ras_mm: Optional[Tuple[float, float, float]] = None
    uncertainty_mm: Optional[float] = None
    confidence: Optional[float] = None
    withheld_reason: Optional[str] = None
    caveats: List[str] = field(default_factory=list)
    manual_edit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": None if self.value is None else round(float(self.value), 3),
            "unit": self.unit,
            "source_type": normalize_provenance(
                self.source_type, manually_corrected=self.manual_edit, value=self.value
            ),
            "method": self.method,
            "structure": self.structure,
            "fdi": self.fdi,
            "location_ras_mm": (
                [round(float(v), 3) for v in self.location_ras_mm]
                if self.location_ras_mm is not None
                else None
            ),
            "uncertainty_mm": (
                None if self.uncertainty_mm is None else round(float(self.uncertainty_mm), 3)
            ),
            "confidence": None if self.confidence is None else round(float(self.confidence), 3),
            "withheld_reason": self.withheld_reason,
            "caveats": list(self.caveats),
            "manual_edit": bool(self.manual_edit),
        }


@dataclass
class UncertaintyClearance:
    """Uncertainty-aware geometric clearance between a planned device and an anatomical structure."""

    structure_id: str
    structure_label: str
    nominal_clearance_mm: Optional[float]
    estimated_uncertainty_mm: Optional[float]
    effective_clearance_mm: Optional[float]
    closest_anatomical_point_ras: Optional[Tuple[float, float, float]] = None
    closest_device_point_ras: Optional[Tuple[float, float, float]] = None
    method: str = "capsule_sdf"
    data_quality: str = SOURCE_MEASURED
    withheld_reason: Optional[str] = None
    limitation_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "structure_label": self.structure_label,
            "nominal_clearance_mm": (
                None if self.nominal_clearance_mm is None else round(float(self.nominal_clearance_mm), 3)
            ),
            "estimated_uncertainty_mm": (
                None
                if self.estimated_uncertainty_mm is None
                else round(float(self.estimated_uncertainty_mm), 3)
            ),
            "effective_clearance_mm": (
                None
                if self.effective_clearance_mm is None
                else round(float(self.effective_clearance_mm), 3)
            ),
            "closest_anatomical_point_ras": (
                [round(float(v), 3) for v in self.closest_anatomical_point_ras]
                if self.closest_anatomical_point_ras is not None
                else None
            ),
            "closest_device_point_ras": (
                [round(float(v), 3) for v in self.closest_device_point_ras]
                if self.closest_device_point_ras is not None
                else None
            ),
            "method": self.method,
            "data_quality": self.data_quality,
            "withheld_reason": self.withheld_reason,
            "limitation_note": self.limitation_note,
            "unit": UNIT_MM,
        }


@dataclass
class StructuredReportEnvelope:
    """Top-level provenance-stamped report envelope for clinical & headless JSON export."""

    schema_version: str = "cbctplanner-report/1.0"
    software_version: str = field(default_factory=read_module_version)
    generated_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    measurements: List[StructuredMeasurementEntry] = field(default_factory=list)
    clearances: List[UncertaintyClearance] = field(default_factory=list)
    manual_edit_history: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "software_version": self.software_version,
            "generated_utc": self.generated_utc,
            "measurements": [m.to_dict() for m in self.measurements],
            "clearances": [c.to_dict() for c in self.clearances],
            "manual_edit_history": list(self.manual_edit_history),
            "notes": list(self.notes),
        }
