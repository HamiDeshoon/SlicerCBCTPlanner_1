"""Clinical delivery diagnostics: preflight checklist, calibration guard, span sizing,
measurement provenance hashing and a plain-text report.

Pure Python/NumPy so the same functions run headlessly in tests and inside 3D Slicer.
Nothing here measures anatomy; it only reports what the measurement engine produced.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .labels import get_tooth_label, is_mandibular_fdi
from .measurements import PROVENANCE_ESTIMATED, PROVENANCE_INTERPOLATED

# CBCT reconstructions are sub-millimetre; a spacing outside this window means the geometry
# was lost (or is not millimetres) and every printed distance would be wrong.
MIN_PLAUSIBLE_SPACING_MM = 0.05
MAX_PLAUSIBLE_SPACING_MM = 1.5


def calibration_status(volume: Any) -> Tuple[str, str]:
    """Return ``(status, detail)`` for a CBCT volume: ``"ok"`` or ``"unverified"``."""
    try:
        spacing = tuple(float(v) for v in getattr(volume, "spacing", ()))
    except (TypeError, ValueError):
        return "unverified", "voxel spacing unavailable"
    if len(spacing) != 3 or any(not math.isfinite(v) or v <= 0.0 for v in spacing):
        return "unverified", "voxel spacing unavailable"
    if max(spacing) > MAX_PLAUSIBLE_SPACING_MM or min(spacing) < MIN_PLAUSIBLE_SPACING_MM:
        return "unverified", "voxel spacing " + " x ".join(f"{v:g}" for v in spacing) + " mm is implausible"
    return "ok", "voxel spacing " + " x ".join(f"{v:g}" for v in spacing) + " mm"


def calibration_stamp(volume: Any) -> str:
    """One-line film stamp, e.g. ``SCALE OK 0.3 x 0.3 x 0.3 mm``."""
    status, detail = calibration_status(volume)
    return f"SCALE {'OK' if status == 'ok' else 'UNVERIFIED'} - {detail}"


def summarize_span_worst_case(measurements: Iterable[Any]) -> Dict[str, Any]:
    """Worst-case sizing across every measured site of a span.

    A single restoration spanning several sites is limited by the narrowest / shortest
    site, so the summary reports the minimum values and which tooth limits them.
    """
    heights: List[Tuple[float, str, bool]] = []
    crest_widths: List[Tuple[float, str]] = []
    diameters: List[Tuple[float, str]] = []
    lengths: List[Tuple[float, str]] = []
    sites = 0
    withheld: List[Tuple[str, str]] = []
    for meas in measurements or ():
        if meas is None:
            continue
        sites += 1
        fdi = str(getattr(meas, "tooth_fdi", ""))
        height = getattr(meas, "bone_height_mm", None)
        crest = getattr(meas, "crest_width_mm", None)
        diameter = getattr(meas, "suggested_diameter_mm", None)
        length = getattr(meas, "suggested_length_mm", None)
        if height is not None:
            heights.append(
                (
                    float(height),
                    fdi,
                    str(getattr(meas, "provenance", "")) in (PROVENANCE_ESTIMATED, PROVENANCE_INTERPOLATED),
                )
            )
        if crest is not None:
            crest_widths.append((float(crest), fdi))
        if diameter is not None:
            diameters.append((float(diameter), fdi))
        if length is not None:
            lengths.append((float(length), fdi))
        reason = getattr(meas, "withheld_reason", None)
        if reason:
            withheld.append((fdi, str(reason)))
    limiting_height = min(heights) if heights else (None, None, False)
    limiting_width = min(crest_widths) if crest_widths else (None, None)
    limiting_diameter = min(diameters) if diameters else (None, None)
    limiting_length = min(lengths) if lengths else (None, None)
    return {
        "sites": sites,
        "min_height_mm": limiting_height[0],
        "limiting_height_site": limiting_height[1],
        # True when the limiting site's height was interpolated/estimated, not label-exact.
        "min_height_estimated": bool(limiting_height[2]),
        "min_crest_width_mm": limiting_width[0],
        "limiting_width_site": limiting_width[1],
        "max_span_diameter_mm": limiting_diameter[0],
        "limiting_diameter_site": limiting_diameter[1],
        "max_span_length_mm": limiting_length[0],
        "limiting_length_site": limiting_length[1],
        "withheld": withheld,
    }


def _record_line(record: Any) -> str:
    return "|".join(
        [
            str(getattr(record, "measurement_type", "")),
            "" if getattr(record, "value_mm", None) is None else f"{float(record.value_mm):.2f}",
            ""
            if getattr(record, "start_uv_mm", None) is None
            else f"{float(record.start_uv_mm[0]):.3f},{float(record.start_uv_mm[1]):.3f}",
            ""
            if getattr(record, "end_uv_mm", None) is None
            else f"{float(record.end_uv_mm[0]):.3f},{float(record.end_uv_mm[1]):.3f}",
            str(getattr(record, "provenance", "")),
            str(getattr(record, "withheld_reason", "") or ""),
        ]
    )


def provenance_hash(records: Any) -> str:
    """Stable SHA-256 over measurement values, endpoints and provenance.

    ``records`` may be a mapping ``name -> MeasurementRecord`` or an
    ``ImplantSiteMeasurement``. Identical plans always hash identically.
    """
    mapping: Mapping[str, Any] = {}
    if hasattr(records, "measurements"):
        mapping = dict(getattr(records, "measurements", None) or {})
    elif isinstance(records, Mapping):
        mapping = dict(records)
    lines = [f"{name}:{_record_line(mapping[name])}" for name in sorted(mapping)]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return digest[:16]


def measurement_report_text(meas: Any, cross_section: Any = None) -> str:
    """Structured plain-text report for one site (values only, never inferred)."""
    if meas is None:
        reason = getattr(cross_section, "withheld_reason", None) or "not measured"
        return f"site: --\nwithheld: {reason}\n"
    fdi = str(getattr(meas, "tooth_fdi", ""))
    info = get_tooth_label(fdi)
    lines = [
        f"site: #{fdi}" + (f" ({info.name})" if info is not None else ""),
        f"jaw: {'lower' if is_mandibular_fdi(fdi) else 'upper'}",
        f"height_mm: {_fmt(getattr(meas, 'bone_height_mm', None))}",
        f"crest_width_mm: {_fmt(getattr(meas, 'crest_width_mm', None))}",
        f"w3_mm: {_fmt((getattr(meas, 'bone_width_at_depths_mm', None) or {}).get(3.0))}",
        f"w5_mm: {_fmt((getattr(meas, 'bone_width_at_depths_mm', None) or {}).get(5.0))}",
        f"vital_structure: {getattr(meas, 'vital_structure_name', '') or '--'}",
        f"vital_distance_mm: {_fmt(getattr(meas, 'distance_to_vital_structure_mm', None))}",
        f"site_source: {getattr(meas, 'site_position_source', '')}"
        + (" (approximate)" if getattr(meas, "is_approximate_site", False) else ""),
        f"provenance: {getattr(meas, 'provenance', '')} | quality: {getattr(meas, 'quality_status', '')}",
        f"withheld: {getattr(meas, 'withheld_reason', None) or 'none'}",
        f"provenance_hash: {provenance_hash(meas)}",
    ]
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def build_preflight_checklist(engine: Any, fdis: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """Check the minimum prerequisites for an honest film before measuring/exporting."""
    checks: List[Dict[str, Any]] = []
    status, detail = calibration_status(getattr(engine, "volume", None))
    checks.append({"check": "calibration", "ok": status == "ok", "detail": detail})

    mask = getattr(engine, "mask", None)
    try:
        present = sorted(int(v) for v in np.unique(mask) if int(v) > 0)
    except Exception:
        present = []
    checks.append(
        {
            "check": "segmentation",
            "ok": bool(present),
            "detail": f"{len(present)} label(s) present" if present else "empty or missing label mask",
        }
    )

    jaws = {"lower" if is_mandibular_fdi(f) else "upper" for f in (fdis or [])}
    arches = getattr(engine, "arches", None) or {}
    for jaw in sorted(jaws):
        arch = arches.get(jaw) or getattr(engine, "arch", None)
        points = 0
        if arch is not None:
            dense = getattr(arch, "pts_dense", None)
            points = int(getattr(arch, "num_points", 0) or (len(dense) if dense is not None else 0))
        checks.append(
            {
                "check": f"{jaw}_arch",
                "ok": points >= 4,
                "detail": f"{points} curve point(s)" if arch is not None else "no fitted arch",
            }
        )

    for fdi in fdis or []:
        try:
            meas = engine.measure_implant_site(str(int(fdi)))
            reason = getattr(engine, "last_withheld_reason", None)
        except Exception as exc:  # pragma: no cover - defensive
            meas, reason = None, f"measurement raised {type(exc).__name__}"
        ok = meas is not None and getattr(meas, "bone_height_mm", None) is not None
        checks.append(
            {
                "check": f"site_{int(fdi)}",
                "ok": bool(ok),
                "detail": "measured" if meas is not None else str(reason or "not measured"),
            }
        )
    return checks


def checklist_text(checks: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"[{'PASS' if c.get('ok') else 'CHECK'}] {c.get('check')}: {c.get('detail')}" for c in checks]
    failed = sum(1 for c in checks if not c.get("ok"))
    lines.append(f"preflight: {len(checks) - failed}/{len(checks)} passed")
    return "\n".join(lines) + "\n"


def write_diagnostics_text(
    engine: Any,
    path: str | Path,
    fdis: Optional[Sequence[int]] = None,
) -> Path:
    """Write the preflight checklist plus per-site reasons and provenance to ``path``."""
    out = Path(path)
    parts = [
        "CBCT Planner diagnostics\n=======================\n",
        checklist_text(build_preflight_checklist(engine, fdis)),
    ]
    if fdis:
        parts.append("\nsites\n-----\n")
        for fdi in fdis:
            try:
                meas = engine.measure_implant_site(str(int(fdi)))
            except Exception as exc:  # pragma: no cover
                parts.append(f"site: #{int(fdi)}\nwithheld: measurement raised {type(exc).__name__}\n")
                continue
            reason = getattr(engine, "last_withheld_reason", None)
            if meas is None:
                parts.append(f"site: #{int(fdi)}\nwithheld: {reason or 'not measured'}\n")
            else:
                parts.append(measurement_report_text(meas))
            parts.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out


__all__ = [
    "MAX_PLAUSIBLE_SPACING_MM",
    "MIN_PLAUSIBLE_SPACING_MM",
    "build_preflight_checklist",
    "calibration_stamp",
    "calibration_status",
    "checklist_text",
    "measurement_report_text",
    "provenance_hash",
    "summarize_span_worst_case",
    "write_diagnostics_text",
]
