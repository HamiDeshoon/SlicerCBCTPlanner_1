"""Pure case-state & keypoint override serialization helpers (UI/Slicer independent)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Tuple


def serialize_corrected_keypoints_jsonl(overrides_by_fdi: Dict[int, Dict[str, Any]]) -> str:
    """Serialize corrected H/W0 keypoints and custom overlays as newline-delimited JSON."""
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines: List[str] = []
    for fdi in sorted(overrides_by_fdi):
        ov = overrides_by_fdi[fdi] or {}
        record: Dict[str, Any] = {"fdi": int(fdi), "timestamp": ts}
        for key in ("h_mm", "w0_mm"):
            if ov.get(key) is not None:
                record[key] = round(float(ov[key]), 3)
        for key in ("h_start_uv", "h_end_uv", "w0_start_uv", "w0_end_uv"):
            pt = ov.get(key)
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                record[key] = [round(float(pt[0]), 3), round(float(pt[1]), 3)]
        if ov.get("custom_overlays"):
            record["custom_overlays"] = [dict(x) for x in ov["custom_overlays"]]
        lines.append(json.dumps(record, ensure_ascii=False))
    return ("\n".join(lines) + "\n") if lines else ""


def serialize_overrides_payload(overrides_by_fdi: Dict[int, Dict[str, Any]]) -> str:
    """Serialize `_measurementOverridesByFDI` into compact JSON for ParameterNode/sidecar storage."""
    payload: Dict[str, Any] = {}
    for fdi, ov in (overrides_by_fdi or {}).items():
        if not isinstance(ov, dict):
            continue
        entry: Dict[str, Any] = {}
        for k in ("h_mm", "w0_mm"):
            if ov.get(k) is not None:
                entry[k] = float(ov[k])
        for k in ("h_start_uv", "h_end_uv", "w0_start_uv", "w0_end_uv"):
            pt = ov.get(k)
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                entry[k] = [float(pt[0]), float(pt[1])]
        if ov.get("custom_overlays"):
            entry["custom_overlays"] = [dict(x) for x in ov["custom_overlays"]]
        if entry:
            payload[str(int(fdi))] = entry
    return json.dumps(payload, ensure_ascii=False)


def deserialize_overrides_payload(raw_json: str) -> Dict[int, Dict[str, Any]]:
    """Deserialize compact override JSON back into `{fdi_int: override_dict}` safely."""
    raw = str(raw_json or "").strip()
    if not raw or raw == "{}":
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    restored: Dict[int, Dict[str, Any]] = {}
    for fdi_str, ov in parsed.items():
        try:
            fdi_i = int(fdi_str)
        except (TypeError, ValueError):
            continue
        if not isinstance(ov, dict):
            continue
        entry: Dict[str, Any] = {}
        for k in ("h_mm", "w0_mm"):
            if ov.get(k) is not None:
                try:
                    entry[k] = float(ov[k])
                except (TypeError, ValueError):
                    continue
        for k in ("h_start_uv", "h_end_uv", "w0_start_uv", "w0_end_uv"):
            pt = ov.get(k)
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                try:
                    pt_t: Tuple[float, float] = (float(pt[0]), float(pt[1]))
                    entry[k] = pt_t
                except (TypeError, ValueError):
                    continue
        if isinstance(ov.get("custom_overlays"), list):
            entry["custom_overlays"] = [dict(x) for x in ov["custom_overlays"] if isinstance(x, dict)]
        if entry:
            restored[fdi_i] = entry
    return restored
