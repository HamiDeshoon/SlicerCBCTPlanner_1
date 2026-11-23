"""
Anatomical Label Registry for Dental CBCT Segmentation (ToothFairy3 / FDI + DentalSegmentator).
Includes standard adult mesiodistal crown widths (B2), vivid SlicerDentalSegmentator-inspired RGB
colors for every anatomical class and FDI tooth, safe 1-arg/2-arg VTK segment tag reading, and
deterministic segment matching with explicit anatomical precedence (B19: pulp > tooth > canal > sinus > jaw).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class AnatomicalLabel:
    id: int
    name: str
    category: str
    fdi: Optional[str] = None
    quadrant: Optional[int] = None
    tooth_number: Optional[int] = None
    color: Tuple[float, float, float] = (0.95, 0.92, 0.82)


LOWER_JAW_ID = 1
UPPER_JAW_ID = 2
LEFT_IAN_ID = 3
RIGHT_IAN_ID = 4
LEFT_SINUS_ID = 5
RIGHT_SINUS_ID = 6
PHARYNX_ID = 7
BRIDGE_ID = 8
CROWN_ID = 9
IMPLANT_ID = 10
LINGUAL_CANAL_ID = 103
LEFT_INCISIVE_CANAL_ID = 104
RIGHT_INCISIVE_CANAL_ID = 105

# DentalSegmentator 5-class group IDs:
# 1 = Maxilla & Upper Skull, 2 = Mandible, 3 = Upper Teeth, 4 = Lower Teeth, 5 = Mandibular canal
DENTALSEGMENTATOR_MAXILLA_ID = 1
DENTALSEGMENTATOR_MANDIBLE_ID = 2
DENTALSEGMENTATOR_UPPER_TEETH_ID = 3
DENTALSEGMENTATOR_LOWER_TEETH_ID = 4
DENTALSEGMENTATOR_CANAL_ID = 5

# B19: Explicit voxel overwrite precedence (higher number wins when segments overlap)
SEGMENT_PRECEDENCE: Dict[str, int] = {
    "unknown": 0,
    "jaw": 1,
    "sinus": 2,
    "canal": 3,
    "tooth": 4,
    "pulp": 5,
}

# B2: Standard adult anatomical mesiodistal crown widths in mm (Wheeler's Dental Anatomy)
ANATOMICAL_MESIODISTAL_WIDTHS_MM: Dict[str, float] = {
    # Maxillary Right (Q1: 18 -> 11)
    "18": 8.5,
    "17": 9.0,
    "16": 10.0,
    "15": 7.0,
    "14": 7.0,
    "13": 7.5,
    "12": 6.5,
    "11": 8.5,
    # Maxillary Left (Q2: 21 -> 28)
    "21": 8.5,
    "22": 6.5,
    "23": 7.5,
    "24": 7.0,
    "25": 7.0,
    "26": 10.0,
    "27": 9.0,
    "28": 8.5,
    # Mandibular Right (Q4: 48 -> 41)
    "48": 10.0,
    "47": 10.5,
    "46": 11.0,
    "45": 7.0,
    "44": 7.0,
    "43": 7.0,
    "42": 6.0,
    "41": 5.0,
    # Mandibular Left (Q3: 31 -> 38)
    "31": 5.0,
    "32": 6.0,
    "33": 7.0,
    "34": 7.0,
    "35": 7.0,
    "36": 11.0,
    "37": 10.5,
    "38": 10.0,
}

_QUADRANT_NAMES = {
    1: "Upper Right",
    2: "Upper Left",
    3: "Lower Left",
    4: "Lower Right",
}

_TOOTH_TYPE_NAMES = {
    1: "Central Incisor",
    2: "Lateral Incisor",
    3: "Canine",
    4: "First Premolar",
    5: "Second Premolar",
    6: "First Molar",
    7: "Second Molar",
    8: "Third Molar (Wisdom)",
}

# Vivid SlicerDentalSegmentator-inspired tooth palette per quadrant and tooth number
# Makes crowns and roots vivid and easily distinguishable in both 3D and 2D views.
_VIVID_TOOTH_COLORS_BY_QUADRANT: Dict[int, Dict[int, Tuple[float, float, float]]] = {
    # Q1 (Upper Right): Warm Vivid Enamel Gold / Coral Ivory
    1: {
        1: (1.00, 0.88, 0.65),
        2: (0.98, 0.82, 0.56),
        3: (0.96, 0.74, 0.46),
        4: (0.95, 0.68, 0.42),
        5: (0.93, 0.63, 0.38),
        6: (0.91, 0.57, 0.34),
        7: (0.88, 0.52, 0.30),
        8: (0.85, 0.47, 0.27),
    },
    # Q2 (Upper Left): Vivid Amber / Golden Pearlescent
    2: {
        1: (1.00, 0.90, 0.68),
        2: (0.99, 0.85, 0.58),
        3: (0.97, 0.78, 0.48),
        4: (0.96, 0.72, 0.44),
        5: (0.94, 0.66, 0.40),
        6: (0.92, 0.60, 0.36),
        7: (0.89, 0.55, 0.32),
        8: (0.86, 0.50, 0.28),
    },
    # Q3 (Lower Left): Vivid Pearl Ivory / Cream White
    3: {
        1: (0.99, 0.97, 0.86),
        2: (0.97, 0.94, 0.80),
        3: (0.95, 0.90, 0.73),
        4: (0.93, 0.87, 0.68),
        5: (0.91, 0.84, 0.63),
        6: (0.89, 0.80, 0.58),
        7: (0.86, 0.76, 0.53),
        8: (0.83, 0.72, 0.48),
    },
    # Q4 (Lower Right): Vivid Champagne Ivory / Warm Pearl
    4: {
        1: (0.98, 0.96, 0.84),
        2: (0.96, 0.92, 0.78),
        3: (0.94, 0.88, 0.71),
        4: (0.92, 0.85, 0.66),
        5: (0.90, 0.82, 0.61),
        6: (0.88, 0.78, 0.56),
        7: (0.85, 0.74, 0.51),
        8: (0.82, 0.70, 0.46),
    },
}

LABELS_BY_ID: Dict[int, AnatomicalLabel] = {
    LOWER_JAW_ID: AnatomicalLabel(
        LOWER_JAW_ID, "Lower Jawbone (Mandible)", "jaw", color=(0.83, 0.63, 0.90)
    ),
    UPPER_JAW_ID: AnatomicalLabel(
        UPPER_JAW_ID, "Upper Jawbone (Maxilla)", "jaw", color=(0.89, 0.87, 0.56)
    ),
    LEFT_IAN_ID: AnatomicalLabel(
        LEFT_IAN_ID, "Left Inferior Alveolar Canal", "canal", color=(0.98, 0.24, 0.18)
    ),
    RIGHT_IAN_ID: AnatomicalLabel(
        RIGHT_IAN_ID, "Right Inferior Alveolar Canal", "canal", color=(1.00, 0.32, 0.16)
    ),
    LEFT_SINUS_ID: AnatomicalLabel(
        LEFT_SINUS_ID, "Left Maxillary Sinus", "sinus", color=(0.22, 0.74, 0.97)
    ),
    RIGHT_SINUS_ID: AnatomicalLabel(
        RIGHT_SINUS_ID, "Right Maxillary Sinus", "sinus", color=(0.18, 0.68, 0.94)
    ),
    PHARYNX_ID: AnatomicalLabel(PHARYNX_ID, "Pharynx", "airway", color=(0.35, 0.85, 0.65)),
    BRIDGE_ID: AnatomicalLabel(BRIDGE_ID, "Bridge", "restoration", color=(0.30, 0.82, 0.95)),
    CROWN_ID: AnatomicalLabel(CROWN_ID, "Crown", "restoration", color=(0.40, 0.90, 0.95)),
    IMPLANT_ID: AnatomicalLabel(IMPLANT_ID, "Implant", "restoration", color=(0.20, 0.92, 0.55)),
    LINGUAL_CANAL_ID: AnatomicalLabel(
        LINGUAL_CANAL_ID, "Lingual Canal", "canal", color=(1.00, 0.45, 0.20)
    ),
    LEFT_INCISIVE_CANAL_ID: AnatomicalLabel(
        LEFT_INCISIVE_CANAL_ID, "Left Incisive Canal", "canal", color=(1.00, 0.50, 0.25)
    ),
    RIGHT_INCISIVE_CANAL_ID: AnatomicalLabel(
        RIGHT_INCISIVE_CANAL_ID, "Right Incisive Canal", "canal", color=(1.00, 0.50, 0.25)
    ),
}

FDI_TO_TOOTH_ID: Dict[str, int] = {}
TOOTH_ID_TO_FDI: Dict[int, str] = {}
FDI_TO_PULP_ID: Dict[str, int] = {}
PULP_ID_TO_FDI: Dict[int, str] = {}

for _q in range(1, 5):
    for _t in range(1, 9):
        _fdi = f"{_q}{_t}"
        _tooth_id = _q * 10 + _t
        _pulp_id = 100 + _tooth_id
        _q_name = _QUADRANT_NAMES[_q]
        _t_name = _TOOTH_TYPE_NAMES[_t]
        _t_col = _VIVID_TOOTH_COLORS_BY_QUADRANT[_q][_t]

        LABELS_BY_ID[_tooth_id] = AnatomicalLabel(
            id=_tooth_id,
            name=f"{_q_name} {_t_name} (#{_fdi})",
            category="tooth",
            fdi=_fdi,
            quadrant=_q,
            tooth_number=_t,
            color=_t_col,
        )
        LABELS_BY_ID[_pulp_id] = AnatomicalLabel(
            id=_pulp_id,
            name=f"Pulp of #{_fdi}",
            category="pulp",
            fdi=_fdi,
            quadrant=_q,
            tooth_number=_t,
            color=(1.00, 0.12, 0.56),
        )
        FDI_TO_TOOTH_ID[_fdi] = _tooth_id
        TOOTH_ID_TO_FDI[_tooth_id] = _fdi
        FDI_TO_PULP_ID[_fdi] = _pulp_id
        PULP_ID_TO_FDI[_pulp_id] = _fdi

ALL_TOOTH_IDS: List[int] = sorted(TOOTH_ID_TO_FDI.keys())
ALL_PULP_IDS: List[int] = sorted(PULP_ID_TO_FDI.keys())
MANDIBULAR_TEETH_IDS: List[int] = [tid for fdi, tid in FDI_TO_TOOTH_ID.items() if fdi[0] in ("3", "4")]
MAXILLARY_TEETH_IDS: List[int] = [tid for fdi, tid in FDI_TO_TOOTH_ID.items() if fdi[0] in ("1", "2")]
WISDOM_TEETH_FDI: List[str] = ["18", "28", "38", "48"]
CANAL_IDS: List[int] = [
    LEFT_IAN_ID,
    RIGHT_IAN_ID,
    LINGUAL_CANAL_ID,
    LEFT_INCISIVE_CANAL_ID,
    RIGHT_INCISIVE_CANAL_ID,
]
SINUS_IDS: List[int] = [LEFT_SINUS_ID, RIGHT_SINUS_ID]

# Regex patterns for B19 exact-token segment matching:
# Matches explicit tooth indicators like "tooth_11", "tooth 11", "fdi_11", "#11", or standalone "11"
# while rejecting generic Slicer auto-names like "Segment_11" or "Segment 11" or "label_110".
_GENERIC_SEGMENT_RE = re.compile(r"^\s*segment[\s_\-]*\d+\s*$", re.IGNORECASE)
_EXPLICIT_TOOTH_FDI_RE = re.compile(
    r"(?:(?<![a-zA-Z0-9])(?:tooth|fdi|t)|#)[\s_\-]*([1-4][1-8])(?![0-9])", re.IGNORECASE
)
_EXACT_FDI_TOKEN_RE = re.compile(r"(?:^|[^a-zA-Z0-9_])([1-4][1-8])(?:$|[^a-zA-Z0-9_])")


def safe_get_segment_tag(segment: Any, tag_name: str) -> Optional[str]:
    """Read a tag from a Slicer `vtkSegment` supporting both VTK C++ 2-arg `vtk.reference` and 1-arg test mocks."""
    if segment is None:
        return None
    if hasattr(segment, "GetTag"):
        # 1. Try 2-arg VTK C++ signature first
        try:
            import vtk  # type: ignore

            if hasattr(vtk, "reference"):
                ref_val = vtk.reference("")
                if segment.GetTag(tag_name, ref_val):
                    val_str = str(ref_val).strip()
                    if val_str:
                        return val_str
        except TypeError:
            pass
        except Exception:
            pass
        # 2. Fallback to 1-arg Python/mock signature
        try:
            val = segment.GetTag(tag_name)
            if val is not None and str(val).strip():
                return str(val).strip()
        except Exception:
            pass
    if hasattr(segment, "GetAttribute"):
        try:
            val = segment.GetAttribute(tag_name)
            if val is not None and str(val).strip():
                return str(val).strip()
        except Exception:
            pass
    return None


def get_label_info(label_id: int) -> Optional[AnatomicalLabel]:
    return LABELS_BY_ID.get(label_id)


def get_tooth_label(fdi: str) -> Optional[AnatomicalLabel]:
    tid = FDI_TO_TOOTH_ID.get(str(fdi))
    return LABELS_BY_ID.get(tid) if tid is not None else None


def match_segment_to_label_and_precedence(
    seg_id: str,
    seg_name: str,
    fdi_attribute: Optional[str] = None,
    terminology_str: Optional[str] = None,
) -> Tuple[int, str, bool, int]:
    """Deterministically classify a segmentation segment into `(label_val, category, is_5class, precedence)`.

    B19 Guarantees:
      1. Checks explicit `fdi_attribute` and `terminology_str` first.
      2. Never misclassifies generic Slicer names like `"Segment_11"` as FDI tooth `#11`.
      3. Enforces strict category precedence: `pulp (5) > tooth (4) > canal (3) > sinus (2) > jaw (1)`.
    """
    sid_clean = (seg_id or "").strip()
    name_clean = (seg_name or "").strip()
    name_low = name_clean.lower()
    term_low = (terminology_str or "").lower()

    # 1. Explicit FDI attribute on segment or terminology tag
    if fdi_attribute and str(fdi_attribute).strip() in FDI_TO_TOOTH_ID:
        fdi_code = str(fdi_attribute).strip()
        if "pulp" in name_low or "pulp" in term_low:
            return FDI_TO_PULP_ID[fdi_code], "pulp", False, SEGMENT_PRECEDENCE["pulp"]
        return FDI_TO_TOOTH_ID[fdi_code], "tooth", False, SEGMENT_PRECEDENCE["tooth"]

    for candidate_text in (term_low, name_low, sid_clean.lower()):
        if not candidate_text or _GENERIC_SEGMENT_RE.match(candidate_text):
            continue
        m_exp = _EXPLICIT_TOOTH_FDI_RE.search(candidate_text)
        if m_exp:
            fdi_code = m_exp.group(1)
            if "pulp" in candidate_text:
                return FDI_TO_PULP_ID[fdi_code], "pulp", False, SEGMENT_PRECEDENCE["pulp"]
            return FDI_TO_TOOTH_ID[fdi_code], "tooth", False, SEGMENT_PRECEDENCE["tooth"]

    # 2. Check DentalSegmentator 5-class grouped names
    if "upper teeth" in name_low or "maxillary teeth" in name_low:
        return DENTALSEGMENTATOR_UPPER_TEETH_ID, "tooth", True, SEGMENT_PRECEDENCE["tooth"]
    if "lower teeth" in name_low or "mandibular teeth" in name_low:
        return DENTALSEGMENTATOR_LOWER_TEETH_ID, "tooth", True, SEGMENT_PRECEDENCE["tooth"]

    # 3. Check exact FDI token only if NOT a generic "Segment_11" style identifier
    if not _GENERIC_SEGMENT_RE.match(name_clean):
        if name_clean in FDI_TO_TOOTH_ID:
            return FDI_TO_TOOTH_ID[name_clean], "tooth", False, SEGMENT_PRECEDENCE["tooth"]
        m_tok = _EXACT_FDI_TOKEN_RE.search(name_clean)
        if m_tok:
            fdi_code = m_tok.group(1)
            if "pulp" in name_low:
                return FDI_TO_PULP_ID[fdi_code], "pulp", False, SEGMENT_PRECEDENCE["pulp"]
            return FDI_TO_TOOTH_ID[fdi_code], "tooth", False, SEGMENT_PRECEDENCE["tooth"]

    # 4. Canal / Nerve (precedence 3)
    if "incisive" in name_low:
        if "right" in name_low:
            return RIGHT_INCISIVE_CANAL_ID, "canal", False, SEGMENT_PRECEDENCE["canal"]
        return LEFT_INCISIVE_CANAL_ID, "canal", False, SEGMENT_PRECEDENCE["canal"]
    if "lingual" in name_low:
        return LINGUAL_CANAL_ID, "canal", False, SEGMENT_PRECEDENCE["canal"]
    if "canal" in name_low or "nerve" in name_low or "ian" in name_low or "alveolar" in name_low:
        if "left" in name_low and "incisive" not in name_low:
            return LEFT_IAN_ID, "canal", False, SEGMENT_PRECEDENCE["canal"]
        if "right" in name_low and "incisive" not in name_low:
            return RIGHT_IAN_ID, "canal", False, SEGMENT_PRECEDENCE["canal"]
        return DENTALSEGMENTATOR_CANAL_ID, "canal", True, SEGMENT_PRECEDENCE["canal"]

    # 5. Sinus (precedence 2)
    if "sinus" in name_low:
        if "right" in name_low:
            return RIGHT_SINUS_ID, "sinus", False, SEGMENT_PRECEDENCE["sinus"]
        return LEFT_SINUS_ID, "sinus", False, SEGMENT_PRECEDENCE["sinus"]

    # 6. Jawbone (precedence 1 — lowest anatomical priority so it never overwrites teeth or canal)
    if "maxilla" in name_low or "skull" in name_low or "upper jaw" in name_low:
        return DENTALSEGMENTATOR_MAXILLA_ID, "jaw", True, SEGMENT_PRECEDENCE["jaw"]
    if "mandible" in name_low or "lower jaw" in name_low:
        return DENTALSEGMENTATOR_MANDIBLE_ID, "jaw", True, SEGMENT_PRECEDENCE["jaw"]
    # 7. Generic Segment_<N> / Label_<N> / TF3_<N> or trailing integer matching canonical ToothFairy3 ID
    for candidate in (name_clean, sid_clean):
        if not candidate:
            continue
        m_num = re.match(r"^(?:segment|label|tf3|toothfairy3?)[_\-\s]*(\d+)$", candidate, re.IGNORECASE)
        if not m_num:
            m_num = re.search(r"(?:^|[_\-\s])(\d+)$", candidate)
        if m_num:
            cand_id = int(m_num.group(1))
            info = LABELS_BY_ID.get(cand_id)
            if info is not None:
                return info.id, info.category, False, SEGMENT_PRECEDENCE.get(info.category, 1)

    return 0, "unknown", False, SEGMENT_PRECEDENCE["unknown"]


def resolve_combined_mask_scheme(
    matched_records: Sequence[Tuple[int, str, bool, int]],
) -> bool:
    """Determine whether a set of matched `(label_val, category, seg_is_5class, prec)` segments is truly 5-class.

    If ANY segment maps to an individual FDI tooth (`11..48`), pulp (`111..148`), split Left/Right IAN
    (`LEFT_IAN_ID=3`, `RIGHT_IAN_ID=4` with `seg_is_5class=False`), or sinus (`5, 6` with `seg_is_5class=False`),
    the segmentation uses the canonical ToothFairy3 / multi-class numbering (`is_5class = False`).
    """
    has_multi_class_token = any(
        (not is_5c and lbl > 0) or (lbl >= 11)
        for lbl, _cat, is_5c, _prec in matched_records
    )
    if has_multi_class_token:
        return False
    return any(is_5c for _lbl, _cat, is_5c, _prec in matched_records)


_FDI_TOOTH_ORDER = [
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
]

CONSECUTIVE_TF3_TO_CANONICAL_ID: Dict[int, int] = {
    **{i: i for i in range(1, 11)},
    **{11 + idx: fdi for idx, fdi in enumerate(_FDI_TOOTH_ORDER)},
    43: LINGUAL_CANAL_ID,
    44: LEFT_INCISIVE_CANAL_ID,
    45: RIGHT_INCISIVE_CANAL_ID,
    **{46 + idx: 100 + fdi for idx, fdi in enumerate(_FDI_TOOTH_ORDER)},
}


def remap_toothfairy3_array(mask: np.ndarray, dataset_labels: Optional[Dict] = None) -> np.ndarray:
    """Remap consecutive nnU-Net ToothFairy3 class indices (1..77) to canonical IDs."""
    arr = np.asarray(mask)
    if arr.size == 0:
        return arr
    unique_vals = np.unique(arr)
    unique_vals = unique_vals[unique_vals > 0]
    if unique_vals.size == 0:
        return arr

    use_consecutive = False
    if isinstance(dataset_labels, dict) and dataset_labels:
        int_vals = [
            int(v) for v in dataset_labels.values() if isinstance(v, (int, float, str)) and str(v).isdigit()
        ]
        if int_vals and max(int_vals) <= 77 and any(v > 48 for v in int_vals):
            use_consecutive = True

    max_val = int(unique_vals.max())
    if not use_consecutive:
        if max_val <= 77 and np.any(unique_vals > 48):
            use_consecutive = True

    if not use_consecutive:
        return arr

    out = np.zeros(arr.shape, dtype=np.int32)
    for src_id, dst_id in CONSECUTIVE_TF3_TO_CANONICAL_ID.items():
        if src_id in unique_vals:
            out[arr == src_id] = dst_id
    for v in unique_vals:
        iv = int(v)
        if iv not in CONSECUTIVE_TF3_TO_CANONICAL_ID:
            out[arr == iv] = iv
    return out


_Q1_ORDER = [str(q) for q in range(11, 19)]
_Q2_ORDER = [str(q) for q in range(21, 29)]
_Q4_ORDER = [str(q) for q in range(48, 40, -1)]
_Q3_ORDER = [str(q) for q in range(38, 30, -1)]


def sort_teeth_anatomically(teeth_list: List[int]) -> List[int]:
    """Sort FDI teeth Patient Right -> Patient Left: Q1, Q2, Q4, Q3."""
    teeth_set = {int(t) for t in teeth_list}
    ordered: List[int] = []
    for group in (_Q1_ORDER, _Q2_ORDER, _Q4_ORDER, _Q3_ORDER):
        ordered.extend(int(fdi) for fdi in group if int(fdi) in teeth_set)
    return ordered

RIGHT_SIDE_TEETH_IDS: List[int] = [tid for fdi, tid in FDI_TO_TOOTH_ID.items() if fdi[0] in ("1", "4")]
LEFT_SIDE_TEETH_IDS: List[int] = [tid for fdi, tid in FDI_TO_TOOTH_ID.items() if fdi[0] in ("2", "3")]


def resolve_fdi_number(fdi_or_id: Any) -> Optional[str]:
    """Normalize an FDI string/int, tooth label ID (11..48), or pulp ID (111..148) to canonical FDI string."""
    if fdi_or_id is None:
        return None
    s = str(fdi_or_id).strip().lstrip("#")
    if s in FDI_TO_TOOTH_ID:
        return s
    try:
        iv = int(s)
    except (TypeError, ValueError):
        return None
    if iv in TOOTH_ID_TO_FDI:
        return TOOTH_ID_TO_FDI[iv]
    if iv in PULP_ID_TO_FDI:
        return PULP_ID_TO_FDI[iv]
    return None


def get_fdi_quadrant(fdi_or_id: Any) -> Optional[int]:
    """Return the FDI quadrant (1..4) for a valid tooth code or label ID."""
    fdi = resolve_fdi_number(fdi_or_id)
    if fdi is None:
        return None
    return int(fdi[0])


def get_fdi_side(fdi_or_id: Any) -> Optional[str]:
    """Return anatomical side ('right' for Q1/Q4, 'left' for Q2/Q3) for a valid FDI tooth."""
    q = get_fdi_quadrant(fdi_or_id)
    if q in (1, 4):
        return "right"
    if q in (2, 3):
        return "left"
    return None


def is_mandibular_fdi(fdi_or_id: Any) -> bool:
    """Return True if the FDI tooth belongs to the mandible (Quadrants 3 or 4)."""
    q = get_fdi_quadrant(fdi_or_id)
    return q in (3, 4)


def is_maxillary_fdi(fdi_or_id: Any) -> bool:
    """Return True if the FDI tooth belongs to the maxilla (Quadrants 1 or 2)."""
    q = get_fdi_quadrant(fdi_or_id)
    return q in (1, 2)


def get_jaw_label_id(is_mandible: bool, is_5class: bool = False) -> int:
    """Return the canonical jawbone label ID for ToothFairy3 or DentalSegmentator 5-class schemes."""
    if is_5class:
        return DENTALSEGMENTATOR_MANDIBLE_ID if is_mandible else DENTALSEGMENTATOR_MAXILLA_ID
    return LOWER_JAW_ID if is_mandible else UPPER_JAW_ID


def get_tooth_group_label_id(is_mandible: bool, is_5class: bool = True) -> int:
    """Return the grouped teeth label ID for 5-class DentalSegmentator masks."""
    if not is_5class:
        raise ValueError("Grouped tooth label IDs apply only to 5-class DentalSegmentator masks.")
    return DENTALSEGMENTATOR_LOWER_TEETH_ID if is_mandible else DENTALSEGMENTATOR_UPPER_TEETH_ID


def get_ipsilateral_canal_ids(fdi_or_quadrant: Any, is_5class: bool = False) -> List[int]:
    """Return only the ipsilateral mandibular canal label ID(s) for a mandibular quadrant/FDI.

    Never returns contralateral canal IDs or canal IDs for maxillary quadrants.
    """
    q = int(fdi_or_quadrant) if isinstance(fdi_or_quadrant, int) and fdi_or_quadrant in (1, 2, 3, 4) else get_fdi_quadrant(fdi_or_quadrant)
    if q not in (3, 4):
        return []
    if is_5class:
        return [DENTALSEGMENTATOR_CANAL_ID]
    return [LEFT_IAN_ID] if q == 3 else [RIGHT_IAN_ID]


def get_ipsilateral_sinus_ids(fdi_or_quadrant: Any, is_5class: bool = False) -> List[int]:
    """Return only the ipsilateral maxillary sinus label ID(s) for a maxillary quadrant/FDI.

    Never returns contralateral sinus IDs or sinus IDs for mandibular quadrants.
    """
    q = int(fdi_or_quadrant) if isinstance(fdi_or_quadrant, int) and fdi_or_quadrant in (1, 2, 3, 4) else get_fdi_quadrant(fdi_or_quadrant)
    if q not in (1, 2) or is_5class:
        return []
    return [RIGHT_SINUS_ID] if q == 1 else [LEFT_SINUS_ID]


def is_tooth_label_id(label_id: int, is_5class: bool = False) -> bool:
    """Return True if `label_id` identifies a tooth in the active labeling scheme."""
    lid = int(label_id)
    if is_5class:
        return lid in (DENTALSEGMENTATOR_UPPER_TEETH_ID, DENTALSEGMENTATOR_LOWER_TEETH_ID)
    return lid in TOOTH_ID_TO_FDI


def is_jaw_label_id(label_id: int, is_5class: bool = False) -> bool:
    """Return True if `label_id` identifies maxilla or mandible in the active scheme."""
    lid = int(label_id)
    if is_5class:
        return lid in (DENTALSEGMENTATOR_MAXILLA_ID, DENTALSEGMENTATOR_MANDIBLE_ID)
    return lid in (LOWER_JAW_ID, UPPER_JAW_ID)


def is_canal_label_id(label_id: int, is_5class: bool = False) -> bool:
    """Return True if `label_id` identifies an inferior alveolar / mandibular canal."""
    lid = int(label_id)
    if is_5class:
        return lid == DENTALSEGMENTATOR_CANAL_ID
    return lid in CANAL_IDS


def is_sinus_label_id(label_id: int, is_5class: bool = False) -> bool:
    """Return True if `label_id` identifies a maxillary sinus."""
    lid = int(label_id)
    if is_5class:
        return False
    return lid in SINUS_IDS


def infer_mask_is_5class(mask: np.ndarray) -> bool:
    """Infer whether a combined label array uses 5-class DentalSegmentator IDs vs ToothFairy3."""
    arr = np.asarray(mask)
    if arr.size == 0:
        return False
    max_val = int(arr.max())
    return 0 < max_val <= 5


def tooth_mask_from_labels(mask: np.ndarray, is_5class: Optional[bool] = None) -> np.ndarray:
    """Return a boolean array selecting all tooth voxels using the canonical label registry."""
    arr = np.asarray(mask)
    use_5class = infer_mask_is_5class(arr) if is_5class is None else bool(is_5class)
    if use_5class:
        return (arr == DENTALSEGMENTATOR_UPPER_TEETH_ID) | (arr == DENTALSEGMENTATOR_LOWER_TEETH_ID)
    return np.isin(arr, ALL_TOOTH_IDS)


def jaw_mask_from_labels(
    mask: np.ndarray,
    is_mandible: Optional[bool] = None,
    is_5class: Optional[bool] = None,
) -> np.ndarray:
    """Return a boolean array selecting jawbone voxels using the canonical label registry."""
    arr = np.asarray(mask)
    use_5class = infer_mask_is_5class(arr) if is_5class is None else bool(is_5class)
    if is_mandible is None:
        return (arr == get_jaw_label_id(True, use_5class)) | (arr == get_jaw_label_id(False, use_5class))
    return arr == get_jaw_label_id(bool(is_mandible), use_5class)


def combine_labeled_segment_masks(
    segments: Sequence[Tuple[str, str, np.ndarray, Optional[str], Optional[str]]],
) -> Tuple[np.ndarray, bool]:
    """Deterministically combine individual binary segment masks into one canonical label volume.

    Each entry is `(seg_id, seg_name, binary_mask, fdi_attribute, terminology_str)`.
    Independent of segment insertion order: enforces `SEGMENT_PRECEDENCE`
    (`pulp > tooth > canal > sinus > jaw`) and resolves 5-class vs ToothFairy3 jaw/canal IDs.
    """
    if not segments:
        return np.zeros((0, 0, 0), dtype=np.int32), False

    matched: List[Tuple[int, str, bool, int, np.ndarray, str]] = []
    shape = None
    for seg_id, seg_name, bin_mask, fdi_attr, term_str in segments:
        arr = np.asarray(bin_mask, dtype=bool)
        if shape is None:
            shape = arr.shape
        lbl_val, category, seg_is_5c, prec = match_segment_to_label_and_precedence(
            seg_id, seg_name, fdi_attribute=fdi_attr, terminology_str=term_str
        )
        if lbl_val > 0:
            matched.append((lbl_val, category, seg_is_5c, prec, arr, seg_name.lower()))

    if shape is None:
        return np.zeros((0, 0, 0), dtype=np.int32), False

    is_5class = resolve_combined_mask_scheme([(m[0], m[1], m[2], m[3]) for m in matched])
    combined = np.zeros(shape, dtype=np.int32)

    # Sort deterministically by (precedence ascending, canonical label ascending) so higher
    # precedence always wins regardless of original segment insertion order.
    resolved_items: List[Tuple[int, int, np.ndarray]] = []
    for lbl_val, category, seg_is_5c, prec, arr, name_low in matched:
        final_lbl = lbl_val
        if not is_5class and seg_is_5c and category == "jaw":
            if lbl_val == DENTALSEGMENTATOR_MANDIBLE_ID or "mandible" in name_low or "lower jaw" in name_low:
                final_lbl = LOWER_JAW_ID
            elif lbl_val == DENTALSEGMENTATOR_MAXILLA_ID or "maxilla" in name_low or "upper jaw" in name_low:
                final_lbl = UPPER_JAW_ID
        resolved_items.append((prec, final_lbl, arr))

    resolved_items.sort(key=lambda item: (item[0], item[1]))
    for _prec, final_lbl, arr in resolved_items:
        combined[arr] = int(final_lbl)
    return combined, is_5class
