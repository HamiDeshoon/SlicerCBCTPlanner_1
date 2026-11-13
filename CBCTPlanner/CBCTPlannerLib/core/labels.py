"""
Anatomical Label Registry for Dental CBCT Segmentation (ToothFairy3 / FDI + DentalSegmentator).
Includes standard adult mesiodistal crown widths (B2) and deterministic segment matching with
explicit anatomical precedence (B19: pulp > tooth > canal > sinus > jaw).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AnatomicalLabel:
    id: int
    name: str
    category: str
    fdi: Optional[str] = None
    quadrant: Optional[int] = None
    tooth_number: Optional[int] = None


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

LABELS_BY_ID: Dict[int, AnatomicalLabel] = {
    LOWER_JAW_ID: AnatomicalLabel(LOWER_JAW_ID, "Lower Jawbone (Mandible)", "jaw"),
    UPPER_JAW_ID: AnatomicalLabel(UPPER_JAW_ID, "Upper Jawbone (Maxilla)", "jaw"),
    LEFT_IAN_ID: AnatomicalLabel(LEFT_IAN_ID, "Left Inferior Alveolar Canal", "canal"),
    RIGHT_IAN_ID: AnatomicalLabel(RIGHT_IAN_ID, "Right Inferior Alveolar Canal", "canal"),
    LEFT_SINUS_ID: AnatomicalLabel(LEFT_SINUS_ID, "Left Maxillary Sinus", "sinus"),
    RIGHT_SINUS_ID: AnatomicalLabel(RIGHT_SINUS_ID, "Right Maxillary Sinus", "sinus"),
    PHARYNX_ID: AnatomicalLabel(PHARYNX_ID, "Pharynx", "airway"),
    BRIDGE_ID: AnatomicalLabel(BRIDGE_ID, "Bridge", "restoration"),
    CROWN_ID: AnatomicalLabel(CROWN_ID, "Crown", "restoration"),
    IMPLANT_ID: AnatomicalLabel(IMPLANT_ID, "Implant", "restoration"),
    LINGUAL_CANAL_ID: AnatomicalLabel(LINGUAL_CANAL_ID, "Lingual Canal", "canal"),
    LEFT_INCISIVE_CANAL_ID: AnatomicalLabel(LEFT_INCISIVE_CANAL_ID, "Left Incisive Canal", "canal"),
    RIGHT_INCISIVE_CANAL_ID: AnatomicalLabel(RIGHT_INCISIVE_CANAL_ID, "Right Incisive Canal", "canal"),
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

        LABELS_BY_ID[_tooth_id] = AnatomicalLabel(
            id=_tooth_id,
            name=f"{_q_name} {_t_name} (#{_fdi})",
            category="tooth",
            fdi=_fdi,
            quadrant=_q,
            tooth_number=_t,
        )
        LABELS_BY_ID[_pulp_id] = AnatomicalLabel(
            id=_pulp_id,
            name=f"Pulp of #{_fdi}",
            category="pulp",
            fdi=_fdi,
            quadrant=_q,
            tooth_number=_t,
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

    return 0, "unknown", False, SEGMENT_PRECEDENCE["unknown"]
