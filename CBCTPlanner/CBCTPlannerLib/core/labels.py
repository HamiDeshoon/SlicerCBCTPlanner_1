"""
Dental CBCT Anatomical Labels and FDI Tooth Notation Definitions.
Based on the ToothFairy3 challenge dataset schema (78 classes + background + ignore).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass(frozen=True)
class LabelInfo:
    id: int
    name: str
    category: str  # "bone", "nerve", "sinus", "airway", "prosthetic", "tooth", "pulp", "canal"
    fdi: Optional[str] = None
    quadrant: Optional[int] = None
    tooth_number: Optional[int] = None
    is_tooth: bool = False
    is_pulp: bool = False
    color_rgb: Tuple[int, int, int] = (200, 200, 200)

# Tooth types in anatomical order (1 to 8 in each quadrant)
TOOTH_NAMES = [
    "Central Incisor",
    "Lateral Incisor",
    "Canine",
    "First Premolar",
    "Second Premolar",
    "First Molar",
    "Second Molar",
    "Third Molar (Wisdom Tooth)"
]

QUADRANTS = {
    1: ("Upper Right", "Maxillary Right", 10),
    2: ("Upper Left", "Maxillary Left", 20),
    3: ("Lower Left", "Mandibular Left", 30),
    4: ("Lower Right", "Mandibular Right", 40),
}

# Anatomical structures with original ToothFairy3 IDs
STRUCTURE_LABELS: Dict[str, Tuple[int, str, Tuple[int, int, int]]] = {
    "background": (0, "airway", (0, 0, 0)),
    "Lower Jawbone": (1, "bone", (230, 220, 200)),
    "Upper Jawbone": (2, "bone", (220, 210, 190)),
    "Left Inferior Alveolar Canal": (3, "nerve", (255, 60, 60)),
    "Right Inferior Alveolar Canal": (4, "nerve", (255, 100, 100)),
    "Left Maxillary Sinus": (5, "sinus", (80, 180, 255)),
    "Right Maxillary Sinus": (6, "sinus", (100, 200, 255)),
    "Pharynx": (7, "airway", (160, 100, 200)),
    "Bridge": (8, "prosthetic", (255, 215, 0)),
    "Crown": (9, "prosthetic", (218, 165, 32)),
    "Implant": (10, "prosthetic", (0, 230, 230)),
    "Left Mandibular Incisive Canal": (103, "canal", (255, 140, 0)),
    "Right Mandibular Incisive Canal": (104, "canal", (255, 165, 0)),
    "Lingual Canal": (105, "canal", (240, 128, 128)),
}

def _build_label_registry() -> Dict[int, LabelInfo]:
    registry: Dict[int, LabelInfo] = {}

    # 1. Non-tooth structures
    for name, (orig_id, category, rgb) in STRUCTURE_LABELS.items():
        registry[orig_id] = LabelInfo(
            id=orig_id,
            name=name,
            category=category,
            color_rgb=rgb,
        )

    # 2. Teeth and pulps for all 4 quadrants
    quadrant_palettes = {
        1: (180, 220, 240),  # Upper right - light cyan/blue
        2: (240, 200, 220),  # Upper left - light pink/magenta
        3: (200, 240, 200),  # Lower left - light green
        4: (240, 230, 180),  # Lower right - light yellow
    }

    for quad_num, (quad_name, _, base) in QUADRANTS.items():
        base_color = quadrant_palettes[quad_num]
        for t_idx, t_name in enumerate(TOOTH_NAMES, 1):
            fdi_code = f"{quad_num}{t_idx}"
            tooth_id = base + t_idx
            pulp_id = base + 100 + t_idx

            # Slightly vary shade for each tooth in quadrant
            shade_shift = (t_idx - 1) * 8
            tooth_rgb = (
                max(0, min(255, base_color[0] - shade_shift)),
                max(0, min(255, base_color[1] - shade_shift)),
                max(0, min(255, base_color[2] - shade_shift)),
            )
            pulp_rgb = (255, 50, 120)  # Bright crimson/magenta for pulp

            full_tooth_name = f"{quad_name} {t_name}"
            registry[tooth_id] = LabelInfo(
                id=tooth_id,
                name=full_tooth_name,
                category="tooth",
                fdi=fdi_code,
                quadrant=quad_num,
                tooth_number=t_idx,
                is_tooth=True,
                is_pulp=False,
                color_rgb=tooth_rgb,
            )

            registry[pulp_id] = LabelInfo(
                id=pulp_id,
                name=f"{full_tooth_name} Pulp",
                category="pulp",
                fdi=f"{fdi_code}P",
                quadrant=quad_num,
                tooth_number=t_idx,
                is_tooth=False,
                is_pulp=True,
                color_rgb=pulp_rgb,
            )

    return registry

LABEL_REGISTRY: Dict[int, LabelInfo] = _build_label_registry()

# Quick lookups
NAME_TO_ID: Dict[str, int] = {info.name: lid for lid, info in LABEL_REGISTRY.items()}
FDI_TO_TOOTH_ID: Dict[str, int] = {
    info.fdi: lid for lid, info in LABEL_REGISTRY.items() if info.is_tooth and info.fdi
}
FDI_TO_PULP_ID: Dict[str, int] = {
    info.fdi[:-1]: lid for lid, info in LABEL_REGISTRY.items() if info.is_pulp and info.fdi
}

# Groups
ALL_TEETH_IDS: List[int] = [lid for lid, info in LABEL_REGISTRY.items() if info.is_tooth]
ALL_PULP_IDS: List[int] = [lid for lid, info in LABEL_REGISTRY.items() if info.is_pulp]
MANDIBULAR_TEETH_IDS: List[int] = [
    lid for lid, info in LABEL_REGISTRY.items() if info.is_tooth and info.quadrant in (3, 4)
]
MAXILLARY_TEETH_IDS: List[int] = [
    lid for lid, info in LABEL_REGISTRY.items() if info.is_tooth and info.quadrant in (1, 2)
]
WISDOM_TEETH_FDI = ["18", "28", "38", "48"]

# Critical anatomical labels
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
LEFT_INCISIVE_CANAL_ID = 103
RIGHT_INCISIVE_CANAL_ID = 104
LINGUAL_CANAL_ID = 105

def get_tooth_label(fdi: str) -> Optional[LabelInfo]:
    """Lookup tooth by FDI string e.g. '36' or '48'."""
    tid = FDI_TO_TOOTH_ID.get(str(fdi))
    return LABEL_REGISTRY.get(tid) if tid is not None else None

def get_related_structures_for_tooth(fdi: str) -> List[int]:
    """Return critical anatomical structure IDs relevant to the given tooth."""
    info = get_tooth_label(fdi)
    if not info:
        return []
    
    related = []
    if info.quadrant in (1, 2):  # Maxilla
        related.append(UPPER_JAW_ID)
        if info.quadrant == 1 and info.tooth_number in (4, 5, 6, 7, 8):
            related.append(RIGHT_SINUS_ID)
        elif info.quadrant == 2 and info.tooth_number in (4, 5, 6, 7, 8):
            related.append(LEFT_SINUS_ID)
        if info.tooth_number in (1, 2):  # Incisors
            related.extend([105])  # Nasopalatine/incisive
    else:  # Mandible
        related.append(LOWER_JAW_ID)
        if info.quadrant == 3:
            related.append(LEFT_IAN_ID)
            if info.tooth_number in (1, 2, 3):
                related.append(LEFT_INCISIVE_CANAL_ID)
        else:
            related.append(RIGHT_IAN_ID)
            if info.tooth_number in (1, 2, 3):
                related.append(RIGHT_INCISIVE_CANAL_ID)
        if info.tooth_number in (1, 2):
            related.append(LINGUAL_CANAL_ID)

    return related
