"""Clinician- and agent-authored annotations for planning sites.

The store is deliberately free of Qt/Slicer imports so it can be unit-tested, serialised to
JSON next to a treatment plan, and handed to/from a remote review agent without dragging the
GUI along. Each annotation may be attached to one FDI tooth and/or an arch arc position, which
is how the Step 5/8 tables group them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import uuid

# Annotation kinds. `comment` is a plain clinician note; `ai_comment` came back from a review
# agent; `flag` is a safety/attention marker; `question` is an open item; `plan_change` records
# an edit the agent proposed or applied to the plan.
KIND_COMMENT = "comment"
KIND_AI_COMMENT = "ai_comment"
KIND_FLAG = "flag"
KIND_QUESTION = "question"
KIND_PLAN_CHANGE = "plan_change"
VALID_KINDS: Tuple[str, ...] = (
    KIND_COMMENT,
    KIND_AI_COMMENT,
    KIND_FLAG,
    KIND_QUESTION,
    KIND_PLAN_CHANGE,
)

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"
VALID_SEVERITIES: Tuple[str, ...] = (SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_CRITICAL)

AUTHOR_CLINICIAN = "clinician"
AUTHOR_AGENT = "agent"

MAX_TEXT_CHARS = 2000
MIN_FDI = 11
MAX_FDI = 48


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_fdi(value: Any) -> Optional[int]:
    """Return a valid FDI number (11-48) or None; never raises on junk input."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        stripped = value.strip().replace("#", "")
        if not stripped:
            return None
        if not stripped.isdigit():
            return None
        value = int(stripped)
    try:
        fdi = int(value)
    except (TypeError, ValueError):
        return None
    if MIN_FDI <= fdi <= MAX_FDI:
        return fdi
    return None


def normalize_severity(value: Any) -> str:
    low = str(value or "").strip().lower()
    if low in ("warn", "warning", "caution", "attention"):
        return SEVERITY_WARNING
    if low in ("critical", "error", "severe", "danger"):
        return SEVERITY_CRITICAL
    if low in VALID_SEVERITIES:
        return low
    return SEVERITY_INFO


def normalize_kind(value: Any, *, default: str = KIND_COMMENT) -> str:
    low = str(value or "").strip().lower()
    aliases = {
        "note": KIND_COMMENT,
        "comment": KIND_COMMENT,
        "ai": KIND_AI_COMMENT,
        "agent": KIND_AI_COMMENT,
        "ai_comment": KIND_AI_COMMENT,
        "flag": KIND_FLAG,
        "warning": KIND_FLAG,
        "question": KIND_QUESTION,
        "action": KIND_PLAN_CHANGE,
        "plan": KIND_PLAN_CHANGE,
        "plan_change": KIND_PLAN_CHANGE,
    }
    return aliases.get(low, low if low in VALID_KINDS else default)


@dataclass
class Annotation:
    """One reviewable note attached to a site (and optionally an arc position)."""

    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    fdi: Optional[int] = None
    arc_mm: Optional[float] = None
    kind: str = KIND_COMMENT
    author: str = AUTHOR_CLINICIAN
    severity: str = SEVERITY_INFO
    created: str = field(default_factory=_utc_now_iso)
    edited: Optional[str] = None
    source: str = ""

    def __post_init__(self) -> None:
        self.text = " ".join(str(self.text or "").split())[:MAX_TEXT_CHARS]
        self.fdi = normalize_fdi(self.fdi)
        self.kind = normalize_kind(self.kind)
        self.severity = normalize_severity(self.severity)
        self.author = (str(self.author or AUTHOR_CLINICIAN).strip() or AUTHOR_CLINICIAN)[:64]
        self.source = str(self.source or "")[:96]
        try:
            self.arc_mm = None if self.arc_mm is None else max(0.0, float(self.arc_mm))
        except (TypeError, ValueError):
            self.arc_mm = None
        if not str(self.id or "").strip():
            self.id = uuid.uuid4().hex[:12]

    @property
    def is_agent(self) -> bool:
        return self.author == AUTHOR_AGENT or self.kind == KIND_AI_COMMENT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "fdi": self.fdi,
            "arc_mm": None if self.arc_mm is None else round(float(self.arc_mm), 3),
            "kind": self.kind,
            "author": self.author,
            "severity": self.severity,
            "created": self.created,
            "edited": self.edited,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["Annotation"]:
        if not isinstance(data, dict):
            return None
        text = data.get("text") or data.get("comment") or data.get("message") or ""
        if not str(text).strip():
            return None
        ann = cls(
            text=str(text),
            id=str(data.get("id") or "") or uuid.uuid4().hex[:12],
            fdi=data.get("fdi", data.get("tooth")),
            arc_mm=data.get("arc_mm", data.get("arc")),
            kind=data.get("kind", data.get("type", KIND_COMMENT)),
            author=data.get("author", AUTHOR_CLINICIAN),
            severity=data.get("severity", SEVERITY_INFO),
            source=data.get("source", ""),
        )
        created = data.get("created")
        if isinstance(created, str) and created.strip():
            ann.created = created.strip()
        edited = data.get("edited")
        if isinstance(edited, str) and edited.strip():
            ann.edited = edited.strip()
        return ann


class AnnotationStore:
    """Ordered, id-addressable collection of :class:`Annotation` objects."""

    def __init__(self, annotations: Optional[Iterable[Annotation]] = None) -> None:
        self._items: List[Annotation] = []
        self.revision = 0
        for ann in annotations or ():
            if isinstance(ann, Annotation):
                self._items.append(ann)

    # ------------------------------------------------------------------ mutation
    def add(
        self,
        text: str,
        *,
        fdi: Any = None,
        arc_mm: Any = None,
        kind: str = KIND_COMMENT,
        author: str = AUTHOR_CLINICIAN,
        severity: str = SEVERITY_INFO,
        source: str = "",
        annotation_id: Optional[str] = None,
    ) -> Annotation:
        """Append an annotation. Raises ValueError when the text is empty."""
        clean = " ".join(str(text or "").split())
        if not clean:
            raise ValueError("annotation text must not be empty")
        ann = Annotation(
            text=clean,
            id=(annotation_id or uuid.uuid4().hex[:12]),
            fdi=fdi,
            arc_mm=arc_mm,
            kind=kind,
            author=author,
            severity=severity,
            source=source,
        )
        if self.get(ann.id) is not None:
            ann.id = uuid.uuid4().hex[:12]
        self._items.append(ann)
        self.revision += 1
        return ann

    def upsert(self, ann: Annotation) -> Annotation:
        """Insert `ann`, or replace the existing annotation sharing its id."""
        existing = self.get(ann.id)
        if existing is None:
            self._items.append(ann)
        else:
            index = self._items.index(existing)
            ann.created = existing.created
            ann.edited = _utc_now_iso()
            self._items[index] = ann
        self.revision += 1
        return ann

    def update(self, annotation_id: str, **fields: Any) -> bool:
        """Patch text/severity/kind/fdi/arc_mm/author of one annotation."""
        ann = self.get(annotation_id)
        if ann is None:
            return False
        patchable = {"text", "severity", "kind", "fdi", "arc_mm", "author"}
        applied = {k: v for k, v in fields.items() if k in patchable}
        if not applied:
            return False
        if "text" in applied:
            clean = " ".join(str(applied.pop("text") or "").split())
            if not clean:
                return False
            ann.text = clean[:MAX_TEXT_CHARS]
        if "severity" in applied:
            ann.severity = normalize_severity(applied.pop("severity"))
        if "kind" in applied:
            ann.kind = normalize_kind(applied.pop("kind"), default=ann.kind)
        if "author" in applied:
            ann.author = (str(applied.pop("author") or ann.author).strip() or ann.author)[:64]
        if "fdi" in applied:
            ann.fdi = normalize_fdi(applied.pop("fdi"))
        if "arc_mm" in applied:
            try:
                value = applied.pop("arc_mm")
                ann.arc_mm = None if value is None else max(0.0, float(value))
            except (TypeError, ValueError):
                pass
        ann.edited = _utc_now_iso()
        self.revision += 1
        return True

    def remove(self, annotation_id: str) -> bool:
        ann = self.get(annotation_id)
        if ann is None:
            return False
        self._items.remove(ann)
        self.revision += 1
        return True

    def clear(self, *, source: Optional[str] = None, kind: Optional[str] = None) -> int:
        """Remove annotations, optionally limited to one `source` or `kind`. Returns the count."""
        before = len(self._items)

        def keep(ann: Annotation) -> bool:
            if source is not None and ann.source != source:
                return True
            if kind is not None and ann.kind != kind:
                return True
            return False

        self._items = [ann for ann in self._items if keep(ann)]
        removed = before - len(self._items)
        if removed:
            self.revision += 1
        return removed

    # ------------------------------------------------------------------ queries
    def get(self, annotation_id: str) -> Optional[Annotation]:
        key = str(annotation_id or "")
        for ann in self._items:
            if ann.id == key:
                return ann
        return None

    def list(
        self,
        *,
        fdi: Optional[Any] = None,
        kind: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Annotation]:
        wanted_fdi = normalize_fdi(fdi) if fdi is not None else None
        out = []
        for ann in self._items:
            if fdi is not None and ann.fdi != wanted_fdi:
                continue
            if kind is not None and ann.kind != normalize_kind(kind, default=kind):
                continue
            if source is not None and ann.source != source:
                continue
            out.append(ann)
        return out

    def by_fdi(self) -> Dict[Optional[int], List[Annotation]]:
        grouped: Dict[Optional[int], List[Annotation]] = {}
        for ann in self._items:
            grouped.setdefault(ann.fdi, []).append(ann)
        return grouped

    def counts(self) -> Dict[str, int]:
        counts = {kind: 0 for kind in VALID_KINDS}
        for ann in self._items:
            counts[ann.kind] = counts.get(ann.kind, 0) + 1
        counts["total"] = len(self._items)
        return counts

    def agent_comment_count(self) -> int:
        return sum(1 for ann in self._items if ann.is_agent)

    def summary_lines(self, *, limit: int = 12) -> List[str]:
        """Compact `#FDI [severity] text` lines for reports and agent prompts."""
        ordered = sorted(
            self._items,
            key=lambda a: (a.fdi is None, a.fdi or 0, a.created),
        )
        lines = []
        for ann in ordered[: max(0, int(limit))]:
            tag = f"#{ann.fdi}" if ann.fdi is not None else "general"
            sev = "" if ann.severity == SEVERITY_INFO else f" ({ann.severity})"
            lines.append(f"{tag}{sev}: {ann.text}")
        if len(ordered) > limit:
            lines.append(f"... {len(ordered) - limit} more")
        return lines

    # ------------------------------------------------------------------ serialisation
    def to_dicts(self) -> List[Dict[str, Any]]:
        return [ann.to_dict() for ann in self._items]

    def to_json(self, *, indent: int = 2) -> str:
        payload = {"version": 1, "revision": self.revision, "annotations": self.to_dicts()}
        return json.dumps(payload, indent=indent, ensure_ascii=False)

    @classmethod
    def from_dicts(cls, data: Any) -> "AnnotationStore":
        items = data
        if isinstance(data, dict):
            items = data.get("annotations", [])
        store = cls()
        for entry in items if isinstance(items, (list, tuple)) else []:
            ann = Annotation.from_dict(entry)
            if ann is not None:
                store._items.append(ann)
        return store

    @classmethod
    def from_json(cls, text: Any) -> "AnnotationStore":
        try:
            return cls.from_dicts(json.loads(str(text or "{}")))
        except (ValueError, TypeError):
            return cls()

    def save(self, path: Any) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Any) -> "AnnotationStore":
        source = Path(path)
        if not source.exists():
            return cls()
        return cls.from_json(source.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ dunders
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(list(self._items))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"AnnotationStore(n={len(self._items)}, revision={self.revision})"
