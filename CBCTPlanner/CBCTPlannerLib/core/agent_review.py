"""Remote review-agent bridge (URL + API key).

The agent receives the case context (measurements, provenance, warnings, existing
annotations) plus rendered slice images, and returns a structured review: a summary,
comments, and optional actions that add/edit/remove annotations or change the tooth plan.

Wire format is the widely implemented OpenAI-compatible chat-completions schema::

    POST {url}/chat/completions
    Authorization: Bearer <key>
    {"model": ..., "messages": [{"role": "system", ...}, {"role": "user", "content": [...]}]}

so OpenAI, Azure gateways, vLLM, Ollama's OpenAI shim and internal proxies all work with
nothing more than a base URL and a key.

Security notes
--------------
* The API key only ever travels in the ``Authorization`` header. It is never put in the
  request body, the parsed result, the annotation store, or an error message
  (``redact_secrets`` scrubs it out of transport/HTTP error text).
* Images are sent as base64 PNG data URIs; they contain the rendered cross-section only
  (no patient identifiers are burned into the pixels).
* Everything returned by the agent is untrusted: actions are validated against a fixed
  whitelist (:data:`ALLOWED_ACTION_OPS`) before anything is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import socket
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.error
import urllib.parse
import urllib.request

from .annotations import (
    Annotation,
    AnnotationStore,
    AUTHOR_AGENT,
    KIND_AI_COMMENT,
    SEVERITY_INFO,
    normalize_fdi,
    normalize_kind,
    normalize_severity,
)

logger = logging.getLogger("CBCTPlanner.agent_review")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_S = 45.0
MAX_CONTEXT_CHARS = 14000
MAX_IMAGES = 8
KEY_PLACEHOLDER = "********"

SYSTEM_PROMPT = (
    "You are a dental CBCT review agent embedded in a surgical planning workstation. "
    "You get anonymised case context (arch, measured implant sites, provenance, warnings) "
    "and, when enabled, rendered cross-section images of the requested teeth. "
    "Review the case like a senior implantologist: flag anatomical risks (thin crest, "
    "insufficient height, canal or sinus proximity, unfavourable angulation), comment on "
    "the stated plan, and answer the clinician's question if one is given.\n\n"
    "Answer with ONE JSON object and nothing else (no prose, no code fence):\n"
    '{"summary": "<2-4 sentences>",\n'
    ' "comments": [{"fdi": 46, "severity": "info|warning|critical", "text": "..."}],\n'
    ' "actions": [{"op": "add_annotation", "fdi": 46, "text": "...", "severity": "warning"}],\n'
    ' "question": "<optional clarifying question>"}\n\n'
    "Allowed action ops: add_annotation, update_annotation, remove_annotation, add_tooth, "
    "remove_tooth, set_implant_diameter, flag_site. Use ids you were given for "
    "update_annotation/remove_annotation. Keep every comment specific, clinically useful and "
    "under 300 characters. Never invent measurements that are not in the context."
)

ALLOWED_ACTION_OPS: Tuple[str, ...] = (
    "add_annotation",
    "update_annotation",
    "remove_annotation",
    "add_tooth",
    "remove_tooth",
    "set_implant_diameter",
    "flag_site",
)

_ERROR_KIND_CONFIG = "config"
_ERROR_KIND_AUTH = "auth"
_ERROR_KIND_RATE_LIMIT = "rate_limit"
_ERROR_KIND_NETWORK = "network"
_ERROR_KIND_SERVER = "server"
_ERROR_KIND_HTTP = "http"
_ERROR_KIND_PARSE = "parse"


class AgentError(RuntimeError):
    """Transport/parse failure with a coarse ``kind`` for the UI to act on."""

    def __init__(self, message: str, *, kind: str = _ERROR_KIND_HTTP, status: Optional[int] = None):
        super().__init__(message)
        self.kind = kind
        self.status = status

    @property
    def is_retryable(self) -> bool:
        return self.kind in (_ERROR_KIND_NETWORK, _ERROR_KIND_RATE_LIMIT, _ERROR_KIND_SERVER)


def redact_secrets(text: Any, secrets: Iterable[Any] = ()) -> str:
    """Remove any secret value from a message so it can be shown/logged safely."""
    out = str(text or "")
    for secret in secrets:
        value = str(secret or "").strip()
        if len(value) >= 4:
            out = out.replace(value, KEY_PLACEHOLDER)
    # Common header/body patterns even when the exact value is unknown.
    out = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}", r"\1" + KEY_PLACEHOLDER, out)
    out = re.sub(r'(?i)("?(?:api[_-]?key|authorization|token)"?\s*[:=]\s*"?)[^"\s,}]{8,}', r"\1" + KEY_PLACEHOLDER, out)
    return out


# ----------------------------------------------------------------------------
# Endpoint configuration
# ----------------------------------------------------------------------------
@dataclass
class AgentEndpointConfig:
    """Where to send the review and how to authenticate to it."""

    url: str = ""
    api_key: str = ""
    model: str = DEFAULT_MODEL
    timeout_s: float = DEFAULT_TIMEOUT_S
    with_images: bool = True
    max_images: int = 4

    def __post_init__(self) -> None:
        self.url = str(self.url or "").strip()
        self.api_key = str(self.api_key or "").strip()
        self.model = str(self.model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        try:
            self.timeout_s = float(self.timeout_s)
        except (TypeError, ValueError):
            self.timeout_s = DEFAULT_TIMEOUT_S
        self.timeout_s = min(300.0, max(2.0, self.timeout_s))
        try:
            self.max_images = int(self.max_images)
        except (TypeError, ValueError):
            self.max_images = 4
        self.max_images = min(MAX_IMAGES, max(0, self.max_images))

    def is_configured(self) -> bool:
        return bool(self.url) and self.url.startswith(("http://", "https://")) and bool(self.api_key)

    def resolved_url(self) -> str:
        """Return the chat-completions endpoint for the configured base URL.

        Accepts any of ``https://host``, ``https://host/v1``, ``https://host/v1/`` or a full
        ``.../chat/completions`` URL, which is what most providers hand out.
        """
        raw = self.url.strip()
        if not raw:
            return ""
        base = raw.rstrip("/")
        if not base.startswith(("http://", "https://")):
            base = "https://" + base.lstrip("/")
        low = base.lower()
        if low.endswith("/chat/completions"):
            return base
        if re.search(r"/v\d+(?:beta)?$", low) or low.endswith("/openai"):
            return base + "/chat/completions"
        # Bare host (or a host with an unrelated path): assume an OpenAI-compatible /v1 root.
        path = urllib.parse.urlsplit(base).path
        if path in ("", "/"):
            return base + "/v1/chat/completions"
        return base + "/chat/completions"

    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def redacted(self) -> Dict[str, Any]:
        return {
            "url": self.resolved_url(),
            "model": self.model,
            "api_key": KEY_PLACEHOLDER if self.api_key else "",
            "timeout_s": self.timeout_s,
            "with_images": self.with_images,
            "max_images": self.max_images,
        }


# ----------------------------------------------------------------------------
# Request payload
# ----------------------------------------------------------------------------
@dataclass
class AgentImage:
    """One rendered image part sent to the agent."""

    label: str
    png_bytes: bytes
    mime: str = "image/png"

    def data_uri(self) -> str:
        import base64

        return f"data:{self.mime};base64,{base64.b64encode(self.png_bytes).decode('ascii')}"


def _fmt_mm(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def site_line(site: Dict[str, Any]) -> str:
    """One compact line per measured site ("the agent's important stuff")."""
    fdi = site.get("fdi", "?")
    bits = [f"FDI {fdi}"]
    if site.get("h_mm") is not None:
        bits.append(f"H {_fmt_mm(site.get('h_mm'))} mm")
    for key, label in (("w0_mm", "W0"), ("w3_mm", "W3"), ("w5_mm", "W5")):
        if site.get(key) is not None:
            bits.append(f"{label} {_fmt_mm(site.get(key))} mm")
    if site.get("ian_mm") is not None:
        bits.append(f"IAN clearance {_fmt_mm(site.get('ian_mm'))} mm")
    if site.get("sinus_mm") is not None:
        bits.append(f"sinus clearance {_fmt_mm(site.get('sinus_mm'))} mm")
    if site.get("implant"):
        bits.append(f"suggested {site.get('implant')}")
    if site.get("ridge_deg") is not None:
        bits.append(f"ridge angle {_fmt_mm(site.get('ridge_deg'), 0)} deg")
    if site.get("provenance"):
        bits.append(f"provenance {site.get('provenance')}")
    if site.get("withheld"):
        bits.append(f"WITHHELD: {site.get('withheld')}")
    return " | ".join(bits)


def context_to_prompt_text(context: Dict[str, Any]) -> str:
    """Serialise the case context as the prompt body (capped, deterministic ordering)."""
    lines: List[str] = []
    patient = context.get("patient") or {}
    lines.append("## Case")
    lines.append(
        f"patient: {patient.get('name') or 'Anonymous'} (ID {patient.get('id') or 'Anonymous'})"
    )
    if patient.get("study_date"):
        lines.append(f"study date: {patient.get('study_date')}")
    if context.get("modality"):
        lines.append(f"modality: {context.get('modality')}")
    arch = context.get("arch") or {}
    if arch:
        lines.append(
            f"jaw: {arch.get('jaw', '?')} | arch length {_fmt_mm(arch.get('length_mm'))} mm"
            f" | occlusal normal {arch.get('occlusal_normal')}"
        )
    if context.get("selected_teeth"):
        lines.append(f"planned teeth: {context.get('selected_teeth')}")
    if context.get("active_arc_mm") is not None:
        lines.append(f"cursor arc position: {_fmt_mm(context.get('active_arc_mm'))} mm")
    if context.get("workspace"):
        lines.append(f"workspace: {context.get('workspace')}")

    sites = context.get("sites") or []
    if sites:
        lines.append("")
        lines.append("## Measured sites")
        for site in sites:
            lines.append(f"- {site_line(site)}")

    annotations = context.get("annotations") or []
    if annotations:
        lines.append("")
        lines.append("## Existing annotations (id | fdi | kind | severity | author | text)")
        for ann in annotations:
            lines.append(
                f"- {ann.get('id')} | {ann.get('fdi')} | {ann.get('kind')} | "
                f"{ann.get('severity')} | {ann.get('author')} | {ann.get('text')}"
            )

    warnings = context.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("## Automated warnings")
        for warning in warnings:
            lines.append(f"- {warning}")

    if context.get("question"):
        lines.append("")
        lines.append("## Clinician question")
        lines.append(str(context.get("question")))

    images = context.get("image_labels") or []
    if images:
        lines.append("")
        lines.append("## Attached images (in order)")
        for label in images:
            lines.append(f"- {label}")

    lines.append("")
    lines.append("Return the single JSON object described in your instructions.")
    text = "\n".join(lines)
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS] + "\n... (context truncated)"
    return text


def build_review_payload(
    config: AgentEndpointConfig,
    context: Dict[str, Any],
    images: Sequence[AgentImage] = (),
) -> Dict[str, Any]:
    """Build the OpenAI-compatible chat payload for a review request."""
    body = context_to_prompt_text(context)
    parts: List[Dict[str, Any]] = [{"type": "text", "text": body}]
    for image in list(images)[: config.max_images]:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": image.data_uri(), "detail": "low"},
            }
        )
    return {
        "model": config.model,
        "temperature": 0.2,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": parts},
        ],
    }


def build_ping_payload(config: AgentEndpointConfig) -> Dict[str, Any]:
    """Smallest possible request used by "Test connection"."""
    return {
        "model": config.model,
        "temperature": 0.0,
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "You are a connectivity probe."},
            {"role": "user", "content": "Reply with the single word OK."},
        ],
    }


# ----------------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------------
@dataclass
class AgentAction:
    """One validated plan/annotation action proposed by the agent."""

    op: str
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def fdi(self) -> Optional[int]:
        return normalize_fdi(self.params.get("fdi"))

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, **{k: v for k, v in self.params.items()}}

    @classmethod
    def from_dict(cls, data: Any) -> Tuple[Optional["AgentAction"], str]:
        """Return ``(action, "")`` or ``(None, reason)``."""
        if not isinstance(data, dict):
            return None, "action is not an object"
        op = str(data.get("op") or data.get("action") or data.get("type") or "").strip().lower()
        if op not in ALLOWED_ACTION_OPS:
            return None, f"unsupported op {op or '<missing>'}"
        params = {k: v for k, v in data.items() if k not in ("op", "action", "type")}
        if op in ("add_annotation", "flag_site") and not str(params.get("text") or "").strip():
            return None, "missing text"
        if op in ("update_annotation", "remove_annotation") and not str(
            params.get("id") or params.get("annotation_id") or ""
        ).strip():
            return None, "missing annotation id"
        if op in ("add_tooth", "remove_tooth", "set_implant_diameter") and normalize_fdi(
            params.get("fdi")
        ) is None:
            return None, "missing/invalid fdi"
        return cls(op=op, params=params), ""


@dataclass
class AgentReviewResult:
    """Parsed agent answer plus the raw text it came from."""

    summary: str = ""
    comments: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)
    question: str = ""
    raw_text: str = ""
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    skipped_actions: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def has_actions(self) -> bool:
        return bool(self.actions)

    def annotations(self) -> List[Annotation]:
        """Comments as annotations (author=agent), for preview before applying anything."""
        out: List[Annotation] = []
        for comment in self.comments:
            ann = Annotation(
                text=str(comment.get("text") or ""),
                fdi=comment.get("fdi"),
                kind=KIND_AI_COMMENT,
                author=AUTHOR_AGENT,
                severity=comment.get("severity", SEVERITY_INFO),
                source=f"agent:{self.model or 'remote'}",
            )
            if ann.text:
                out.append(ann)
        return out


def _balanced_json_slice(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block in `text` (string-aware)."""
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : index + 1]
    return None


def extract_agent_json(text: Any) -> Dict[str, Any]:
    """Tolerantly pull the agent's JSON object out of a reply (fences, prose, bare lists)."""
    raw = str(text or "").strip()
    if not raw:
        raise AgentError("agent returned an empty message", kind=_ERROR_KIND_PARSE)
    candidate = raw
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    for attempt in (candidate, raw):
        try:
            parsed = json.loads(attempt)
        except (ValueError, TypeError):
            block = _balanced_json_slice(attempt)
            if block is None:
                continue
            try:
                parsed = json.loads(block)
            except (ValueError, TypeError):
                continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"comments": parsed}
    raise AgentError(
        "agent reply did not contain a JSON object; first 200 chars: "
        + redact_secrets(raw[:200]),
        kind=_ERROR_KIND_PARSE,
    )


def parse_agent_response(raw_text: str, *, model: str = "") -> AgentReviewResult:
    """Convert the agent's raw message into a structured review."""
    payload = extract_agent_json(raw_text)
    result = AgentReviewResult(raw_text=str(raw_text or ""), model=model or "")

    summary = payload.get("summary") or payload.get("review") or payload.get("assessment") or ""
    if isinstance(summary, list):
        summary = " ".join(str(part) for part in summary)
    result.summary = str(summary).strip()

    question = payload.get("question") or payload.get("clarify") or ""
    result.question = str(question).strip() if isinstance(question, str) else ""

    comments = payload.get("comments") or payload.get("findings") or payload.get("notes") or []
    if isinstance(comments, dict):
        comments = [comments]
    for entry in comments if isinstance(comments, list) else []:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                result.comments.append({"text": text, "severity": SEVERITY_INFO})
            continue
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or entry.get("comment") or entry.get("message") or "").strip()
        if not text:
            continue
        result.comments.append(
            {
                "text": text,
                "fdi": normalize_fdi(entry.get("fdi", entry.get("tooth"))),
                "severity": normalize_severity(entry.get("severity")),
                "kind": normalize_kind(entry.get("kind"), default=KIND_AI_COMMENT),
            }
        )

    actions = payload.get("actions") or payload.get("edits") or []
    if isinstance(actions, dict):
        actions = [actions]
    for entry in actions if isinstance(actions, list) else []:
        action, reason = AgentAction.from_dict(entry)
        if action is None:
            result.skipped_actions.append((str(entry.get("op") if isinstance(entry, dict) else entry), reason))
        else:
            result.actions.append(action)

    # Comments-only replies still deserve annotations, but apply_agent_actions() is what
    # actually mutates the store; parsing never touches application state.
    return result


# ----------------------------------------------------------------------------
# Transport
# ----------------------------------------------------------------------------
Transport = Callable[[str, Dict[str, Any], Dict[str, str], float], Dict[str, Any]]


def urllib_transport(
    url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout_s: float
) -> Dict[str, Any]:
    """POST JSON via the standard library and return the decoded JSON response."""
    token = str(headers.get("Authorization", "")).replace("Bearer ", "").strip()
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # justification: error body is best-effort diagnostics only
            detail = ""
        status = int(getattr(exc, "code", 0) or 0)
        kind = _ERROR_KIND_HTTP
        if status in (401, 403):
            kind = _ERROR_KIND_AUTH
        elif status == 429:
            kind = _ERROR_KIND_RATE_LIMIT
        elif status >= 500:
            kind = _ERROR_KIND_SERVER
        raise AgentError(
            redact_secrets(f"HTTP {status}: {detail or exc.reason}", [token]),
            kind=kind,
            status=status,
        ) from exc
    except urllib.error.URLError as exc:
        raise AgentError(
            redact_secrets(f"network error: {exc.reason}", [token]),
            kind=_ERROR_KIND_NETWORK,
        ) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise AgentError(f"request timed out after {timeout_s:.0f}s", kind=_ERROR_KIND_NETWORK) from exc
    try:
        return json.loads(body)
    except (ValueError, TypeError) as exc:
        raise AgentError(
            redact_secrets(f"response was not JSON: {body[:200]}", [token]), kind=_ERROR_KIND_PARSE
        ) from exc


class MockTransport:
    """Deterministic transport for tests (no sockets)."""

    def __init__(self, responses: Optional[Sequence[Any]] = None) -> None:
        self.responses = list(responses or [])
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self, url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout_s: float
    ) -> Dict[str, Any]:
        self.calls.append({"url": url, "payload": payload, "headers": headers, "timeout_s": timeout_s})
        if not self.responses:
            return {"choices": [{"message": {"content": '{"summary": "ok", "comments": []}'}}]}
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(payload)
        return item


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------
def _message_text(response: Dict[str, Any]) -> str:
    """Extract the assistant text from a chat-completions response."""
    try:
        choices = response.get("choices") or []
        message = choices[0].get("message") or {}
    except (AttributeError, IndexError, TypeError):
        raise AgentError("response missing choices[0].message", kind=_ERROR_KIND_PARSE)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in (None, "text"):
                chunks.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                chunks.append(part)
        return "".join(chunks)
    if content is None:
        raise AgentError("assistant message had no content", kind=_ERROR_KIND_PARSE)
    return str(content)


class AgentClient:
    """Thin client around a chat-completions endpoint."""

    def __init__(self, config: AgentEndpointConfig, transport: Optional[Transport] = None) -> None:
        self.config = config
        self._transport: Transport = transport or urllib_transport

    # -- helpers ----------------------------------------------------------
    def _require_configured(self) -> str:
        if not self.config.is_configured():
            raise AgentError(
                "agent endpoint is not configured (need an http(s) URL and an API key)",
                kind=_ERROR_KIND_CONFIG,
            )
        return self.config.resolved_url()

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._require_configured()
        headers = self.config.headers()
        started = time.perf_counter()
        response = self._transport(url, payload, headers, self.config.timeout_s)
        self.last_latency_s = time.perf_counter() - started
        if not isinstance(response, dict):
            raise AgentError("agent returned a non-object response", kind=_ERROR_KIND_PARSE)
        if response.get("error") and not response.get("choices"):
            message = response.get("error")
            if isinstance(message, dict):
                message = message.get("message") or message.get("code") or message
            raise AgentError(
                redact_secrets(f"agent error: {message}"), kind=_ERROR_KIND_SERVER
            )
        return response

    # -- public API -------------------------------------------------------
    def test_connection(self) -> Tuple[bool, str]:
        """Send a tiny probe. Returns ``(ok, human readable message)``; never raises."""
        try:
            response = self._post(build_ping_payload(self.config))
            text = _message_text(response).strip()
        except AgentError as exc:
            return False, f"{exc.kind}: {exc}"
        except Exception as exc:  # justification: UI probe must never raise into Qt
            logger.debug("agent probe failed", exc_info=True)
            return False, redact_secrets(f"unexpected error: {exc}", [self.config.api_key])
        usage = response.get("usage") or {}
        detail = ", ".join(f"{k}={v}" for k, v in usage.items() if isinstance(v, (int, float))) or "no usage"
        return True, f"connected in {getattr(self, 'last_latency_s', 0.0):.2f}s - replied {text[:60]!r} ({detail})"

    def review(self, context: Dict[str, Any], images: Sequence[AgentImage] = ()) -> AgentReviewResult:
        """Send the context (+ images) and parse the structured answer."""
        sent_images = list(images)[: self.config.max_images] if self.config.with_images else []
        enriched = dict(context)
        enriched["image_labels"] = [f"{img.label}" for img in sent_images]
        payload = build_review_payload(self.config, enriched, sent_images)
        response = self._post(payload)
        text = _message_text(response)
        result = parse_agent_response(text, model=self.config.model)
        result.usage = dict(response.get("usage") or {})
        if not result.model:
            result.model = str(response.get("model") or self.config.model)
        return result


# ----------------------------------------------------------------------------
# Applying what the agent proposed
# ----------------------------------------------------------------------------
@dataclass
class ActionResult:
    applied: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    def summary(self) -> str:
        text = f"{len(self.applied)} change(s) applied"
        if self.skipped:
            text += f", {len(self.skipped)} skipped"
        return text


def apply_agent_actions(
    result: AgentReviewResult,
    store: AnnotationStore,
    plan: Any = None,
    *,
    add_comments: bool = True,
) -> ActionResult:
    """Apply the agent's comments/actions to the annotation store and (optionally) the plan.

    ``plan`` is any object exposing ``add_tooth(fdi)``, ``remove_tooth(fdi)`` and
    ``set_implant_diameter(fdi, mm)``; missing methods simply skip those ops.
    """
    outcome = ActionResult()
    for reason_op, reason in result.skipped_actions:
        outcome.skipped.append((str(reason_op or "?"), reason))

    if add_comments:
        for comment in result.comments:
            try:
                ann = store.add(
                    comment.get("text", ""),
                    fdi=comment.get("fdi"),
                    kind=comment.get("kind") or KIND_AI_COMMENT,
                    author=AUTHOR_AGENT,
                    severity=comment.get("severity", SEVERITY_INFO),
                    source=f"agent:{result.model or 'remote'}",
                )
            except ValueError as exc:
                outcome.skipped.append(("comment", str(exc)))
                continue
            outcome.applied.append({"op": "comment", "id": ann.id, "fdi": ann.fdi})

    for action in result.actions:
        op = action.op
        params = dict(action.params)
        if op in ("add_annotation", "flag_site"):
            try:
                ann = store.add(
                    params.get("text", ""),
                    fdi=params.get("fdi"),
                    arc_mm=params.get("arc_mm"),
                    kind=KIND_AI_COMMENT if op == "add_annotation" else "flag",
                    author=AUTHOR_AGENT,
                    severity=params.get("severity", SEVERITY_INFO),
                    source=f"agent:{result.model or 'remote'}",
                )
            except ValueError as exc:
                outcome.skipped.append((op, str(exc)))
                continue
            outcome.applied.append({"op": op, "id": ann.id, "fdi": ann.fdi})
        elif op == "update_annotation":
            ann_id = str(params.get("id") or params.get("annotation_id") or "")
            patch = {
                k: params[k]
                for k in ("text", "severity", "kind", "fdi", "arc_mm")
                if k in params
            }
            if not store.update(ann_id, **patch):
                outcome.skipped.append((op, f"annotation {ann_id} not found or nothing to update"))
                continue
            outcome.applied.append({"op": op, "id": ann_id})
        elif op == "remove_annotation":
            ann_id = str(params.get("id") or params.get("annotation_id") or "")
            if not store.remove(ann_id):
                outcome.skipped.append((op, f"annotation {ann_id} not found"))
                continue
            outcome.applied.append({"op": op, "id": ann_id})
        elif op in ("add_tooth", "remove_tooth"):
            fdi = action.fdi
            handler = getattr(plan, "add_tooth" if op == "add_tooth" else "remove_tooth", None)
            if handler is None or fdi is None:
                outcome.skipped.append((op, "plan adapter does not support tooth membership"))
                continue
            try:
                changed = bool(handler(int(fdi)))
            except Exception as exc:  # justification: a plan edit must not abort the review
                logger.debug("agent plan action failed", exc_info=True)
                outcome.skipped.append((op, f"plan error: {exc}"))
                continue
            if changed:
                outcome.applied.append({"op": op, "fdi": int(fdi)})
            else:
                outcome.skipped.append((op, f"no change for FDI {fdi}"))
        elif op == "set_implant_diameter":
            handler = getattr(plan, "set_implant_diameter", None)
            fdi = action.fdi
            try:
                diameter = float(params.get("diameter_mm") or params.get("diameter") or 0.0)
            except (TypeError, ValueError):
                diameter = 0.0
            if handler is None or fdi is None or diameter <= 0.0:
                outcome.skipped.append((op, "implant diameter edits are not supported by this plan"))
                continue
            try:
                changed = bool(handler(int(fdi), diameter))
            except Exception as exc:  # justification: a plan edit must not abort the review
                logger.debug("agent implant action failed", exc_info=True)
                outcome.skipped.append((op, f"plan error: {exc}"))
                continue
            if changed:
                outcome.applied.append({"op": op, "fdi": int(fdi), "diameter_mm": diameter})
            else:
                outcome.skipped.append((op, f"no plan row for FDI {fdi}"))
        else:  # pragma: no cover - guarded by AgentAction validation
            outcome.skipped.append((op, "unsupported op"))

    return outcome


def context_warnings(context: Dict[str, Any]) -> List[str]:
    """Derive reviewer-facing warnings from the measured sites (no agent needed)."""
    warnings: List[str] = []
    for site in context.get("sites") or []:
        fdi = site.get("fdi")
        if site.get("withheld"):
            warnings.append(f"FDI {fdi}: measurement withheld ({site.get('withheld')})")
        w0 = site.get("w0_mm")
        if isinstance(w0, (int, float)) and w0 < 4.0:
            warnings.append(f"FDI {fdi}: crest width {w0:.1f} mm is below the 4 mm guided-implant floor")
        ian = site.get("ian_mm")
        if isinstance(ian, (int, float)) and ian < 2.0:
            warnings.append(f"FDI {fdi}: IAN clearance {ian:.1f} mm is under the 2 mm safety margin")
        h = site.get("h_mm")
        if isinstance(h, (int, float)) and h < 8.0:
            warnings.append(f"FDI {fdi}: available height {h:.1f} mm is short for a standard implant")
    return warnings
