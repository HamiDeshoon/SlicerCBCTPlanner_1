"""Tests for the remote review-agent bridge and the annotation store."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "CBCTPlanner"))

from CBCTPlannerLib.core.agent_review import (  # noqa: E402
    ALLOWED_ACTION_OPS,
    KEY_PLACEHOLDER,
    AgentAction,
    AgentClient,
    AgentEndpointConfig,
    AgentError,
    AgentImage,
    AgentReviewResult,
    MockTransport,
    apply_agent_actions,
    build_review_payload,
    context_to_prompt_text,
    context_warnings,
    extract_agent_json,
    parse_agent_response,
    redact_secrets,
)
from CBCTPlannerLib.core.annotations import (  # noqa: E402
    AUTHOR_AGENT,
    KIND_AI_COMMENT,
    KIND_COMMENT,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    Annotation,
    AnnotationStore,
)


class TestAnnotationStore(unittest.TestCase):
    def test_add_requires_text_and_normalises(self) -> None:
        store = AnnotationStore()
        with self.assertRaises(ValueError):
            store.add("   ")
        ann = store.add("  thin buccal   plate ", fdi="46", severity="WARN", kind="Note")
        self.assertEqual(ann.text, "thin buccal plate")
        self.assertEqual(ann.fdi, 46)
        self.assertEqual(ann.severity, SEVERITY_WARNING)
        self.assertEqual(ann.kind, KIND_COMMENT)
        self.assertEqual(len(store), 1)

    def test_invalid_fdi_is_dropped(self) -> None:
        ann = Annotation(text="general note", fdi=99)
        self.assertIsNone(ann.fdi)
        self.assertIsNone(Annotation(text="x", fdi="abc").fdi)

    def test_update_remove_and_clear(self) -> None:
        store = AnnotationStore()
        ann = store.add("first", fdi=11)
        self.assertTrue(store.update(ann.id, text="second", severity=SEVERITY_CRITICAL))
        self.assertEqual(store.get(ann.id).text, "second")
        self.assertEqual(store.get(ann.id).severity, SEVERITY_CRITICAL)
        self.assertFalse(store.update(ann.id, text="   "))
        self.assertFalse(store.update("missing", text="x"))
        self.assertFalse(store.remove("missing"))
        self.assertTrue(store.remove(ann.id))
        self.assertEqual(len(store), 0)

    def test_upsert_replaces_same_id(self) -> None:
        store = AnnotationStore()
        ann = store.add("original", fdi=21)
        replacement = Annotation(text="edited by agent", id=ann.id, fdi=21, author=AUTHOR_AGENT)
        store.upsert(replacement)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.get(ann.id).text, "edited by agent")
        self.assertIsNotNone(store.get(ann.id).edited)

    def test_roundtrip_json(self) -> None:
        store = AnnotationStore()
        store.add("a", fdi=16, kind=KIND_AI_COMMENT, author=AUTHOR_AGENT, source="agent:test")
        store.add("b")
        text = store.to_json()
        restored = AnnotationStore.from_json(text)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored.list(fdi=16)[0].text, "a")
        self.assertEqual(restored.agent_comment_count(), 1)
        self.assertEqual(AnnotationStore.from_json("not json").counts()["total"], 0)

    def test_grouping_and_summary(self) -> None:
        store = AnnotationStore()
        store.add("site note", fdi=36)
        store.add("general")
        grouped = store.by_fdi()
        self.assertIn(36, grouped)
        self.assertIn(None, grouped)
        lines = store.summary_lines()
        self.assertTrue(any(line.startswith("#36") for line in lines))
        self.assertTrue(any(line.startswith("general") for line in lines))


class TestEndpointConfig(unittest.TestCase):
    def test_resolved_url_variants(self) -> None:
        cases = {
            "https://api.example.com": "https://api.example.com/v1/chat/completions",
            "https://api.example.com/": "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1": "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1/": "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1/chat/completions": "https://api.example.com/v1/chat/completions",
            "api.example.com/v1": "https://api.example.com/v1/chat/completions",
            "https://gw.example.com/openai": "https://gw.example.com/openai/chat/completions",
            "http://localhost:8000/v1": "http://localhost:8000/v1/chat/completions",
        }
        for given, expected in cases.items():
            self.assertEqual(AgentEndpointConfig(url=given, api_key="k").resolved_url(), expected, given)

    def test_is_configured_requires_url_and_key(self) -> None:
        self.assertFalse(AgentEndpointConfig(url="https://x", api_key="").is_configured())
        self.assertFalse(AgentEndpointConfig(url="", api_key="k").is_configured())
        self.assertFalse(AgentEndpointConfig(url="ftp://x", api_key="k").is_configured())
        self.assertTrue(AgentEndpointConfig(url="https://x", api_key="k").is_configured())

    def test_clamps_and_redaction(self) -> None:
        cfg = AgentEndpointConfig(url="https://x/v1", api_key="sk-secret-value", timeout_s=1e9, max_images=99)
        self.assertLessEqual(cfg.timeout_s, 300.0)
        self.assertEqual(cfg.max_images, 8)
        redacted = cfg.redacted()
        self.assertNotIn("sk-secret-value", json.dumps(redacted))
        self.assertEqual(redacted["api_key"], KEY_PLACEHOLDER)


class TestPayloadAndParsing(unittest.TestCase):
    def _context(self) -> dict:
        return {
            "patient": {"name": "Anonymous", "id": "anon-1", "study_date": "2026-01-01"},
            "arch": {"jaw": "lower", "length_mm": 110.0, "occlusal_normal": [0, 0, 1]},
            "selected_teeth": [46],
            "sites": [
                {"fdi": 46, "h_mm": 12.3, "w0_mm": 7.1, "w3_mm": 8.0, "ian_mm": 1.4, "provenance": "measured"},
                {"fdi": 47, "withheld": "no canal"},
            ],
            "annotations": [{"id": "abc", "fdi": 46, "kind": "comment", "severity": "info", "author": "clinician", "text": "review"}],
            "question": "Is 46 wide enough for a 4.1 mm implant?",
        }

    def test_prompt_text_contains_sites_and_question(self) -> None:
        text = context_to_prompt_text(self._context())
        self.assertIn("FDI 46", text)
        self.assertIn("IAN clearance 1.4 mm", text)
        self.assertIn("WITHHELD", text)
        self.assertIn("4.1 mm implant", text)
        self.assertLessEqual(len(context_to_prompt_text({"sites": [{"fdi": i} for i in range(400)]})), 14200)

    def test_payload_structure_and_image_limit(self) -> None:
        cfg = AgentEndpointConfig(url="https://x/v1", api_key="sk-key", model="m1", max_images=2)
        images = [AgentImage(label=f"s{i}", png_bytes=b"\x89PNG" + bytes([i])) for i in range(4)]
        payload = build_review_payload(cfg, self._context(), images)
        self.assertEqual(payload["model"], "m1")
        self.assertEqual(payload["messages"][0]["role"], "system")
        parts = payload["messages"][1]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(sum(1 for p in parts if p["type"] == "image_url"), 2)
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn("sk-key", json.dumps(payload))

    def test_extract_json_tolerances(self) -> None:
        fenced = 'Here you go:\n```json\n{"summary": "ok", "comments": []}\n```\nThanks'
        self.assertEqual(extract_agent_json(fenced)["summary"], "ok")
        prose = 'Assessment: {"summary": "a", "comments": [{"text": "b"}]} end'
        self.assertEqual(extract_agent_json(prose)["comments"][0]["text"], "b")
        self.assertEqual(extract_agent_json('[{"text": "only comments"}]')["comments"][0]["text"], "only comments")
        nested = '{"summary": "brace } inside", "comments": [{"text": "ok"}]}'
        self.assertEqual(extract_agent_json(nested)["summary"], "brace } inside")
        with self.assertRaises(AgentError):
            extract_agent_json("no json here at all")

    def test_parse_comments_actions_and_rejections(self) -> None:
        raw = json.dumps(
            {
                "summary": "Ridge is adequate.",
                "comments": [
                    {"fdi": 46, "severity": "warning", "text": "Thin buccal plate."},
                    "free text finding",
                    {"severity": "info"},
                ],
                "actions": [
                    {"op": "add_annotation", "fdi": 46, "text": "Verify with CBCT re-cut"},
                    {"op": "delete_everything"},
                    {"op": "add_annotation", "fdi": 46, "text": "  "},
                    {"op": "add_tooth", "fdi": "47"},
                ],
                "question": "Any graft planned?",
            }
        )
        result = parse_agent_response(raw, model="m")
        self.assertEqual(result.summary, "Ridge is adequate.")
        self.assertEqual(len(result.comments), 2)
        self.assertEqual(result.comments[0]["severity"], SEVERITY_WARNING)
        self.assertEqual(result.question, "Any graft planned?")
        self.assertEqual([a.op for a in result.actions], ["add_annotation", "add_tooth"])
        self.assertEqual(result.actions[1].fdi, 47)
        self.assertEqual(len(result.skipped_actions), 2)
        self.assertTrue(all(op not in ALLOWED_ACTION_OPS or "text" in reason for op, reason in result.skipped_actions))

    def test_action_validation(self) -> None:
        self.assertIsNotNone(AgentAction.from_dict({"op": "add_annotation", "text": "t"})[0])
        self.assertIsNone(AgentAction.from_dict({"op": "explode"})[0])
        self.assertIsNone(AgentAction.from_dict({"op": "update_annotation"})[0])
        self.assertIsNone(AgentAction.from_dict({"op": "remove_tooth", "fdi": 99})[0])
        self.assertIsNone(AgentAction.from_dict("nope")[0])

    def test_redact_secrets(self) -> None:
        text = "failed: Bearer sk-abcdef123456 and api_key=sk-abcdef123456"
        out = redact_secrets(text, ["sk-abcdef123456"])
        self.assertNotIn("sk-abcdef123456", out)
        self.assertIn(KEY_PLACEHOLDER, out)


class TestApplyActions(unittest.TestCase):
    class _Plan:
        def __init__(self) -> None:
            self.teeth = {46}
            self.calls = []

        def add_tooth(self, fdi: int) -> bool:
            self.calls.append(("add", fdi))
            if fdi in self.teeth:
                return False
            self.teeth.add(fdi)
            return True

        def remove_tooth(self, fdi: int) -> bool:
            self.calls.append(("remove", fdi))
            if fdi not in self.teeth:
                return False
            self.teeth.discard(fdi)
            return True

    def test_comments_become_agent_annotations(self) -> None:
        result = AgentReviewResult(
            comments=[{"text": "Crest is knife-edged.", "fdi": 46, "severity": "warning"}],
            model="m",
        )
        store = AnnotationStore()
        outcome = apply_agent_actions(result, store)
        self.assertEqual(len(outcome.applied), 1)
        stored = store.list(fdi=46)[0]
        self.assertEqual(stored.author, AUTHOR_AGENT)
        self.assertEqual(stored.kind, KIND_AI_COMMENT)
        self.assertEqual(stored.source, "agent:m")

    def test_store_actions(self) -> None:
        store = AnnotationStore()
        manual = store.add("manual note", fdi=11)
        result = AgentReviewResult(
            actions=[
                AgentAction("add_annotation", {"fdi": 11, "text": "agent added"}),
                AgentAction("flag_site", {"fdi": 11, "text": "flag", "severity": "critical"}),
                AgentAction("update_annotation", {"id": manual.id, "text": "edited by agent"}),
                AgentAction("remove_annotation", {"id": manual.id}),
                AgentAction("remove_annotation", {"id": "does-not-exist"}),
            ],
            model="m",
        )
        outcome = apply_agent_actions(result, store, add_comments=False)
        self.assertEqual(len(outcome.applied), 4)
        self.assertEqual(len(outcome.skipped), 1)
        self.assertIsNone(store.get(manual.id))
        kinds = sorted(a.kind for a in store)
        self.assertEqual(kinds, ["ai_comment", "flag"])

    def test_plan_actions(self) -> None:
        plan = self._Plan()
        result = AgentReviewResult(
            actions=[
                AgentAction("add_tooth", {"fdi": 47}),
                AgentAction("add_tooth", {"fdi": 46}),
                AgentAction("remove_tooth", {"fdi": 46}),
                AgentAction("set_implant_diameter", {"fdi": 46, "diameter_mm": 4.1}),
            ],
            model="m",
        )
        outcome = apply_agent_actions(result, AnnotationStore(), plan, add_comments=False)
        self.assertEqual([a["op"] for a in outcome.applied], ["add_tooth", "remove_tooth"])
        self.assertEqual(plan.teeth, {47})
        self.assertEqual(len(outcome.skipped), 2)
        self.assertTrue(any("diameter" in reason for _op, reason in outcome.skipped))

    def test_plan_absent_is_reported_not_raised(self) -> None:
        result = AgentReviewResult(actions=[AgentAction("add_tooth", {"fdi": 36})], model="m")
        outcome = apply_agent_actions(result, AnnotationStore(), None, add_comments=False)
        self.assertFalse(outcome.changed)
        self.assertEqual(outcome.skipped[0][0], "add_tooth")


class TestAgentClient(unittest.TestCase):
    def _client(self, responses) -> tuple[AgentClient, MockTransport]:
        cfg = AgentEndpointConfig(url="https://api.example.com/v1", api_key="sk-secret", model="m1")
        transport = MockTransport(responses)
        return AgentClient(cfg, transport=transport), transport

    def test_review_sends_key_only_in_header(self) -> None:
        body = {"choices": [{"message": {"content": '{"summary": "looks fine", "comments": []}'}}], "usage": {"total_tokens": 12}}
        client, transport = self._client([body])
        result = client.review({"selected_teeth": [46]})
        self.assertEqual(result.summary, "looks fine")
        self.assertEqual(result.model, "m1")
        self.assertEqual(result.usage, {"total_tokens": 12})
        call = transport.calls[0]
        self.assertEqual(call["url"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer sk-secret")
        self.assertNotIn("sk-secret", json.dumps(call["payload"]))
        self.assertEqual(result.raw_text, '{"summary": "looks fine", "comments": []}')

    def test_review_honours_image_switch(self) -> None:
        body = {"choices": [{"message": {"content": '{"summary": "ok"}'}}]}
        client, transport = self._client([body])
        client.config.with_images = False
        client.review({}, [AgentImage(label="s", png_bytes=b"x")])
        parts = transport.calls[0]["payload"]["messages"][1]["content"]
        self.assertEqual(len(parts), 1)

    def test_parse_errors_surface_as_agent_error(self) -> None:
        client, _ = self._client([{"choices": [{"message": {"content": "I cannot help with that."}}]}])
        with self.assertRaises(AgentError) as ctx:
            client.review({})
        self.assertEqual(ctx.exception.kind, "parse")

    def test_transport_errors_are_mapped(self) -> None:
        for kind in ("auth", "rate_limit", "network", "server"):
            client, _ = self._client([AgentError("boom", kind=kind)])
            with self.assertRaises(AgentError) as ctx:
                client.review({})
            self.assertEqual(ctx.exception.kind, kind)
            self.assertTrue(ctx.exception.is_retryable == (kind != "auth"))

    def test_unconfigured_client_fails_fast(self) -> None:
        client = AgentClient(AgentEndpointConfig())
        with self.assertRaises(AgentError) as ctx:
            client.review({})
        self.assertEqual(ctx.exception.kind, "config")

    def test_test_connection_success_and_failure(self) -> None:
        client, transport = self._client([{"choices": [{"message": {"content": "OK"}}], "usage": {"total_tokens": 3}}])
        ok, message = client.test_connection()
        self.assertTrue(ok)
        self.assertIn("OK", message)
        self.assertEqual(transport.calls[0]["payload"]["max_tokens"], 8)

        bad, bad_message = self._client([AgentError("bad key", kind="auth")])
        ok2, message2 = bad.test_connection()
        self.assertFalse(ok2)
        self.assertIn("auth", message2)
        self.assertNotIn("sk-secret", message2)


class TestContextWarnings(unittest.TestCase):
    def test_thresholds(self) -> None:
        warnings = context_warnings(
            {
                "sites": [
                    {"fdi": 46, "w0_mm": 3.2},
                    {"fdi": 47, "ian_mm": 1.0},
                    {"fdi": 36, "h_mm": 7.0},
                    {"fdi": 37, "withheld": "no canal"},
                ]
            }
        )
        text = " || ".join(warnings)
        self.assertIn("FDI 46", text)
        self.assertIn("FDI 47", text)
        self.assertIn("FDI 36", text)
        self.assertIn("FDI 37", text)

    def test_clean_case_has_no_warnings(self) -> None:
        self.assertEqual(context_warnings({"sites": [{"fdi": 46, "w0_mm": 9.0, "h_mm": 14.0, "ian_mm": 5.0}]}), [])


if __name__ == "__main__":
    unittest.main()
