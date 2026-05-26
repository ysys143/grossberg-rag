"""Unit tests for router.py — _parse() pure logic + route() with mocked llm."""
import json
from unittest.mock import AsyncMock

import pytest

from grag import router


class TestParse:
    def test_valid_full_json(self):
        raw = json.dumps({
            "in_scope": True, "needs_retrieval": True,
            "effort": "medium", "needs_clarification": False, "clarification": "",
        })
        r = router._parse(raw)
        assert r["in_scope"] is True
        assert r["needs_retrieval"] is True
        assert r["effort"] == "medium"
        assert r["needs_clarification"] is False
        assert r["clarification"] == ""

    def test_effort_low_preserved(self):
        assert router._parse('{"effort": "low"}')["effort"] == "low"

    def test_effort_high_preserved(self):
        assert router._parse('{"effort": "high"}')["effort"] == "high"

    def test_invalid_effort_string_defaults_to_medium(self):
        assert router._parse('{"effort": "extreme"}')["effort"] == "medium"

    def test_effort_integer_defaults_to_medium(self):
        # Model occasionally emits numeric effort
        assert router._parse('{"effort": 3}')["effort"] == "medium"

    def test_effort_null_defaults_to_medium(self):
        assert router._parse('{"effort": null}')["effort"] == "medium"

    def test_empty_object_safe_defaults(self):
        r = router._parse("{}")
        assert r["in_scope"] is True
        assert r["needs_retrieval"] is True
        assert r["effort"] == "medium"
        assert r["needs_clarification"] is False

    def test_no_json_at_all_safe_defaults(self):
        r = router._parse("sorry, I cannot produce JSON")
        assert r["in_scope"] is True
        assert r["needs_retrieval"] is True
        assert r["effort"] == "medium"

    def test_empty_string_safe_defaults(self):
        r = router._parse("")
        assert r["in_scope"] is True

    def test_json_embedded_in_prose(self):
        raw = 'Sure! {"in_scope": false, "needs_retrieval": false, "effort": "low"} There you go.'
        r = router._parse(raw)
        assert r["in_scope"] is False
        assert r["effort"] == "low"

    def test_json_in_markdown_fence_still_extracted(self):
        # re.DOTALL means newlines are consumed; the outer {} is still found
        raw = '```json\n{"in_scope": true, "effort": "high"}\n```'
        r = router._parse(raw)
        assert r["in_scope"] is True
        assert r["effort"] == "high"

    def test_in_scope_false_needs_retrieval_false(self):
        raw = '{"in_scope": false, "needs_retrieval": false, "effort": "low"}'
        r = router._parse(raw)
        assert r["in_scope"] is False
        assert r["needs_retrieval"] is False

    def test_clarification_true_with_question_preserved(self):
        raw = '{"needs_clarification": true, "clarification": "어떤 모델을 말씀하시나요?"}'
        r = router._parse(raw)
        assert r["needs_clarification"] is True
        assert r["clarification"] == "어떤 모델을 말씀하시나요?"

    def test_clarification_true_empty_string_drops_flag(self):
        raw = '{"needs_clarification": true, "clarification": ""}'
        r = router._parse(raw)
        assert r["needs_clarification"] is False

    def test_clarification_true_whitespace_only_drops_flag(self):
        raw = '{"needs_clarification": true, "clarification": "   "}'
        r = router._parse(raw)
        assert r["needs_clarification"] is False

    def test_clarification_true_null_drops_flag(self):
        raw = '{"needs_clarification": true, "clarification": null}'
        r = router._parse(raw)
        assert r["needs_clarification"] is False

    def test_clarification_false_with_question_string_preserved(self):
        # flag=false so we don't care what clarification contains
        raw = '{"needs_clarification": false, "clarification": "something"}'
        r = router._parse(raw)
        assert r["needs_clarification"] is False

    def test_result_always_has_reason_key(self):
        assert "reason" in router._parse("{}")

    def test_in_scope_truthy_int_coerced(self):
        # JSON true/false, but just in case model emits 1/0
        raw = '{"in_scope": 1, "needs_retrieval": 0}'
        r = router._parse(raw)
        assert r["in_scope"] is True
        assert r["needs_retrieval"] is False

    def test_multiline_json_parsed(self):
        raw = '{\n  "in_scope": true,\n  "effort": "high"\n}'
        r = router._parse(raw)
        assert r["effort"] == "high"

    def test_extra_keys_ignored(self):
        raw = '{"in_scope": true, "effort": "low", "model": "gpt-5", "unknown": 99}'
        r = router._parse(raw)
        assert r["effort"] == "low"


class TestRoute:
    async def test_router_disabled_returns_high_effort_passthrough(self, monkeypatch):
        monkeypatch.setattr(router, "ROUTER_ENABLED", False)
        result = await router.route("아무 질문", [])
        assert result["in_scope"] is True
        assert result["needs_retrieval"] is True
        assert result["effort"] == "high"
        assert result["needs_clarification"] is False

    async def test_successful_route_parses_llm_response(self, monkeypatch):
        from grag import llm as llm_mod
        payload = json.dumps({
            "in_scope": True, "needs_retrieval": True,
            "effort": "medium", "needs_clarification": False, "clarification": "",
        })
        monkeypatch.setattr(llm_mod, "generate", AsyncMock(return_value=payload))
        result = await router.route("FACADE 이론이란?")
        assert result["in_scope"] is True
        assert result["effort"] == "medium"

    async def test_llm_exception_fails_open(self, monkeypatch):
        from grag import llm as llm_mod
        async def boom(*a, **kw):
            raise RuntimeError("network timeout")
        monkeypatch.setattr(llm_mod, "generate", boom)
        result = await router.route("무언가")
        assert result["in_scope"] is True
        assert result["needs_retrieval"] is True
        assert result["effort"] == "medium"
        assert "router-failopen" in result["reason"]

    async def test_llm_json_exception_fails_open(self, monkeypatch):
        from grag import llm as llm_mod
        async def returns_garbage(*a, **kw):
            raise json.JSONDecodeError("err", "", 0)
        monkeypatch.setattr(llm_mod, "generate", returns_garbage)
        result = await router.route("질문")
        assert result["in_scope"] is True

    async def test_history_passed_to_llm_prompt(self, monkeypatch):
        from grag import llm as llm_mod
        captured = []
        async def capture(model, prompt, **kw):
            captured.append(prompt)
            return '{"in_scope": true, "needs_retrieval": true, "effort": "low"}'
        monkeypatch.setattr(llm_mod, "generate", capture)
        history = [
            {"role": "user", "content": "이전 질문이에요"},
            {"role": "assistant", "content": "이전 답변이에요"},
        ]
        await router.route("후속 질문", history)
        assert "이전 질문이에요" in captured[0]

    async def test_history_trimmed_to_last_four(self, monkeypatch):
        from grag import llm as llm_mod
        captured = []
        async def capture(model, prompt, **kw):
            captured.append(prompt)
            return '{"in_scope": true, "needs_retrieval": false, "effort": "low"}'
        monkeypatch.setattr(llm_mod, "generate", capture)
        # 6 turns — only last 4 messages should appear
        history = [{"role": "user" if i % 2 == 0 else "assistant",
                    "content": f"msg{i}"} for i in range(6)]
        await router.route("질문", history)
        prompt = captured[0]
        assert "msg0" not in prompt   # oldest two trimmed
        assert "msg5" in prompt

    async def test_empty_history_does_not_prepend_context(self, monkeypatch):
        from grag import llm as llm_mod
        captured = []
        async def capture(model, prompt, **kw):
            captured.append(prompt)
            return '{"in_scope": true, "needs_retrieval": true, "effort": "low"}'
        monkeypatch.setattr(llm_mod, "generate", capture)
        await router.route("질문", [])
        assert "Recent conversation" not in captured[0]

    async def test_router_disabled_ignores_llm(self, monkeypatch):
        from grag import llm as llm_mod
        monkeypatch.setattr(router, "ROUTER_ENABLED", False)
        called = []
        monkeypatch.setattr(llm_mod, "generate", AsyncMock(side_effect=lambda *a, **kw: called.append(1)))
        await router.route("질문")
        assert not called
