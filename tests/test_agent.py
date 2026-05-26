"""Unit tests for agent.py — _extract_tool_calls() pure + run_agent_stream() routing."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from grag import agent


# ---------------------------------------------------------------------------
# Async generator helpers
# ---------------------------------------------------------------------------

async def _async_items(*items):
    for item in items:
        yield item


async def collect(gen):
    return [item async for item in gen]


# ---------------------------------------------------------------------------
# _extract_tool_calls — pure
# ---------------------------------------------------------------------------

class TestExtractToolCalls:
    def test_returns_function_call_items_only(self):
        output = [
            {"type": "function_call", "name": "search_knowledge", "call_id": "c1", "arguments": "{}"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
            {"type": "reasoning", "summary": []},
        ]
        result = agent._extract_tool_calls(output)
        assert len(result) == 1
        assert result[0]["name"] == "search_knowledge"

    def test_empty_output_returns_empty(self):
        assert agent._extract_tool_calls([]) == []

    def test_no_function_calls_returns_empty(self):
        output = [{"type": "message"}, {"type": "reasoning"}]
        assert agent._extract_tool_calls(output) == []

    def test_multiple_function_calls_all_returned(self):
        output = [
            {"type": "function_call", "name": "search_knowledge", "call_id": "c1"},
            {"type": "function_call", "name": "web_search_preview", "call_id": "c2"},
            {"type": "message"},
        ]
        result = agent._extract_tool_calls(output)
        assert len(result) == 2

    def test_order_preserved(self):
        output = [
            {"type": "function_call", "name": "first", "call_id": "c1"},
            {"type": "message"},
            {"type": "function_call", "name": "second", "call_id": "c2"},
        ]
        result = agent._extract_tool_calls(output)
        assert result[0]["name"] == "first"
        assert result[1]["name"] == "second"

    def test_missing_type_key_not_included(self):
        output = [{"name": "orphan"}, {"type": "function_call", "name": "valid"}]
        result = agent._extract_tool_calls(output)
        assert len(result) == 1
        assert result[0]["name"] == "valid"


# ---------------------------------------------------------------------------
# Fixtures for routing tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kb_class(monkeypatch):
    """Replace KBTool with a mock that records close() calls."""
    close_calls = []

    class _MockKB:
        def __init__(self, *a, **kw): pass
        async def call(self, args, **kw): return "retrieved context"
        async def close(self): close_calls.append(True)

    monkeypatch.setattr(agent, "KBTool", _MockKB)
    return close_calls


@pytest.fixture
def mock_responses_turn_no_tools(monkeypatch):
    """_responses_turn returns no function_calls → loop exits immediately."""
    async def _turn(*a, **kw):
        return [{"type": "message", "content": []}]
    monkeypatch.setattr(agent, "_responses_turn", _turn)


@pytest.fixture
def mock_responses_stream_answer(monkeypatch):
    async def _stream(*a, **kw):
        yield {"type": "answer", "delta": "final answer text"}
    monkeypatch.setattr(agent, "_responses_stream", _stream)


# ---------------------------------------------------------------------------
# run_agent_stream — routing
# ---------------------------------------------------------------------------

class TestRunAgentStreamRouting:
    async def test_out_of_scope_yields_decline_answer_and_stops(self, monkeypatch):
        async def mock_route(*a, **kw):
            return {"in_scope": False, "needs_clarification": False,
                    "needs_retrieval": True, "effort": "low"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)
        chunks = await collect(agent.run_agent_stream("오늘 날씨는?"))
        types = [c["type"] for c in chunks]
        assert types == ["answer"]
        assert "Grossberg" in chunks[0]["delta"] or "시스템" in chunks[0]["delta"]

    async def test_needs_clarification_yields_clarification_text_and_stops(self, monkeypatch):
        async def mock_route(*a, **kw):
            return {"in_scope": True, "needs_clarification": True,
                    "clarification": "어떤 신경 모델을 말씀하시나요?",
                    "needs_retrieval": True, "effort": "low"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)
        chunks = await collect(agent.run_agent_stream("그거 설명해줘"))
        answer = "".join(c["delta"] for c in chunks if c["type"] == "answer")
        assert "어떤 신경 모델" in answer

    async def test_no_retrieval_bypasses_tool_loop(
        self, monkeypatch, mock_kb_class
    ):
        async def mock_route(*a, **kw):
            return {"in_scope": True, "needs_clarification": False,
                    "needs_retrieval": False, "effort": "low"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)
        stream_called = []
        async def mock_answer_stream(*a, **kw):
            stream_called.append(True)
            yield {"type": "answer", "delta": "직접 답변"}
        monkeypatch.setattr(agent, "answer_model_stream", mock_answer_stream)
        chunks = await collect(agent.run_agent_stream("안녕하세요"))
        assert stream_called
        assert any(c["delta"] == "직접 답변" for c in chunks)
        # KBTool.close() should NOT be called on the no-retrieval path
        # (KBTool is never instantiated in this branch)
        assert not mock_kb_class

    async def test_kb_close_called_in_finally_on_success(
        self, monkeypatch, mock_kb_class,
        mock_responses_turn_no_tools, mock_responses_stream_answer
    ):
        async def mock_route(*a, **kw):
            return {"in_scope": True, "needs_clarification": False,
                    "needs_retrieval": True, "effort": "medium"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)
        await collect(agent.run_agent_stream("BCS 설명해줘"))
        assert mock_kb_class, "KBTool.close() must be called after tool loop"

    async def test_kb_close_called_in_finally_on_exception(
        self, monkeypatch, mock_kb_class, mock_responses_stream_answer
    ):
        async def mock_route(*a, **kw):
            return {"in_scope": True, "needs_clarification": False,
                    "needs_retrieval": True, "effort": "medium"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)
        async def exploding_turn(*a, **kw):
            raise RuntimeError("simulated API failure")
        monkeypatch.setattr(agent, "_responses_turn", exploding_turn)
        with pytest.raises(RuntimeError):
            await collect(agent.run_agent_stream("BCS?"))
        assert mock_kb_class, "KBTool.close() must fire even when tool loop raises"

    async def test_status_events_emitted_each_round(
        self, monkeypatch, mock_kb_class,
        mock_responses_turn_no_tools, mock_responses_stream_answer
    ):
        async def mock_route(*a, **kw):
            return {"in_scope": True, "needs_clarification": False,
                    "needs_retrieval": True, "effort": "low"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)
        chunks = await collect(agent.run_agent_stream("FCS?"))
        status = [c for c in chunks if c["type"] == "status"]
        assert len(status) >= 1

    async def test_tool_call_dispatched_to_kb(
        self, monkeypatch, mock_responses_stream_answer
    ):
        async def mock_route(*a, **kw):
            return {"in_scope": True, "needs_clarification": False,
                    "needs_retrieval": True, "effort": "low"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)

        kb_calls = []

        class _TracingKB:
            def __init__(self, *a, **kw): pass
            async def call(self, args, **kw):
                kb_calls.append(args)
                return "context"
            async def close(self): pass

        monkeypatch.setattr(agent, "KBTool", _TracingKB)

        call_count = 0
        async def turn_then_done(input_msgs, tools, effort, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{
                    "type": "function_call",
                    "name": "search_knowledge",
                    "call_id": "c1",
                    "arguments": json.dumps({"query": "BCS 경계 완성"}),
                }]
            return []  # second call: no more tools
        monkeypatch.setattr(agent, "_responses_turn", turn_then_done)

        await collect(agent.run_agent_stream("BCS 경계 완성 설명"))
        assert kb_calls
        assert kb_calls[0]["query"] == "BCS 경계 완성"

    async def test_high_effort_appends_plan_instruction(
        self, monkeypatch, mock_kb_class,
        mock_responses_turn_no_tools, mock_responses_stream_answer
    ):
        async def mock_route(*a, **kw):
            return {"in_scope": True, "needs_clarification": False,
                    "needs_retrieval": True, "effort": "high"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)
        captured = []
        orig_turn = agent._responses_turn
        async def capture_turn(input_msgs, tools, effort, **kw):
            captured.append(input_msgs)
            return []
        monkeypatch.setattr(agent, "_responses_turn", capture_turn)
        await collect(agent.run_agent_stream("복잡한 질문"))
        # The system message should contain the planning instruction
        sys_msg = captured[0][0]
        assert "sub-question" in sys_msg["content"] or "sub_question" in sys_msg["content"] or "sub-questions" in sys_msg["content"]

    async def test_unknown_tool_call_produces_placeholder_result(
        self, monkeypatch, mock_responses_stream_answer
    ):
        async def mock_route(*a, **kw):
            return {"in_scope": True, "needs_clarification": False,
                    "needs_retrieval": True, "effort": "low"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)

        class _NoOpKB:
            def __init__(self, *a, **kw): pass
            async def call(self, *a, **kw): return "ctx"
            async def close(self): pass

        monkeypatch.setattr(agent, "KBTool", _NoOpKB)

        call_count = 0
        input_accumulated = []

        async def turn_with_unknown_tool(input_msgs, tools, effort, **kw):
            nonlocal call_count
            call_count += 1
            input_accumulated.extend(input_msgs)
            if call_count == 1:
                return [{
                    "type": "function_call",
                    "name": "some_future_tool",
                    "call_id": "c99",
                    "arguments": "{}",
                }]
            return []
        monkeypatch.setattr(agent, "_responses_turn", turn_with_unknown_tool)

        await collect(agent.run_agent_stream("q"))
        # The tool output for unknown tool should still be appended
        outputs = [m for m in input_accumulated if m.get("type") == "function_call_output"]
        assert any("c99" == o.get("call_id") for o in outputs)

    async def test_max_tool_rounds_respected(
        self, monkeypatch, mock_kb_class, mock_responses_stream_answer
    ):
        async def mock_route(*a, **kw):
            return {"in_scope": True, "needs_clarification": False,
                    "needs_retrieval": True, "effort": "low"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)
        monkeypatch.setattr(agent, "MAX_TOOL_ROUNDS", 2)

        rounds = []
        async def always_tools(input_msgs, tools, effort, **kw):
            rounds.append(True)
            return [{
                "type": "function_call",
                "name": "search_knowledge",
                "call_id": f"c{len(rounds)}",
                "arguments": json.dumps({"query": "q"}),
            }]
        monkeypatch.setattr(agent, "_responses_turn", always_tools)

        await collect(agent.run_agent_stream("계속 도구를 쓰는 질문"))
        assert len(rounds) == 2  # loop capped at MAX_TOOL_ROUNDS

    async def test_provider_forwarded_to_no_retrieval_stream(self, monkeypatch):
        async def mock_route(*a, **kw):
            return {"in_scope": True, "needs_clarification": False,
                    "needs_retrieval": False, "effort": "low"}
        monkeypatch.setattr(agent.router_mod, "route", mock_route)
        captured = {}
        async def mock_answer_stream(*a, provider=None, effort=None, **kw):
            captured["provider"] = provider
            captured["effort"] = effort
            yield {"type": "answer", "delta": "ok"}
        monkeypatch.setattr(agent, "answer_model_stream", mock_answer_stream)
        await collect(agent.run_agent_stream("hi", provider="gemini"))
        assert captured["provider"] == "gemini"
