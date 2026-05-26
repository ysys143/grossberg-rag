"""
Agentic query loop — gpt-5.5 (Responses API) with tools.

Flow:
  1. router.route()           — pre-filter: in_scope / needs_retrieval / effort / clarification
  2. tool-calling loop        — non-streaming Responses API turns; model decides
                                 which tool(s) to call and with what args
       tools:
         search_knowledge     — LightRAG context retrieval (kb_tool.py)
         web_search_preview   — OpenAI built-in (zero extra deps); Phase-2 TODO:
                                 replace with custom Tavily function tool so we can
                                 run summarisation middleware on raw results before
                                 they reach the model context
  3. answer_model_stream()    — streaming final answer (identical to query.py Stage 2)

No SDK dependency — all HTTP via httpx (consistent with llm.py / embedding.py).
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
import yaml

from . import llm
from . import router as router_mod
from .kb_tool import KBTool, SCHEMA as KB_SCHEMA
from .models import answer_model_stream, ANSWER_OPENAI, ANSWER_PROVIDER_DEFAULT, ANSWER_STYLE_PROMPT
from .paths import CONFIG_PATH, ENV_PATH, APIKEY_ENV

_cfg = yaml.safe_load(CONFIG_PATH.read_text())
_acfg = _cfg.get("agent", {})

MAX_TOOL_ROUNDS: int = _acfg.get("max_tool_rounds", 4)
WEB_SEARCH_ENABLED: bool = _acfg.get("web_search", True)

_OPENAI_BASE = "https://api.openai.com/v1"

# Planning instruction injected into system prompt for high-effort queries.
# The model handles sub-question decomposition in its reasoning — no extra API call.
_PLAN_INSTRUCTION = (
    "\n\nThis is a complex multi-system question. Before answering, identify the key "
    "sub-questions, then call search_knowledge for each one systematically. "
    "Synthesise all retrieved context into a single coherent answer."
)

# ---------------------------------------------------------------------------
# Non-streaming Responses API turn (tool-calling phase)
# ---------------------------------------------------------------------------

async def _responses_turn(
    input_messages: list[dict],
    tools: list[dict],
    effort: str,
    model: str = ANSWER_OPENAI,
) -> list[dict]:
    """One non-streaming Responses API call. Returns the output item list.

    Output items are either:
      {"type": "function_call", "name": ..., "arguments": "<json>", "call_id": ...}
      {"type": "message", "role": "assistant", "content": [{"type": "output_text", ...}]}
      {"type": "reasoning", ...}
    """
    body: dict = {
        "model": model,
        "input": input_messages,
        "tools": tools,
        "reasoning": {"summary": "auto", "effort": effort},
        "tool_choice": "auto",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{_OPENAI_BASE}/responses",
            headers={"Authorization": f"Bearer {llm._openai_key()}"},
            json=body,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI {resp.status_code}: {resp.text[:400]}")
        return resp.json().get("output", [])


def _extract_tool_calls(output: list[dict]) -> list[dict]:
    return [item for item in output if item.get("type") == "function_call"]


async def _responses_stream(
    input_messages: list[dict],
    effort: str,
    model: str = ANSWER_OPENAI,
) -> AsyncIterator[dict]:
    """Stream the final answer by continuing the tool-calling conversation.

    No tools are passed — the model synthesises from the accumulated context
    (system prompt + user question + tool calls + tool results).
    Yields {"type": "reasoning"|"answer", "delta": str}.
    """
    body: dict = {
        "model": model,
        "input": input_messages,
        "reasoning": {"summary": "auto", "effort": effort},
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{_OPENAI_BASE}/responses",
            headers={"Authorization": f"Bearer {llm._openai_key()}"},
            json=body,
        ) as resp:
            if resp.status_code >= 400:
                body_bytes = await resp.aread()
                raise RuntimeError(f"OpenAI {resp.status_code}: {body_bytes.decode()[:400]}")
            event: str | None = None
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:].strip()
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if event in (
                        "response.reasoning_summary_text.delta",
                        "response.output_text.delta",
                    ):
                        delta = data.get("delta", "")
                        if not delta:
                            continue
                        kind = (
                            "reasoning"
                            if event == "response.reasoning_summary_text.delta"
                            else "answer"
                        )
                        yield {"type": kind, "delta": delta}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_agent_stream(
    question: str,
    working_dir: str | None = None,
    provider: str | None = None,
    conversation_history: list[dict] | None = None,
) -> AsyncIterator[dict]:
    """Agentic query — yields {"type": "reasoning"|"answer"|"status", "delta": str}."""

    provider = provider or ANSWER_PROVIDER_DEFAULT
    history = conversation_history or []

    # ------------------------------------------------------------------
    # Stage 0: router pre-filter
    # ------------------------------------------------------------------
    route = await router_mod.route(question, history)

    if not route["in_scope"]:
        yield {"type": "answer", "delta": "이 시스템은 Grossberg 4장(시각 지각 신경회로) 내용에만 답변합니다."}
        return

    if route.get("needs_clarification"):
        yield {"type": "answer", "delta": route.get("clarification", "질문을 좀 더 구체적으로 말씀해 주세요.")}
        return

    if not route.get("needs_retrieval", True):
        # meta / greeting — answer directly without retrieval
        async for chunk in answer_model_stream(
            prompt=question,
            system_prompt=None,
            provider=provider,
            effort=route.get("effort", "low"),
        ):
            yield chunk
        return

    effort: str = route.get("effort", "medium")

    # ------------------------------------------------------------------
    # Stage 1: tool-calling loop
    # ------------------------------------------------------------------
    tools: list[dict] = [KB_SCHEMA]
    if WEB_SEARCH_ENABLED:
        tools.append({"type": "web_search_preview"})

    # Build initial system message.
    # Style guide is injected here so it applies to the final answer generation
    # (which continues this same conversation rather than starting a fresh call).
    # Planning instruction is appended for high-effort queries.
    tool_sys = (
        "You are a research assistant for Grossberg Chapter 4. "
        "Use search_knowledge to retrieve relevant context before answering. "
        "Always cite sources using the [src:] markers present in retrieved context."
    )
    if effort == "high":
        tool_sys += _PLAN_INSTRUCTION
    sys_content = f"{ANSWER_STYLE_PROMPT}\n\n---\n\n{tool_sys}" if ANSWER_STYLE_PROMPT else tool_sys

    input_messages: list[dict] = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": question},
    ]

    kb = KBTool(working_dir)

    try:
        for _round in range(MAX_TOOL_ROUNDS):
            yield {"type": "status", "delta": f"[도구 호출 {_round + 1}회차]"}
            output = await _responses_turn(input_messages, tools, effort)

            tool_calls = _extract_tool_calls(output)
            if not tool_calls:
                # Model decided no more tools needed — stream final answer below
                break

            # Append model output to input for next turn
            input_messages.extend(output)

            # Execute each tool call
            for tc in tool_calls:
                name = tc.get("name", "")
                call_id = tc.get("call_id", "")
                args = json.loads(tc.get("arguments", "{}"))

                if name == "search_knowledge":
                    yield {"type": "status", "delta": f"[KB 검색: {args.get('query', '')[:60]}]"}
                    result = await kb.call(args, conversation_history=history)
                else:
                    result = f"(tool {name} executed)"

                input_messages.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                })
    finally:
        await kb.close()

    # ------------------------------------------------------------------
    # Stage 2: stream final answer continuing the tool-calling conversation
    # ------------------------------------------------------------------
    # input_messages now contains the full context: system prompt (with style
    # guide), user question, all tool calls, and all tool results. The model
    # synthesises directly from this — no fresh API call needed.
    async for chunk in _responses_stream(input_messages, effort):
        yield chunk


if __name__ == "__main__":
    import asyncio
    import sys
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH)
    load_dotenv(APIKEY_ENV, override=False)

    _DIM   = "\033[2m"
    _BOLD  = "\033[1m"
    _RESET = "\033[0m"

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "BCS와 FCS의 차이를 설명해줘."

    async def _run():
        print(f"Q: {question}\n")
        current = None
        async for chunk in run_agent_stream(question):
            t = chunk["type"]
            d = chunk["delta"]
            if t == "status":
                print(f"{_DIM}{d}{_RESET}", flush=True)
            elif t == "reasoning":
                if current != "reasoning":
                    print(f"\n{_DIM}[Reasoning]{_RESET}")
                    current = "reasoning"
                print(f"{_DIM}{d}{_RESET}", end="", flush=True)
            elif t == "answer":
                if current != "answer":
                    print(f"\n{_BOLD}[Answer]{_RESET}\n")
                    current = "answer"
                print(d, end="", flush=True)
        print()

    asyncio.run(_run())
