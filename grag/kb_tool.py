"""
LightRAG knowledge-base tool for agentic use.

Two return modes, selected via return_answer:
  False (default) — only_need_context=True: raw retrieved context with [src:]
    markers intact. The calling agent synthesises the answer, preserving citations.
  True — LightRAG's internal LLM generates the answer. Use for quick sub-lookups;
    prefer False when the agent model is the designated answerer (gpt-5.5).

Keyword injection (concepts/entities) bypasses LightRAG's internal keyword-
extraction LLM call (one less gemini-3.5-flash round-trip) and lets the agent
explicitly steer retrieval strategy:
  concepts → hl_keywords  (thematic/high-level → feeds global KG retrieval)
  entities → ll_keywords  (specific terms     → feeds local entity retrieval)

SCHEMA is the OpenAI Responses API / Chat Completions function definition to
include in a tools list.
"""
from __future__ import annotations

from typing import Literal

import yaml
from lightrag import LightRAG, QueryParam

from .models import llm_model_func, embedding_func
from .rerank import rerank as rerank_oneshot
from .paths import CONFIG_PATH

_cfg = yaml.safe_load(CONFIG_PATH.read_text())

# ---------------------------------------------------------------------------
# OpenAI function tool schema (Responses API + Chat Completions compatible)
# ---------------------------------------------------------------------------
SCHEMA: dict = {
    "type": "function",
    "name": "search_knowledge",
    "description": (
        "Search the Grossberg Chapter 4 knowledge base about neural models of visual "
        "perception: FACADE theory, BCS (Boundary Contour System), FCS (Feature "
        "Contour System), LAMINART, boundary completion, surface filling-in, "
        "brightness, depth, figure-ground separation, competitive learning, and "
        "related mechanisms. Use this for any question about the document's content. "
        "Supply concepts (thematic terms) and/or entities (specific model names / "
        "technical terms) to steer retrieval strategy and bypass internal keyword "
        "extraction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language question or search query.",
            },
            "concepts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "High-level thematic terms (e.g. 'boundary completion', "
                    "'surface filling-in', 'competitive learning'). "
                    "Guides global/thematic KG retrieval. Leave empty if query "
                    "is entity-specific."
                ),
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific technical terms or model names "
                    "(e.g. 'BCS', 'FACADE', 'FCS', 'LAMINART'). "
                    "Guides local/entity-centric KG retrieval. Leave empty for "
                    "broad thematic questions."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["local", "global", "hybrid", "mix"],
                "description": (
                    "Retrieval strategy. "
                    "local=entity-specific lookups, "
                    "global=thematic synthesis, "
                    "hybrid=balanced (default), "
                    "mix=KG+vector."
                ),
            },
            "return_answer": {
                "type": "boolean",
                "description": (
                    "If false (default), returns retrieved context with citation "
                    "markers for the agent to synthesise. If true, returns "
                    "LightRAG's generated answer directly (quick lookup mode)."
                ),
            },
        },
        "required": ["query"],
    },
}


class KBTool:
    """Stateful LightRAG wrapper — one instance per working_dir.

    Holds a single LightRAG object across tool calls so initialize_storages()
    (disk I/O + index loading) only runs once per agent session.
    """

    def __init__(self, working_dir: str | None = None) -> None:
        self._wdir = working_dir or _cfg["storage"]["working_dir"]
        self._rag: LightRAG | None = None

    async def _get_rag(self) -> LightRAG:
        if self._rag is None:
            self._rag = LightRAG(
                working_dir=self._wdir,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                rerank_model_func=rerank_oneshot,
            )
            await self._rag.initialize_storages()
        return self._rag

    async def search(
        self,
        query: str,
        concepts: list[str] | None = None,
        entities: list[str] | None = None,
        mode: Literal["local", "global", "hybrid", "mix"] = "hybrid",
        return_answer: bool = False,
        conversation_history: list[dict] | None = None,
    ) -> str:
        rag = await self._get_rag()
        param = QueryParam(
            mode=mode,
            only_need_context=not return_answer,
            hl_keywords=concepts or [],
            ll_keywords=entities or [],
            conversation_history=conversation_history or [],
            enable_rerank=True,
        )
        result = await rag.aquery(query, param=param)
        return result or "(no results)"

    async def call(
        self,
        args: dict,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Dispatch from parsed OpenAI function-call arguments dict."""
        return await self.search(
            query=args["query"],
            concepts=args.get("concepts") or [],
            entities=args.get("entities") or [],
            mode=args.get("mode", "hybrid"),
            return_answer=bool(args.get("return_answer", False)),
            conversation_history=conversation_history,
        )

    async def close(self) -> None:
        if self._rag is not None:
            await self._rag.finalize_storages()
            self._rag = None
