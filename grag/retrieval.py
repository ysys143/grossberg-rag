"""
Single source of truth for constructing a retrieval-ready LightRAG.

Both the chat engine (`engine.ChatSession`) and the KB tool (`kb_tool.KBTool`, used by the
agent loop and the grossberg-ask skill) build their rag here. Any *construction-time*
retrieval enhancement — rerank wiring, BM25 hybrid seeding, and anything added later — is
applied once in `build_rag` and inherited by every consumer, so the paths can't drift.
"""
from __future__ import annotations

import logging

import yaml
from lightrag import LightRAG

from .models import llm_model_func, embedding_func
from .rerank import rerank as rerank_oneshot, rerank_batched
from . import hybrid_seed
from .paths import CONFIG_PATH

logger = logging.getLogger("grag.retrieval")

_cfg = yaml.safe_load(CONFIG_PATH.read_text())

# rerank mode -> LightRAG rerank_model_func (None disables reranking)
RERANK_FUNCS = {"none": None, "oneshot": rerank_oneshot, "batched": rerank_batched}


async def build_rag(working_dir: str, rerank_mode: str = "oneshot") -> LightRAG:
    """Construct a LightRAG, initialize its storages, and apply every configured
    construction-time retrieval enhancement. The only place a retrieval-ready rag is built."""
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        rerank_model_func=RERANK_FUNCS[rerank_mode],
    )
    await rag.initialize_storages()

    if _cfg["query"].get("hybrid_seed"):  # BM25 lexical entity seeds unioned with vector
        try:
            hybrid_seed.attach_hybrid_seed(
                rag, working_dir, top_k=int(_cfg["query"].get("hybrid_seed_top_k", 10)))
        except Exception as e:  # fail-open: an enhancement, never block construction
            logger.warning(f"hybrid_seed disabled (init failed): {type(e).__name__}: {e}")

    return rag
