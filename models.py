"""
RAGAnything adapter layer.
Reads config.yaml and wires llm.py + embedding.py into RAGAnything-compatible async functions.
"""
from pathlib import Path

import numpy as np
import yaml
from lightrag.utils import EmbeddingFunc

import llm
import embedding as emb

_cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
_m = _cfg["models"]

LLM_MODEL = _m["llm"]
VISION_MODEL = _m["vision"]
_ans = _m["answer"]
ANSWER_PROVIDER_DEFAULT = _ans["provider"]   # openai | gemini
ANSWER_OPENAI = _ans["openai"]
ANSWER_GEMINI = _ans["gemini"]
EMBED_MODEL = _m["embedding"]
EMBED_DIM = _m["embedding_dim"]
EMBED_MAX_TOKENS = _m["embedding_max_tokens"]

# Load style guide once at import. Swappable: point config.yaml at a different .md
_style_path = Path(__file__).parent / _ans.get("system_prompt", "prompts/answer_system.md")
ANSWER_STYLE_PROMPT = _style_path.read_text() if _style_path.exists() else ""


async def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str:
    """LightRAG-facing LLM (KG extraction + keyword extraction).

    Thinking is disabled (thinking_budget=0): these are structured/deterministic
    tasks where thinking tokens were inflating cost ~2.3x with no quality gain.
    For tasks that benefit from reasoning, use `gemini_generate_stream` directly.
    """
    return await llm.generate(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history=history_messages,
        thinking_budget=0,
    )


async def vision_model_func(
    prompt: str,
    images: list | None = None,
    system_prompt: str | None = None,
    messages: list | None = None,
    **kwargs,
) -> str:
    return await llm.generate_with_vision(
        model=VISION_MODEL,
        prompt=prompt,
        images=images,
        system_prompt=system_prompt,
        raw_messages=messages,
    )


async def answer_model_stream(
    prompt: str,
    system_prompt: str | None = None,
    provider: str | None = None,
):
    """Stream reasoning + answer from selected provider.

    The style guide (ANSWER_STYLE_PROMPT) is prepended to the caller's
    system_prompt to enforce a consistent tone/format across providers.

    Yields: {"type": "reasoning"|"answer", "delta": str}
    """
    provider = provider or ANSWER_PROVIDER_DEFAULT

    # Prepend style guide to LightRAG's rag_response system prompt
    if ANSWER_STYLE_PROMPT and system_prompt:
        final_sys = f"{ANSWER_STYLE_PROMPT}\n\n---\n\n{system_prompt}"
    else:
        final_sys = ANSWER_STYLE_PROMPT or system_prompt

    if provider == "openai":
        gen = llm.openai_generate_stream(model=ANSWER_OPENAI, prompt=prompt, system_prompt=final_sys)
    elif provider == "gemini":
        gen = llm.gemini_generate_stream(model=ANSWER_GEMINI, prompt=prompt, system_prompt=final_sys)
    else:
        raise ValueError(f"Unknown answer provider: {provider}")
    async for chunk in gen:
        yield chunk


async def _embedding_func(texts: list[str]) -> np.ndarray:
    vectors = await emb.embed_texts(model=EMBED_MODEL, texts=texts)
    return np.array(vectors, dtype=np.float32)


embedding_func = EmbeddingFunc(
    embedding_dim=EMBED_DIM,
    max_token_size=EMBED_MAX_TOKENS,
    func=_embedding_func,
)
