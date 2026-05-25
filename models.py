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
VISION_MODELS = set(_ans.get("vision_models", []))   # answer models that accept image input


def is_vision_capable(provider: str | None) -> bool:
    """True if the active answer model for `provider` can accept image input."""
    provider = provider or ANSWER_PROVIDER_DEFAULT
    model = ANSWER_OPENAI if provider == "openai" else ANSWER_GEMINI
    return model in VISION_MODELS
EMBED_MODEL = _m["embedding"]
EMBED_DIM = _m["embedding_dim"]
EMBED_MAX_TOKENS = _m["embedding_max_tokens"]

# Abstract router effort -> Gemini thinking_budget. OpenAI takes the effort string
# directly (reasoning.effort), so this map only realizes effort for the Gemini path.
_EFFORT_BUDGET = _cfg.get("router", {}).get(
    "thinking_budget", {"low": 512, "medium": 4096, "high": -1}
)

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
    image_data: str | None = None,
    **kwargs,
) -> str:
    # RAGAnything's modal processor passes the figure as `image_data` (raw base64),
    # NOT as `images`/`messages`. Earlier this landed in **kwargs and was dropped, so
    # every figure was described WITHOUT its image — captions were hallucinated from
    # the figure caption + domain knowledge. Forward it as a data-URI so the model
    # actually sees the figure.
    if image_data and not images:
        images = [image_data if image_data.startswith("data:")
                  else f"data:image/jpeg;base64,{image_data}"]
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
    effort: str | None = None,
    images: list | None = None,
):
    """Stream reasoning + answer from selected provider.

    The style guide (ANSWER_STYLE_PROMPT) is prepended to the caller's
    system_prompt to enforce a consistent tone/format across providers.

    effort: abstract reasoning level from the router (low|medium|high). OpenAI
    receives it as reasoning.effort; Gemini gets the mapped thinking_budget.
    images: optional figure pixels (data-URIs / file paths) injected at query time.

    Yields: {"type": "reasoning"|"answer", "delta": str}
    """
    provider = provider or ANSWER_PROVIDER_DEFAULT

    # Prepend style guide to LightRAG's rag_response system prompt
    if ANSWER_STYLE_PROMPT and system_prompt:
        final_sys = f"{ANSWER_STYLE_PROMPT}\n\n---\n\n{system_prompt}"
    else:
        final_sys = ANSWER_STYLE_PROMPT or system_prompt

    if provider == "openai":
        gen = llm.openai_generate_stream(
            model=ANSWER_OPENAI, prompt=prompt, system_prompt=final_sys, effort=effort,
            images=images,
        )
    elif provider == "gemini":
        budget = _EFFORT_BUDGET.get(effort, -1) if effort else -1
        gen = llm.gemini_generate_stream(
            model=ANSWER_GEMINI, prompt=prompt, system_prompt=final_sys, thinking_budget=budget,
            images=images,
        )
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
