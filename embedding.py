"""
Gemini Embedding via raw HTTP (no SDK).
Model: gemini-embedding-2 (3072 dim, multimodal)
Endpoint: POST /v1beta/models/{model}:batchEmbedContents
"""
import os

import httpx

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_TIMEOUT = 60


def _api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    return key


async def embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts. Returns list of float vectors."""
    import time
    model_path = model if model.startswith("models/") else f"models/{model}"
    requests = [
        {"model": model_path, "content": {"parts": [{"text": t}]}}
        for t in texts
    ]

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_BASE}/{model_path}:batchEmbedContents",
            params={"key": _api_key()},
            json={"requests": requests},
        )
        resp.raise_for_status()
        result = [e["values"] for e in resp.json()["embeddings"]]

    try:
        import tracing
        tracing.emit_llm_span(
            fn_name="embed_texts", model=model,
            prompt=f"{len(texts)} texts", response=f"{len(result)} vectors",
            usage=None, elapsed_s=time.monotonic() - t0, status="ok",
            provider="google", kind="EMBEDDING",
        )
    except Exception:
        pass
    return result
