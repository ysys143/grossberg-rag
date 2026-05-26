"""
Gemini Embedding via raw HTTP (no SDK).
Model: gemini-embedding-2 (3072 dim, multimodal)
Endpoint: POST /v1beta/models/{model}:batchEmbedContents
"""
import asyncio
import os
import time

import httpx
import yaml

from .paths import CONFIG_PATH

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_TIMEOUT = 60

# Cap concurrent embedding calls so bulk re-ingest (hundreds of entity/relation
# embeddings fired at once) stays under the gemini-embedding rate limit (429).
_cfg = yaml.safe_load(CONFIG_PATH.read_text())
_CONCURRENCY = int(_cfg.get("models", {}).get("embedding_concurrency", 4))
_MAX_RETRIES = 5
_sem = asyncio.Semaphore(_CONCURRENCY)


def _api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    return key


def _mask(text: str, key: str) -> str:
    """Redact the API key from error text so it never lands in logs."""
    return text.replace(key, "***") if key else text


async def embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts. Returns list of float vectors.

    Concurrency-capped (semaphore) + exponential backoff on 429, so bulk
    ingestion stays under the embedding rate limit instead of relying solely on
    LightRAG's upstream retry.
    """
    model_path = model if model.startswith("models/") else f"models/{model}"
    requests = [
        {"model": model_path, "content": {"parts": [{"text": t}]}}
        for t in texts
    ]
    key = _api_key()

    t0 = time.monotonic()
    result: list[list[float]] = []
    async with _sem:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for attempt in range(_MAX_RETRIES):
                resp = await client.post(
                    f"{_BASE}/{model_path}:batchEmbedContents",
                    params={"key": key},
                    json={"requests": requests},
                )
                if resp.status_code == 429 and attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)  # 1,2,4,8s backoff
                    continue
                if resp.status_code >= 400:
                    # Strip the URL (carries ?key=) from the error to avoid leaking the key.
                    raise RuntimeError(
                        f"Embedding {resp.status_code}: {_mask(resp.text, key)[:300]}"
                    )
                result = [e["values"] for e in resp.json()["embeddings"]]
                break

    try:
        from . import tracing
        tracing.emit_llm_span(
            fn_name="embed_texts", model=model,
            prompt=f"{len(texts)} texts", response=f"{len(result)} vectors",
            usage=None, elapsed_s=time.monotonic() - t0, status="ok",
            provider="google", kind="EMBEDDING",
        )
    except Exception:
        pass
    return result
