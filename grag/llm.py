"""
LLM HTTP layer (no SDK).

Functions:
  generate / generate_with_vision  — Gemini, single-shot
  openai_generate                  — OpenAI Chat Completions, single-shot
  gemini_generate_stream           — Gemini streamGenerateContent (reasoning + answer parts)
  openai_generate_stream           — OpenAI Responses API (reasoning_summary + output streaming)

Streaming functions yield: {"type": "reasoning"|"answer", "delta": str}

Middleware:
  - `@with_logging` records every (non-stream) call to logs/llm_calls.jsonl.
    Set LLM_LOG_FULL=1 to log full content (no truncation).
  - Stream functions log inline once the iterator is exhausted.
"""
import asyncio
import base64
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import AsyncIterator

import httpx

from .paths import LOGS_DIR

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_OPENAI_BASE = "https://api.openai.com/v1"
_TIMEOUT = 120

_LOG_PATH = LOGS_DIR / "llm_calls.jsonl"
_LOG_FULL = os.environ.get("LLM_LOG_FULL", "0") == "1"
_PREVIEW_CHARS = 300


def _provider_of(model: str) -> str:
    m = (model or "").lower()
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    return "unknown"


def _api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    return key


def _openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return key


def _history_to_contents(history: list) -> list:
    result = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        result.append({"role": role, "parts": [{"text": msg["content"]}]})
    return result


def _preview(s: str | None) -> str | None:
    if s is None or _LOG_FULL:
        return s
    if len(s) <= _PREVIEW_CHARS:
        return s
    return s[:_PREVIEW_CHARS] + f"... [+{len(s) - _PREVIEW_CHARS} chars]"


def _append_log(entry: dict) -> None:
    _LOG_PATH.parent.mkdir(exist_ok=True)
    with _LOG_PATH.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ─── Gemini cachedContents: indexing-time prefix caching ──────────────────────
# LightRAG sends the same entity_extraction system_prompt 200+ times during
# indexing. We cache it once and reference by name → 75% discount on cached
# tokens.

_GEMINI_CACHE_TTL_SECS = 3600  # 1h — longer than typical indexing run
_GEMINI_CACHE_MIN_CHARS = 4000  # ~1000 tokens; below this Gemini rejects caching
_gemini_cache_registry: dict[str, tuple[str, float]] = {}  # hash → (name, expiry_epoch)
_gemini_cache_locks: dict[str, asyncio.Lock] = {}


async def _ensure_gemini_cache(model: str, system_prompt: str) -> str | None:
    """Return 'cachedContents/...' name, creating cache if needed.

    Returns None if the prompt is too small to cache or creation failed.
    Thread-safe across concurrent indexing calls via per-hash asyncio lock.
    """
    if not system_prompt or len(system_prompt) < _GEMINI_CACHE_MIN_CHARS:
        return None

    key = hashlib.sha256(f"{model}|{system_prompt}".encode()).hexdigest()

    lock = _gemini_cache_locks.setdefault(key, asyncio.Lock())
    async with lock:
        now = time.time()
        entry = _gemini_cache_registry.get(key)
        if entry and entry[1] > now + 60:  # ≥1min remaining
            return entry[0]

        body = {
            "model": f"models/{model}",
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "ttl": f"{_GEMINI_CACHE_TTL_SECS}s",
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{_BASE}/cachedContents",
                    params={"key": _api_key()},
                    json=body,
                )
                if resp.status_code >= 400:
                    # Too small / unsupported model / quota: fall back silently
                    _append_log({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "fn": "_ensure_gemini_cache",
                        "model": model,
                        "status": "skip",
                        "reason": f"{resp.status_code}: {resp.text[:200]}",
                    })
                    return None
                name = resp.json()["name"]
        except Exception as e:
            _append_log({
                "ts": datetime.now(timezone.utc).isoformat(),
                "fn": "_ensure_gemini_cache",
                "model": model,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            })
            return None

        _gemini_cache_registry[key] = (name, now + _GEMINI_CACHE_TTL_SECS)
        _append_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "fn": "_ensure_gemini_cache",
            "model": model,
            "status": "created",
            "cache_name": name,
            "system_prompt_chars": len(system_prompt),
        })
        return name


def _log_call(
    fn_name: str,
    model: str,
    prompt: str | None,
    response_text: str | None,
    usage: dict | None,
    elapsed: float,
    status: str,
    error: str | None = None,
    extra: dict | None = None,
    images: list | None = None,
) -> None:
    """Unified per-call log entry. Captures provider usage stats for cache analysis."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "fn": fn_name,
        "model": model,
        "prompt": _preview(prompt),
        "prompt_chars": len(prompt) if isinstance(prompt, str) else None,
        "response": _preview(response_text),
        "response_chars": len(response_text) if isinstance(response_text, str) else None,
        "usage": usage,
        "elapsed_s": round(elapsed, 3),
        "status": status,
    }
    if error:
        entry["error"] = error
    if extra:
        entry.update(extra)
    _append_log(entry)

    # Emit OpenInference span (no-op if tracing disabled)
    try:
        from . import tracing
        tracing.emit_llm_span(
            fn_name=fn_name, model=model, prompt=prompt, response=response_text,
            usage=usage, elapsed_s=elapsed, status=status,
            provider=_provider_of(model), images=images,
        )
    except Exception:
        pass


def with_logging(fn):
    """Middleware: log request/response + elapsed time to JSONL.

    Designed to compose with other middlewares (retry, cache, metrics) by
    stacking decorators. Closest decorator runs innermost.
    """
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        model = kwargs.get("model") if "model" in kwargs else (args[0] if args else None)
        prompt = kwargs.get("prompt") if "prompt" in kwargs else (args[1] if len(args) > 1 else None)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "fn": fn.__name__,
            "model": model,
            "prompt": _preview(prompt),
            "prompt_chars": len(prompt) if isinstance(prompt, str) else None,
        }
        t0 = time.monotonic()
        try:
            response = await fn(*args, **kwargs)
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["elapsed_s"] = round(time.monotonic() - t0, 3)
            _append_log(entry)
            raise
        entry["status"] = "ok"
        entry["response"] = _preview(response)
        entry["response_chars"] = len(response) if isinstance(response, str) else None
        entry["elapsed_s"] = round(time.monotonic() - t0, 3)
        _append_log(entry)
        return response

    return wrapper


async def generate(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history: list | None = None,
    temperature: float = 0.7,
    thinking_budget: int | None = None,
) -> str:
    """Non-streaming Gemini call.

    thinking_budget: int | None
        None  → Gemini default (thinking enabled, dynamic budget)
        0     → disable thinking (faster, cheaper; use for deterministic structured tasks)
        N     → cap at N tokens
    """
    contents = _history_to_contents(history or [])
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    gen_cfg: dict = {"temperature": temperature}
    if thinking_budget is not None:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}

    body: dict = {"contents": contents, "generationConfig": gen_cfg}

    # If system_prompt is long+stable, reference cached content instead of inlining.
    # On cache hit, the cached tokens are billed at ~25% rate (75% discount).
    cache_name = await _ensure_gemini_cache(model, system_prompt) if system_prompt else None
    if cache_name:
        body["cachedContent"] = cache_name  # cache contains systemInstruction
    elif system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{_BASE}/models/{model}:generateContent",
                params={"key": _api_key()},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata")
    except Exception as e:
        _log_call("generate", model, prompt, None, None, time.monotonic() - t0,
                  "error", f"{type(e).__name__}: {e}")
        raise
    _log_call("generate", model, prompt, text, usage, time.monotonic() - t0, "ok")
    return text


def _encode_image(img: str | Path) -> tuple[str, str]:
    """Return (data_uri, mime_type) from a data-URI string or a local file path.

    File bytes are base64-encoded; mime is inferred from the suffix (our figures
    are .jpg). Shared by the vision + streaming answer paths.
    """
    if isinstance(img, str) and img.startswith("data:"):
        return img, img.split(":", 1)[1].split(";", 1)[0]
    raw = Path(img).read_bytes()
    b64 = base64.b64encode(raw).decode()
    mime = "image/png" if str(img).lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{b64}", mime


def _gemini_image_part(data_uri: str, mime: str) -> dict:
    """data-URI -> Gemini inline_data part."""
    return {"inline_data": {"mime_type": mime, "data": data_uri.split(",", 1)[1]}}


async def generate_with_vision(
    model: str,
    prompt: str,
    images: list | None = None,
    system_prompt: str | None = None,
    raw_messages: list | None = None,
) -> str:
    """
    Generate with optional images.
    images: list of data-URIs or local file paths.
    raw_messages: pre-formatted list (OpenAI format) — converted to Gemini contents.
    """
    img_uris: list[str] = []  # collected for tracing (OpenInference multimodal)
    if raw_messages:
        contents = []
        for msg in raw_messages:
            if msg is None:
                continue
            role = "model" if msg.get("role") == "assistant" else "user"
            raw_content = msg.get("content", "")
            if isinstance(raw_content, str):
                parts = [{"text": raw_content}]
            else:
                parts = []
                for block in raw_content:
                    if block.get("type") == "text":
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image_url":
                        url = block["image_url"]["url"]
                        if url.startswith("data:"):
                            img_uris.append(url)
                            header, data = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                            parts.append({"inline_data": {"mime_type": mime, "data": data}})
            contents.append({"role": role, "parts": parts})
    else:
        parts = [{"text": prompt}]
        for img in images or []:
            if isinstance(img, str) and img.startswith("data:") or Path(str(img)).exists():
                data_uri, mime = _encode_image(img)
                img_uris.append(data_uri)
                parts.append(_gemini_image_part(data_uri, mime))
        contents = [{"role": "user", "parts": parts}]

    body: dict = {"contents": contents}
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{_BASE}/models/{model}:generateContent",
                params={"key": _api_key()},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata")
    except Exception as e:
        _log_call("generate_with_vision", model, prompt, None, None,
                  time.monotonic() - t0, "error", f"{type(e).__name__}: {e}",
                  extra={"images": len(img_uris)}, images=img_uris)
        raise
    _log_call("generate_with_vision", model, prompt, text, usage,
              time.monotonic() - t0, "ok",
              extra={"images": len(img_uris)}, images=img_uris)
    return text


async def openai_generate(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> str:
    """OpenAI Chat Completions via raw HTTP (no SDK).

    Reserved for final answer synthesis (high-reasoning model) — separated
    from Gemini calls so we can route by step: cheap Gemini for KG/keywords,
    high-reasoning OpenAI for the final user-facing response.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: dict = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature  # reasoning models (gpt-5*, o-series) reject this

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=300) as client:  # reasoning models can be slow
            resp = await client.post(
                f"{_OPENAI_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {_openai_key()}"},
                json=body,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"OpenAI {resp.status_code}: {resp.text}")
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage")
    except Exception as e:
        _log_call("openai_generate", model, prompt, None, None,
                  time.monotonic() - t0, "error", f"{type(e).__name__}: {e}")
        raise
    _log_call("openai_generate", model, prompt, text, usage,
              time.monotonic() - t0, "ok")
    return text


async def _log_stream_summary(fn_name: str, model: str, prompt: str | None,
                              reasoning_buf: list, answer_buf: list,
                              elapsed: float, status: str,
                              usage: dict | None = None,
                              error: str | None = None,
                              system_prompt: str | None = None,
                              images: list | None = None) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "fn": fn_name,
        "model": model,
        "prompt": _preview(prompt),
        "prompt_chars": len(prompt) if isinstance(prompt, str) else None,
        "reasoning": _preview("".join(reasoning_buf)),
        "reasoning_chars": sum(len(c) for c in reasoning_buf),
        "answer": _preview("".join(answer_buf)),
        "answer_chars": sum(len(c) for c in answer_buf),
        "usage": usage,
        "elapsed_s": round(elapsed, 3),
        "status": status,
    }
    if error:
        entry["error"] = error
    _append_log(entry)

    # Span input = the FULL injected prompt (system context + user query), so the
    # trace shows "how it was injected", not just the bare question.
    span_input = (
        f"{system_prompt}\n\n---User Query---\n\n{prompt}" if system_prompt else prompt
    )
    try:
        from . import tracing
        tracing.emit_llm_span(
            fn_name=fn_name, model=model, prompt=span_input,
            response="".join(answer_buf), usage=usage, elapsed_s=elapsed,
            status=status, provider=_provider_of(model), images=images,
        )
    except Exception:
        pass


async def gemini_generate_stream(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    include_thoughts: bool = True,
    thinking_budget: int = -1,
    images: list | None = None,
) -> AsyncIterator[dict]:
    """Stream Gemini response with thought parts (reasoning) + answer parts.

    images: optional list of data-URIs / local file paths injected alongside the
    text prompt (query-time figure re-injection).

    Yields: {"type": "reasoning"|"answer", "delta": str}
    """
    parts: list = [{"text": prompt}]
    img_uris: list[str] = []
    for img in images or []:
        data_uri, mime = _encode_image(img)
        img_uris.append(data_uri)
        parts.append(_gemini_image_part(data_uri, mime))
    body: dict = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "thinkingConfig": {
                "includeThoughts": include_thoughts,
                "thinkingBudget": thinking_budget,
            }
        },
    }
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    reasoning_buf: list[str] = []
    answer_buf: list[str] = []
    usage: dict | None = None
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{_BASE}/models/{model}:streamGenerateContent",
                params={"key": _api_key(), "alt": "sse"},
                json=body,
            ) as resp:
                if resp.status_code >= 400:
                    body_bytes = await resp.aread()
                    raise RuntimeError(f"Gemini {resp.status_code}: {body_bytes.decode()}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if "usageMetadata" in data:
                        usage = data["usageMetadata"]  # last chunk usually carries totals
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", []) or []
                    for part in parts:
                        text = part.get("text")
                        if not text:
                            continue
                        kind = "reasoning" if part.get("thought") else "answer"
                        (reasoning_buf if kind == "reasoning" else answer_buf).append(text)
                        yield {"type": kind, "delta": text}
    except Exception as e:
        await _log_stream_summary(
            "gemini_generate_stream", model, prompt, reasoning_buf, answer_buf,
            time.monotonic() - t0, "error", usage=usage, error=f"{type(e).__name__}: {e}",
            system_prompt=system_prompt, images=img_uris,
        )
        raise
    await _log_stream_summary(
        "gemini_generate_stream", model, prompt, reasoning_buf, answer_buf,
        time.monotonic() - t0, "ok", usage=usage, system_prompt=system_prompt,
        images=img_uris,
    )


async def openai_generate_stream(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    effort: str | None = None,
    images: list | None = None,
) -> AsyncIterator[dict]:
    """Stream OpenAI Responses API with reasoning summary + final output.

    effort: None -> model default; "low"|"medium"|"high" sets reasoning.effort,
    letting the router dial reasoning depth per question.
    images: optional list of data-URIs / local file paths injected as input_image
    blocks on the user turn (query-time figure re-injection).

    Yields: {"type": "reasoning"|"answer", "delta": str}
    """
    input_messages: list = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})

    img_uris: list[str] = []
    if images:
        user_content: list = [{"type": "input_text", "text": prompt}]
        for img in images:
            data_uri, _ = _encode_image(img)
            img_uris.append(data_uri)
            user_content.append({"type": "input_image", "image_url": data_uri})
        input_messages.append({"role": "user", "content": user_content})
    else:
        input_messages.append({"role": "user", "content": prompt})

    reasoning_cfg: dict = {"summary": "auto"}
    if effort:
        reasoning_cfg["effort"] = effort
    body = {
        "model": model,
        "input": input_messages,
        "reasoning": reasoning_cfg,
        "stream": True,
    }

    reasoning_buf: list[str] = []
    answer_buf: list[str] = []
    usage: dict | None = None
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{_OPENAI_BASE}/responses",
                headers={"Authorization": f"Bearer {_openai_key()}"},
                json=body,
            ) as resp:
                if resp.status_code >= 400:
                    body_bytes = await resp.aread()
                    raise RuntimeError(f"OpenAI {resp.status_code}: {body_bytes.decode()}")

                event: str | None = None
                async for line in resp.aiter_lines():
                    if line.startswith("event: "):
                        event = line[7:].strip()
                    elif line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        if event in ("response.reasoning_summary_text.delta", "response.output_text.delta"):
                            delta = data.get("delta", "")
                            if not delta:
                                continue
                            kind = "reasoning" if event == "response.reasoning_summary_text.delta" else "answer"
                            (reasoning_buf if kind == "reasoning" else answer_buf).append(delta)
                            yield {"type": kind, "delta": delta}
                        elif event == "response.completed":
                            # final event carries totals incl. input_tokens_details.cached_tokens
                            usage = data.get("response", {}).get("usage")
                        elif event in ("error", "response.failed"):
                            # Responses API can stream a failure (HTTP 200) — e.g.
                            # insufficient_quota. Surface it instead of returning a
                            # silently-empty answer.
                            err = (data.get("error")
                                   or data.get("response", {}).get("error") or {})
                            raise RuntimeError(
                                f"OpenAI stream failed: {err.get('code', '?')}: {err.get('message', '')}"
                            )
    except Exception as e:
        await _log_stream_summary(
            "openai_generate_stream", model, prompt, reasoning_buf, answer_buf,
            time.monotonic() - t0, "error", usage=usage, error=f"{type(e).__name__}: {e}",
            system_prompt=system_prompt, images=img_uris,
        )
        raise
    await _log_stream_summary(
        "openai_generate_stream", model, prompt, reasoning_buf, answer_buf,
        time.monotonic() - t0, "ok", usage=usage, system_prompt=system_prompt,
        images=img_uris,
    )
