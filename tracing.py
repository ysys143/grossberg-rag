"""
Arize AX tracing (OpenInference manual spans).

Why manual: this project calls Gemini/OpenAI via raw httpx (no SDK), so the
openinference auto-instrumentors (which patch SDK clients) cannot see our calls.
We emit spans manually at the same boundary where usage/timing is already
captured (llm._log_call / _log_stream_summary), plus CHAIN/RETRIEVER/RERANKER
spans around the query flow in query.py.

Credentials (read from env, loaded via dotenv from apikey.env):
  ARIZE_API_KEY, ARIZE_SPACE_ID

Enable with env var TRACING=1. No-ops silently when disabled or unconfigured,
so the pipeline runs unchanged without Arize.
"""
import os
import time

_ENABLED = os.environ.get("TRACING", "0") == "1"
_tracer = None
_provider = None


def init_tracing(project_name: str = "grossberg-rag"):
    """Register Arize OTLP exporter. Idempotent. Returns tracer or None."""
    global _tracer, _provider
    if not _ENABLED or _tracer is not None:
        return _tracer
    if not os.environ.get("ARIZE_API_KEY") or not os.environ.get("ARIZE_SPACE_ID"):
        return None

    from arize.otel import register
    from opentelemetry.trace import get_tracer

    _provider = register(
        space_id=os.environ["ARIZE_SPACE_ID"],
        api_key=os.environ["ARIZE_API_KEY"],
        project_name=project_name,
    )
    _tracer = get_tracer(__name__)
    return _tracer


def shutdown_tracing() -> None:
    """Flush pending spans before CLI exit (async OTLP exports are dropped otherwise)."""
    if _provider is not None:
        try:
            _provider.force_flush()
        finally:
            _provider.shutdown()


def get_tracer():
    return _tracer


from contextlib import contextmanager


@contextmanager
def span(name: str, kind: str, input_value: str | None = None):
    """Context manager for a nested span. Yields the span (or None if disabled).

    Child LLM spans emitted within the block nest under this span via OTel context.
    """
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        if input_value is not None:
            s.set_attribute("input.value", input_value)
        yield s


def emit_llm_span(
    fn_name: str,
    model: str,
    prompt: str | None,
    response: str | None,
    usage: dict | None,
    elapsed_s: float,
    status: str,
    provider: str | None = None,
    kind: str = "LLM",
    images: list | None = None,
) -> None:
    """Retroactively create+end a span at the logging boundary.

    Uses explicit start/end timestamps from elapsed_s. Becomes a child of any
    active span (e.g. the query CHAIN span) via OTel context propagation.

    images: list of data-URI strings. When present, the span is marked as a
    multimodal input — each image is attached via OpenInference message-content
    conventions so Arize renders the actual image the vision model described.
    """
    if _tracer is None:
        return

    from opentelemetry.trace import Status, StatusCode

    end_ns = time.time_ns()
    start_ns = end_ns - int(elapsed_s * 1e9)
    span = _tracer.start_span(fn_name, start_time=start_ns)

    span.set_attribute("openinference.span.kind", kind)
    span.set_attribute("llm.model_name", model)
    if provider:
        span.set_attribute("llm.provider", provider)
    if prompt is not None:
        span.set_attribute("input.value", prompt)
    if response is not None:
        span.set_attribute("output.value", response)

    # Multimodal: attach images as OpenInference message-content blocks
    if images:
        span.set_attribute("metadata.image_count", len(images))
        span.set_attribute("metadata.modality", "multimodal")
        base = "llm.input_messages.0.message"
        span.set_attribute(f"{base}.role", "user")
        # content[0] = the text prompt
        span.set_attribute(f"{base}.contents.0.message_content.type", "text")
        if prompt is not None:
            span.set_attribute(f"{base}.contents.0.message_content.text", prompt[:2000])
        # content[1..] = images (cap embedded size to avoid oversized spans)
        slot = 1
        for img in images:
            if not isinstance(img, str):
                continue
            span.set_attribute(f"{base}.contents.{slot}.message_content.type", "image")
            if len(img) <= 1_500_000:  # ~1.5MB cap on a single data URI
                span.set_attribute(
                    f"{base}.contents.{slot}.message_content.image.image.url", img
                )
            else:
                span.set_attribute(
                    f"{base}.contents.{slot}.message_content.image.image.url",
                    "[image too large to embed]",
                )
            slot += 1

    if usage:
        # Normalize both Gemini and OpenAI usage schemas to OpenInference token counts
        prompt_toks = usage.get("promptTokenCount") or usage.get("input_tokens")
        out_toks = usage.get("candidatesTokenCount") or usage.get("output_tokens")
        cached = usage.get("cachedContentTokenCount")
        if cached is None:
            cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
        thoughts = usage.get("thoughtsTokenCount")
        if prompt_toks is not None:
            span.set_attribute("llm.token_count.prompt", int(prompt_toks))
        if out_toks is not None:
            span.set_attribute("llm.token_count.completion", int(out_toks))
        if prompt_toks is not None and out_toks is not None:
            span.set_attribute("llm.token_count.total", int(prompt_toks) + int(out_toks))
        if cached is not None:
            span.set_attribute("llm.token_count.prompt_details.cache_read", int(cached))
        if thoughts is not None:
            span.set_attribute("llm.token_count.completion_details.reasoning", int(thoughts))

    if status == "ok":
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_status(Status(StatusCode.ERROR))

    span.end(end_time=end_ns)
