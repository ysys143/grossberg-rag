"""
LLM-based reranker using gemini-3.1-flash-lite.

Interface (LightRAG-standard):
  async def rerank(query: str, documents: list[str], top_n: int | None = None, **_)
      -> list[{"index": int, "relevance_score": float in [0,1]}]

Strategy:
  Single list-wise call — model sees all documents at once and scores 0-100.
  Deterministic (temperature=0, thinking_budget=0). Truncates very long docs
  to keep the prompt within budget.
"""
import json

from . import llm

_RERANK_MODEL = "gemini-3.1-flash-lite"
_MAX_DOC_CHARS = 1500

_PROMPT_TEMPLATE = """Rate the relevance of each document to the query on a 0-100 integer scale (higher = more relevant). Documents that directly answer the query should score 80+; tangentially related ones 30-60; unrelated ones 0-20.

Query: {query}

Documents:
{docs_block}

Output ONLY a JSON array, one object per document in original order:
[{{"index": 0, "score": <int>}}, {{"index": 1, "score": <int>}}, ...]
No prose, no markdown fences, JSON only.
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # ```json\n[...]\n``` → take middle
        parts = text.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            return inner.strip()
    return text


async def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    **_kwargs,
) -> list[dict]:
    if not documents:
        return []

    from . import tracing
    with tracing.span("rerank", "RERANKER", input_value=query) as rspan:
        if rspan is not None:
            rspan.set_attribute("reranker.query", query)
            rspan.set_attribute("reranker.model_name", _RERANK_MODEL)
            rspan.set_attribute("reranker.top_k", top_n or len(documents))
            rspan.set_attribute("metadata.input_doc_count", len(documents))

        parts = []
        for i, doc in enumerate(documents):
            body = doc if len(doc) <= _MAX_DOC_CHARS else doc[:_MAX_DOC_CHARS] + "..."
            parts.append(f"[{i}] {body}")
        docs_block = "\n\n".join(parts)

        prompt = _PROMPT_TEMPLATE.format(query=query, docs_block=docs_block)

        text = await llm.generate(
            model=_RERANK_MODEL,
            prompt=prompt,
            thinking_budget=0,
            temperature=0,
        )

        try:
            scored = json.loads(_strip_fences(text))
        except json.JSONDecodeError:
            fallback = [{"index": i, "relevance_score": 0.5} for i in range(len(documents))]
            if rspan is not None:
                rspan.set_attribute("metadata.parse_error", True)
                rspan.set_attribute("metadata.output_doc_count", len(fallback))
            return fallback

        out: list[dict] = []
        for s in scored:
            idx = s.get("index")
            score = s.get("score", 0)
            if isinstance(idx, int) and 0 <= idx < len(documents):
                out.append({"index": idx, "relevance_score": float(score) / 100.0})

        out.sort(key=lambda x: x["relevance_score"], reverse=True)
        result = out[:top_n] if top_n else out

        if rspan is not None:
            rspan.set_attribute("metadata.output_doc_count", len(result))
            rspan.set_attribute(
                "output.value",
                json.dumps([{"index": r["index"], "score": round(r["relevance_score"], 3)}
                            for r in result], ensure_ascii=False),
            )
        return result


async def rerank_batched(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    batch_size: int = 20,
    **_kwargs,
) -> list[dict]:
    """Two-stage batched rerank to mitigate lost-in-the-middle.

    Stage 1: split N docs into batches of B → rerank each → keep top B/2 per batch.
    Stage 2: merge survivors → one final list-wise rerank over the shorter set.

    Calls: ceil(N/B) + 1. Compared to one-shot: better attention per doc,
    less position bias, but B/2 cutoff in stage 1 risks dropping false negatives.
    """
    if not documents:
        return []
    final_top = top_n or len(documents)

    if len(documents) <= batch_size:
        return await rerank(query, documents, top_n=final_top)

    # Stage 1: per-batch rerank
    survivors_idx: list[int] = []
    survivors_score: dict[int, float] = {}
    for offset in range(0, len(documents), batch_size):
        batch = documents[offset:offset + batch_size]
        keep = max(1, len(batch) // 2)
        local_ranked = await rerank(query, batch, top_n=keep)
        for r in local_ranked:
            g_idx = offset + r["index"]
            survivors_idx.append(g_idx)
            survivors_score[g_idx] = r["relevance_score"]

    # Stage 2: final list-wise rerank over survivors
    if len(survivors_idx) <= final_top:
        ordered = sorted(survivors_idx, key=lambda i: survivors_score[i], reverse=True)
        return [{"index": i, "relevance_score": survivors_score[i]} for i in ordered]

    survivor_docs = [documents[i] for i in survivors_idx]
    final_ranked = await rerank(query, survivor_docs, top_n=final_top)
    return [
        {"index": survivors_idx[r["index"]], "relevance_score": r["relevance_score"]}
        for r in final_ranked
    ]
