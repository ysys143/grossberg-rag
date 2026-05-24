"""
A/B test rerank modes for a given question.

Usage:
    python ab_test.py "your question here"
    python ab_test.py --pdf grossberg_ch4_p3.pdf "question"

Runs the same question through three pipelines:
  (a) none      — no rerank, LightRAG's vector-similarity ordering only
  (b) oneshot   — single list-wise LLM rerank over all candidates (current default)
  (c) batched   — two-stage: per-batch rerank + final list-wise over survivors

For each run, captures:
  - which chunks landed in the final assembled prompt (by reference_id)
  - the streamed final answer
  - cumulative token/cost stats from logs/llm_calls.jsonl

Then prints a side-by-side digest.
"""
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path.home() / ".oh-my-zsh/custom/apikey.env", override=False)

import yaml
from lightrag import LightRAG, QueryParam

from models import (
    llm_model_func,
    embedding_func,
    answer_model_stream,
    ANSWER_PROVIDER_DEFAULT,
)
from rerank import rerank as rerank_oneshot, rerank_batched

_cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())

_LOG_PATH = Path(__file__).parent / "logs" / "llm_calls.jsonl"


def _log_offset() -> int:
    return _LOG_PATH.stat().st_size if _LOG_PATH.exists() else 0


def _read_log_after(offset: int) -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    with _LOG_PATH.open("rb") as f:
        f.seek(offset)
        raw = f.read().decode("utf-8", errors="ignore")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


async def run_mode(question: str, working_dir: str, mode: str) -> dict:
    """Run one mode end-to-end. Returns {answer, chunks_used, tokens, elapsed}."""
    rerank_fn = {"none": None, "oneshot": rerank_oneshot, "batched": rerank_batched}[mode]

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        rerank_model_func=rerank_fn,
    )
    await rag.initialize_storages()

    offset_before = _log_offset()
    try:
        retrieval_mode = _cfg["query"]["default_mode"]
        assembled = await rag.aquery(
            question,
            param=QueryParam(
                mode=retrieval_mode,
                only_need_prompt=True,
                enable_rerank=(rerank_fn is not None),
            ),
        )
        marker = "\n\n---User Query---\n\n"
        if marker in assembled:
            sys_prompt, user_query = assembled.split(marker, 1)
        else:
            sys_prompt, user_query = assembled, question

        # Extract reference_ids that survived into the assembled context
        ref_ids = [int(r) for r in __import__("re").findall(r'"reference_id":\s*(\d+)', sys_prompt)]

        # Synthesize final answer (streamed → accumulated here for comparison)
        answer_parts: list[str] = []
        async for chunk in answer_model_stream(prompt=user_query, system_prompt=sys_prompt):
            if chunk["type"] == "answer":
                answer_parts.append(chunk["delta"])
        answer = "".join(answer_parts)
    finally:
        await rag.finalize_storages()

    new_logs = _read_log_after(offset_before)
    total_prompt_toks = 0
    total_cached_toks = 0
    total_output_toks = 0
    rerank_calls = 0
    for e in new_logs:
        u = e.get("usage") or {}
        # Gemini schema
        if "promptTokenCount" in u:
            total_prompt_toks += u.get("promptTokenCount") or 0
            total_cached_toks += u.get("cachedContentTokenCount") or 0
            total_output_toks += u.get("candidatesTokenCount") or 0
        # OpenAI schema
        elif "input_tokens" in u:
            total_prompt_toks += u.get("input_tokens") or 0
            total_cached_toks += (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
            total_output_toks += u.get("output_tokens") or 0
        if e.get("model", "").endswith("flash-lite") and e.get("fn") == "generate":
            rerank_calls += 1

    return {
        "answer": answer,
        "ref_ids": sorted(set(ref_ids)),
        "rerank_calls": rerank_calls,
        "tokens_in": total_prompt_toks,
        "tokens_cached": total_cached_toks,
        "tokens_out": total_output_toks,
    }


def _parse_args() -> tuple[str | None, list[str]]:
    args = sys.argv[1:]
    pdf_stem = None
    if "--pdf" in args:
        i = args.index("--pdf")
        pdf_stem = Path(args[i + 1]).stem
        args = args[:i] + args[i + 2:]
    return pdf_stem, args


async def main():
    pdf_stem, args = _parse_args()
    if not args:
        print("Usage: python ab_test.py [--pdf FILE.pdf] \"question\"")
        sys.exit(1)
    question = " ".join(args)

    base = _cfg["storage"]["working_dir"]
    wdir = base + (f"_{pdf_stem}" if pdf_stem else "")

    if not Path(wdir).exists():
        print(f"ERROR: Storage not found — {wdir}")
        sys.exit(1)

    print(f"Q: {question}")
    print(f"Storage: {wdir}  |  Answer provider: {ANSWER_PROVIDER_DEFAULT}\n")

    results = {}
    for mode in ["none", "oneshot", "batched"]:
        print(f"--- Running mode: {mode} ---")
        results[mode] = await run_mode(question, wdir, mode)
        r = results[mode]
        print(f"  refs={r['ref_ids']}  rerank_calls={r['rerank_calls']}  "
              f"in={r['tokens_in']} cached={r['tokens_cached']} out={r['tokens_out']}\n")

    # Final digest
    print("=" * 72)
    print("DIGEST")
    print("=" * 72)
    for mode in ["none", "oneshot", "batched"]:
        r = results[mode]
        print(f"\n[{mode}]")
        print(f"  chunks used (ref_ids): {r['ref_ids']}")
        print(f"  rerank LLM calls:      {r['rerank_calls']}")
        print(f"  tokens (in/cached/out): {r['tokens_in']}/{r['tokens_cached']}/{r['tokens_out']}")
        print(f"  answer ({len(r['answer'])} chars):")
        for line in r["answer"].splitlines():
            print(f"    | {line}")

    # Overlap analysis
    none_set = set(results["none"]["ref_ids"])
    one_set = set(results["oneshot"]["ref_ids"])
    bat_set = set(results["batched"]["ref_ids"])
    print("\n" + "=" * 72)
    print("CHUNK SET OVERLAP")
    print("=" * 72)
    print(f"  none ∩ oneshot:  {len(none_set & one_set)}/{len(none_set | one_set)}")
    print(f"  none ∩ batched:  {len(none_set & bat_set)}/{len(none_set | bat_set)}")
    print(f"  oneshot ∩ batched: {len(one_set & bat_set)}/{len(one_set | bat_set)}")
    print(f"  in oneshot only:  {sorted(one_set - none_set - bat_set)}")
    print(f"  in batched only:  {sorted(bat_set - none_set - one_set)}")
    print(f"  in none only:     {sorted(none_set - one_set - bat_set)}")


if __name__ == "__main__":
    asyncio.run(main())
