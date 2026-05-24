"""
Grossberg Ch4 — interactive query (uses existing LightRAG storage)
Usage:
  python query.py                              # interactive, default PDF
  python query.py "your question"              # single query
  python query.py --pdf grossberg_ch4_p1.pdf "question"  # specific storage
"""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path.home() / ".oh-my-zsh/custom/apikey.env", override=False)

import yaml
from lightrag import LightRAG, QueryParam

from models import llm_model_func, embedding_func, answer_model_stream, ANSWER_PROVIDER_DEFAULT
from rerank import rerank as rerank_oneshot, rerank_batched

_cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())

_SAMPLE_QUESTIONS = [
    "What is the main theoretical framework introduced in this chapter?",
    "How does competitive learning and self-organization work in Grossberg's model?",
    "What role do on-center off-surround networks play?",
    "Explain the mathematical equations for lateral inhibition.",
    "What are the key differences between excitatory and inhibitory connections?",
]


def _parse_args() -> tuple[str | None, str | None, str, list[str]]:
    """Returns (pdf_stem, provider_override, rerank_mode, remaining_args)."""
    args = sys.argv[1:]
    pdf_stem = None
    provider = None
    rerank_mode = "oneshot"
    if "--pdf" in args:
        idx = args.index("--pdf")
        if idx + 1 < len(args):
            pdf_stem = Path(args[idx + 1]).stem
            args = args[:idx] + args[idx + 2:]
    if "--provider" in args:
        idx = args.index("--provider")
        if idx + 1 < len(args):
            provider = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
    if "--rerank" in args:
        idx = args.index("--rerank")
        if idx + 1 < len(args):
            rerank_mode = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
    return pdf_stem, provider, rerank_mode, args


def _working_dir(pdf_stem: str | None) -> str:
    base = _cfg["storage"]["working_dir"]
    return base + f"_{pdf_stem}" if pdf_stem else base


_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


_RERANK_FUNCS = {
    "none": None,
    "oneshot": rerank_oneshot,
    "batched": rerank_batched,
}


async def run_query_stream(
    question: str,
    working_dir: str,
    provider: str | None = None,
    rerank_mode: str = "oneshot",
):
    if rerank_mode not in _RERANK_FUNCS:
        raise ValueError(f"rerank_mode must be one of {list(_RERANK_FUNCS)}")
    fn = _RERANK_FUNCS[rerank_mode]

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        rerank_model_func=fn,
    )
    await rag.initialize_storages()
    try:
        mode = _cfg["query"]["default_mode"]
        # Stage 1: retrieval + prompt assembly (cheap intermediate LLM)
        assembled = await rag.aquery(
            question,
            param=QueryParam(mode=mode, only_need_prompt=True, enable_rerank=(fn is not None)),
        )
        marker = "\n\n---User Query---\n\n"
        if marker in assembled:
            sys_prompt, user_query = assembled.split(marker, 1)
        else:
            sys_prompt, user_query = assembled, question

        # Stage 2: streamed answer + reasoning from chosen provider
        print(f"{_DIM}[Reasoning]{_RESET}")
        current_kind = "reasoning"
        async for chunk in answer_model_stream(
            prompt=user_query, system_prompt=sys_prompt, provider=provider
        ):
            if chunk["type"] != current_kind:
                # Switch section header
                print(f"\n{_BOLD}[Answer]{_RESET}" if chunk["type"] == "answer" else f"\n{_DIM}[Reasoning]{_RESET}")
                current_kind = chunk["type"]
            if chunk["type"] == "reasoning":
                print(f"{_DIM}{chunk['delta']}{_RESET}", end="", flush=True)
            else:
                print(chunk["delta"], end="", flush=True)
        print()  # final newline
    finally:
        await rag.finalize_storages()


async def main():
    pdf_stem, provider, rerank_mode, args = _parse_args()
    wdir = _working_dir(pdf_stem)
    effective_provider = provider or ANSWER_PROVIDER_DEFAULT

    if not Path(wdir).exists():
        print(f"ERROR: Storage not found — {wdir}")
        print("       Run ingest.py first.")
        sys.exit(1)

    if args:
        question = " ".join(args)
        print(f"Q: {question}  ({_DIM}provider: {effective_provider} / rerank: {rerank_mode}{_RESET})\n")
        await run_query_stream(question, wdir, provider=provider, rerank_mode=rerank_mode)
        return

    # Interactive mode
    print(f"Grossberg RAG — interactive query (storage: {wdir}, provider: {effective_provider}, rerank: {rerank_mode})")
    print("Sample questions:")
    for i, q in enumerate(_SAMPLE_QUESTIONS, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        try:
            question = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not question:
            continue
        if question.isdigit() and 1 <= int(question) <= len(_SAMPLE_QUESTIONS):
            question = _SAMPLE_QUESTIONS[int(question) - 1]
            print(f"   -> {question}")
        await run_query_stream(question, wdir, provider=provider, rerank_mode=rerank_mode)
        print()


if __name__ == "__main__":
    asyncio.run(main())
