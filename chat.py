"""
Conversational RAG CLI — multi-turn search over the Grossberg index.

Unlike query.py (each question independent), chat.py keeps conversation history
so follow-ups resolve against prior turns. History is passed to LightRAG
(conversation_history) for context-aware retrieval AND folded into the assembled
prompt the answer model sees.

Usage:
  python chat.py [--pdf FILE.pdf] [--provider openai|gemini] [--rerank none|oneshot|batched]

In-session commands:
  /help                 show commands
  /provider <name>      switch answer provider (openai | gemini)
  /rerank <mode>        switch rerank (none | oneshot | batched)
  /sources              show sources cited in the last answer
  /history              show the conversation so far
  /clear                reset conversation history
  /exit (or Ctrl-D)     quit
"""
import asyncio
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path.home() / ".oh-my-zsh/custom/apikey.env", override=False)

import yaml
from lightrag import LightRAG, QueryParam

from models import llm_model_func, embedding_func, answer_model_stream, ANSWER_PROVIDER_DEFAULT
from rerank import rerank as rerank_oneshot, rerank_batched
import llm
import tracing

tracing.init_tracing("grossberg-rag")
_tracer = tracing.get_tracer()

_cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())

_DIM, _BOLD, _CYAN, _GREEN, _RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[0m"
_MARKER = "\n\n---User Query---\n\n"
_RERANK_FUNCS = {"none": None, "oneshot": rerank_oneshot, "batched": rerank_batched}
_HISTORY_TURNS = 6  # how many prior turns LightRAG folds into retrieval/context
_SUMMARIZE_MODEL = "gemini-3.1-flash-lite"  # cheap, thinking-off summarizer


async def _summarize_cited(question: str, cited: list[str]) -> str:
    """One-shot summary of the cited chunks for this turn (append-only history).

    Keeps [src: ...] markers on the key facts so attribution survives into
    follow-up turns. Uses the fast flash-lite model with thinking disabled.
    """
    joined = "\n\n".join(cited)
    prompt = (
        "Summarize these cited source excerpts into 1-2 sentences capturing the "
        "key facts that answered the question. Preserve the [src: ...] marker for "
        "each key fact. Output only the summary.\n\n"
        f"Question: {question}\n\nCited excerpts:\n{joined}"
    )
    return await llm.generate(
        model=_SUMMARIZE_MODEL, prompt=prompt, thinking_budget=0, temperature=0
    )


class ChatSession:
    def __init__(self, working_dir: str, provider: str | None, rerank_mode: str):
        self.working_dir = working_dir
        self.provider = provider or ANSWER_PROVIDER_DEFAULT
        self.rerank_mode = rerank_mode
        self.history: list[dict] = []   # [{"role": "user"|"assistant", "content": str}]
        self.last_sources: list[str] = []
        self.rag: LightRAG | None = None

    async def setup(self):
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            rerank_model_func=_RERANK_FUNCS[self.rerank_mode],
        )
        await self.rag.initialize_storages()

    async def teardown(self):
        if self.rag is not None:
            await self.rag.finalize_storages()

    def set_rerank(self, mode: str):
        self.rerank_mode = mode
        self.rag.rerank_model_func = _RERANK_FUNCS[mode]  # live swap, no rebuild

    async def ask(self, question: str):
        mode = _cfg["query"]["default_mode"]
        fn = _RERANK_FUNCS[self.rerank_mode]

        chain_cm = (
            _tracer.start_as_current_span("chat_turn") if _tracer else _nullctx()
        )
        with chain_cm as span:
            if span is not None:
                span.set_attribute("openinference.span.kind", "CHAIN")
                span.set_attribute("input.value", question)
                span.set_attribute("metadata.turn", len(self.history) // 2 + 1)

            assembled = await self.rag.aquery(
                question,
                param=QueryParam(
                    mode=mode,
                    only_need_prompt=True,
                    enable_rerank=(fn is not None),
                    conversation_history=self.history[-_HISTORY_TURNS * 2:],
                    history_turns=_HISTORY_TURNS,
                ),
            )
            sys_prompt, user_query = (
                assembled.split(_MARKER, 1) if _MARKER in assembled else (assembled, question)
            )
            self.last_sources = _extract_sources(sys_prompt)

            print(f"{_DIM}[Reasoning]{_RESET}")
            answer_parts: list[str] = []
            current = "reasoning"
            async for chunk in answer_model_stream(
                prompt=user_query, system_prompt=sys_prompt, provider=self.provider
            ):
                if chunk["type"] != current:
                    print(f"\n{_BOLD}[Answer]{_RESET}" if chunk["type"] == "answer"
                          else f"\n{_DIM}[Reasoning]{_RESET}")
                    current = chunk["type"]
                if chunk["type"] == "reasoning":
                    print(f"{_DIM}{chunk['delta']}{_RESET}", end="", flush=True)
                else:
                    answer_parts.append(chunk["delta"])
                    print(chunk["delta"], end="", flush=True)
            print()
            answer = "".join(answer_parts)
            if span is not None:
                span.set_attribute("output.value", answer)

        # Append-only history: each turn stores a one-shot SUMMARY of its cited
        # chunks (never the verbose prose, never rewritten later). Past turns stay
        # frozen so the prompt prefix is stable -> provider prompt caching keeps
        # working. The current turn's full evidence always comes from fresh
        # retrieval, so summarizing the past loses no answer fidelity.
        cited = _cited_chunks(answer, sys_prompt)
        summary = await _summarize_cited(question, cited) if cited else "(no sources cited)"
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": summary})
        print(f"{_DIM}({len(cited)} cited chunks -> summary carried){_RESET}")


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _extract_sources(sys_prompt: str) -> list[str]:
    """Pull distinct [src: ...] markers from the assembled context."""
    seen, out = set(), []
    for m in re.findall(r"\[src:[^\]]+\]", sys_prompt):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _cited_pages(answer: str) -> set[int]:
    """Pages the answer cited: matches p.N and ranges p.N-M / p.N–M."""
    pages: set[int] = set()
    for a, b in re.findall(r"p\.\s*(\d+)\s*[-–]\s*(\d+)", answer):
        pages.update(range(int(a), int(b) + 1))
    for n in re.findall(r"p\.\s*(\d+)", answer):
        pages.add(int(n))
    return pages


def _cited_chunks(answer: str, sys_prompt: str) -> list[str]:
    """Return retrieved units (entity/relation/chunk) whose [src: ... p.N] page was
    cited in the answer.

    In hybrid/mix mode the Document Chunks block is often empty — retrieval is
    entity/relation-centric, and the [src:] markers live inside entity/relation
    DESCRIPTIONS (KG extraction absorbed them from the source text). So we scan
    every JSON object across all context blocks, not just Document Chunks. Page is
    the match key (clean integers vs OCR-noisy section names).
    """
    pages = _cited_pages(answer)
    if not pages:
        return []
    out, seen = [], set()
    for line in sys_prompt.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get("content") or obj.get("description") or ""
        pm = re.search(r"\[src:[^\]]*\|\s*p\.(\d+)", text)
        if pm and int(pm.group(1)) in pages and text not in seen:
            seen.add(text)
            out.append(text[:1000])
    return out


def _parse_args() -> tuple[str, str | None, str]:
    args = sys.argv[1:]
    pdf_stem, provider, rerank_mode = None, None, "oneshot"
    if "--pdf" in args:
        i = args.index("--pdf"); pdf_stem = Path(args[i + 1]).stem
    if "--provider" in args:
        i = args.index("--provider"); provider = args[i + 1]
    if "--rerank" in args:
        i = args.index("--rerank"); rerank_mode = args[i + 1]
    base = _cfg["storage"]["working_dir"]
    wdir = base + (f"_{pdf_stem}" if pdf_stem else "")
    return wdir, provider, rerank_mode


_HELP = """commands:
  /help              this help
  /provider <name>   switch answer provider (openai | gemini)
  /rerank <mode>     switch rerank (none | oneshot | batched)
  /sources           sources cited in the last answer
  /history           conversation so far
  /clear             reset conversation history
  /exit              quit (or Ctrl-D)"""


async def _handle_command(line: str, sess: ChatSession) -> bool:
    """Returns True if the loop should continue, False to exit."""
    parts = line.split()
    cmd = parts[0]
    if cmd in ("/exit", "/quit"):
        return False
    if cmd == "/help":
        print(_HELP)
    elif cmd == "/provider" and len(parts) > 1:
        sess.provider = parts[1]
        print(f"{_GREEN}provider -> {sess.provider}{_RESET}")
    elif cmd == "/rerank" and len(parts) > 1 and parts[1] in _RERANK_FUNCS:
        sess.set_rerank(parts[1])
        print(f"{_GREEN}rerank -> {parts[1]}{_RESET}")
    elif cmd == "/sources":
        if sess.last_sources:
            for s in sess.last_sources:
                print(f"  {s}")
        else:
            print(f"{_DIM}(no sources yet){_RESET}")
    elif cmd == "/history":
        for h in sess.history:
            tag = "Q" if h["role"] == "user" else "A"
            print(f"  {tag}: {h['content'][:100]}")
    elif cmd == "/clear":
        sess.history.clear()
        print(f"{_GREEN}history cleared{_RESET}")
    else:
        print(f"{_DIM}unknown command -- /help{_RESET}")
    return True


async def main():
    wdir, provider, rerank_mode = _parse_args()
    if not Path(wdir).exists():
        print(f"ERROR: Storage not found -- {wdir}\n       Run ingest.py first.")
        sys.exit(1)

    sess = ChatSession(wdir, provider, rerank_mode)
    await sess.setup()
    print(f"{_CYAN}{_BOLD}Grossberg RAG -- conversational search{_RESET}")
    print(f"{_DIM}storage={wdir} · provider={sess.provider} · rerank={sess.rerank_mode} · /help for commands{_RESET}\n")

    try:
        while True:
            try:
                line = input(f"{_CYAN}you >{_RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                break
            if not line:
                continue
            if line.startswith("/"):
                if not await _handle_command(line, sess):
                    print("bye.")
                    break
                continue
            await sess.ask(line)
            print()
    finally:
        await sess.teardown()
        tracing.shutdown_tracing()


if __name__ == "__main__":
    asyncio.run(main())
