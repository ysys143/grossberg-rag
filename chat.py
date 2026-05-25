"""
Conversational RAG CLI — multi-turn search over the Grossberg index.

Thin consumer of `engine.ask_events` (shared with server.py): this file renders
the engine's event stream to the terminal. Logic lives in engine.py; presentation
(ANSI prints, Korean status, readline, /commands) lives here. Changing chat
*behavior* means editing engine.ask_events, not this file — that keeps the CLI and
the web app from drifting.

Usage:
  python chat.py [--pdf FILE.pdf] [--provider openai|gemini] [--rerank none|oneshot|batched]
                 [--agent] [--session NAME] [--resume [ID]]

In-session commands:
  /help  /provider <name>  /rerank <mode>  /sources  /history  /sessions  /clear  /exit
"""
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    import readline  # noqa: F401 — enables backspace/arrow-key/history line editing in input()
except ImportError:
    pass

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path.home() / ".oh-my-zsh/custom/apikey.env", override=False)

import yaml

from engine import ChatSession, ask_events, _RERANK_FUNCS, _SESSIONS_DIR
import agent as agent_mod
import tracing

# Pure-noise loggers stay silenced; the lightrag logger is re-routed to a Korean
# status handler below (after the color constants it depends on are defined).
for _name in ("nano-vectordb", "nano_vectordb", "httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

tracing.init_tracing("grossberg-rag")

_cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())

_DIM, _BOLD, _CYAN, _GREEN, _RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[0m"
# readline-safe input prompt: wrap non-printing ANSI in \001..\002 so backspace /
# arrow keys / cursor position aren't thrown off by counting invisible bytes.
_INPUT_PROMPT = "\001\033[36m\002you >\001\033[0m\002 "


# LightRAG's English INFO logs are translated to Korean status events inside
# engine.ask_events (single source); cli_ask renders them like any other status.


async def cli_ask(sess: ChatSession, question: str, skip_clarify: bool = False) -> None:
    """Render engine.ask_events to the terminal (CLI presentation of one turn)."""
    current: str | None = None
    streamed = False
    async for ev in ask_events(sess, question, skip_clarify):
        t = ev["type"]
        if t == "status":
            pre = "·   " if ev.get("detail") else "· "
            print(f"{_DIM}{pre}{ev['msg']}{_RESET}")
        elif t == "routing":
            need = ev["needs_retrieval"]
            print(f"{_DIM}·   판단: 검색 {'필요' if need else '불필요'} · 추론 강도 {ev['effort']}{_RESET}")
        elif t == "decline":
            print(f"\n{ev['msg']}\n")
        elif t == "clarify":
            print(f"\n{_BOLD}· 질문을 좀 더 구체화해 주세요:{_RESET}")
            print(f"  {ev['question']}\n")
        elif t in ("reasoning", "answer"):
            streamed = True
            if t != current:
                print(f"\n{_DIM}[Reasoning]{_RESET}" if t == "reasoning"
                      else f"\n{_BOLD}[Answer]{_RESET}")
                current = t
            if t == "reasoning":
                print(f"{_DIM}{ev['delta']}{_RESET}", end="", flush=True)
            else:
                print(ev["delta"], end="", flush=True)
        elif t == "done":
            if streamed:
                print()
            if ev.get("summarized"):
                print(f"{_DIM}· 인용된 근거 {ev['cited']}개를 요약해 대화 기억에 추가했습니다{_RESET}")
        # images / sources events: ignored by the CLI (sources shown via /sources)


def _parse_args() -> tuple[str, str | None, str, Path, bool]:
    args = sys.argv[1:]
    pdf_stem, provider, rerank_mode = None, None, "oneshot"
    session_name, resume = None, False
    agent_mode = "--agent" in args
    if "--pdf" in args:
        i = args.index("--pdf"); pdf_stem = Path(args[i + 1]).stem
    if "--provider" in args:
        i = args.index("--provider"); provider = args[i + 1]
    if "--rerank" in args:
        i = args.index("--rerank"); rerank_mode = args[i + 1]
    if "--session" in args:
        i = args.index("--session"); session_name = args[i + 1]
    for flag in ("--resume", "--continue", "-c"):
        if flag in args:
            i = args.index(flag)
            nxt = args[i + 1] if i + 1 < len(args) else None
            if nxt and not nxt.startswith("-"):
                session_name = nxt   # --resume <id>: resume a specific session
            else:
                resume = True        # --resume (bare): resume most recent
            break
    base = _cfg["storage"]["working_dir"]
    wdir = base + (f"_{pdf_stem}" if pdf_stem else "")

    # Resolve which session file to use
    if session_name:
        session_path = _SESSIONS_DIR / f"{session_name}.json"
    elif resume:
        existing = sorted(_SESSIONS_DIR.glob("*.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        session_path = existing[0] if existing else _SESSIONS_DIR / f"{_session_ts()}.json"
    else:
        session_path = _SESSIONS_DIR / f"{_session_ts()}.json"
    return wdir, provider, rerank_mode, session_path, agent_mode


def _session_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


_HELP = """commands:
  /help              this help
  /provider <name>   switch answer provider (openai | gemini)
  /rerank <mode>     switch rerank (none | oneshot | batched)
  /sources           sources cited in the last answer
  /history           conversation so far
  /sessions          list saved sessions (* = current)
  /clear             reset conversation history (saved)
  /exit              quit (or 'exit', Ctrl-D)
session: auto-saved each turn -> sessions/<name>.json
  --session NAME              resume/create a named session
  --resume|--continue|-c [id] continue most recent, or a specific <id> if given
  --agent                     use the agentic loop (multi-step retrieval + web search)"""


async def _run_agent_turn(sess: ChatSession, question: str) -> None:
    """Handle one question turn via the agentic loop (agent.py).

    The agent manages its own KBTool/LightRAG internally; sess.rag is unused
    in this path. sess.history and sess.provider are passed through for
    multi-turn context and provider selection.
    """
    answer_parts: list[str] = []
    current: str | None = None

    async for chunk in agent_mod.run_agent_stream(
        question=question,
        working_dir=sess.working_dir,
        provider=sess.provider,
        conversation_history=sess.history,
    ):
        t = chunk["type"]
        d = chunk["delta"]
        if t == "status":
            print(f"{_DIM}{d}{_RESET}", flush=True)
        elif t == "reasoning":
            if current != "reasoning":
                print(f"\n{_DIM}[Reasoning]{_RESET}")
                current = "reasoning"
            print(f"{_DIM}{d}{_RESET}", end="", flush=True)
        elif t == "answer":
            if current != "answer":
                print(f"\n{_BOLD}[Answer]{_RESET}\n")
                current = "answer"
            answer_parts.append(d)
            print(d, end="", flush=True)
    print()

    answer = "".join(answer_parts)
    sess.history.append({"role": "user", "content": question})
    sess.history.append({"role": "assistant", "content": answer[:300]})
    sess.save()


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
            injected = set(sess.last_image_sources)
            for s in sess.last_sources:
                tag = f"  {_GREEN}[이미지 주입됨]{_RESET}" if s in injected else ""
                print(f"  {s}{tag}")
        else:
            print(f"{_DIM}(no sources yet){_RESET}")
    elif cmd == "/history":
        for h in sess.history:
            tag = "Q" if h["role"] == "user" else "A"
            print(f"  {tag}: {h['content'][:100]}")
    elif cmd == "/clear":
        sess.history.clear()
        sess.pending_question = None
        sess.save()
        print(f"{_GREEN}history cleared{_RESET}")
    elif cmd == "/sessions":
        files = sorted(_SESSIONS_DIR.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print(f"{_DIM}(no saved sessions){_RESET}")
        for p in files[:15]:
            try:
                d = json.loads(p.read_text())
                turns = len(d.get("history", [])) // 2
                mark = " *" if p == sess.session_path else ""
                print(f"  {p.stem}  ({turns} turns, {d.get('updated_at', '?')}){mark}")
            except (json.JSONDecodeError, OSError):
                pass
    else:
        print(f"{_DIM}unknown command -- /help{_RESET}")
    return True


async def main():
    wdir, provider, rerank_mode, session_path, agent_mode = _parse_args()
    if not Path(wdir).exists():
        print(f"ERROR: Storage not found -- {wdir}\n       Run ingest.py first.")
        sys.exit(1)

    sess = ChatSession(wdir, provider, rerank_mode, session_path)
    prior = sess.load()
    if not agent_mode:
        await sess.setup()

    print(f"{_CYAN}{_BOLD}Grossberg RAG -- conversational search{_RESET}")
    mode_tag = "agent=true" if agent_mode else f"rerank={sess.rerank_mode}"
    print(f"{_DIM}storage={wdir} · provider={sess.provider} · {mode_tag} · session={session_path.stem} · /help{_RESET}")
    if prior:
        print(f"{_DIM}· 이전 세션 {prior}개 턴을 이어갑니다{_RESET}")
    print()

    try:
        while True:
            try:
                line = input(_INPUT_PROMPT).strip()
            except EOFError:           # Ctrl-D -> exit
                print("\nbye.")
                break
            except KeyboardInterrupt:  # Ctrl-C at the prompt -> exit
                print("\nbye.")
                break
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                print("bye.")
                break
            if line.startswith("/"):
                if not await _handle_command(line, sess):
                    print("bye.")
                    break
                continue
            try:
                if agent_mode:
                    await _run_agent_turn(sess, line)
                elif sess.pending_question is not None:
                    # This line answers a prior clarification request: merge it with
                    # the original question and force past the gate (skip_clarify).
                    merged = f"{sess.pending_question}\n\n[사용자 보충 설명] {line}"
                    sess.pending_question = None
                    await cli_ask(sess, merged, skip_clarify=True)
                else:
                    await cli_ask(sess, line)
            except KeyboardInterrupt:  # Ctrl-C during streaming -> cancel this turn
                print(f"\n{_DIM}· 생성을 중단했습니다 (종료: /exit 또는 빈 입력에서 Ctrl-C){_RESET}")
                continue
            print()
    finally:
        if not agent_mode:
            await sess.teardown()
        tracing.shutdown_tracing()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:  # final guard: clean exit, no traceback
        print("\nbye.")
