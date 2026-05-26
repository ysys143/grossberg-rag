"""
Shared chat engine — event-yielding core consumed by both the CLI (chat.py) and
the web server (server.py).

`ask_events()` is the single source of truth for a chat turn: router → (clarify
gate) → retrieval → query-time image selection → answer streaming → append-only
cited-summary memory. Instead of printing, it *yields* typed events so each
consumer renders them its own way (CLI prints; web emits SSE). This keeps CLI and
web behavior identical by construction.

Event types yielded by ask_events:
  {"type":"status","msg":str,"detail":bool}        progress line (detail=indented sub-line)
  {"type":"routing","needs_retrieval":bool,"effort":str}
  {"type":"clarify","question":str}                vague question -> stream ends, await reply
  {"type":"decline","msg":str}                     out-of-scope -> stream ends
  {"type":"images","items":[{hash,section,page,marker,path}]}
  {"type":"reasoning","delta":str} / {"type":"answer","delta":str}
  {"type":"sources","items":[{"marker":str,"injected":bool}]}
  {"type":"done","cited":int,"summarized":bool}
"""
import asyncio
import glob
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import yaml
from lightrag import LightRAG, QueryParam

from .models import answer_model_stream, ANSWER_PROVIDER_DEFAULT, is_vision_capable
from . import image_gate
from . import llm
from . import router
from . import expand
from . import retrieval
from . import tracing
from .paths import CONFIG_PATH, SESSIONS_DIR, DATA_DIR

_cfg = yaml.safe_load(CONFIG_PATH.read_text())
_MARKER = "\n\n---User Query---\n\n"
_RERANK_FUNCS = retrieval.RERANK_FUNCS  # rerank mode -> func (shared with kb_tool via retrieval)
_HISTORY_TURNS = 6  # how many prior turns LightRAG folds into retrieval/context
_SUMMARIZE_MODEL = "gemini-3.1-flash-lite"  # cheap, thinking-off summarizer
_SESSIONS_DIR = SESSIONS_DIR  # persisted conversation history
_INJECT_IMAGES = bool(_cfg["query"].get("inject_images", False))  # query-time figure re-injection

# LightRAG emits English INFO logs during retrieval. We silence its console and,
# during aquery, capture+translate the query-stage lines into Korean status events
# (shared by CLI and web — single source, no duplicated handler in chat.py).
_lr_logger = logging.getLogger("lightrag")
_lr_logger.setLevel(logging.INFO)
_lr_logger.propagate = False  # no console spam; ask_events attaches a capturing handler

_LR_MAP = [
    (re.compile(r"Query nodes: (.+?) \(top_k"), lambda m: f"엔티티 검색 키워드: {m.group(1)}"),
    (re.compile(r"Query edges: (.+?) \(top_k"), lambda m: f"관계 검색 키워드: {m.group(1)}"),
    (re.compile(r"Local query: (\d+) entites?, (\d+) relations"),
     lambda m: f"지역 검색(엔티티 중심): 엔티티 {m.group(1)}개, 관계 {m.group(2)}개"),
    (re.compile(r"Global query: (\d+) entites?, (\d+) relations"),
     lambda m: f"전역 검색(관계 중심): 엔티티 {m.group(1)}개, 관계 {m.group(2)}개"),
    (re.compile(r"Raw search results: (\d+) entities, (\d+) relations, (\d+) vector chunks"),
     lambda m: f"원시 검색 결과: 엔티티 {m.group(1)}개, 관계 {m.group(2)}개, 벡터청크 {m.group(3)}개"),
    (re.compile(r"After truncation: (\d+) entities, (\d+) relations"),
     lambda m: f"토큰 한도 적용 후: 엔티티 {m.group(1)}개, 관계 {m.group(2)}개"),
    (re.compile(r"Round-robin merged chunks: (\d+) -> (\d+) \(deduplicated (\d+)\)"),
     lambda m: f"청크 병합: {m.group(1)} → {m.group(2)}개 (중복 {m.group(3)}개 제거)"),
    (re.compile(r"Successfully reranked: (\d+) chunks from (\d+) original chunks"),
     lambda m: f"관련도 재정렬: {m.group(2)}개 → {m.group(1)}개"),
    (re.compile(r"Final context: (\d+) entities, (\d+) relations, (\d+) chunks"),
     lambda m: f"최종 컨텍스트: 엔티티 {m.group(1)}개, 관계 {m.group(2)}개, 청크 {m.group(3)}개"),
]


class _LRCapture(logging.Handler):
    """Collects translated LightRAG query-stage lines during aquery (yielded after)."""
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        for pat, fn in _LR_MAP:
            m = pat.search(msg)
            if m:
                self.lines.append(fn(m))
                return


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


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


def _extract_sources(sys_prompt: str) -> list[str]:
    """Pull distinct, well-formed [src: ...] markers from the assembled context.

    Bounded length + no nested brackets so a truncated marker can't greedily
    swallow following JSON content into the match.
    """
    seen, out = set(), []
    for m in re.findall(r"\[src:[^\[\]]{1,200}\]", sys_prompt):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _source_contents(sys_prompt: str) -> dict:
    """Map each [src:] marker -> the chunk text it appears in (path-stripped,
    bounded), so the UI can show a source's content on click."""
    out: dict = {}
    for line in sys_prompt.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get("content") or obj.get("description") or ""
        if not text:
            continue
        clean = re.sub(r"Image Path:\s*\S+", "Image Path: (생략)", text)
        for mk in re.findall(r"\[src:[^\[\]]{1,200}\]", text):
            out.setdefault(mk, clean[:1500])
    return out


def _resolve_image_path(img_hash: str, content: str) -> str | None:
    """Recover the original figure file: prefer the absolute path embedded in the
    chunk ("Image Path: /.../<hash>.jpg"); else glob the output images dir by hash."""
    m = re.search(r"(/\S+?/images/" + re.escape(img_hash) + r"\.\w+)", content)
    if m and Path(m.group(1)).exists():
        return m.group(1)
    hits = glob.glob(str(DATA_DIR / "output*" / "**" / "images" / f"{img_hash}.*"), recursive=True)
    return hits[0] if hits else None


def _image_candidates(sys_prompt: str) -> list[dict]:
    """Distinct figure chunks in the assembled context, as gate candidates:
    {hash, caption, section, page, marker, path}. Drops any whose file can't be found."""
    out, seen = [], set()
    for line in sys_prompt.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = obj.get("content") or obj.get("description") or ""
        if "Image Path" not in content:
            continue
        hm = re.search(r"images/([a-f0-9]+)", content)
        if not hm or hm.group(1) in seen:
            continue
        img_hash = hm.group(1)
        path = _resolve_image_path(img_hash, content)
        if not path:
            continue
        seen.add(img_hash)
        marker_m = re.search(r"\[src:[^\]]*image\]", content)
        marker = marker_m.group(0) if marker_m else ""
        sec_m = re.search(r"§(.+?)\s*\|\s*p\.(\d+)", marker or content)
        out.append({
            "hash": img_hash,
            "caption": content[:4000],  # gate truncates to 400; extra feeds full figure title/desc
            "section": sec_m.group(1).strip() if sec_m else "?",
            "page": int(sec_m.group(2)) if sec_m else 0,
            "marker": marker,
            "path": path,
        })
    return out


def _figure_meta(caption: str) -> tuple[str, str]:
    """(figure title line, visual-analysis description) parsed from a chunk excerpt,
    for the image-preview overlay."""
    fig = re.search(r"FIGURE\s+[\d.]+[^\n]*", caption)
    title = fig.group(0).strip() if fig else ""
    va = caption.find("Visual Analysis")
    desc = caption[va + len("Visual Analysis:"):].strip() if va >= 0 else ""
    return title, desc[:3000]


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
    cited in the answer. Scans every JSON object across all context blocks (hybrid
    mode keeps [src:] markers inside entity/relation descriptions)."""
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


class ChatSession:
    """Conversation state shared by the CLI and the web server."""

    def __init__(self, working_dir: str, provider: str | None, rerank_mode: str,
                 session_path: Path):
        self.working_dir = working_dir
        self.provider = provider or ANSWER_PROVIDER_DEFAULT
        self.rerank_mode = rerank_mode
        self.session_path = session_path
        self.history: list[dict] = []   # [{"role": "user"|"assistant", "content": str}]
        self.last_sources: list[str] = []
        self.last_image_sources: list[str] = []   # [src:...|image] markers injected last turn
        self.rag: LightRAG | None = None
        self.pending_question: str | None = None  # original Q awaiting HITL clarification
        self.corpus_lang: str = "en"  # language of the index; target for keyword expansion
        self.glossary: str = ""       # corpus characteristic terms injected into expansion

    def load(self) -> int:
        """Restore history. Returns prior turn count."""
        if not self.session_path.exists():
            return 0
        data = json.loads(self.session_path.read_text())
        self.history = data.get("history", [])
        return len(self.history) // 2

    def save(self) -> None:
        """Atomic write of the session so a crash mid-turn can't corrupt it."""
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "working_dir": self.working_dir,
            "provider": self.provider,
            "rerank_mode": self.rerank_mode,
            "history": self.history,
        }
        tmp = self.session_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(self.session_path)

    async def setup(self):
        # Shared factory applies construction-time enhancements (rerank, hybrid_seed).
        self.rag = await retrieval.build_rag(self.working_dir, self.rerank_mode)
        if _cfg["query"].get("expand_keywords"):
            cl = _cfg["query"].get("expand_lang", "auto")
            self.corpus_lang = expand.detect_corpus_lang(self.working_dir) if cl == "auto" else cl
            if _cfg["query"].get("expand_glossary"):
                self.glossary = expand.build_glossary(
                    self.working_dir,
                    hub_n=int(_cfg["query"].get("glossary_hub_n", 100)),
                    distinct_n=int(_cfg["query"].get("glossary_distinct_n", 120)),
                    df_min=int(_cfg["query"].get("glossary_df_min", 2)),
                )

    async def teardown(self):
        if self.rag is not None:
            await self.rag.finalize_storages()

    def set_rerank(self, mode: str):
        self.rerank_mode = mode
        self.rag.rerank_model_func = _RERANK_FUNCS[mode]  # live swap, no rebuild


async def ask_events(session: ChatSession, question: str,
                     skip_clarify: bool = False) -> AsyncIterator[dict]:
    """Run one chat turn, yielding typed events (see module docstring)."""
    mode = _cfg["query"]["default_mode"]
    fn = _RERANK_FUNCS[session.rerank_mode]
    tracer = tracing.get_tracer()

    yield {"type": "status", "msg": "질문 유형을 분석하고 있습니다...", "detail": False}
    # Run the router and (optionally) corpus-language keyword expansion concurrently —
    # both are cheap flash-lite calls, so the expansion adds ~no wall-clock latency.
    if _cfg["query"].get("expand_keywords"):
        decision, kw = await asyncio.gather(
            router.route(question, session.history),
            expand.expand_keywords(question, session.corpus_lang, session.history,
                                   glossary=session.glossary),
        )
    else:
        decision = await router.route(question, session.history)
        kw = {"concepts": [], "entities": []}

    if not decision["in_scope"]:
        msg = ("이 질문은 문서(Grossberg 시각 지각 신경 모델) 범위를 벗어납니다. "
               "문서 내용에 대해 질문해 주세요.")
        session.history.append({"role": "user", "content": question})
        session.history.append({"role": "assistant", "content": "(범위 밖 질문 — 거절)"})
        session.save()
        yield {"type": "decline", "msg": msg}
        return

    if decision["needs_clarification"] and not skip_clarify:
        session.pending_question = question
        yield {"type": "clarify", "question": decision["clarification"]}
        return  # not recorded in history; resumes when the user replies

    effort = decision["effort"]
    need = decision["needs_retrieval"]
    yield {"type": "routing", "needs_retrieval": need, "effort": effort}

    chain_cm = tracer.start_as_current_span("chat_turn") if tracer else _nullctx()
    with chain_cm as span:
        if span is not None:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("input.value", question)
            span.set_attribute("metadata.turn", len(session.history) // 2 + 1)
            span.set_attribute("metadata.router.needs_retrieval", need)
            span.set_attribute("metadata.router.effort", effort)

        if need:
            yield {"type": "status",
                   "msg": f"「{question}」 관련 내용을 검색 중입니다...", "detail": False}
            # Inject corpus-language keywords when expansion produced any; empty lists are
            # equivalent to not passing them (LightRAG then runs its own extractor).
            hl, ll = kw.get("concepts") or [], kw.get("entities") or []
            if hl or ll:
                yield {"type": "status",
                       "msg": f"검색 키워드({session.corpus_lang}): {', '.join((hl + ll)[:8])}",
                       "detail": True}
            cap = _LRCapture()  # collect LightRAG's query-stage logs during aquery
            _lr_logger.addHandler(cap)
            try:
                assembled = await session.rag.aquery(
                    question,
                    param=QueryParam(
                        mode=mode,
                        only_need_prompt=True,
                        enable_rerank=(fn is not None),
                        conversation_history=session.history[-_HISTORY_TURNS * 2:],
                        history_turns=_HISTORY_TURNS,
                        hl_keywords=hl,
                        ll_keywords=ll,
                    ),
                )
            finally:
                _lr_logger.removeHandler(cap)
            for line in cap.lines:  # surface each retrieval stage as a stacked status line
                yield {"type": "status", "msg": line, "detail": True}
            sys_prompt, user_query = (
                assembled.split(_MARKER, 1) if _MARKER in assembled else (assembled, question)
            )
        else:
            hist = session.history[-_HISTORY_TURNS * 2:]
            sys_prompt = ("이전 대화:\n" + "\n".join(
                f"{h['role']}: {h['content']}" for h in hist)) if hist else ""
            user_query = question

        session.last_sources = _extract_sources(sys_prompt)

        # Query-time figure re-injection: pick only figures DIRECTLY relevant and
        # feed their pixels to the (multimodal) answer model.
        images = None
        session.last_image_sources = []
        if need and _INJECT_IMAGES:
            cands = _image_candidates(sys_prompt)
            if cands:
                if not is_vision_capable(session.provider):
                    yield {"type": "status",
                           "msg": (f"[경고] inject_images=true이지만 현재 답변 모델"
                                   f"({session.provider})이 이미지 입력을 지원하지 않아 "
                                   f"텍스트로만 답변합니다"),
                           "detail": False}
                else:
                    yield {"type": "status",
                           "msg": f"관련 이미지를 선별하고 있습니다 (후보 {len(cands)}개)...",
                           "detail": False}
                    selected = await image_gate.select_relevant_images(question, cands)
                    if selected:
                        by_hash = {c["hash"]: c for c in cands}
                        chosen = [by_hash[h] for h in selected if h in by_hash]
                        session.last_image_sources = [c["marker"] for c in chosen if c["marker"]]
                        images = [c["path"] for c in chosen]
                        img_items = []
                        for c in chosen:
                            title, desc = _figure_meta(c["caption"])
                            img_items.append({
                                **{k: c[k] for k in ("hash", "section", "page", "marker", "path")},
                                "title": title, "desc": desc,
                            })
                        yield {"type": "images", "items": img_items}
                        yield {"type": "status",
                               "msg": f"이미지 {len(chosen)}개를 답변 컨텍스트에 주입합니다",
                               "detail": True}
                    else:
                        yield {"type": "status",
                               "msg": "직접 관련된 이미지가 없어 주입하지 않습니다", "detail": True}
        if span is not None:
            span.set_attribute("metadata.injected_images", len(images or []))

        prov = session.provider
        yield {"type": "status",
               "msg": f"{prov} 모델로 답변을 생성하고 있습니다 (강도 {effort})...", "detail": False}

        answer_parts: list[str] = []
        try:
            async for chunk in answer_model_stream(
                prompt=user_query, system_prompt=sys_prompt,
                provider=prov, effort=effort, images=images,
            ):
                if chunk["type"] == "answer":
                    answer_parts.append(chunk["delta"])
                yield chunk  # {"type": "reasoning"|"answer", "delta": str}
        except Exception as e:
            # Provider failed BEFORE producing an answer (e.g. OpenAI insufficient_quota).
            # Fall back to gemini and notify the client via a modal. If some answer
            # already streamed, or we were already on gemini, don't silently retry.
            if answer_parts or prov == "gemini":
                raise
            yield {"type": "fallback", "from": prov, "to": "gemini", "reason": str(e)}
            yield {"type": "status", "msg": "gemini 모델로 대체해 답변을 생성합니다...", "detail": False}
            async for chunk in answer_model_stream(
                prompt=user_query, system_prompt=sys_prompt,
                provider="gemini", effort=effort, images=images,
            ):
                if chunk["type"] == "answer":
                    answer_parts.append(chunk["delta"])
                yield chunk
        answer = "".join(answer_parts)
        if span is not None:
            span.set_attribute("output.value", answer)

    # All well-formed retrieved markers + their chunk content + injected flag.
    # The client splits these into numbered citations (matched to the answer's
    # footnotes) vs "기타 출처" (retrieved but not cited).
    injected = set(session.last_image_sources)
    contents = _source_contents(sys_prompt)
    yield {"type": "sources",
           "items": [{"marker": m, "injected": m in injected, "content": contents.get(m, "")}
                     for m in session.last_sources]}

    # Append-only history: store a one-shot SUMMARY of cited chunks (stable prefix
    # keeps provider prompt caching working; fresh retrieval each turn loses nothing).
    if need:
        cited = _cited_chunks(answer, sys_prompt)
        summary = await _summarize_cited(question, cited) if cited else "(no sources cited)"
        cited_n = len(cited)
    else:
        summary = answer[:300]
        cited_n = 0
    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": summary})
    session.save()
    yield {"type": "done", "cited": cited_n, "summarized": need}
