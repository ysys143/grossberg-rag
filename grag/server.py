"""
Web server for grossberg-rag — a thin SSE consumer of engine.ask_events.

Mirrors the CLI (chat.py) by streaming the SAME engine events; the only
difference is presentation (HTML + inline figure images instead of ANSI text).
Single-user local tool: one in-memory ChatSession, serialized with a lock.

Run:  uvicorn server:app   (then open http://127.0.0.1:8000)
"""
import asyncio
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

from .paths import ENV_PATH, APIKEY_ENV, CONFIG_PATH, STATIC_DIR

load_dotenv(ENV_PATH)
load_dotenv(APIKEY_ENV, override=False)

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .engine import ChatSession, ask_events, _SESSIONS_DIR, _resolve_image_path
from . import tracing

_cfg = yaml.safe_load(CONFIG_PATH.read_text())
_STATIC = STATIC_DIR
_HASH_RE = re.compile(r"^[a-f0-9]{8,}$")  # guard against path traversal in /img/{hash}
_lock = asyncio.Lock()  # serialize turns (shared session state is not concurrency-safe)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracing.init_tracing("grossberg-rag")
    wdir = _cfg["storage"]["working_dir"]
    if not Path(wdir).exists():
        raise RuntimeError(f"Storage not found: {wdir} — run ingest.py first.")
    # One shared read-only LightRAG index; each chat session is a lightweight
    # ChatSession (own history/file) that reuses this rag.
    master = ChatSession(wdir, provider=None, rerank_mode="oneshot",
                         session_path=_SESSIONS_DIR / "web.json")
    master.load()
    await master.setup()
    app.state.wdir = wdir
    app.state.rag = master.rag
    app.state.master = master
    app.state.sessions = {"default": master}  # session_id -> ChatSession
    yield
    await master.teardown()
    tracing.shutdown_tracing()


app = FastAPI(lifespan=lifespan)

_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _get_session(sid: str) -> ChatSession:
    sid = _ID_RE.sub("", sid or "default") or "default"
    sessions = app.state.sessions
    if sid not in sessions:
        s = ChatSession(app.state.wdir, provider=None, rerank_mode="oneshot",
                        session_path=_SESSIONS_DIR / f"web_{sid}.json")
        s.rag = app.state.rag  # share the initialized retrieval index
        s.load()
        sessions[sid] = s
    return sessions[sid]


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


class ResetRequest(BaseModel):
    session_id: str = "default"


def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"


async def _event_stream(sess: ChatSession, question: str):
    """Yield engine events as SSE frames. Rewrites image file paths to /img URLs
    so server-side paths never leak to the browser."""
    # Mirror the CLI's pending-question merge: a reply to a prior clarification is
    # joined to the original question and forced past the gate.
    skip = False
    if sess.pending_question is not None:
        question = f"{sess.pending_question}\n\n[사용자 보충 설명] {question}"
        sess.pending_question = None
        skip = True
    try:
        async for ev in ask_events(sess, question, skip_clarify=skip):
            if ev.get("type") == "images":
                ev = {"type": "images", "items": [
                    {"hash": it["hash"], "section": it["section"],
                     "page": it["page"], "marker": it["marker"],
                     "title": it.get("title", ""), "desc": it.get("desc", ""),
                     "url": f"/img/{it['hash']}"}
                    for it in ev["items"]
                ]}
            yield _sse(ev)
    except Exception as e:  # surface engine errors to the client instead of hanging
        yield _sse({"type": "error", "msg": f"{type(e).__name__}: {e}"})


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    sess = _get_session(req.session_id)

    async def gen():
        async with _lock:  # one turn at a time (shared rag)
            async for frame in _event_stream(sess, req.question):
                yield frame

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/reset")
async def api_reset(req: ResetRequest):
    sess = _get_session(req.session_id)
    sess.history.clear()
    sess.pending_question = None
    sess.save()
    return {"ok": True}


@app.get("/img/{img_hash}")
async def get_image(img_hash: str):
    if not _HASH_RE.match(img_hash):
        raise HTTPException(status_code=400, detail="invalid hash")
    path = _resolve_image_path(img_hash, "")
    if not path:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path)


@app.get("/")
async def index():
    return FileResponse(_STATIC / "index.html")
