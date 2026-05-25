"""
Front-of-pipeline router.

A cheap classifier (flash-lite, thinking off) that runs BEFORE the expensive
retrieval + answer path and decides three things:

  - in_scope        : is the question about the document's topic (or a meta
                      question about this conversation)? If not, we decline
                      instead of retrieving / answering.
  - needs_retrieval : does answering actually require searching the document?
                      Greetings, thanks, and "summarize your last answer" don't.
  - effort          : reasoning demand (low | medium | high). The answer model
                      (gpt-5.5) is fixed; the router only dials its effort.

The model is configurable in config.yaml (router.model); it defaults to
flash-lite. The router emits an *abstract* effort level — how each provider
realizes it (OpenAI reasoning.effort vs Gemini thinking_budget) lives in
models.py, not here.

Fail-open: on any transport/parse error we return in_scope + needs_retrieval +
medium effort, so the router can never silently swallow a real question.
"""
import json
import re
from pathlib import Path

import yaml

import llm

_cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
_rcfg = _cfg.get("router", {})
ROUTER_MODEL = _rcfg.get("model", "gemini-3.1-flash-lite")
ROUTER_ENABLED = _rcfg.get("enabled", True)

_VALID_EFFORT = ("low", "medium", "high")

_SYSTEM = """You are a routing classifier for a RAG system whose ONLY knowledge
source is a document about Stephen Grossberg's neural models of visual
perception: boundary completion, surface filling-in, FACADE theory, BCS/FCS,
brightness, depth, figure-ground separation, and related neural mechanisms.

Classify the user's question. Reply with ONLY a JSON object, no prose, no
markdown fences:
{"in_scope": <bool>, "needs_retrieval": <bool>, "effort": "low"|"medium"|"high",
 "needs_clarification": <bool>, "clarification": "<string>"}

Field meaning:
- in_scope: true if the question is about visual perception / neural models /
  the document's topics, OR a meta-question about THIS conversation
  (e.g. "summarize your last answer", "what did you just say"). false for
  unrelated topics (weather, coding help, current events, general trivia).
- needs_retrieval: true if answering requires looking up facts in the document.
  false for greetings, thanks, or meta-questions answerable from the
  conversation history alone.
- effort: reasoning demand.
    "low"    = a definition or single-fact lookup.
    "medium" = explain one mechanism or concept.
    "high"   = compare/synthesize multiple systems, trace causal chains, or
               evaluate competing theories.
- needs_clarification: true ONLY when the question is too vague or
  underspecified to retrieve usefully AND the conversation history does NOT
  resolve it — e.g. "그거 설명해줘"/"이건 어때?" with no resolvable referent,
  or a one-word topic so broad that retrieval would be unfocused. If history
  already resolves the referent (demonstratives like "그게"), set false.
  When in doubt, prefer false (do not over-interrupt the user).
- clarification: when needs_clarification is true, a SINGLE concise Korean
  question asking for the missing specifics. Otherwise "".
"""


def _parse(raw: str) -> dict:
    """Extract the JSON object from the model output, with safe defaults."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    obj = json.loads(m.group(0)) if m else {}
    effort = obj.get("effort", "medium")
    if effort not in _VALID_EFFORT:
        effort = "medium"
    needs_clarification = bool(obj.get("needs_clarification", False))
    clarification = str(obj.get("clarification", "") or "")
    # A clarification flag with no actual question is useless — drop it.
    if needs_clarification and not clarification.strip():
        needs_clarification = False
    return {
        "in_scope": bool(obj.get("in_scope", True)),
        "needs_retrieval": bool(obj.get("needs_retrieval", True)),
        "effort": effort,
        "needs_clarification": needs_clarification,
        "clarification": clarification.strip(),
        "reason": "",
    }


async def route(question: str, history: list[dict] | None = None) -> dict:
    """Classify a question. Returns
    {"in_scope": bool, "needs_retrieval": bool, "effort": str, "reason": str}.
    """
    if not ROUTER_ENABLED:
        return {"in_scope": True, "needs_retrieval": True, "effort": "high",
                "needs_clarification": False, "clarification": "",
                "reason": "router disabled"}

    ctx = ""
    if history:
        recent = history[-4:]  # last ~2 turns is enough to resolve follow-ups
        ctx = "Recent conversation:\n" + "\n".join(
            f"{h['role']}: {h['content'][:200]}" for h in recent
        ) + "\n\n"
    prompt = f"{ctx}User question: {question}"

    try:
        raw = await llm.generate(
            model=ROUTER_MODEL,
            prompt=prompt,
            system_prompt=_SYSTEM,
            thinking_budget=0,
            temperature=0,
        )
        return _parse(raw)
    except Exception as e:  # fail open — never swallow a real question
        return {"in_scope": True, "needs_retrieval": True, "effort": "medium",
                "needs_clarification": False, "clarification": "",
                "reason": f"router-failopen: {type(e).__name__}"}
