"""
Query-time image relevance gate.

After retrieval, some retrieved chunks are figure descriptions. This gate decides
which of those figures (if any) are DIRECTLY relevant enough to the question to be
worth re-injecting as actual pixels into the answer model — and ruthlessly excludes
the rest. Runs only when image candidates exist; one cheap flash-lite call.

Fail-CLOSED: on any transport/parse error, select NO images. An injected image is an
enhancement, not load-bearing context (the text description is still present), so the
safe failure is to fall back to the text-only answer — the opposite of router.py,
which fails open because dropping a real question would be worse.
"""
import json
import re

import yaml

from . import llm
from .paths import CONFIG_PATH

_cfg = yaml.safe_load(CONFIG_PATH.read_text())
_qcfg = _cfg.get("query", {})
GATE_MODEL = _qcfg.get("image_gate_model", "gemini-3.1-flash-lite")
MAX_IMAGES = int(_qcfg.get("max_injected_images", 5))

_SYSTEM = """You decide which figures from a neuroscience document must be SEEN
(actual image) to answer the user's question well.

You are given the question and a numbered list of figure captions. A figure is
relevant ONLY if seeing its pixels would materially help answer THIS question —
e.g. the question asks about a circuit's structure, spatial arrangement, arrow
directions, color coding, or a specific figure. A figure is NOT relevant if it is
merely topically adjacent or if the text already suffices.

Reply with ONLY a JSON object, no prose, no markdown fences:
{"relevant": [<indices>]}

- indices: the 0-based indices of directly-relevant figures, most relevant first.
- Include ALL figures that are directly relevant — if several figures illustrate
  the asked phenomenon or mechanism, include them all (they reinforce the answer).
  Still EXCLUDE figures that are only tangentially or topically related. Return []
  only if none genuinely help. The list is capped, so don't pad with marginal figures.
"""


def _parse(raw: str, n: int) -> list[int]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    obj = json.loads(m.group(0)) if m else {}
    out: list[int] = []
    for i in obj.get("relevant", []):
        try:
            idx = int(i)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n and idx not in out:
            out.append(idx)
    return out


async def select_relevant_images(question: str, candidates: list[dict]) -> list[str]:
    """Pick the directly-relevant figures to re-inject as pixels.

    candidates: [{"hash": str, "caption": str, "section": str, "page": int}, ...]
    Returns: list of selected hashes (<= MAX_IMAGES), most relevant first. May be empty.
    """
    if not candidates:
        return []

    listing = "\n".join(
        f"[{i}] (§{c.get('section', '?')}, p.{c.get('page', '?')}) {(c.get('caption') or '')[:400]}"
        for i, c in enumerate(candidates)
    )
    prompt = f"Question: {question}\n\nFigures:\n{listing}"

    try:
        raw = await llm.generate(
            model=GATE_MODEL, prompt=prompt, system_prompt=_SYSTEM,
            thinking_budget=0, temperature=0,
        )
        idxs = _parse(raw, len(candidates))
    except Exception:
        return []  # fail closed -> text-only answer
    return [candidates[i]["hash"] for i in idxs[:MAX_IMAGES]]
