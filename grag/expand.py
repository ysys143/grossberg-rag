"""
Corpus-language query keyword expansion (cross-lingual retrieval bridge).

The index is in the document's language (entity names, descriptions). A query in another
language fails retrieval on BOTH arms — vector (cross-lingual embedding gap) and BM25
(lexical, same-language only). LightRAG's standard path extracts keywords in the *query*
language; this module instead expands the query into **corpus-language** keywords and the
caller injects them as explicit `QueryParam(hl_keywords=…, ll_keywords=…)` (the path
`kb_tool.py` already uses), so both arms match the index.

`detect_corpus_lang` infers the index language from the stored entity text (dependency-free
Unicode-script heuristic). `expand_keywords` is one cheap flash-lite call; fail-open to
empty so the caller falls back to LightRAG's internal extractor.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml

from . import llm
from .paths import CONFIG_PATH

logger = logging.getLogger("grag.expand")

_cfg = yaml.safe_load(CONFIG_PATH.read_text())
_qcfg = _cfg.get("query", {})
EXPAND_MODEL = _qcfg.get("expand_model") or _cfg.get("router", {}).get("model", "gemini-3.1-flash-lite")

_LANG_NAMES = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}

# Unicode ranges per script (dominant script of the index text -> target language).
_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_KANA = re.compile(r"[぀-ヿ]")
_HAN = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")

_lang_cache: dict[str, str] = {}


def detect_corpus_lang(working_dir: str, sample: int = 400) -> str:
    """Infer the index language from a sample of entity `content` (the searchable text),
    by dominant Unicode script. Returns an ISO-ish code ("en"/"ko"/"ja"/"zh"). Cached.
    Falls back to 'en' when the store is missing/empty/ambiguous."""
    if working_dir in _lang_cache:
        return _lang_cache[working_dir]
    lang = "en"
    try:
        records = json.loads((Path(working_dir) / "vdb_entities.json").read_text()).get("data", [])
        text = " ".join((r.get("content") or "") for r in records[:sample])
        counts = {
            "ko": len(_HANGUL.findall(text)),
            "ja": len(_KANA.findall(text)),
            "zh": len(_HAN.findall(text)),
            "en": len(_LATIN.findall(text)),
        }
        # A doc with kana is Japanese even if it also uses Han; check kana before Han.
        if counts["ko"] and counts["ko"] >= max(counts["en"], counts["ja"], counts["zh"]):
            lang = "ko"
        elif counts["ja"]:
            lang = "ja"
        elif counts["zh"] and counts["zh"] >= counts["en"]:
            lang = "zh"
        elif any(counts.values()):
            lang = "en"
    except Exception as e:
        logger.warning(f"detect_corpus_lang fallback to 'en': {type(e).__name__}: {e}")
    _lang_cache[working_dir] = lang
    return lang


def _system_prompt(lang: str) -> str:
    name = _LANG_NAMES.get(lang, "English")
    return f"""You expand a user's question into search keywords for a knowledge base whose
text is written in {name}. Reply with ONLY a JSON object, no prose, no markdown fences:
{{"concepts": [<strings>], "entities": [<strings>]}}

- concepts: high-level themes / topics, for global graph retrieval. Expressed in {name}.
- entities: specific named things — models, mechanisms, figures, people, acronyms — for
  local entity retrieval. Expressed in {name}.
- If the question is in another language, TRANSLATE its meaning into {name}.
- CRITICAL: keep acronyms, proper nouns, figure numbers, and established technical terms
  VERBATIM — never translate or paraphrase them (e.g. FACADE, BCS, LAMINART, Kanizsa,
  Figure 4.25). These are language-independent and must match the index exactly.
- 3-8 items per list, most important first. Omit a list (use []) only if truly nothing fits.
"""


def _parse(raw: str) -> dict:
    """Extract {concepts, entities} from model output, defensively (mirrors router._parse)."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    obj = json.loads(m.group(0)) if m else {}

    def _clean(key: str) -> list[str]:
        out, seen = [], set()
        for v in obj.get(key, []) or []:
            s = str(v).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out[:8]

    return {"concepts": _clean("concepts"), "entities": _clean("entities")}


async def expand_keywords(question: str, corpus_lang: str, history: list[dict] | None = None) -> dict:
    """Expand a query into corpus-language {concepts (hl), entities (ll)} keywords.
    Fail-open: returns empty lists on any transport/parse error so the caller falls back
    to LightRAG's internal keyword extraction."""
    ctx = ""
    if history:
        recent = history[-4:]
        ctx = "Recent conversation:\n" + "\n".join(
            f"{h['role']}: {h['content'][:200]}" for h in recent
        ) + "\n\n"
    prompt = f"{ctx}User question: {question}"
    try:
        raw = await llm.generate(
            model=EXPAND_MODEL, prompt=prompt, system_prompt=_system_prompt(corpus_lang),
            thinking_budget=0, temperature=0,
        )
        return _parse(raw)
    except Exception as e:  # fail-open -> caller uses LightRAG's own extractor
        logger.warning(f"expand_keywords fail-open: {type(e).__name__}: {e}")
        return {"concepts": [], "entities": []}
