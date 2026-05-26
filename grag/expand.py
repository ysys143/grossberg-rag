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
import math
import re
from pathlib import Path

import yaml

from . import llm
from .paths import CONFIG_PATH, PROJECT_ROOT

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


# ── Glossary: corpus's characteristic terms, injected into the expansion prompt ──────
# A mix of common hub terms (canonical spelling of core concepts) and distinctive jargon
# (rare terms the LLM/vector miss). Auto-derived from the index -> no cold start. Built
# once and sorted deterministically so it forms a stable system-prompt prefix (Gemini
# prefix caching in llm._ensure_gemini_cache then discounts repeated expansion calls).

# Lightweight, mecab-free tokenizer for scoring entity NAMES (latin / figure-number / CJK runs).
_NAME_TOK = re.compile(r"[A-Za-z]+|\d+\.\d+|\d+|[가-힣]+|[一-鿿]+|[぀-ヿ]+")
_glossary_cache: dict[str, str] = {}


def _name_tokens(s: str) -> list[str]:
    return [t.lower() for t in _NAME_TOK.findall(s or "")]


def _is_noise_name(name: str) -> bool:
    n = name.strip()
    if len(n) < 2:
        return True
    if not re.search(r"[A-Za-z가-힣一-鿿]", n):  # no letters at all (pure symbols/numbers)
        return True
    if n.lower().startswith("section "):  # heading fragments like "Section BIPOLEPROPERTY"
        return True
    return False


def _load_overlay() -> list[str]:
    """Optional hand-curated terms (config query.glossary_overlay). Accepts a YAML list of
    strings or of {canonical, aliases}. Fail-open to []."""
    rel = _qcfg.get("glossary_overlay")
    if not rel:
        return []
    p = Path(rel)
    if not p.is_absolute():
        p = PROJECT_ROOT / rel
    try:
        data = yaml.safe_load(p.read_text()) or []
        terms = []
        for it in data:
            if isinstance(it, str) and it.strip():
                terms.append(it.strip())
            elif isinstance(it, dict) and (it.get("canonical") or "").strip():
                terms.append(it["canonical"].strip())
        return terms
    except Exception as e:
        logger.warning(f"glossary overlay skipped: {type(e).__name__}: {e}")
        return []


def build_glossary(working_dir: str, hub_n: int = 100, distinct_n: int = 120, df_min: int = 2) -> str:
    """Build a deterministic, sorted glossary text block from the entity store: top hub
    terms (by chunk frequency) UNION top distinctive terms (by mean name-token IDF, with a
    chunk-frequency floor + noise filter) UNION an optional curated overlay. Cached per
    working_dir. Fail-open to '' (caller then expands without a glossary)."""
    if working_dir in _glossary_cache:
        return _glossary_cache[working_dir]
    glossary = ""
    try:
        records = json.loads((Path(working_dir) / "vdb_entities.json").read_text()).get("data", [])
        ents: list[tuple[str, int]] = []  # (entity_name, chunk_frequency)
        for r in records:
            name = (r.get("entity_name") or "").strip()
            if _is_noise_name(name):
                continue
            sid = r.get("source_id") or ""
            freq = len(sid.split("<SEP>")) if sid else 1
            ents.append((name, freq))

        n = len(ents)
        df: dict[str, int] = {}
        toks_by: dict[str, set[str]] = {}
        for name, _ in ents:
            ts = set(_name_tokens(name))
            toks_by[name] = ts
            for t in ts:
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

        hub_names = [name for name, _ in sorted(ents, key=lambda x: x[1], reverse=True)[:hub_n]]

        def _distinct_score(name: str) -> float:
            ts = toks_by[name]
            return sum(idf[t] for t in ts) / len(ts) if ts else 0.0

        distinct_cand = [name for name, f in ents if f >= df_min]
        distinct_names = sorted(distinct_cand, key=_distinct_score, reverse=True)[:distinct_n]

        terms = set(hub_names) | set(distinct_names) | set(_load_overlay())
        glossary = "\n".join(sorted(terms))  # sorted -> byte-stable prefix for cache hits
    except Exception as e:
        logger.warning(f"build_glossary fail-open: {type(e).__name__}: {e}")
    _glossary_cache[working_dir] = glossary
    return glossary


def _system_prompt(lang: str, glossary: str = "") -> str:
    name = _LANG_NAMES.get(lang, "English")
    gloss = ""
    if glossary:
        gloss = (f"\n\nCharacteristic terms in this knowledge base — when the query refers to "
                 f"any of these, use the EXACT spelling shown (do not paraphrase):\n{glossary}\n")
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
- 3-8 items per list, most important first. Omit a list (use []) only if truly nothing fits.{gloss}
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


async def expand_keywords(question: str, corpus_lang: str, history: list[dict] | None = None,
                          glossary: str = "") -> dict:
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
            model=EXPAND_MODEL, prompt=prompt, system_prompt=_system_prompt(corpus_lang, glossary),
            thinking_budget=0, temperature=0,
        )
        return _parse(raw)
    except Exception as e:  # fail-open -> caller uses LightRAG's own extractor
        logger.warning(f"expand_keywords fail-open: {type(e).__name__}: {e}")
        return {"concepts": [], "entities": []}
