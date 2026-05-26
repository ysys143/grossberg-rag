"""
Hybrid (lexical + vector) entity seeding for LightRAG retrieval.

LightRAG seeds graph expansion purely by vector kNN over entity-name embeddings
(`operate.py` `_get_node_data` -> `entities_vdb.query`). Exact-name entities, acronyms,
and figure numbers the embedding ranks low never become seeds, so their whole subgraph
(relations, neighbors, chunks) drops out of context. This module adds a BM25 lexical seed
layer and unions it with the vector seeds by wrapping the entity vector store's `query`
method; the unioned seeds then flow through the *unchanged* `_get_node_data` (graph fetch
+ edge expansion) — so BM25-found entities get full graph treatment for free.

Opt-in via config `query.hybrid_seed`. Self-contained: a small Okapi BM25 (Lucene-style
non-negative idf) + a mecab-ko (Korean) / regex (latin, figure-number) tokenizer. No
external retrieval library.
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

logger = logging.getLogger("grag.hybrid_seed")

# Latin words, bare integers, and figure numbers ("4.25"). The high-value exact-match
# tokens in this corpus are English acronyms / names and figure references — mecab-ko
# mangles those, so a regex pass owns them and mecab owns Korean morphemes.
_LATIN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+\.\d+|\d+")

# mecab-ko POS tags worth keeping from the Korean side: 일반명사 / 고유명사 / 한자.
# (Latin SL / numeric SN are left to the regex pass to avoid double-counting.)
_KEEP_POS = {"NNG", "NNP", "SH"}


class Tokenizer:
    """mecab-ko Korean noun morphemes + latin/figure-number regex tokens (lowercased)."""

    def __init__(self):
        from mecab import MeCab  # python-mecab-ko; lazy so only needed when enabled

        self._mecab = MeCab()

    def __call__(self, text: str) -> list[str]:
        text = text or ""
        toks: list[str] = []
        for surface, tag in self._mecab.pos(text):
            if tag.split("+", 1)[0] in _KEEP_POS:  # compound tags e.g. "NNG+JKB"
                toks.append(surface.lower())
        toks.extend(m.group(0).lower() for m in _LATIN_RE.finditer(text))
        return toks


class BM25Index:
    """Okapi BM25 with Lucene-style idf `log(1 + (N-df+0.5)/(df+0.5))` (always >= 0,
    so common terms can't produce negative scores — no separate idf floor needed)."""

    def __init__(self, names: list[str], docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.names = names
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.doc_len = [len(t) for t in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tfs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for toks in docs_tokens:
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.tfs.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - dfi + 0.5) / (dfi + 0.5)) for t, dfi in df.items()}

    def search(self, query_text: str, top_k: int, tokenizer: Tokenizer) -> list[str]:
        q_terms = set(tokenizer(query_text))  # dedup query terms
        if not q_terms or self.N == 0 or self.avgdl == 0:
            return []
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self.tfs):
            dl = self.doc_len[i]
            s = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / denom
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [self.names[i] for _, i in scored[:top_k]]


def build_entity_index(working_dir: str) -> tuple[BM25Index, dict[str, str], Tokenizer]:
    """Build a BM25 index over entity `content` (= name + description, the same text the
    embedding saw) from the persisted entity vector store. Returns (index, created_at map,
    tokenizer). Reads the on-disk JSON (stable format) rather than LightRAG internals."""
    path = Path(working_dir) / "vdb_entities.json"
    records = json.loads(path.read_text()).get("data", [])
    tok = Tokenizer()
    names: list[str] = []
    docs: list[list[str]] = []
    created: dict[str, str] = {}
    for rec in records:
        name = rec.get("entity_name")
        if not name:
            continue
        names.append(name)
        docs.append(tok(rec.get("content") or name))
        created[name] = rec.get("__created_at__")
    return BM25Index(names, docs), created, tok


def attach_hybrid_seed(rag, working_dir: str, top_k: int = 10) -> int:
    """Wrap `rag.entities_vdb.query` so entity seeds = vector hits UNION BM25 hits.

    Returns the number of entities indexed. Idempotent (won't double-wrap). The wrapped
    method matches `_get_node_data`'s call: `query(text, top_k=..., query_embedding=...)`.
    BM25 hits are synthesized as result dicts with the fields `_get_node_data` reads
    (`entity_name`, `created_at`); downstream graph fetch + edge expansion are unchanged.
    """
    vdb = rag.entities_vdb
    if getattr(vdb, "_grag_hybrid_attached", False):
        return getattr(vdb, "_grag_hybrid_n", 0)

    index, created_map, tok = build_entity_index(working_dir)
    orig_query = vdb.query
    bm25_k = top_k

    async def hybrid_query(query, top_k, query_embedding=None):
        results = await orig_query(query, top_k, query_embedding=query_embedding)
        seen = {r.get("entity_name") for r in results}
        added = 0
        for name in index.search(query, bm25_k, tok):
            if name in seen:
                continue
            results.append({
                "entity_name": name,
                "id": None,
                "distance": 0.0,
                "created_at": created_map.get(name),
                "_grag_bm25": True,
            })
            seen.add(name)
            added += 1
        if added:
            logger.info(f"[hybrid_seed] +{added} BM25 entity seed(s) for query='{query[:60]}'")
        return results

    vdb.query = hybrid_query
    vdb._grag_hybrid_attached = True
    vdb._grag_hybrid_n = index.N
    logger.info(f"[hybrid_seed] attached: {index.N} entities indexed, bm25 top_k={bm25_k}")
    return index.N
