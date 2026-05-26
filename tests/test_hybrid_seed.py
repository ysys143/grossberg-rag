"""Unit tests for hybrid BM25 entity seeding + guard tests pinning LightRAG behavior."""
import inspect
import json
from pathlib import Path

import pytest

from grag.hybrid_seed import BM25Index, Tokenizer, build_entity_index, attach_hybrid_seed


# ---------------------------------------------------------------------------
# Tokenizer (mecab-ko Korean nouns + latin/figure-number regex)
# ---------------------------------------------------------------------------

class TestTokenizer:
    @pytest.fixture(scope="class")
    def tok(self):
        return Tokenizer()

    def test_english_name_lowercased(self, tok):
        out = tok("Boundary Cortical Stream (BCS)")
        assert {"boundary", "cortical", "stream", "bcs"} <= set(out)

    def test_figure_number_kept_as_token(self, tok):
        out = tok("see Figure 4.25 and 3.14")
        assert "4.25" in out and "3.14" in out

    def test_mixed_korean_english(self, tok):
        # Korean noun morphemes + English tokens both surface.
        out = tok("초복합세포가 end cut을 만든다 Figure 4.25")
        assert "end" in out and "cut" in out and "4.25" in out
        assert any("가" <= ch <= "힣" for t in out for ch in t)  # some Hangul token

    def test_empty_string(self, tok):
        assert tok("") == []


# ---------------------------------------------------------------------------
# BM25Index (pre-tokenized docs, to isolate scoring from tokenization)
# ---------------------------------------------------------------------------

def _toy_index():
    names = ["d1", "d2", "d3"]
    docs = [
        ["boundary", "contour", "system", "bcs"],
        ["feature", "contour", "system", "fcs"],
        ["the", "cat", "sat", "on", "the", "mat"],
    ]
    return BM25Index(names, docs)


class TestBM25Index:
    def test_exact_term_ranks_owning_doc_first(self):
        idx = _toy_index()
        assert idx.search("bcs", 3, _IdentityTok()) == ["d1"]

    def test_shared_term_returns_both(self):
        idx = _toy_index()
        out = idx.search("contour system", 3, _IdentityTok())
        assert set(out) == {"d1", "d2"}

    def test_unknown_term_returns_empty(self):
        idx = _toy_index()
        assert idx.search("quantum", 3, _IdentityTok()) == []

    def test_top_k_caps_results(self):
        idx = _toy_index()
        assert len(idx.search("contour system the cat", 1, _IdentityTok())) == 1

    def test_idf_non_negative_even_for_common_terms(self):
        # Lucene-style idf log(1 + (N-df+0.5)/(df+0.5)) is >= 0 by construction,
        # so a term in every doc can never produce a negative score.
        idx = _toy_index()
        assert all(v >= 0 for v in idx.idf.values())

    def test_empty_index_search_is_safe(self):
        idx = BM25Index([], [])
        assert idx.search("anything", 5, _IdentityTok()) == []


class _IdentityTok:
    """Whitespace tokenizer so BM25 tests don't depend on mecab."""

    def __call__(self, text: str) -> list[str]:
        return text.lower().split()


# ---------------------------------------------------------------------------
# Guard tests — pin the LightRAG seed mechanism our monkeypatch relies on.
# If a future upgrade changes either, these fail loudly (see plan: pin + guard).
# ---------------------------------------------------------------------------

class TestLightRAGSeedContract:
    def test_pinned_version(self):
        import lightrag
        assert lightrag.__version__ == "1.4.16", (
            "lightrag-hku version changed; re-verify the hybrid-seed monkeypatch "
            "against the new _get_node_data / entities_vdb.query.")

    def test_get_node_data_seeds_via_entities_vdb_query(self):
        from lightrag import operate
        src = inspect.getsource(operate._get_node_data)
        assert "entities_vdb.query(" in src, (
            "_get_node_data no longer seeds via entities_vdb.query; the wrap point "
            "in grag.hybrid_seed.attach_hybrid_seed must be updated.")


# ---------------------------------------------------------------------------
# Tokenizer — adversarial / edge inputs (real mecab-ko)
# ---------------------------------------------------------------------------

class TestTokenizerEdge:
    @pytest.fixture(scope="class")
    def tok(self):
        return Tokenizer()

    def test_whitespace_only_is_empty(self, tok):
        assert tok("   \n\t  ") == []

    def test_pure_punctuation_is_empty(self, tok):
        assert tok("!@#$%^&*()_+-=[]{};:,.<>/?") == []

    def test_hyphenated_word_splits(self, tok):
        out = tok("Long-Range Cooperation")
        assert "long" in out and "range" in out and "cooperation" in out

    def test_apostrophe_kept_in_token(self, tok):
        # regex [A-Za-z]+(?:'[A-Za-z]+)? keeps an internal apostrophe.
        assert "hubel's" in tok("Hubel's law")

    def test_token_frequency_preserved_not_deduped(self, tok):
        # documents must keep term frequency for BM25 (no dedup at tokenize time).
        assert tok("end end end").count("end") == 3

    def test_bare_integer_and_figure_number(self, tok):
        out = tok("page 23 and Figure 4.25")
        assert "23" in out and "4.25" in out

    def test_chained_decimal_splits_into_figure_plus_int(self, tok):
        # "4.25.6" -> \d+\.\d+ eats "4.25", remainder "6" matches \d+.
        out = tok("section 4.25.6")
        assert "4.25" in out and "6" in out

    def test_greek_and_symbols_do_not_crash(self, tok):
        # Greek letters aren't captured (no latin/hangul), but must not error.
        assert isinstance(tok("the αβγ gain is ↑ high"), list)

    def test_long_input_does_not_crash(self, tok):
        out = tok(("boundary contour system " * 500) + "초복합세포 " * 200)
        assert "boundary" in out and len(out) > 1000

    def test_idempotent_lowercasing(self, tok):
        assert tok("FACADE") == tok("facade") == ["facade"]


# ---------------------------------------------------------------------------
# BM25Index — numeric properties / boundaries (identity tokenizer)
# ---------------------------------------------------------------------------

class TestBM25Edge:
    def test_top_k_zero_returns_empty(self):
        idx = _toy_index()
        assert idx.search("bcs", 0, _IdentityTok()) == []

    def test_all_empty_docs_no_zero_division(self):
        # avgdl == 0 path must be guarded (no ZeroDivisionError).
        idx = BM25Index(["a", "b"], [[], []])
        assert idx.avgdl == 0
        assert idx.search("anything", 5, _IdentityTok()) == []

    def test_single_doc_corpus(self):
        idx = BM25Index(["only"], [["alpha", "beta"]])
        assert idx.search("alpha", 3, _IdentityTok()) == ["only"]
        assert idx.search("gamma", 3, _IdentityTok()) == []

    def test_rarer_term_has_higher_idf(self):
        # "rare" in 1 of 3 docs; "common" in all 3 -> idf(rare) > idf(common).
        idx = BM25Index(
            ["d1", "d2", "d3"],
            [["rare", "common"], ["common", "x"], ["common", "y"]],
        )
        assert idx.idf["rare"] > idx.idf["common"]

    def test_length_normalization_favors_shorter_doc(self):
        # both docs contain "target" exactly once; the shorter doc should rank first
        # (BM25 length normalization via b).
        idx = BM25Index(
            ["short", "long"],
            [["target"], ["target"] + ["pad"] * 50],
        )
        assert idx.search("target", 2, _IdentityTok())[0] == "short"

    def test_doc_with_term_outranks_doc_without(self):
        idx = _toy_index()
        ranked = idx.search("bcs cat", 3, _IdentityTok())
        assert ranked[0] in {"d1", "d3"}  # both contain one query term
        assert "d2" not in ranked          # d2 has neither bcs nor cat

    def test_repeated_query_terms_deduped(self):
        # query terms are set()-deduped; repeating a term must not change ranking.
        idx = _toy_index()
        assert idx.search("bcs bcs bcs", 3, _IdentityTok()) == idx.search("bcs", 3, _IdentityTok())

    def test_unicode_korean_tokens_scored(self):
        idx = BM25Index(["k1", "k2"], [["세포", "복합"], ["경계", "윤곽"]])
        assert idx.search("복합 세포", 2, _IdentityTok()) == ["k1"]

    def test_tf_saturation_monotonic(self):
        # more occurrences never lowers the score (k1 saturation is monotonic up).
        idx = BM25Index(["one", "many"], [["t"], ["t", "t", "t", "t", "t"]])
        ranked = idx.search("t", 2, _IdentityTok())
        assert ranked[0] == "many"  # more occurrences -> higher (same length scaling)


# ---------------------------------------------------------------------------
# build_entity_index — sourcing from the persisted entity store
# ---------------------------------------------------------------------------

def _write_vdb(tmp_path: Path, records: list[dict]) -> str:
    (tmp_path / "vdb_entities.json").write_text(json.dumps({"data": records}))
    return str(tmp_path)


class TestBuildEntityIndex:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_entity_index(str(tmp_path))  # no vdb_entities.json

    def test_empty_data_builds_empty_index(self, tmp_path):
        wd = _write_vdb(tmp_path, [])
        idx, created, tok = build_entity_index(wd)
        assert idx.N == 0 and created == {}

    def test_records_without_entity_name_skipped(self, tmp_path):
        wd = _write_vdb(tmp_path, [
            {"entity_name": "Good Entity", "content": "good entity desc", "__created_at__": "1"},
            {"content": "orphan with no name", "__created_at__": "2"},
            {"entity_name": "", "content": "empty name", "__created_at__": "3"},
        ])
        idx, created, _ = build_entity_index(wd)
        assert idx.N == 1 and set(created) == {"Good Entity"}

    def test_content_falls_back_to_name_when_missing(self, tmp_path):
        wd = _write_vdb(tmp_path, [{"entity_name": "FACADE", "__created_at__": "1"}])
        idx, _, tok = build_entity_index(wd)
        assert idx.search("facade", 1, tok) == ["FACADE"]

    def test_created_at_mapped(self, tmp_path):
        wd = _write_vdb(tmp_path, [{"entity_name": "E", "content": "alpha", "__created_at__": "777"}])
        _, created, _ = build_entity_index(wd)
        assert created["E"] == "777"


# ---------------------------------------------------------------------------
# attach_hybrid_seed — the entities_vdb.query wrapper (union / dedup / idempotency)
# ---------------------------------------------------------------------------

class _FakeVDB:
    """Stands in for LightRAG's entity vector store. .query returns fixed vector hits."""

    def __init__(self, vector_hits):
        self._hits = vector_hits

    async def query(self, query, top_k, query_embedding=None):
        return [dict(h) for h in self._hits[:top_k]]


class _FakeRAG:
    def __init__(self, vector_hits):
        self.entities_vdb = _FakeVDB(vector_hits)


def _corpus(tmp_path):
    return _write_vdb(tmp_path, [
        {"entity_name": "BCS Double Filter", "content": "BCS Double Filter the double filter stage", "__created_at__": "111"},
        {"entity_name": "End Cut", "content": "End Cut end cut generation", "__created_at__": "222"},
        {"entity_name": "Unrelated Thing", "content": "lorem ipsum dolor sit", "__created_at__": "333"},
    ])


class TestAttachHybridSeed:
    async def test_unions_bm25_seed_missing_from_vector(self, tmp_path):
        wd = _corpus(tmp_path)
        rag = _FakeRAG([{"entity_name": "End Cut", "created_at": "222", "distance": 0.1}])
        attach_hybrid_seed(rag, wd, top_k=10)
        out = await rag.entities_vdb.query("double filter", top_k=60)
        names = [r["entity_name"] for r in out]
        assert "End Cut" in names            # vector hit preserved
        assert "BCS Double Filter" in names  # BM25 added (vector missed it)

    async def test_vector_hits_come_first(self, tmp_path):
        wd = _corpus(tmp_path)
        rag = _FakeRAG([{"entity_name": "End Cut", "created_at": "222"}])
        attach_hybrid_seed(rag, wd, top_k=10)
        out = await rag.entities_vdb.query("double filter", top_k=60)
        assert out[0]["entity_name"] == "End Cut"  # vector hit kept at front

    async def test_no_duplicate_when_bm25_overlaps_vector(self, tmp_path):
        wd = _corpus(tmp_path)
        # vector already returns the same entity BM25 would find
        rag = _FakeRAG([{"entity_name": "BCS Double Filter", "created_at": "111"}])
        attach_hybrid_seed(rag, wd, top_k=10)
        out = await rag.entities_vdb.query("double filter", top_k=60)
        names = [r["entity_name"] for r in out]
        assert names.count("BCS Double Filter") == 1

    async def test_synthesized_hit_shape(self, tmp_path):
        wd = _corpus(tmp_path)
        rag = _FakeRAG([])  # no vector hits -> any seed is BM25-synthesized
        attach_hybrid_seed(rag, wd, top_k=10)
        out = await rag.entities_vdb.query("double filter", top_k=60)
        hit = next(r for r in out if r["entity_name"] == "BCS Double Filter")
        assert hit["_grag_bm25"] is True
        assert hit["created_at"] == "111"  # mapped from the index

    async def test_no_new_seeds_when_query_has_no_lexical_match(self, tmp_path):
        wd = _corpus(tmp_path)
        rag = _FakeRAG([{"entity_name": "End Cut", "created_at": "222"}])
        attach_hybrid_seed(rag, wd, top_k=10)
        out = await rag.entities_vdb.query("zzzznonexistent", top_k=60)
        assert [r["entity_name"] for r in out] == ["End Cut"]  # unchanged

    async def test_idempotent_no_double_wrap(self, tmp_path):
        wd = _corpus(tmp_path)
        rag = _FakeRAG([{"entity_name": "End Cut", "created_at": "222"}])
        n1 = attach_hybrid_seed(rag, wd, top_k=10)
        wrapped_once = rag.entities_vdb.query
        n2 = attach_hybrid_seed(rag, wd, top_k=10)  # second call must be a no-op
        assert n1 == n2 == 3
        assert rag.entities_vdb.query is wrapped_once  # not re-wrapped
        out = await rag.entities_vdb.query("double filter", top_k=60)
        assert [r["entity_name"] for r in out].count("BCS Double Filter") == 1

    async def test_respects_top_k_caller_limit(self, tmp_path):
        # the caller's top_k still bounds the vector slice; BM25 adds up to its own cap.
        wd = _corpus(tmp_path)
        rag = _FakeRAG([{"entity_name": f"V{i}", "created_at": str(i)} for i in range(5)])
        attach_hybrid_seed(rag, wd, top_k=2)
        out = await rag.entities_vdb.query("double filter end cut", top_k=3)
        # 3 vector hits (top_k=3) + up to 2 BM25 additions, deduped
        assert sum(1 for r in out if r.get("_grag_bm25")) <= 2
        assert sum(1 for r in out if not r.get("_grag_bm25")) == 3
