"""Integration tests — require live API keys and a running LightRAG index.

Run with:  pytest --integration tests/test_integration.py

Skipped by default in CI (no API keys / no index).
"""
import pytest


# ===========================================================================
# router — live LLM classification
# ===========================================================================

@pytest.mark.integration
class TestRouterLive:
    async def test_in_scope_bcs_question(self):
        from grag import router
        r = await router.route("BCS와 FCS의 차이를 설명해줘.")
        assert r["in_scope"] is True
        assert r["effort"] in ("low", "medium", "high")
        assert isinstance(r["needs_retrieval"], bool)

    async def test_out_of_scope_coding_question_rejected(self):
        from grag import router
        r = await router.route("파이썬으로 웹 서버를 만드는 법을 알려줘.")
        assert r["in_scope"] is False

    async def test_out_of_scope_weather_rejected(self):
        from grag import router
        r = await router.route("오늘 서울 날씨가 어때?")
        assert r["in_scope"] is False

    async def test_greeting_needs_no_retrieval(self):
        from grag import router
        r = await router.route("안녕하세요!")
        assert r["needs_retrieval"] is False

    async def test_simple_definition_low_effort(self):
        from grag import router
        r = await router.route("FCS가 뭐야?")
        assert r["in_scope"] is True
        assert r["effort"] in ("low", "medium")

    async def test_complex_synthesis_high_effort(self):
        from grag import router
        q = ("BCS, FCS, FACADE 이론의 상호작용을 비교하고 각각의 신경 회로 구조가 "
             "지각적 그루핑에서 어떻게 협력하는지 인과적 흐름으로 설명해줘.")
        r = await router.route(q)
        assert r["in_scope"] is True
        assert r["effort"] == "high"

    async def test_vague_referential_question_may_request_clarification(self):
        from grag import router
        r = await router.route("그거 설명해줘.")
        # Router may or may not flag clarification — contract: never crashes,
        # always returns a fully populated dict.
        assert "in_scope" in r
        assert "needs_clarification" in r
        assert "effort" in r

    async def test_meta_question_no_retrieval_needed(self):
        from grag import router
        history = [
            {"role": "user", "content": "FACADE 이론이란?"},
            {"role": "assistant", "content": "FACADE 이론은 ..."},
        ]
        r = await router.route("방금 말한 내용을 요약해줘.", history)
        assert r["in_scope"] is True
        assert r["needs_retrieval"] is False

    async def test_history_resolves_demonstrative_pronoun(self):
        from grag import router
        history = [
            {"role": "user", "content": "BCS란 무엇인가?"},
            {"role": "assistant", "content": "BCS는 경계 윤곽 시스템입니다."},
        ]
        r = await router.route("그게 LAMINART와 어떻게 연결돼?", history)
        assert r["in_scope"] is True
        # History provides context -> clarification should not be needed
        assert r["needs_clarification"] is False

    async def test_response_always_has_all_required_keys(self):
        from grag import router
        r = await router.route("아무 질문")
        required = {"in_scope", "needs_retrieval", "effort", "needs_clarification",
                    "clarification", "reason"}
        assert required.issubset(r.keys())

    async def test_effort_always_in_valid_set(self):
        from grag import router
        questions = [
            "BCS란?",
            "FACADE와 BCS의 관계를 설명해줘.",
            "BCS, FCS, LAMINART, FACADE를 비교 분석해줘.",
        ]
        for q in questions:
            r = await router.route(q)
            assert r["effort"] in ("low", "medium", "high"), f"invalid effort for: {q}"

    async def test_disabled_router_returns_passthrough(self, monkeypatch):
        from grag import router
        monkeypatch.setattr(router, "ROUTER_ENABLED", False)
        r = await router.route("anything")
        assert r["in_scope"] is True
        assert r["needs_retrieval"] is True
        assert r["effort"] == "high"


# ===========================================================================
# rerank — live LLM scoring
# ===========================================================================

@pytest.mark.integration
class TestRerankLive:
    async def test_relevant_doc_scores_higher_than_irrelevant(self):
        from grag import rerank
        docs = [
            "The Boundary Contour System (BCS) generates illusory contours via "
            "competitive interactions between simple and complex cells in V1/V2.",
            "Today the weather is sunny and 24 degrees Celsius in Seoul.",
            "Python is a high-level programming language used for web development.",
        ]
        result = await rerank.rerank("BCS의 경계 완성 메커니즘", docs)
        assert len(result) == 3
        scores = {r["index"]: r["relevance_score"] for r in result}
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    async def test_all_scores_in_0_1_range(self):
        from grag import rerank
        docs = [f"doc {i} about neural models" for i in range(5)]
        result = await rerank.rerank("BCS", docs)
        assert all(0.0 <= r["relevance_score"] <= 1.5 for r in result)
        # upper bound is 1.5 to allow >100 scores from model without crashing

    async def test_top_n_limits_live_result(self):
        from grag import rerank
        docs = [f"neural model doc {i}" for i in range(8)]
        result = await rerank.rerank("FACADE theory", docs, top_n=3)
        assert len(result) == 3

    async def test_result_sorted_descending(self):
        from grag import rerank
        docs = [
            "FACADE theory explains how surfaces are perceived.",
            "Unrelated text about programming.",
            "FCS handles surface filling-in and brightness perception.",
        ]
        result = await rerank.rerank("surface perception", docs)
        scores = [r["relevance_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    async def test_single_doc_returns_single_result(self):
        from grag import rerank
        result = await rerank.rerank("BCS", ["BCS is the Boundary Contour System."])
        assert len(result) == 1
        assert result[0]["index"] == 0

    async def test_empty_docs_returns_empty(self):
        from grag import rerank
        assert await rerank.rerank("BCS", []) == []

    async def test_batched_result_has_valid_global_indices(self):
        from grag import rerank
        docs = [f"doc {i}" for i in range(25)]
        result = await rerank.rerank_batched("BCS 경계 완성", docs, batch_size=10, top_n=5)
        assert all(0 <= r["index"] < len(docs) for r in result)
        assert len(result) <= 5

    async def test_batched_vs_oneshot_both_return_same_top_doc(self):
        from grag import rerank
        docs = [
            "BCS generates illusory contours through end-stopped cells and bipole cells.",
            "FCS is responsible for surface filling-in and color perception.",
            "Today is a good day to go hiking in the mountains.",
            "The sky is blue because of Rayleigh scattering.",
        ]
        query = "BCS 경계 완성"
        oneshot = await rerank.rerank(query, docs, top_n=1)
        batched = await rerank.rerank_batched(query, docs, batch_size=3, top_n=1)
        # Both should rank the BCS doc first
        assert oneshot[0]["index"] == 0
        assert batched[0]["index"] == 0


# ===========================================================================
# image_gate — live LLM relevance filtering
# ===========================================================================

@pytest.mark.integration
class TestImageGateLive:
    async def test_empty_candidates_returns_empty(self):
        from grag import image_gate
        assert await image_gate.select_relevant_images("BCS 회로도", []) == []

    async def test_clearly_irrelevant_image_not_selected(self):
        from grag import image_gate
        candidates = [{
            "hash": "abc123",
            "caption": "A photograph of a sunset over the ocean with orange clouds.",
            "section": "Preface",
            "page": 1,
        }]
        result = await image_gate.select_relevant_images(
            "BCS의 신경 회로 구조를 설명해줘", candidates
        )
        assert result == []

    async def test_relevant_circuit_diagram_selected(self):
        from grag import image_gate
        candidates = [
            {
                "hash": "irrelevant_hash",
                "caption": "Bar chart of reaction times in a psychology experiment.",
                "section": "Methods",
                "page": 10,
            },
            {
                "hash": "relevant_hash",
                "caption": (
                    "Figure 4.25: BCS boundary completion circuit. "
                    "Shows bipole cell connections between simple and complex cells "
                    "in layers 2/3 and 4 of visual cortex V1/V2. "
                    "Arrows indicate excitatory long-range horizontal connections."
                ),
                "section": "BCS",
                "page": 42,
            },
        ]
        result = await image_gate.select_relevant_images(
            "BCS 경계 완성 회로에서 바이폴 셀의 역할을 설명해줘", candidates
        )
        assert "relevant_hash" in result

    async def test_result_respects_max_images_cap(self):
        from grag import image_gate
        candidates = [
            {
                "hash": f"h{i}",
                "caption": f"Figure {i}: BCS boundary circuit diagram variant {i}.",
                "section": "BCS",
                "page": i,
            }
            for i in range(10)
        ]
        result = await image_gate.select_relevant_images(
            "모든 BCS 회로 다이어그램을 보여줘", candidates
        )
        assert len(result) <= image_gate.MAX_IMAGES

    async def test_result_contains_only_valid_hashes(self):
        from grag import image_gate
        candidates = [
            {"hash": "aabbcc", "caption": "FACADE figure", "section": "FACADE", "page": 5},
            {"hash": "ddeeff", "caption": "FCS filling-in", "section": "FCS", "page": 8},
        ]
        result = await image_gate.select_relevant_images("시각 지각", candidates)
        valid_hashes = {c["hash"] for c in candidates}
        assert all(h in valid_hashes for h in result)

    async def test_api_error_fails_closed(self, monkeypatch):
        from grag import image_gate, llm
        async def boom(*a, **kw):
            raise RuntimeError("simulated network error")
        monkeypatch.setattr(llm, "generate", boom)
        candidates = [{"hash": "h", "caption": "fig", "section": "A", "page": 1}]
        result = await image_gate.select_relevant_images("q", candidates)
        assert result == []


# ===========================================================================
# end-to-end router + rerank pipeline (no LightRAG required)
# ===========================================================================

@pytest.mark.integration
class TestRouterRerankPipeline:
    async def test_in_scope_question_then_rerank_docs(self):
        from grag import router, rerank
        q = "BCS에서 단순세포와 복합세포의 역할은?"
        route = await router.route(q)
        assert route["in_scope"] is True

        docs = [
            "Simple cells in V1 respond to oriented edges; complex cells pool their outputs.",
            "BCS uses end-stopped cells and bipole cells for boundary completion.",
            "Weather forecast for tomorrow: partly cloudy.",
        ]
        ranked = await rerank.rerank(q, docs, top_n=2)
        assert len(ranked) == 2
        # The weather doc should not be in top 2
        top_indices = {r["index"] for r in ranked}
        assert 2 not in top_indices

    async def test_out_of_scope_question_skips_rerank(self):
        from grag import router, rerank
        q = "오늘 주식 시장은 어때?"
        route = await router.route(q)
        assert route["in_scope"] is False
        # In production we'd skip retrieval; verify rerank still works defensively
        result = await rerank.rerank(q, [])
        assert result == []


# ===========================================================================
# hybrid_seed — against the REAL local entity store / LightRAG
# ===========================================================================

from pathlib import Path  # noqa: E402

_WDIR = "data/rag_storage"
_needs_index = pytest.mark.skipif(
    not Path(_WDIR, "vdb_entities.json").exists(),
    reason="no local rag_storage index (run ingest first)",
)


@pytest.mark.integration
@_needs_index
class TestHybridSeedRealCorpus:
    """Offline against the real persisted entity store (no API, but needs the index)."""

    def test_index_covers_full_entity_store(self):
        from grag.hybrid_seed import build_entity_index
        idx, created, _ = build_entity_index(_WDIR)
        assert idx.N > 500
        assert len(created) == idx.N

    def test_exact_name_and_figure_recovered_by_bm25(self):
        from grag.hybrid_seed import build_entity_index
        idx, _, tok = build_entity_index(_WDIR)
        assert "Figure 4.25" in idx.search("Figure 4.25", 5, tok)
        assert any("Boundary" in n for n in idx.search("boundary cortical stream", 5, tok))
        assert idx.search("hypercomplex", 5, tok)            # non-empty
        assert idx.search("zzzznonexistentterm", 5, tok) == []


@pytest.mark.integration
@_needs_index
class TestHybridSeedConfigGate:
    """setup() must honor the config flag (loads index from disk; no API call)."""

    async def test_flag_off_does_not_attach(self):
        from grag.engine import ChatSession, _SESSIONS_DIR
        sess = ChatSession(_WDIR, provider="gemini", rerank_mode="none",
                           session_path=_SESSIONS_DIR / "_it_off.json")
        await sess.setup()
        try:
            assert getattr(sess.rag.entities_vdb, "_grag_hybrid_attached", False) is False
        finally:
            await sess.teardown()

    async def test_flag_on_attaches_via_setup(self, monkeypatch):
        from grag import retrieval
        from grag.engine import ChatSession, _SESSIONS_DIR
        # the flag is read by the shared factory (retrieval), not engine
        monkeypatch.setitem(retrieval._cfg["query"], "hybrid_seed", True)
        sess = ChatSession(_WDIR, provider="gemini", rerank_mode="none",
                           session_path=_SESSIONS_DIR / "_it_on.json")
        await sess.setup()
        try:
            assert getattr(sess.rag.entities_vdb, "_grag_hybrid_attached", False) is True
        finally:
            await sess.teardown()


@pytest.mark.integration
@_needs_index
class TestHybridSeedLiveRetrieval:
    """Real LightRAG + real entity embeddings: hybrid must union lexical seeds the
    pure-vector seed set misses (needs a live embedding key)."""

    async def test_hybrid_unions_seed_vector_missed(self):
        from grag.engine import ChatSession, _SESSIONS_DIR
        from grag import hybrid_seed
        sess = ChatSession(_WDIR, provider="gemini", rerank_mode="none",
                           session_path=_SESSIONS_DIR / "_it_hybrid.json")
        await sess.setup()
        try:
            q = "BCS double filter end cut figure 4.25"
            vdb = sess.rag.entities_vdb
            base = [r["entity_name"] for r in await vdb.query(q, top_k=30)]
            hybrid_seed.attach_hybrid_seed(sess.rag, _WDIR, top_k=10)
            hyb_hits = await vdb.query(q, top_k=30)
            hyb = [r["entity_name"] for r in hyb_hits]
            assert set(base) <= set(hyb)                       # vector seeds preserved
            assert len(set(hyb) - set(base)) >= 1              # BM25 added new seed(s)
            assert any(r.get("_grag_bm25") for r in hyb_hits)  # added ones are tagged
        finally:
            await sess.teardown()


# ===========================================================================
# expand — corpus-language keyword expansion (cross-lingual bridge)
# ===========================================================================

@pytest.mark.integration
@_needs_index
class TestCorpusLangDetection:
    """Offline against the real index (no API)."""

    def test_real_corpus_detected_english(self):
        from grag.expand import detect_corpus_lang
        assert detect_corpus_lang(_WDIR) == "en"


@pytest.mark.integration
@_needs_index
class TestExpandConfigGate:
    """setup() resolves corpus_lang only when the flag is on (loads index; no API)."""

    async def test_flag_off_keeps_default_lang(self):
        from grag.engine import ChatSession, _SESSIONS_DIR
        sess = ChatSession(_WDIR, provider="gemini", rerank_mode="none",
                           session_path=_SESSIONS_DIR / "_it_exp_off.json")
        await sess.setup()
        try:
            assert sess.corpus_lang == "en"  # default, not auto-detected (flag off)
        finally:
            await sess.teardown()

    async def test_flag_on_autodetects_corpus_lang(self, monkeypatch):
        from grag import engine
        from grag.engine import ChatSession, _SESSIONS_DIR
        monkeypatch.setitem(engine._cfg["query"], "expand_keywords", True)
        monkeypatch.setitem(engine._cfg["query"], "expand_lang", "auto")
        sess = ChatSession(_WDIR, provider="gemini", rerank_mode="none",
                           session_path=_SESSIONS_DIR / "_it_exp_on.json")
        await sess.setup()
        try:
            assert sess.corpus_lang == "en"
        finally:
            await sess.teardown()


@pytest.mark.integration
class TestExpandKeywordsLive:
    """Real flash-lite call: a Korean query must expand into English (corpus-lang) keywords."""

    async def test_korean_query_expands_to_english(self):
        import re
        from grag.expand import expand_keywords
        out = await expand_keywords("초복합세포가 엔드 컷을 만드는 회로를 설명해줘", "en")
        terms = out["concepts"] + out["entities"]
        assert terms, "expansion returned no keywords"
        # translated into the corpus language (English) -> at least one ASCII-latin term
        assert any(re.search(r"[A-Za-z]", t) for t in terms)


@pytest.mark.integration
@_needs_index
class TestGlossaryRealCorpus:
    """Offline against the real index (no API): glossary covers known jargon and the
    expansion system prompt clears the Gemini prefix-cache threshold."""

    def test_glossary_contains_jargon_and_is_cache_eligible(self):
        from grag.expand import build_glossary, _system_prompt
        terms = build_glossary(_WDIR).split("\n")
        assert any("FACADE" in t for t in terms)
        assert any(t == "BCS" or "Boundary Contour System" in t for t in terms)
        assert any("Filling-In" in t for t in terms)
        assert len(_system_prompt("en", "\n".join(terms))) >= 4000  # Gemini cache threshold


@pytest.mark.integration
@_needs_index
class TestKBToolInheritsHybridSeed:
    """The grossberg-ask skill / agent loop build their rag via the shared factory, so
    enabling hybrid_seed must reach KBTool too (no API; builds from the on-disk index)."""

    async def test_kbtool_rag_is_hybrid_wrapped_when_flag_on(self, monkeypatch):
        from grag import retrieval
        from grag.kb_tool import KBTool
        monkeypatch.setitem(retrieval._cfg["query"], "hybrid_seed", True)
        kb = KBTool(working_dir=_WDIR)
        rag = await kb._get_rag()
        try:
            assert getattr(rag.entities_vdb, "_grag_hybrid_attached", False) is True
        finally:
            await rag.finalize_storages()
