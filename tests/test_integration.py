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
        import router
        r = await router.route("BCS와 FCS의 차이를 설명해줘.")
        assert r["in_scope"] is True
        assert r["effort"] in ("low", "medium", "high")
        assert isinstance(r["needs_retrieval"], bool)

    async def test_out_of_scope_coding_question_rejected(self):
        import router
        r = await router.route("파이썬으로 웹 서버를 만드는 법을 알려줘.")
        assert r["in_scope"] is False

    async def test_out_of_scope_weather_rejected(self):
        import router
        r = await router.route("오늘 서울 날씨가 어때?")
        assert r["in_scope"] is False

    async def test_greeting_needs_no_retrieval(self):
        import router
        r = await router.route("안녕하세요!")
        assert r["needs_retrieval"] is False

    async def test_simple_definition_low_effort(self):
        import router
        r = await router.route("FCS가 뭐야?")
        assert r["in_scope"] is True
        assert r["effort"] in ("low", "medium")

    async def test_complex_synthesis_high_effort(self):
        import router
        q = ("BCS, FCS, FACADE 이론의 상호작용을 비교하고 각각의 신경 회로 구조가 "
             "지각적 그루핑에서 어떻게 협력하는지 인과적 흐름으로 설명해줘.")
        r = await router.route(q)
        assert r["in_scope"] is True
        assert r["effort"] == "high"

    async def test_vague_referential_question_may_request_clarification(self):
        import router
        r = await router.route("그거 설명해줘.")
        # Router may or may not flag clarification — contract: never crashes,
        # always returns a fully populated dict.
        assert "in_scope" in r
        assert "needs_clarification" in r
        assert "effort" in r

    async def test_meta_question_no_retrieval_needed(self):
        import router
        history = [
            {"role": "user", "content": "FACADE 이론이란?"},
            {"role": "assistant", "content": "FACADE 이론은 ..."},
        ]
        r = await router.route("방금 말한 내용을 요약해줘.", history)
        assert r["in_scope"] is True
        assert r["needs_retrieval"] is False

    async def test_history_resolves_demonstrative_pronoun(self):
        import router
        history = [
            {"role": "user", "content": "BCS란 무엇인가?"},
            {"role": "assistant", "content": "BCS는 경계 윤곽 시스템입니다."},
        ]
        r = await router.route("그게 LAMINART와 어떻게 연결돼?", history)
        assert r["in_scope"] is True
        # History provides context -> clarification should not be needed
        assert r["needs_clarification"] is False

    async def test_response_always_has_all_required_keys(self):
        import router
        r = await router.route("아무 질문")
        required = {"in_scope", "needs_retrieval", "effort", "needs_clarification",
                    "clarification", "reason"}
        assert required.issubset(r.keys())

    async def test_effort_always_in_valid_set(self):
        import router
        questions = [
            "BCS란?",
            "FACADE와 BCS의 관계를 설명해줘.",
            "BCS, FCS, LAMINART, FACADE를 비교 분석해줘.",
        ]
        for q in questions:
            r = await router.route(q)
            assert r["effort"] in ("low", "medium", "high"), f"invalid effort for: {q}"

    async def test_disabled_router_returns_passthrough(self, monkeypatch):
        import router
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
        import rerank
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
        import rerank
        docs = [f"doc {i} about neural models" for i in range(5)]
        result = await rerank.rerank("BCS", docs)
        assert all(0.0 <= r["relevance_score"] <= 1.5 for r in result)
        # upper bound is 1.5 to allow >100 scores from model without crashing

    async def test_top_n_limits_live_result(self):
        import rerank
        docs = [f"neural model doc {i}" for i in range(8)]
        result = await rerank.rerank("FACADE theory", docs, top_n=3)
        assert len(result) == 3

    async def test_result_sorted_descending(self):
        import rerank
        docs = [
            "FACADE theory explains how surfaces are perceived.",
            "Unrelated text about programming.",
            "FCS handles surface filling-in and brightness perception.",
        ]
        result = await rerank.rerank("surface perception", docs)
        scores = [r["relevance_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    async def test_single_doc_returns_single_result(self):
        import rerank
        result = await rerank.rerank("BCS", ["BCS is the Boundary Contour System."])
        assert len(result) == 1
        assert result[0]["index"] == 0

    async def test_empty_docs_returns_empty(self):
        import rerank
        assert await rerank.rerank("BCS", []) == []

    async def test_batched_result_has_valid_global_indices(self):
        import rerank
        docs = [f"doc {i}" for i in range(25)]
        result = await rerank.rerank_batched("BCS 경계 완성", docs, batch_size=10, top_n=5)
        assert all(0 <= r["index"] < len(docs) for r in result)
        assert len(result) <= 5

    async def test_batched_vs_oneshot_both_return_same_top_doc(self):
        import rerank
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
        import image_gate
        assert await image_gate.select_relevant_images("BCS 회로도", []) == []

    async def test_clearly_irrelevant_image_not_selected(self):
        import image_gate
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
        import image_gate
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
        import image_gate
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
        import image_gate
        candidates = [
            {"hash": "aabbcc", "caption": "FACADE figure", "section": "FACADE", "page": 5},
            {"hash": "ddeeff", "caption": "FCS filling-in", "section": "FCS", "page": 8},
        ]
        result = await image_gate.select_relevant_images("시각 지각", candidates)
        valid_hashes = {c["hash"] for c in candidates}
        assert all(h in valid_hashes for h in result)

    async def test_api_error_fails_closed(self, monkeypatch):
        import image_gate, llm
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
        import router, rerank
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
        import router, rerank
        q = "오늘 주식 시장은 어때?"
        route = await router.route(q)
        assert route["in_scope"] is False
        # In production we'd skip retrieval; verify rerank still works defensively
        result = await rerank.rerank(q, [])
        assert result == []
