"""Unit tests for rerank.py — _strip_fences() pure + rerank/rerank_batched mocked."""
import json
from unittest.mock import AsyncMock

import pytest

import rerank


class TestStripFences:
    def test_plain_json_unchanged(self):
        s = '[{"index": 0, "score": 80}]'
        assert rerank._strip_fences(s) == s

    def test_json_fence_removed(self):
        s = '```json\n[{"index": 0}]\n```'
        assert rerank._strip_fences(s) == '[{"index": 0}]'

    def test_plain_fence_no_lang_tag_removed(self):
        s = '```\n[{"index": 0}]\n```'
        assert rerank._strip_fences(s) == '[{"index": 0}]'

    def test_empty_string_unchanged(self):
        assert rerank._strip_fences("") == ""

    def test_leading_trailing_whitespace_stripped(self):
        assert rerank._strip_fences("  [1, 2]  ") == "[1, 2]"

    def test_fence_with_extra_whitespace_in_tag(self):
        # "```json\n" — json tag right after backticks (no space)
        s = '```json\n[{}]\n```'
        result = rerank._strip_fences(s)
        assert "[{}]" in result

    def test_single_backtick_not_treated_as_fence(self):
        s = '`not a fence`'
        # does not start with ```, so returned as-is (stripped)
        assert rerank._strip_fences(s) == '`not a fence`'

    def test_multiple_fences_takes_first_inner(self):
        # Only the outer split matters; inner content is returned
        s = '```json\n[0]\n```\n```json\n[1]\n```'
        result = rerank._strip_fences(s)
        # first split between ``` yields the first inner block
        assert "[0]" in result


class TestRerank:
    async def test_empty_documents_returns_empty(self):
        assert await rerank.rerank("q", []) == []

    async def test_valid_response_normalized_to_0_1(self, monkeypatch):
        import llm
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value='[{"index": 0, "score": 100}]'))
        result = await rerank.rerank("q", ["doc"])
        assert result[0]["relevance_score"] == pytest.approx(1.0)

    async def test_score_50_normalized_to_0_5(self, monkeypatch):
        import llm
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value='[{"index": 0, "score": 50}]'))
        result = await rerank.rerank("q", ["doc"])
        assert result[0]["relevance_score"] == pytest.approx(0.5)

    async def test_results_sorted_descending_by_score(self, monkeypatch):
        import llm
        payload = '[{"index": 0, "score": 30}, {"index": 1, "score": 90}, {"index": 2, "score": 60}]'
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value=payload))
        result = await rerank.rerank("q", ["a", "b", "c"])
        scores = [r["relevance_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    async def test_top_n_limits_output_count(self, monkeypatch):
        import llm
        payload = '[{"index": 0, "score": 80}, {"index": 1, "score": 60}, {"index": 2, "score": 40}]'
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value=payload))
        result = await rerank.rerank("q", ["a", "b", "c"], top_n=2)
        assert len(result) == 2

    async def test_top_n_none_returns_all(self, monkeypatch):
        import llm
        payload = '[{"index": 0, "score": 80}, {"index": 1, "score": 60}]'
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value=payload))
        result = await rerank.rerank("q", ["a", "b"], top_n=None)
        assert len(result) == 2

    async def test_malformed_json_fallback_all_0_5(self, monkeypatch):
        import llm
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value="not json at all"))
        result = await rerank.rerank("q", ["a", "b", "c"])
        assert len(result) == 3
        assert all(r["relevance_score"] == 0.5 for r in result)

    async def test_fenced_json_parsed_correctly(self, monkeypatch):
        import llm
        payload = '```json\n[{"index": 0, "score": 75}]\n```'
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value=payload))
        result = await rerank.rerank("q", ["doc"])
        assert result[0]["relevance_score"] == pytest.approx(0.75)

    async def test_out_of_range_index_silently_dropped(self, monkeypatch):
        import llm
        # index 3 is out of range for 2 docs
        payload = '[{"index": 0, "score": 80}, {"index": 3, "score": 99}]'
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value=payload))
        result = await rerank.rerank("q", ["a", "b"])
        assert all(r["index"] < 2 for r in result)

    async def test_negative_index_silently_dropped(self, monkeypatch):
        import llm
        payload = '[{"index": -1, "score": 99}, {"index": 0, "score": 50}]'
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value=payload))
        result = await rerank.rerank("q", ["a"])
        assert all(r["index"] >= 0 for r in result)

    async def test_duplicate_indices_both_kept(self, monkeypatch):
        import llm
        payload = '[{"index": 0, "score": 80}, {"index": 0, "score": 60}]'
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value=payload))
        result = await rerank.rerank("q", ["a"])
        # both pass the range check; dedup is not rerank's job
        assert len(result) == 2

    async def test_missing_score_defaults_to_0(self, monkeypatch):
        import llm
        payload = '[{"index": 0}]'
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value=payload))
        result = await rerank.rerank("q", ["a"])
        assert result[0]["relevance_score"] == pytest.approx(0.0)

    async def test_score_over_100_not_clamped(self, monkeypatch):
        import llm
        payload = '[{"index": 0, "score": 150}]'
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value=payload))
        result = await rerank.rerank("q", ["a"])
        assert result[0]["relevance_score"] == pytest.approx(1.5)

    async def test_empty_array_response_returns_empty(self, monkeypatch):
        import llm
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value='[]'))
        result = await rerank.rerank("q", ["a", "b"])
        assert result == []

    async def test_long_doc_truncated_in_prompt(self, monkeypatch):
        import llm
        captured = []
        async def capture(model, prompt, **kw):
            captured.append(prompt)
            return '[{"index": 0, "score": 50}]'
        monkeypatch.setattr(llm, "generate", capture)
        long_doc = "x" * 3000
        await rerank.rerank("q", [long_doc])
        # The truncated doc in the prompt should be shorter than the original
        assert len(captured[0]) < 3000 + 500  # prompt overhead is reasonable


class TestRerankBatched:
    async def test_empty_documents_returns_empty(self):
        assert await rerank.rerank_batched("q", []) == []

    async def test_small_set_delegates_single_call(self, monkeypatch):
        import llm
        calls = []
        async def capture(model, prompt, **kw):
            calls.append(True)
            return '[{"index": 0, "score": 80}]'
        monkeypatch.setattr(llm, "generate", capture)
        result = await rerank.rerank_batched("q", ["doc0"], batch_size=20)
        assert len(result) == 1
        assert len(calls) == 1  # single rerank call, no batching

    async def test_large_set_makes_multiple_calls(self, monkeypatch):
        import llm
        calls = []
        async def capture(model, prompt, **kw):
            calls.append(prompt)
            # Count docs in prompt by counting "[N]" prefixes
            n = prompt.count("\n[")
            return json.dumps([{"index": i, "score": 80 - i * 2} for i in range(n)])
        monkeypatch.setattr(llm, "generate", capture)
        docs = [f"doc{i}" for i in range(25)]
        await rerank.rerank_batched("q", docs, batch_size=10, top_n=5)
        # With 25 docs and batch_size=10: 3 stage-1 batches + 1 stage-2 final = 4 calls
        assert len(calls) >= 3

    async def test_top_n_respected_in_output(self, monkeypatch):
        import llm
        async def capture(model, prompt, **kw):
            n = prompt.count("\n[")
            return json.dumps([{"index": i, "score": 80 - i} for i in range(n)])
        monkeypatch.setattr(llm, "generate", capture)
        docs = [f"doc{i}" for i in range(30)]
        result = await rerank.rerank_batched("q", docs, batch_size=10, top_n=3)
        assert len(result) <= 3

    async def test_exactly_batch_size_docs_single_rerank(self, monkeypatch):
        import llm
        calls = []
        async def capture(model, prompt, **kw):
            calls.append(True)
            return '[{"index": 0, "score": 80}]'
        monkeypatch.setattr(llm, "generate", capture)
        docs = [f"doc{i}" for i in range(20)]
        await rerank.rerank_batched("q", docs, batch_size=20)
        assert len(calls) == 1

    async def test_global_indices_preserved_after_batching(self, monkeypatch):
        import llm
        # With 22 docs and batch_size=10: batches [0-9], [10-19], [20-21]
        # After stage-1 each batch keeps top half; indices in result must map
        # to original 0-21 range, not the local batch range
        async def capture(model, prompt, **kw):
            n = prompt.count("\n[")
            return json.dumps([{"index": i, "score": 90 - i * 3} for i in range(n)])
        monkeypatch.setattr(llm, "generate", capture)
        docs = [f"doc{i}" for i in range(22)]
        result = await rerank.rerank_batched("q", docs, batch_size=10, top_n=5)
        # All returned indices must be valid global indices
        assert all(0 <= r["index"] < len(docs) for r in result)
