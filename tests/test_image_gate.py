"""Unit tests for image_gate.py — _parse() pure + select_relevant_images() mocked."""
from unittest.mock import AsyncMock

import pytest

from grag import image_gate


class TestParse:
    def test_valid_single_index(self):
        result = image_gate._parse('{"relevant": [0]}', n=3)
        assert result == [0]

    def test_valid_multiple_indices_order_preserved(self):
        result = image_gate._parse('{"relevant": [2, 0, 1]}', n=3)
        assert result == [2, 0, 1]

    def test_empty_relevant_list(self):
        assert image_gate._parse('{"relevant": []}', n=5) == []

    def test_no_relevant_key(self):
        assert image_gate._parse('{}', n=5) == []

    def test_no_json_in_raw(self):
        assert image_gate._parse("no json here", n=5) == []

    def test_out_of_range_index_filtered(self):
        # n=2 → valid range [0, 1]; index 2 is out
        result = image_gate._parse('{"relevant": [0, 2, 5]}', n=2)
        assert result == [0]

    def test_negative_index_filtered(self):
        result = image_gate._parse('{"relevant": [-1, 0]}', n=3)
        assert result == [0]

    def test_duplicate_indices_deduplicated(self):
        result = image_gate._parse('{"relevant": [1, 1, 0, 1]}', n=3)
        assert result == [1, 0]  # first occurrence wins, duplicates dropped

    def test_non_int_index_skipped(self):
        result = image_gate._parse('{"relevant": ["two", 0, null]}', n=3)
        assert result == [0]

    def test_float_index_truncated_to_int(self):
        # int(1.9) == 1
        result = image_gate._parse('{"relevant": [1.9]}', n=3)
        assert result == [1]

    def test_n_equals_zero_all_filtered(self):
        result = image_gate._parse('{"relevant": [0, 1]}', n=0)
        assert result == []

    def test_index_exactly_at_n_minus_1_included(self):
        result = image_gate._parse('{"relevant": [4]}', n=5)
        assert result == [4]

    def test_index_exactly_n_excluded(self):
        result = image_gate._parse('{"relevant": [5]}', n=5)
        assert result == []

    def test_json_in_prose_still_extracted(self):
        raw = 'Here are the relevant figures: {"relevant": [0, 2]} Done.'
        result = image_gate._parse(raw, n=5)
        assert result == [0, 2]

    def test_all_indices_invalid_returns_empty(self):
        result = image_gate._parse('{"relevant": [99, 100, -1]}', n=5)
        assert result == []


class TestSelectRelevantImages:
    async def test_empty_candidates_returns_empty(self):
        result = await image_gate.select_relevant_images("question", [])
        assert result == []

    async def test_llm_error_fails_closed_returns_empty(self, monkeypatch):
        from grag import llm
        async def boom(*a, **kw):
            raise RuntimeError("api error")
        monkeypatch.setattr(llm, "generate", boom)
        candidates = [{"hash": "abc123", "caption": "fig", "section": "A", "page": 1}]
        result = await image_gate.select_relevant_images("q", candidates)
        assert result == []

    async def test_valid_selection_returns_hashes(self, monkeypatch):
        from grag import llm
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value='{"relevant": [1, 0]}'))
        candidates = [
            {"hash": "hash0", "caption": "fig0", "section": "A", "page": 1},
            {"hash": "hash1", "caption": "fig1", "section": "B", "page": 2},
        ]
        result = await image_gate.select_relevant_images("q", candidates)
        assert result == ["hash1", "hash0"]

    async def test_order_matches_parse_order(self, monkeypatch):
        from grag import llm
        # Most relevant first = index 2, then 0
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value='{"relevant": [2, 0]}'))
        candidates = [
            {"hash": "h0", "caption": "c0", "section": "A", "page": 1},
            {"hash": "h1", "caption": "c1", "section": "B", "page": 2},
            {"hash": "h2", "caption": "c2", "section": "C", "page": 3},
        ]
        result = await image_gate.select_relevant_images("q", candidates)
        assert result[0] == "h2"
        assert result[1] == "h0"

    async def test_max_images_cap_enforced(self, monkeypatch):
        from grag import llm
        # Return all 10 indices as relevant
        monkeypatch.setattr(llm, "generate", AsyncMock(
            return_value='{"relevant": [0,1,2,3,4,5,6,7,8,9]}'
        ))
        candidates = [
            {"hash": f"h{i}", "caption": f"fig{i}", "section": "X", "page": i}
            for i in range(10)
        ]
        result = await image_gate.select_relevant_images("q", candidates)
        assert len(result) <= image_gate.MAX_IMAGES

    async def test_no_relevant_returns_empty(self, monkeypatch):
        from grag import llm
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value='{"relevant": []}'))
        candidates = [{"hash": "h0", "caption": "fig", "section": "A", "page": 1}]
        result = await image_gate.select_relevant_images("q", candidates)
        assert result == []

    async def test_out_of_range_indices_silently_dropped(self, monkeypatch):
        from grag import llm
        # Only 1 candidate (index 0 valid), model returns index 5 (invalid)
        monkeypatch.setattr(llm, "generate", AsyncMock(return_value='{"relevant": [5, 0]}'))
        candidates = [{"hash": "h0", "caption": "fig", "section": "A", "page": 1}]
        result = await image_gate.select_relevant_images("q", candidates)
        assert result == ["h0"]

    async def test_llm_network_timeout_fails_closed(self, monkeypatch):
        from grag import llm
        async def slow(*a, **kw):
            raise TimeoutError("timeout")
        monkeypatch.setattr(llm, "generate", slow)
        candidates = [{"hash": "abc", "caption": "fig", "section": "A", "page": 1}]
        result = await image_gate.select_relevant_images("q", candidates)
        assert result == []

    async def test_long_caption_truncated_in_prompt(self, monkeypatch):
        from grag import llm
        captured = []
        async def capture(model, prompt, **kw):
            captured.append(prompt)
            return '{"relevant": []}'
        monkeypatch.setattr(llm, "generate", capture)
        long_caption = "x" * 1000
        candidates = [{"hash": "h0", "caption": long_caption, "section": "A", "page": 1}]
        await image_gate.select_relevant_images("q", candidates)
        # Caption truncated to 400 chars in listing
        assert long_caption not in captured[0]
        assert "x" * 400 in captured[0] or len(captured[0]) < 1000 + 200
