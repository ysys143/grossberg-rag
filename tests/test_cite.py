"""Unit tests for cite.py — pure logic, no mocks needed."""
from pathlib import Path

import pytest

from grag.cite import _is_heading, doc_name_for, enrich_content_list


class TestIsHeading:
    def test_level_1_is_heading(self):
        assert _is_heading({"text_level": 1}) is True

    def test_level_3_is_heading(self):
        assert _is_heading({"text_level": 3}) is True

    def test_level_0_is_not_heading(self):
        assert _is_heading({"text_level": 0}) is False

    def test_missing_level_is_not_heading(self):
        assert _is_heading({}) is False

    def test_string_level_is_not_heading(self):
        assert _is_heading({"text_level": "1"}) is False

    def test_none_level_is_not_heading(self):
        assert _is_heading({"text_level": None}) is False


class TestEnrichContentList:
    def test_text_block_gets_src_marker(self):
        items = [{"type": "text", "text": "hello world", "page_idx": 0}]
        result = enrich_content_list(items, "doc.pdf")
        assert "[src: doc.pdf | §(front matter) | p.1]" in result[0]["text"]
        assert "hello world" in result[0]["text"]

    def test_page_idx_zero_based_converted_to_one_based(self):
        items = [{"type": "text", "text": "x", "page_idx": 4}]
        result = enrich_content_list(items, "doc.pdf")
        assert "p.5" in result[0]["text"]

    def test_heading_updates_current_section(self):
        items = [
            {"type": "text", "text": "BCS Overview", "page_idx": 0, "text_level": 1},
            {"type": "text", "text": "body text", "page_idx": 1},
        ]
        result = enrich_content_list(items, "doc.pdf")
        assert "§BCS Overview" in result[1]["text"]

    def test_section_persists_across_multiple_blocks(self):
        items = [
            {"type": "text", "text": "Intro", "page_idx": 0, "text_level": 1},
            {"type": "text", "text": "para1", "page_idx": 0},
            {"type": "text", "text": "para2", "page_idx": 1},
            {"type": "text", "text": "para3", "page_idx": 2},
        ]
        result = enrich_content_list(items, "doc.pdf")
        assert "§Intro" in result[1]["text"]
        assert "§Intro" in result[2]["text"]
        assert "§Intro" in result[3]["text"]

    def test_new_heading_overrides_section(self):
        items = [
            {"type": "text", "text": "Section A", "page_idx": 0, "text_level": 1},
            {"type": "text", "text": "para", "page_idx": 0},
            {"type": "text", "text": "Section B", "page_idx": 1, "text_level": 1},
            {"type": "text", "text": "para2", "page_idx": 1},
        ]
        result = enrich_content_list(items, "doc.pdf")
        assert "§Section A" in result[1]["text"]
        assert "§Section B" in result[3]["text"]

    def test_default_section_is_front_matter(self):
        items = [{"type": "text", "text": "preface", "page_idx": 0}]
        result = enrich_content_list(items, "doc.pdf")
        assert "(front matter)" in result[0]["text"]

    def test_empty_text_still_gets_marker(self):
        items = [{"type": "text", "text": "", "page_idx": 0}]
        result = enrich_content_list(items, "doc.pdf")
        assert result[0]["text"].startswith("[src:")

    def test_original_items_not_mutated(self):
        items = [{"type": "text", "text": "original", "page_idx": 0}]
        enrich_content_list(items, "doc.pdf")
        assert items[0]["text"] == "original"

    def test_multimodal_image_gets_img_caption_marker(self):
        items = [{"type": "image", "page_idx": 2}]
        result = enrich_content_list(items, "doc.pdf")
        assert "img_caption" in result[0]
        marker = result[0]["img_caption"][0]
        assert marker.startswith("[src: doc.pdf")
        assert "image" in marker
        assert "p.3" in marker

    def test_multimodal_prepends_to_existing_img_caption(self):
        items = [{"type": "image", "page_idx": 0, "img_caption": ["Figure 4.1"]}]
        result = enrich_content_list(items, "doc.pdf")
        caps = result[0]["img_caption"]
        assert len(caps) == 2
        assert caps[0].startswith("[src:")
        assert caps[1] == "Figure 4.1"

    def test_multimodal_prefers_image_caption_field(self):
        items = [{"type": "image", "page_idx": 0, "image_caption": ["Cap A"]}]
        result = enrich_content_list(items, "doc.pdf")
        assert "image_caption" in result[0]
        assert result[0]["image_caption"][0].startswith("[src:")

    def test_multimodal_also_sets_text_marker(self):
        items = [{"type": "image", "page_idx": 0}]
        result = enrich_content_list(items, "doc.pdf")
        assert result[0]["text"].startswith("[src:")

    def test_multimodal_string_caption_wrapped_in_list(self):
        items = [{"type": "image", "page_idx": 0, "img_caption": "single string"}]
        result = enrich_content_list(items, "doc.pdf")
        caps = result[0]["img_caption"]
        assert isinstance(caps, list)
        assert caps[0].startswith("[src:")

    def test_marker_format(self):
        items = [{"type": "text", "text": "x", "page_idx": 9}]
        result = enrich_content_list(items, "grossberg_ch4.pdf")
        marker_line = result[0]["text"].splitlines()[0]
        assert marker_line == "[src: grossberg_ch4.pdf | §(front matter) | p.10]"

    def test_empty_list_returns_empty(self):
        assert enrich_content_list([], "doc.pdf") == []


class TestDocNameFor:
    def test_returns_filename_only(self):
        assert doc_name_for(Path("/some/path/grossberg_ch4.pdf")) == "grossberg_ch4.pdf"

    def test_no_directory_prefix(self):
        result = doc_name_for(Path("relative/path/doc.pdf"))
        assert "/" not in result
        assert result == "doc.pdf"
