"""Unit tests for pure helpers and ChatSession I/O in chat.py."""
import json
import sys
from pathlib import Path

import pytest

from grag import engine, cli
from grag.engine import (
    _extract_sources,
    _cited_pages,
    _cited_chunks,
    _image_candidates,
    ChatSession,
)
from grag.cli import _parse_args, _session_ts


# ---------------------------------------------------------------------------
# _extract_sources
# ---------------------------------------------------------------------------

class TestExtractSources:
    def test_single_marker_returned(self):
        prompt = "some text [src: doc.pdf | §BCS | p.5] more text"
        assert _extract_sources(prompt) == ["[src: doc.pdf | §BCS | p.5]"]

    def test_multiple_distinct_markers(self):
        prompt = "[src: doc.pdf | §A | p.1] ... [src: doc.pdf | §B | p.3]"
        result = _extract_sources(prompt)
        assert len(result) == 2

    def test_duplicate_markers_deduplicated(self):
        marker = "[src: doc.pdf | §BCS | p.5]"
        prompt = f"{marker} some text {marker}"
        result = _extract_sources(prompt)
        assert result == [marker]

    def test_order_of_first_occurrence_preserved(self):
        prompt = "[src: doc.pdf | §B | p.3] [src: doc.pdf | §A | p.1]"
        result = _extract_sources(prompt)
        assert result[0] == "[src: doc.pdf | §B | p.3]"
        assert result[1] == "[src: doc.pdf | §A | p.1]"

    def test_no_markers_returns_empty(self):
        assert _extract_sources("no markers here") == []

    def test_empty_string_returns_empty(self):
        assert _extract_sources("") == []

    def test_marker_with_image_suffix_captured(self):
        prompt = "[src: doc.pdf | §FACADE | p.7 | image]"
        result = _extract_sources(prompt)
        assert result == ["[src: doc.pdf | §FACADE | p.7 | image]"]

    def test_partial_bracket_not_captured(self):
        prompt = "[src: doc.pdf incomplete"
        assert _extract_sources(prompt) == []

    def test_marker_spanning_multiline_is_captured(self):
        # [^\]]+ matches any non-] char including newlines (character class,
        # not dot — re.DOTALL doesn't apply). The regex does cross newlines.
        prompt = "[src: doc.pdf\n| §BCS | p.5]"
        result = _extract_sources(prompt)
        assert len(result) == 1
        assert "doc.pdf" in result[0]


# ---------------------------------------------------------------------------
# _cited_pages
# ---------------------------------------------------------------------------

class TestCitedPages:
    def test_single_page_reference(self):
        assert _cited_pages("see p.5 for details") == {5}

    def test_page_with_space(self):
        assert _cited_pages("see p. 5 for details") == {5}

    def test_range_with_hyphen(self):
        assert _cited_pages("see p.3-7") == {3, 4, 5, 6, 7}

    def test_range_with_en_dash(self):
        assert _cited_pages("see p.3–7") == {3, 4, 5, 6, 7}

    def test_multiple_distinct_pages(self):
        assert _cited_pages("p.1 and p.10") == {1, 10}

    def test_page_in_src_marker(self):
        assert _cited_pages("[src: doc.pdf | §BCS | p.42]") == {42}

    def test_no_pages_returns_empty_set(self):
        assert _cited_pages("no page references here") == set()

    def test_empty_string_returns_empty_set(self):
        assert _cited_pages("") == set()

    def test_range_single_page_range(self):
        # p.5-5 is a degenerate range, still includes 5
        assert _cited_pages("p.5-5") == {5}

    def test_page_zero_included(self):
        assert _cited_pages("p.0") == {0}

    def test_range_and_standalone_combined(self):
        pages = _cited_pages("p.1-3 and also p.10")
        assert pages == {1, 2, 3, 10}


# ---------------------------------------------------------------------------
# _cited_chunks
# ---------------------------------------------------------------------------

def _json_line(content: str) -> str:
    return json.dumps({"content": content})


class TestCitedChunks:
    def test_returns_chunks_on_cited_page(self):
        chunk = _json_line("[src: doc.pdf | §BCS | p.5] BCS defines boundary completion.")
        answer = "as shown on p.5, boundaries are..."
        result = _cited_chunks(answer, chunk)
        assert len(result) == 1
        assert "BCS defines" in result[0]

    def test_ignores_chunks_on_uncited_pages(self):
        chunk = _json_line("[src: doc.pdf | §BCS | p.9] irrelevant content")
        answer = "the answer cites p.5"
        result = _cited_chunks(answer, chunk)
        assert result == []

    def test_no_pages_in_answer_returns_empty(self):
        chunk = _json_line("[src: doc.pdf | §BCS | p.5] BCS content")
        answer = "a general statement with no page reference"
        assert _cited_chunks(answer, chunk) == []

    def test_duplicate_content_deduplicated(self):
        line = _json_line("[src: doc.pdf | §A | p.3] shared content")
        prompt = f"{line}\n{line}"
        answer = "p.3 is cited"
        result = _cited_chunks(answer, prompt)
        assert len(result) == 1

    def test_non_json_lines_skipped(self):
        prompt = "plain text line\n" + _json_line("[src: doc.pdf | §X | p.7] data")
        answer = "p.7 referenced"
        result = _cited_chunks(answer, prompt)
        assert len(result) == 1

    def test_json_without_content_or_description_skipped(self):
        prompt = json.dumps({"other_field": "p.5 data"})
        answer = "p.5 cited"
        result = _cited_chunks(answer, prompt)
        assert result == []

    def test_description_field_also_matched(self):
        line = json.dumps({"description": "[src: doc.pdf | §D | p.2] desc content"})
        answer = "mentioned at p.2"
        result = _cited_chunks(answer, line)
        assert len(result) == 1

    def test_content_truncated_to_1000_chars(self):
        long_text = "x" * 2000
        content = f"[src: doc.pdf | §L | p.1] {long_text}"
        prompt = _json_line(content)
        answer = "p.1 cited"
        result = _cited_chunks(answer, prompt)
        assert len(result) == 1
        assert len(result[0]) <= 1000

    def test_multiple_chunks_same_page_all_returned(self):
        line1 = _json_line("[src: doc.pdf | §A | p.4] first chunk")
        line2 = _json_line("[src: doc.pdf | §B | p.4] second chunk")
        prompt = f"{line1}\n{line2}"
        answer = "see p.4"
        result = _cited_chunks(answer, prompt)
        assert len(result) == 2

    def test_range_cited_includes_all_pages_in_range(self):
        chunk_p3 = _json_line("[src: doc.pdf | §A | p.3] page 3 content")
        chunk_p5 = _json_line("[src: doc.pdf | §A | p.5] page 5 content")
        chunk_p9 = _json_line("[src: doc.pdf | §A | p.9] page 9 content")
        prompt = f"{chunk_p3}\n{chunk_p5}\n{chunk_p9}"
        answer = "see p.3-5 for the full explanation"
        result = _cited_chunks(answer, prompt)
        contents = "\n".join(result)
        assert "page 3 content" in contents
        assert "page 5 content" in contents
        assert "page 9 content" not in contents


# ---------------------------------------------------------------------------
# _image_candidates
# ---------------------------------------------------------------------------

class TestImageCandidates:
    def _make_image_line(self, img_hash: str, section: str = "BCS", page: int = 5) -> str:
        content = (
            f"[src: doc.pdf | §{section} | p.{page} | image] "
            f"Image Path: /output/images/{img_hash}.jpg"
        )
        return json.dumps({"content": content})

    def test_empty_prompt_returns_empty(self):
        assert _image_candidates("") == []

    def test_non_json_lines_skipped(self):
        assert _image_candidates("plain text\nanother line") == []

    def test_json_without_image_path_skipped(self):
        line = json.dumps({"content": "some text without Image Path"})
        assert _image_candidates(line) == []

    def test_json_with_image_path_but_no_hash_skipped(self):
        line = json.dumps({"content": "Image Path: /output/notahex.jpg"})
        assert _image_candidates(line) == []

    def test_valid_image_chunk_included(self, monkeypatch):
        monkeypatch.setattr(engine, "_resolve_image_path",
                            lambda h, c: f"/fake/images/{h}.jpg")
        line = self._make_image_line("abc123def456")
        result = _image_candidates(line)
        assert len(result) == 1
        assert result[0]["hash"] == "abc123def456"
        assert result[0]["path"] == "/fake/images/abc123def456.jpg"

    def test_image_without_resolvable_path_excluded(self, monkeypatch):
        monkeypatch.setattr(engine, "_resolve_image_path", lambda h, c: None)
        line = self._make_image_line("abc123def456")
        assert _image_candidates(line) == []

    def test_duplicate_hash_deduplicated(self, monkeypatch):
        monkeypatch.setattr(engine, "_resolve_image_path",
                            lambda h, c: f"/fake/{h}.jpg")
        line = self._make_image_line("aabbccdd1122")
        prompt = f"{line}\n{line}"
        result = _image_candidates(prompt)
        assert len(result) == 1

    def test_section_and_page_extracted_from_src_marker(self, monkeypatch):
        monkeypatch.setattr(engine, "_resolve_image_path",
                            lambda h, c: f"/fake/{h}.jpg")
        line = self._make_image_line("aabbccdd1122", section="FACADE", page=17)
        result = _image_candidates(line)
        assert result[0]["section"] == "FACADE"
        assert result[0]["page"] == 17

    def test_marker_field_extracted(self, monkeypatch):
        monkeypatch.setattr(engine, "_resolve_image_path",
                            lambda h, c: f"/fake/{h}.jpg")
        line = self._make_image_line("aabbccdd1122")
        result = _image_candidates(line)
        assert "[src:" in result[0]["marker"]
        assert "image" in result[0]["marker"]

    def test_multiple_distinct_images_all_included(self, monkeypatch):
        monkeypatch.setattr(engine, "_resolve_image_path",
                            lambda h, c: f"/fake/{h}.jpg")
        line1 = self._make_image_line("aabb11223344")
        line2 = self._make_image_line("ccdd55667788")
        result = _image_candidates(f"{line1}\n{line2}")
        assert len(result) == 2

    def test_description_field_also_scanned(self, monkeypatch):
        monkeypatch.setattr(engine, "_resolve_image_path",
                            lambda h, c: f"/fake/{h}.jpg")
        content = "[src: d.pdf | §X | p.1 | image] Image Path: /output/images/aabbccddeeff.jpg"
        line = json.dumps({"description": content})
        result = _image_candidates(line)
        assert len(result) == 1
        assert result[0]["hash"] == "aabbccddeeff"


# ---------------------------------------------------------------------------
# ChatSession.load / save
# ---------------------------------------------------------------------------

class TestChatSessionLoad:
    def test_nonexistent_file_returns_zero(self, tmp_path):
        sess = ChatSession("/wdir", "openai", "oneshot", tmp_path / "missing.json")
        assert sess.load() == 0
        assert sess.history == []

    def test_existing_file_restores_history(self, tmp_path):
        session_file = tmp_path / "test.json"
        data = {
            "history": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
        }
        session_file.write_text(json.dumps(data))
        sess = ChatSession("/wdir", "openai", "oneshot", session_file)
        turns = sess.load()
        assert turns == 2
        assert sess.history[0]["content"] == "Q1"

    def test_empty_history_in_file_returns_zero(self, tmp_path):
        session_file = tmp_path / "empty.json"
        session_file.write_text(json.dumps({"history": []}))
        sess = ChatSession("/wdir", "openai", "oneshot", session_file)
        assert sess.load() == 0

    def test_odd_length_history_counted_as_floor_divide(self, tmp_path):
        session_file = tmp_path / "odd.json"
        data = {"history": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "Q2"},  # unpaired
        ]}
        session_file.write_text(json.dumps(data))
        sess = ChatSession("/wdir", None, "oneshot", session_file)
        turns = sess.load()
        assert turns == 1  # 3 // 2

    def test_missing_history_key_defaults_to_empty(self, tmp_path):
        session_file = tmp_path / "no_history.json"
        session_file.write_text(json.dumps({"provider": "openai"}))
        sess = ChatSession("/wdir", None, "oneshot", session_file)
        turns = sess.load()
        assert turns == 0
        assert sess.history == []


class TestChatSessionSave:
    def test_save_creates_session_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "_SESSIONS_DIR", tmp_path)
        sp = tmp_path / "session.json"
        sess = ChatSession("/wdir", "openai", "oneshot", sp)
        sess.history = [{"role": "user", "content": "hi"}]
        sess.save()
        assert sp.exists()

    def test_saved_data_contains_history(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "_SESSIONS_DIR", tmp_path)
        sp = tmp_path / "session.json"
        sess = ChatSession("/wdir", "openai", "oneshot", sp)
        sess.history = [{"role": "user", "content": "question"}]
        sess.save()
        data = json.loads(sp.read_text())
        assert data["history"][0]["content"] == "question"

    def test_saved_data_contains_provider_and_rerank(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "_SESSIONS_DIR", tmp_path)
        sp = tmp_path / "s.json"
        sess = ChatSession("/wdir", "gemini", "batched", sp)
        sess.save()
        data = json.loads(sp.read_text())
        assert data["provider"] == "gemini"
        assert data["rerank_mode"] == "batched"

    def test_save_is_atomic_tmp_file_removed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "_SESSIONS_DIR", tmp_path)
        sp = tmp_path / "session.json"
        sess = ChatSession("/wdir", "openai", "oneshot", sp)
        sess.save()
        tmp = sp.with_suffix(".json.tmp")
        assert not tmp.exists()  # .tmp file replaced by real file

    def test_save_roundtrip_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "_SESSIONS_DIR", tmp_path)
        sp = tmp_path / "roundtrip.json"
        sess = ChatSession("/wdir", "openai", "oneshot", sp)
        sess.history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        sess.save()
        sess2 = ChatSession("/wdir", "openai", "oneshot", sp)
        turns = sess2.load()
        assert turns == 1
        assert sess2.history == sess.history

    def test_save_overwrites_previous(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "_SESSIONS_DIR", tmp_path)
        sp = tmp_path / "overwrite.json"
        sess = ChatSession("/wdir", "openai", "oneshot", sp)
        sess.history = [{"role": "user", "content": "first"}]
        sess.save()
        sess.history = [{"role": "user", "content": "second"}]
        sess.save()
        data = json.loads(sp.read_text())
        assert data["history"][0]["content"] == "second"

    def test_save_creates_sessions_dir_if_missing(self, tmp_path, monkeypatch):
        new_dir = tmp_path / "new_sessions"
        monkeypatch.setattr(engine, "_SESSIONS_DIR", new_dir)
        sp = new_dir / "s.json"
        sess = ChatSession("/wdir", "openai", "oneshot", sp)
        sess.save()
        assert new_dir.exists()
        assert sp.exists()


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["chat.py"])
        monkeypatch.setattr(cli, "_SESSIONS_DIR", tmp_path)
        wdir, provider, rerank_mode, session_path, agent_mode = _parse_args()
        assert provider is None
        assert rerank_mode == "oneshot"
        assert agent_mode is False
        assert session_path.suffix == ".json"

    def test_provider_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["chat.py", "--provider", "gemini"])
        monkeypatch.setattr(cli, "_SESSIONS_DIR", tmp_path)
        _, provider, _, _, _ = _parse_args()
        assert provider == "gemini"

    def test_rerank_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["chat.py", "--rerank", "batched"])
        monkeypatch.setattr(cli, "_SESSIONS_DIR", tmp_path)
        _, _, rerank_mode, _, _ = _parse_args()
        assert rerank_mode == "batched"

    def test_agent_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["chat.py", "--agent"])
        monkeypatch.setattr(cli, "_SESSIONS_DIR", tmp_path)
        _, _, _, _, agent_mode = _parse_args()
        assert agent_mode is True

    def test_session_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["chat.py", "--session", "mysession"])
        monkeypatch.setattr(cli, "_SESSIONS_DIR", tmp_path)
        _, _, _, session_path, _ = _parse_args()
        assert session_path.stem == "mysession"
        assert session_path.suffix == ".json"

    def test_resume_bare_picks_most_recent(self, monkeypatch, tmp_path):
        # Create two session files with different mtimes
        (tmp_path / "older.json").write_text("{}")
        import time; time.sleep(0.01)
        (tmp_path / "newer.json").write_text("{}")
        monkeypatch.setattr(sys, "argv", ["chat.py", "--resume"])
        monkeypatch.setattr(cli, "_SESSIONS_DIR", tmp_path)
        _, _, _, session_path, _ = _parse_args()
        assert session_path.stem == "newer"

    def test_resume_with_id_uses_named_session(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["chat.py", "--resume", "specific_id"])
        monkeypatch.setattr(cli, "_SESSIONS_DIR", tmp_path)
        _, _, _, session_path, _ = _parse_args()
        assert session_path.stem == "specific_id"

    def test_continue_flag_alias(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["chat.py", "--continue"])
        monkeypatch.setattr(cli, "_SESSIONS_DIR", tmp_path)
        # No sessions exist → fresh session path created
        _, _, _, session_path, _ = _parse_args()
        assert session_path.suffix == ".json"

    def test_dash_c_flag_alias(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["chat.py", "-c"])
        monkeypatch.setattr(cli, "_SESSIONS_DIR", tmp_path)
        _, _, _, session_path, _ = _parse_args()
        assert session_path.suffix == ".json"
