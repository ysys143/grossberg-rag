"""Unit tests for corpus-language detection + query keyword expansion (with edge cases)."""
import json
from pathlib import Path
from unittest.mock import AsyncMock

from grag import expand
from grag.expand import detect_corpus_lang, expand_keywords, _parse, _system_prompt


def _mk_corpus(tmp_path: Path, content: str) -> str:
    (tmp_path / "vdb_entities.json").write_text(json.dumps({"data": [{"content": content}]}))
    return str(tmp_path)


# ---------------------------------------------------------------------------
# detect_corpus_lang — dominant Unicode script
# ---------------------------------------------------------------------------

class TestDetectCorpusLang:
    def setup_method(self):
        expand._lang_cache.clear()  # isolate the per-working_dir cache between tests

    def test_english(self, tmp_path):
        assert detect_corpus_lang(_mk_corpus(tmp_path, "Boundary Cortical Stream FACADE theory")) == "en"

    def test_korean(self, tmp_path):
        assert detect_corpus_lang(_mk_corpus(tmp_path, "초복합세포가 경계를 완성하는 신경 회로")) == "ko"

    def test_japanese_kana_beats_han(self, tmp_path):
        # kana present -> Japanese, even though kanji (Han) also appears.
        assert detect_corpus_lang(_mk_corpus(tmp_path, "境界完成の神経回路を説明する")) == "ja"

    def test_chinese_han_no_kana(self, tmp_path):
        assert detect_corpus_lang(_mk_corpus(tmp_path, "边界完成 神经回路 视觉 感知")) == "zh"

    def test_korean_dominant_with_english_jargon(self, tmp_path):
        # mixed but Hangul-dominant -> ko (English jargon present but minority).
        c = "초복합세포는 boundary completion 을 수행하는 신경 세포이며 경계 완성에 핵심이다"
        assert detect_corpus_lang(_mk_corpus(tmp_path, c)) == "ko"

    def test_missing_file_falls_back_to_en(self, tmp_path):
        assert detect_corpus_lang(str(tmp_path)) == "en"  # no vdb_entities.json

    def test_empty_data_falls_back_to_en(self, tmp_path):
        (tmp_path / "vdb_entities.json").write_text(json.dumps({"data": []}))
        assert detect_corpus_lang(str(tmp_path)) == "en"

    def test_result_is_cached(self, tmp_path):
        wd = _mk_corpus(tmp_path, "초복합세포 경계 완성")
        assert detect_corpus_lang(wd) == "ko"
        (tmp_path / "vdb_entities.json").unlink()  # delete; cache should still answer
        assert detect_corpus_lang(wd) == "ko"


# ---------------------------------------------------------------------------
# _parse — defensive JSON extraction
# ---------------------------------------------------------------------------

class TestParse:
    def test_well_formed(self):
        out = _parse('{"concepts": ["boundary completion"], "entities": ["FACADE", "BCS"]}')
        assert out == {"concepts": ["boundary completion"], "entities": ["FACADE", "BCS"]}

    def test_dedup_strip_and_drop_empty(self):
        out = _parse('{"concepts": [" x ", "x", ""], "entities": ["A", "A", "  "]}')
        assert out == {"concepts": ["x"], "entities": ["A"]}

    def test_caps_at_eight(self):
        out = _parse(json.dumps({"concepts": [str(i) for i in range(20)], "entities": []}))
        assert len(out["concepts"]) == 8

    def test_missing_keys_default_empty(self):
        assert _parse('{"concepts": ["a"]}') == {"concepts": ["a"], "entities": []}

    def test_non_string_items_coerced(self):
        out = _parse('{"concepts": [4.25, 7], "entities": []}')
        assert out["concepts"] == ["4.25", "7"]

    def test_surrounding_prose_and_fences_tolerated(self):
        out = _parse('here you go:\n```json\n{"concepts": ["a"], "entities": ["B"]}\n```')
        assert out == {"concepts": ["a"], "entities": ["B"]}

    def test_no_json_returns_empty(self):
        assert _parse("no json here at all") == {"concepts": [], "entities": []}

    def test_null_lists_treated_as_empty(self):
        assert _parse('{"concepts": null, "entities": null}') == {"concepts": [], "entities": []}


# ---------------------------------------------------------------------------
# _system_prompt — language threading + verbatim instruction
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_known_language_named(self):
        assert "Korean" in _system_prompt("ko")
        assert "English" in _system_prompt("en")

    def test_unknown_language_defaults_to_english(self):
        assert "English" in _system_prompt("xx")

    def test_mentions_verbatim_preservation(self):
        p = _system_prompt("ko")
        assert "VERBATIM" in p and "FACADE" in p


# ---------------------------------------------------------------------------
# expand_keywords — async, with mocked llm.generate
# ---------------------------------------------------------------------------

class TestExpandKeywords:
    async def test_parses_model_json(self, monkeypatch):
        monkeypatch.setattr(
            expand.llm, "generate",
            AsyncMock(return_value='{"concepts": ["boundary completion"], "entities": ["FACADE"]}'))
        out = await expand_keywords("초복합세포 회로", "en")
        assert out == {"concepts": ["boundary completion"], "entities": ["FACADE"]}

    async def test_fail_open_on_exception(self, monkeypatch):
        monkeypatch.setattr(expand.llm, "generate", AsyncMock(side_effect=RuntimeError("quota")))
        assert await expand_keywords("q", "en") == {"concepts": [], "entities": []}

    async def test_fail_open_on_junk(self, monkeypatch):
        monkeypatch.setattr(expand.llm, "generate", AsyncMock(return_value="totally not json"))
        assert await expand_keywords("q", "en") == {"concepts": [], "entities": []}

    async def test_fail_open_on_malformed_braces(self, monkeypatch):
        # JSON-looking but invalid -> json.loads raises inside _parse -> caught -> empty.
        monkeypatch.setattr(expand.llm, "generate", AsyncMock(return_value="{concepts: oops}"))
        assert await expand_keywords("q", "en") == {"concepts": [], "entities": []}

    async def test_verbatim_acronym_preserved(self, monkeypatch):
        monkeypatch.setattr(
            expand.llm, "generate",
            AsyncMock(return_value='{"concepts": [], "entities": ["FACADE", "Figure 4.25"]}'))
        out = await expand_keywords("파사드 이론의 그림 4.25", "en")
        assert "FACADE" in out["entities"] and "Figure 4.25" in out["entities"]

    async def test_target_language_threaded_into_system_prompt(self, monkeypatch):
        gen = AsyncMock(return_value='{"concepts": [], "entities": []}')
        monkeypatch.setattr(expand.llm, "generate", gen)
        await expand_keywords("질문", "ko")
        assert "Korean" in gen.call_args.kwargs["system_prompt"]

    async def test_history_included_in_prompt(self, monkeypatch):
        gen = AsyncMock(return_value='{"concepts": [], "entities": []}')
        monkeypatch.setattr(expand.llm, "generate", gen)
        hist = [{"role": "user", "content": "BCS란?"}, {"role": "assistant", "content": "경계 윤곽 시스템"}]
        await expand_keywords("그게 뭐야", "en", history=hist)
        assert "BCS란?" in gen.call_args.kwargs["prompt"]

    async def test_empty_question_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(expand.llm, "generate", AsyncMock(return_value='{"concepts": [], "entities": []}'))
        assert await expand_keywords("", "en") == {"concepts": [], "entities": []}
