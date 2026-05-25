"""
Perverse edge-case tests — inputs that probe boundaries, silent coercions,
and failure modes the happy path never hits.  Every test documents a specific
surprising or dangerous behavior.
"""
import json
import pytest
from unittest.mock import AsyncMock


# ===========================================================================
# router._parse — surprising type coercions and JSON matching edge cases
# ===========================================================================

class TestRouterParseEvil:
    def test_in_scope_null_coerced_to_false(self):
        # bool(None) == False — model returning null means out-of-scope
        import router
        r = router._parse('{"in_scope": null}')
        assert r["in_scope"] is False

    def test_needs_retrieval_zero_coerced_to_false(self):
        # bool(0) == False — integer 0 skips retrieval silently
        import router
        r = router._parse('{"needs_retrieval": 0}')
        assert r["needs_retrieval"] is False

    def test_in_scope_string_false_is_truthy(self):
        # DANGER: bool("false") == True — string "false" != JSON false.
        # Documents that the router does NOT protect against this model error.
        import router
        r = router._parse('{"in_scope": "false"}')
        assert r["in_scope"] is True   # non-empty string is truthy

    def test_effort_empty_string_defaults_to_medium(self):
        import router
        assert router._parse('{"effort": ""}')["effort"] == "medium"

    def test_effort_unicode_garbage_defaults_to_medium(self):
        import router
        assert router._parse('{"effort": "높음"}')["effort"] == "medium"

    def test_two_json_objects_greedy_match_raises_json_decode_error(self):
        # re.search(r"\{.*\}", ..., re.DOTALL) is greedy — with two objects
        # it matches from first { to last }, producing invalid JSON which
        # raises JSONDecodeError from _parse. route() catches this and fails open.
        import router
        raw = '{"effort": "low"} {"effort": "high"}'
        with pytest.raises(json.JSONDecodeError):
            router._parse(raw)

    def test_nested_effort_at_wrong_depth_returns_medium(self):
        # Model wraps response — outer object has no "effort" key
        import router
        r = router._parse('{"result": {"effort": "high"}}')
        assert r["effort"] == "medium"

    async def test_route_catches_parse_json_error_and_fails_open(self, monkeypatch):
        # JSONDecodeError from _parse must be caught by route() and return fail-open.
        import router, llm
        monkeypatch.setattr(llm, "generate",
                            AsyncMock(return_value='{"a": 1} {"b": 2}'))
        result = await router.route("q")
        assert result["in_scope"] is True
        assert "router-failopen" in result["reason"]

    def test_clarification_key_absent_with_flag_true_drops_flag(self):
        import router
        r = router._parse('{"needs_clarification": true}')
        assert r["needs_clarification"] is False

    def test_very_long_effort_value_defaults_to_medium(self):
        import router
        raw = '{"effort": "' + "x" * 10000 + '"}'
        assert router._parse(raw)["effort"] == "medium"


# ===========================================================================
# rerank._strip_fences — fence parsing edge cases
# ===========================================================================

class TestStripFencesEvil:
    def test_no_closing_fence_still_extracts_content(self):
        # split("```") on "```json\n[...]" yields ["", "json\n[...]"]
        # inner[4:] strips "json" — content returned without closing fence.
        import rerank
        s = '```json\n[{"index": 0, "score": 80}]'
        result = rerank._strip_fences(s)
        assert "[" in result

    def test_four_backticks_inner_starts_with_backtick_not_json(self):
        # "````json\n..." split on "```" -> inner starts with "`json", not "json"
        # so the 4-char strip is skipped; backtick stays in output.
        import rerank
        s = '````json\n[{"index": 0}]\n````'
        result = rerank._strip_fences(s)
        assert result  # no crash; content is returned in some form

    def test_fence_with_only_whitespace_inside_returns_empty(self):
        import rerank
        s = '```json\n   \n```'
        result = rerank._strip_fences(s)
        assert result == ""


# ===========================================================================
# rerank.rerank — boundary values and silent type coercions
# ===========================================================================

class TestRerankEvil:
    async def test_top_n_zero_treated_same_as_none_returns_all(self, monkeypatch):
        # Code: `result = out[:top_n] if top_n else out` — if top_n=0, bool(0)
        # is False, so the else branch fires and ALL results are returned.
        # Passing top_n=0 to mean "empty" is a footgun; it acts like top_n=None.
        import rerank, llm
        monkeypatch.setattr(llm, "generate",
                            AsyncMock(return_value='[{"index": 0, "score": 80}]'))
        result = await rerank.rerank("q", ["doc"], top_n=0)
        assert len(result) == 1  # NOT empty — 0 is falsy, all docs returned

    async def test_top_n_larger_than_doc_count_returns_all(self, monkeypatch):
        import rerank, llm
        monkeypatch.setattr(llm, "generate", AsyncMock(
            return_value='[{"index": 0, "score": 80},{"index": 1, "score": 60}]'
        ))
        result = await rerank.rerank("q", ["a", "b"], top_n=999)
        assert len(result) == 2

    async def test_score_as_string_converted_via_float(self, monkeypatch):
        # float("80") / 100 = 0.8 — model emitting quoted numbers works silently.
        import rerank, llm
        monkeypatch.setattr(llm, "generate",
                            AsyncMock(return_value='[{"index": 0, "score": "80"}]'))
        result = await rerank.rerank("q", ["a"])
        assert result[0]["relevance_score"] == pytest.approx(0.80)

    async def test_null_score_raises_type_error(self, monkeypatch):
        # dict.get("score", 0) returns None when key is present with JSON null.
        # (The default 0 is only used when the key is ABSENT.)
        # float(None) -> TypeError. Documents an unguarded rough edge.
        import rerank, llm
        monkeypatch.setattr(llm, "generate",
                            AsyncMock(return_value='[{"index": 0, "score": null}]'))
        with pytest.raises(TypeError):
            await rerank.rerank("q", ["a"])

    async def test_all_docs_same_score_all_returned(self, monkeypatch):
        import rerank, llm
        monkeypatch.setattr(llm, "generate", AsyncMock(
            return_value='[{"index":0,"score":50},{"index":1,"score":50},{"index":2,"score":50}]'
        ))
        result = await rerank.rerank("q", ["a", "b", "c"])
        assert len(result) == 3

    async def test_rerank_batched_batch_size_one_correct_global_indices(self, monkeypatch):
        # batch_size=1: each batch is a single doc; keep=max(1,1//2)=1 -> all survive.
        # Final stage reranks survivors; returned indices must map to original positions.
        import rerank, llm
        async def capture(model, prompt, **kw):
            n = prompt.count("\n[")
            return json.dumps([{"index": i, "score": 90 - i * 10} for i in range(n)])
        monkeypatch.setattr(llm, "generate", capture)
        docs = ["doc0", "doc1", "doc2"]
        result = await rerank.rerank_batched("q", docs, batch_size=1, top_n=2)
        assert all(0 <= r["index"] < len(docs) for r in result)
        assert len(result) <= 2


# ===========================================================================
# image_gate._parse — non-list and type-coercion surprises
# ===========================================================================

class TestImageGateParseEvil:
    def test_relevant_as_bool_raises_type_error(self):
        # {"relevant": true} -> bool is not iterable ->
        # TypeError propagates from _parse (caught upstream by select_relevant_images).
        import image_gate
        with pytest.raises(TypeError):
            image_gate._parse('{"relevant": true}', n=5)

    async def test_select_relevant_images_bool_relevant_fails_closed(self, monkeypatch):
        # Even though _parse raises TypeError, select_relevant_images catches
        # all exceptions and returns [] (fail-closed contract preserved).
        import image_gate, llm
        monkeypatch.setattr(llm, "generate",
                            AsyncMock(return_value='{"relevant": true}'))
        result = await image_gate.select_relevant_images(
            "q", [{"hash": "h0", "caption": "c", "section": "A", "page": 1}]
        )
        assert result == []

    def test_relevant_as_object_iterates_over_keys(self):
        # {"relevant": {"0": "yes"}} -> dict iteration yields key "0" -> int("0")=0
        # Silently "works" and returns [0] if in range. Surprising but harmless.
        import image_gate
        result = image_gate._parse('{"relevant": {"0": "yes"}}', n=3)
        assert result == [0]

    def test_mixed_types_in_relevant_list(self):
        # [1, "two", 3.7, null, -1, 99] for n=5:
        # 1 -> int(1)=1 [OK]; "two" -> ValueError (skipped); 3.7 -> int(3.7)=3 [OK];
        # null -> TypeError (skipped); -1 -> out-of-range; 99 -> out-of-range
        import image_gate
        result = image_gate._parse('{"relevant": [1, "two", 3.7, null, -1, 99]}', n=5)
        assert 1 in result
        assert 3 in result
        assert -1 not in result
        assert 99 not in result

    def test_relevant_list_with_only_invalid_items_returns_empty(self):
        import image_gate
        result = image_gate._parse('{"relevant": ["a", "b", "c"]}', n=5)
        assert result == []

    def test_n_one_only_index_zero_valid(self):
        import image_gate
        result = image_gate._parse('{"relevant": [0, 1, 2]}', n=1)
        assert result == [0]


# ===========================================================================
# _extract_sources — regex boundary cases
# ===========================================================================

class TestExtractSourcesEvil:
    def test_closing_bracket_in_doc_name_truncates_marker(self):
        # [src: doc[1].pdf | §A | p.5] — [^\]]+ stops at the ] in "doc[1]"
        # Result: "[src: doc[1]" is captured (incomplete), not the full marker.
        from chat import _extract_sources
        prompt = "[src: doc[1].pdf | §A | p.5]"
        result = _extract_sources(prompt)
        assert len(result) == 1
        assert result[0] == "[src: doc[1]"

    def test_1000_identical_markers_deduplicated_to_one(self):
        from chat import _extract_sources
        marker = "[src: doc.pdf | §BCS | p.5]"
        result = _extract_sources(" ".join([marker] * 1000))
        assert result == [marker]

    def test_unicode_section_name_captured(self):
        from chat import _extract_sources
        prompt = "[src: doc.pdf | §경계완성 | p.3]"
        result = _extract_sources(prompt)
        assert len(result) == 1
        assert "경계완성" in result[0]

    def test_adjacent_markers_no_separator_both_captured(self):
        from chat import _extract_sources
        prompt = "[src: a.pdf | §X | p.1][src: b.pdf | §Y | p.2]"
        result = _extract_sources(prompt)
        assert len(result) == 2


# ===========================================================================
# _cited_pages — pattern matching surprises
# ===========================================================================

class TestCitedPagesEvil:
    def test_reversed_range_only_first_number_matched_by_standalone(self):
        # "p.7-3": range(7, 4) is empty (nothing added from range pattern).
        # Standalone pattern p\.\s*(\d+) requires "p." prefix — it matches "p.7"
        # (capturing 7) but NOT the trailing "3" after the dash (no "p." there).
        # So only 7 ends up in the set.
        from chat import _cited_pages
        pages = _cited_pages("p.7-3")
        assert 7 in pages
        assert 3 not in pages  # "3" has no p. prefix -> standalone pattern misses it

    def test_pp_5_matches_embedded_p_5(self):
        # re.findall searches anywhere in the string -> finds "p.5" inside "pp.5"
        from chat import _cited_pages
        assert 5 in _cited_pages("see pp.5 for details")

    def test_period_at_end_of_sentence_no_digits_no_match(self):
        from chat import _cited_pages
        assert _cited_pages("end of paragraph p.") == set()

    def test_very_large_page_number(self):
        from chat import _cited_pages
        assert 9999 in _cited_pages("p.9999")

    def test_page_reference_inside_url_still_matched(self):
        # The regex is not context-aware: "example.com/p.5" triggers a match.
        # Documents a known limitation (not a fixable false positive without context).
        from chat import _cited_pages
        assert 5 in _cited_pages("http://example.com/p.5")

    def test_range_with_spaces_around_dash(self):
        from chat import _cited_pages
        assert _cited_pages("p.3 - 7") == {3, 4, 5, 6, 7}

    def test_zero_padded_page_number_parsed_correctly(self):
        # int("05") == 5
        from chat import _cited_pages
        assert 5 in _cited_pages("p.05")


# ===========================================================================
# _cited_chunks — content field type surprises
# ===========================================================================

class TestCitedChunksEvil:
    def test_content_as_integer_raises_type_error(self):
        # {"content": 42} -> text = 42 -> re.search(pattern, 42) -> TypeError.
        # Documents unguarded non-string content in retrieved JSON objects.
        from chat import _cited_chunks
        line = json.dumps({"content": 42})
        with pytest.raises(TypeError):
            _cited_chunks("p.5 cited", line)

    def test_content_wins_over_description_when_both_present(self):
        # obj.get("content") or obj.get("description") — content wins if truthy.
        from chat import _cited_chunks
        line = json.dumps({
            "content": "[src: a.pdf | §A | p.5] content text",
            "description": "[src: b.pdf | §B | p.5] description text",
        })
        result = _cited_chunks("p.5 cited", line)
        assert len(result) == 1
        assert "content text" in result[0]

    def test_empty_content_string_not_included(self):
        from chat import _cited_chunks
        line = json.dumps({"content": "", "description": ""})
        assert _cited_chunks("p.5 cited", line) == []

    def test_chunk_with_no_src_marker_skipped(self):
        from chat import _cited_chunks
        line = json.dumps({"content": "plain text on page 5 with no marker"})
        assert _cited_chunks("p.5 cited", line) == []

    def test_src_marker_without_page_skipped(self):
        # [src: doc.pdf | §BCS] with no p.N — re.search for p.\d+ fails
        from chat import _cited_chunks
        line = json.dumps({"content": "[src: doc.pdf | §BCS] no page number here"})
        assert _cited_chunks("p.5 cited", line) == []


# ===========================================================================
# _image_candidates — regex and field coercion edge cases
# ===========================================================================

class TestImageCandidatesEvil:
    def test_uppercase_hex_hash_not_matched_by_regex(self, monkeypatch):
        # r"images/([a-f0-9]+)" only matches lowercase hex.
        # Uppercase hashes from some imaging tools are silently skipped.
        import chat
        monkeypatch.setattr(chat, "_resolve_image_path",
                            lambda h, c: f"/fake/{h}.jpg")
        content = "[src: d.pdf | §X | p.1 | image] Image Path: /output/images/ABCDEF123456.jpg"
        result = chat._image_candidates(json.dumps({"content": content}))
        assert result == []

    def test_null_content_field_skipped_safely(self, monkeypatch):
        import chat
        monkeypatch.setattr(chat, "_resolve_image_path", lambda h, c: "/fake/h.jpg")
        result = chat._image_candidates(json.dumps({"content": None}))
        assert result == []

    def test_image_path_key_lowercase_not_matched(self, monkeypatch):
        # "image path" vs "Image Path" — case-sensitive check; lowercase skipped.
        import chat
        monkeypatch.setattr(chat, "_resolve_image_path", lambda h, c: "/fake/h.jpg")
        content = "image path: /output/images/aabbccdd1122.jpg"
        result = chat._image_candidates(json.dumps({"content": content}))
        assert result == []

    def test_hash_with_non_hex_chars_not_matched(self, monkeypatch):
        import chat
        monkeypatch.setattr(chat, "_resolve_image_path", lambda h, c: "/fake/h.jpg")
        content = "Image Path: /output/images/ghijklmn1234.jpg"  # g-n not in [a-f]
        result = chat._image_candidates(json.dumps({"content": content}))
        assert result == []

    def test_mixed_case_image_path_keyword_not_matched(self, monkeypatch):
        import chat
        monkeypatch.setattr(chat, "_resolve_image_path", lambda h, c: "/fake/h.jpg")
        content = "IMAGE PATH: /output/images/aabbccdd1122.jpg"
        result = chat._image_candidates(json.dumps({"content": content}))
        assert result == []


# ===========================================================================
# ChatSession — load with corrupt / unexpected data
# ===========================================================================

class TestChatSessionEvil:
    def test_load_corrupted_json_raises_decode_error(self, tmp_path):
        import chat
        sp = tmp_path / "corrupt.json"
        sp.write_text("{this is not json")
        sess = chat.ChatSession("/wdir", "openai", "oneshot", sp)
        with pytest.raises(json.JSONDecodeError):
            sess.load()

    def test_load_empty_file_raises_decode_error(self, tmp_path):
        import chat
        sp = tmp_path / "empty.json"
        sp.write_text("")
        sess = chat.ChatSession("/wdir", "openai", "oneshot", sp)
        with pytest.raises(json.JSONDecodeError):
            sess.load()

    def test_load_null_history_field_causes_type_error_on_len(self, tmp_path):
        # {"history": null}: data.get("history", []) returns None (key exists).
        # self.history = None; len(None)//2 -> TypeError.
        # Documents that explicit null history is not guarded in load().
        import chat
        sp = tmp_path / "null_history.json"
        sp.write_text(json.dumps({"history": None}))
        sess = chat.ChatSession("/wdir", "openai", "oneshot", sp)
        with pytest.raises(TypeError):
            sess.load()

    def test_none_provider_coerced_to_default_at_init(self, tmp_path, monkeypatch):
        # __init__ does: self.provider = provider or ANSWER_PROVIDER_DEFAULT
        # None is falsy -> default provider is substituted immediately.
        # Callers that pass None expecting null in the saved file will be surprised.
        import chat
        monkeypatch.setattr(chat, "_SESSIONS_DIR", tmp_path)
        sp = tmp_path / "none_provider.json"
        sess = chat.ChatSession("/wdir", None, "oneshot", sp)
        assert sess.provider != None   # None was coerced to the default string
        sess.save()
        data = json.loads(sp.read_text())
        assert isinstance(data["provider"], str)  # saved as the default, not null

    def test_load_history_with_non_dict_items_stored_as_is(self, tmp_path):
        # No type validation on load — arbitrary list items are stored.
        import chat
        sp = tmp_path / "weird_history.json"
        sp.write_text(json.dumps({"history": ["string", 42, None]}))
        sess = chat.ChatSession("/wdir", "openai", "oneshot", sp)
        turns = sess.load()
        assert turns == 1   # 3 // 2
        assert sess.history == ["string", 42, None]


# ===========================================================================
# _parse_args — flag interaction edge cases
# ===========================================================================

class TestParseArgsEvil:
    def test_invalid_rerank_value_passes_through_unvalidated(self, monkeypatch, tmp_path):
        # _parse_args does not validate rerank_mode — callers hit KeyError later.
        import sys, chat
        monkeypatch.setattr(sys, "argv", ["chat.py", "--rerank", "superfast"])
        monkeypatch.setattr(chat, "_SESSIONS_DIR", tmp_path)
        _, _, rerank_mode, _, _ = chat._parse_args()
        assert rerank_mode == "superfast"

    def test_agent_flag_combined_with_other_flags_all_parsed(self, monkeypatch, tmp_path):
        import sys, chat
        monkeypatch.setattr(sys, "argv",
                            ["chat.py", "--agent", "--provider", "gemini", "--rerank", "none"])
        monkeypatch.setattr(chat, "_SESSIONS_DIR", tmp_path)
        _, provider, rerank_mode, _, agent_mode = chat._parse_args()
        assert agent_mode is True
        assert provider == "gemini"
        assert rerank_mode == "none"

    def test_resume_followed_by_dash_flag_is_bare_resume(self, monkeypatch, tmp_path):
        # --resume --provider: next arg starts with "-" -> resume=True (bare), no ID.
        import sys, chat
        (tmp_path / "existing.json").write_text("{}")
        monkeypatch.setattr(sys, "argv", ["chat.py", "--resume", "--provider", "openai"])
        monkeypatch.setattr(chat, "_SESSIONS_DIR", tmp_path)
        _, _, _, session_path, _ = chat._parse_args()
        assert session_path.stem == "existing"

    def test_pdf_flag_with_deep_path_uses_stem_only(self, monkeypatch, tmp_path):
        import sys, chat
        monkeypatch.setattr(sys, "argv", ["chat.py", "--pdf", "/deep/path/grossberg_ch4.pdf"])
        monkeypatch.setattr(chat, "_SESSIONS_DIR", tmp_path)
        wdir, _, _, _, _ = chat._parse_args()
        assert wdir.endswith("grossberg_ch4")
        assert "/deep/path" not in wdir

    def test_session_flag_overrides_resume(self, monkeypatch, tmp_path):
        # --session takes precedence; --resume is ignored when --session is present
        # because the code checks session_name first.
        import sys, chat
        (tmp_path / "other.json").write_text("{}")
        monkeypatch.setattr(sys, "argv",
                            ["chat.py", "--session", "named", "--resume"])
        monkeypatch.setattr(chat, "_SESSIONS_DIR", tmp_path)
        _, _, _, session_path, _ = chat._parse_args()
        assert session_path.stem == "named"
