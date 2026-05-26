"""Unit tests for kb_tool.py — KBTool with mocked LightRAG."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grag import kb_tool
from grag.kb_tool import KBTool


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.aquery = AsyncMock(return_value="retrieved context with [src: doc.pdf | §BCS | p.5]")
    return rag


@pytest.fixture
def patched_tool(monkeypatch, mock_rag):
    """KBTool with the shared rag factory (retrieval.build_rag) patched out."""
    monkeypatch.setattr(kb_tool.retrieval, "build_rag", AsyncMock(return_value=mock_rag))
    return KBTool("/fake/working/dir"), mock_rag


class TestKBToolInit:
    def test_working_dir_stored(self):
        tool = KBTool("/some/dir")
        assert tool._wdir == "/some/dir"

    def test_rag_initially_none(self):
        tool = KBTool("/some/dir")
        assert tool._rag is None

    def test_default_working_dir_from_config(self, monkeypatch):
        # When no dir supplied, falls back to config
        tool = KBTool()
        assert tool._wdir  # non-empty


class TestKBToolGetRag:
    # NOTE: LightRAG construction + initialize_storages + hybrid_seed are now the shared
    # factory's job (tested in test_retrieval.py). KBTool only delegates to it and caches.
    async def test_first_call_builds_rag(self, monkeypatch, mock_rag):
        build = AsyncMock(return_value=mock_rag)
        monkeypatch.setattr(kb_tool.retrieval, "build_rag", build)
        tool = KBTool("/fake/dir")
        rag = await tool._get_rag()
        assert rag is mock_rag
        build.assert_called_once()

    async def test_second_call_reuses_same_instance(self, monkeypatch, mock_rag):
        build = AsyncMock(return_value=mock_rag)
        monkeypatch.setattr(kb_tool.retrieval, "build_rag", build)
        tool = KBTool("/fake/dir")
        rag1 = await tool._get_rag()
        rag2 = await tool._get_rag()
        assert rag1 is rag2
        build.assert_called_once()  # built once despite two _get_rag calls

    async def test_build_rag_called_with_working_dir(self, monkeypatch, mock_rag):
        build = AsyncMock(return_value=mock_rag)
        monkeypatch.setattr(kb_tool.retrieval, "build_rag", build)
        tool = KBTool("/my/working/dir")
        await tool._get_rag()
        assert build.call_args[0][0] == "/my/working/dir"


class TestKBToolSearch:
    async def test_search_passes_query_to_aquery(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.search("BCS란 무엇인가?")
        call_args = mock_rag.aquery.call_args
        assert call_args[0][0] == "BCS란 무엇인가?"

    async def test_only_need_context_true_when_not_return_answer(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.search("q", return_answer=False)
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.only_need_context is True

    async def test_only_need_context_false_when_return_answer(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.search("q", return_answer=True)
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.only_need_context is False

    async def test_mode_hybrid_default(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.search("q")
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.mode == "hybrid"

    async def test_mode_local_passed_through(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.search("q", mode="local")
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.mode == "local"

    async def test_concepts_mapped_to_hl_keywords(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.search("q", concepts=["boundary completion", "filling-in"])
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.hl_keywords == ["boundary completion", "filling-in"]

    async def test_entities_mapped_to_ll_keywords(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.search("q", entities=["BCS", "FACADE"])
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.ll_keywords == ["BCS", "FACADE"]

    async def test_none_concepts_yields_empty_hl_keywords(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.search("q", concepts=None)
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.hl_keywords == []

    async def test_conversation_history_forwarded(self, patched_tool):
        tool, mock_rag = patched_tool
        history = [{"role": "user", "content": "이전 질문"}]
        await tool.search("q", conversation_history=history)
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.conversation_history == history

    async def test_empty_result_returns_no_results_string(self, patched_tool):
        tool, mock_rag = patched_tool
        mock_rag.aquery.return_value = ""
        result = await tool.search("q")
        assert result == "(no results)"

    async def test_none_result_returns_no_results_string(self, patched_tool):
        tool, mock_rag = patched_tool
        mock_rag.aquery.return_value = None
        result = await tool.search("q")
        assert result == "(no results)"

    async def test_rerank_enabled(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.search("q")
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.enable_rerank is True


class TestKBToolCall:
    async def test_call_dispatches_query(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.call({"query": "FACADE 이론?"})
        assert mock_rag.aquery.called
        assert mock_rag.aquery.call_args[0][0] == "FACADE 이론?"

    async def test_call_defaults_mode_hybrid(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.call({"query": "q"})
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.mode == "hybrid"

    async def test_call_passes_mode_from_args(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.call({"query": "q", "mode": "global"})
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.mode == "global"

    async def test_call_return_answer_false_by_default(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.call({"query": "q"})
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.only_need_context is True

    async def test_call_passes_concepts_and_entities(self, patched_tool):
        tool, mock_rag = patched_tool
        await tool.call({
            "query": "q",
            "concepts": ["surface filling-in"],
            "entities": ["FCS"],
        })
        param = mock_rag.aquery.call_args[1]["param"]
        assert param.hl_keywords == ["surface filling-in"]
        assert param.ll_keywords == ["FCS"]

    async def test_call_missing_optional_keys_no_error(self, patched_tool):
        tool, mock_rag = patched_tool
        # Only required key is "query"
        result = await tool.call({"query": "minimal"})
        assert result  # non-empty


class TestKBToolClose:
    async def test_close_when_rag_none_is_no_op(self):
        tool = KBTool("/fake")
        await tool.close()  # should not raise

    async def test_close_calls_finalize_storages(self, monkeypatch, mock_rag):
        monkeypatch.setattr(kb_tool.retrieval, "build_rag", AsyncMock(return_value=mock_rag))
        tool = KBTool("/fake")
        await tool._get_rag()          # triggers rag construction
        await tool.close()
        mock_rag.finalize_storages.assert_called_once()

    async def test_close_sets_rag_to_none(self, monkeypatch, mock_rag):
        monkeypatch.setattr(kb_tool.retrieval, "build_rag", AsyncMock(return_value=mock_rag))
        tool = KBTool("/fake")
        await tool._get_rag()
        await tool.close()
        assert tool._rag is None

    async def test_double_close_is_safe(self, monkeypatch, mock_rag):
        monkeypatch.setattr(kb_tool.retrieval, "build_rag", AsyncMock(return_value=mock_rag))
        tool = KBTool("/fake")
        await tool._get_rag()
        await tool.close()
        await tool.close()  # second close with rag=None — must not raise
        # finalize_storages called only once (first close)
        mock_rag.finalize_storages.assert_called_once()
