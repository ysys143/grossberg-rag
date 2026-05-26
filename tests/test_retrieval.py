"""Unit tests for the shared rag factory (build_rag) — the single construction point
that keeps the engine and KBTool retrieval paths from drifting."""
from unittest.mock import Mock

from grag import retrieval


class _FakeRAG:
    def __init__(self, **kw):
        self.kw = kw

    async def initialize_storages(self):
        self._initialized = True


class TestRerankFuncs:
    def test_modes(self):
        assert set(retrieval.RERANK_FUNCS) == {"none", "oneshot", "batched"}

    def test_none_disables_rerank(self):
        assert retrieval.RERANK_FUNCS["none"] is None


class TestBuildRag:
    def _patch(self, monkeypatch, hybrid: bool):
        monkeypatch.setattr(retrieval, "LightRAG", _FakeRAG)
        monkeypatch.setitem(retrieval._cfg["query"], "hybrid_seed", hybrid)

    async def test_constructs_and_initializes(self, monkeypatch):
        self._patch(monkeypatch, hybrid=False)
        rag = await retrieval.build_rag("/wd", "oneshot")
        assert isinstance(rag, _FakeRAG)
        assert rag._initialized is True
        assert rag.kw["working_dir"] == "/wd"

    async def test_rerank_mode_wired(self, monkeypatch):
        self._patch(monkeypatch, hybrid=False)
        assert (await retrieval.build_rag("/wd", "none")).kw["rerank_model_func"] is None
        assert (await retrieval.build_rag("/wd", "batched")).kw["rerank_model_func"] \
            is retrieval.RERANK_FUNCS["batched"]

    async def test_no_attach_when_flag_off(self, monkeypatch):
        self._patch(monkeypatch, hybrid=False)
        spy = Mock()
        monkeypatch.setattr(retrieval.hybrid_seed, "attach_hybrid_seed", spy)
        await retrieval.build_rag("/wd", "oneshot")
        spy.assert_not_called()

    async def test_attaches_when_flag_on(self, monkeypatch):
        self._patch(monkeypatch, hybrid=True)
        monkeypatch.setitem(retrieval._cfg["query"], "hybrid_seed_top_k", 7)
        spy = Mock()
        monkeypatch.setattr(retrieval.hybrid_seed, "attach_hybrid_seed", spy)
        rag = await retrieval.build_rag("/wd", "oneshot")
        spy.assert_called_once()
        args, kwargs = spy.call_args
        assert args[0] is rag and args[1] == "/wd"  # rag + working_dir
        assert kwargs.get("top_k") == 7

    async def test_attach_failure_is_fail_open(self, monkeypatch):
        self._patch(monkeypatch, hybrid=True)
        monkeypatch.setattr(retrieval.hybrid_seed, "attach_hybrid_seed",
                            Mock(side_effect=RuntimeError("mecab missing")))
        rag = await retrieval.build_rag("/wd", "oneshot")  # must not raise
        assert isinstance(rag, _FakeRAG)
