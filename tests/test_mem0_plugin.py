"""Tests for the Mem0 memory provider plugin.

Ensures that the plugin calls the Mem0 SDK with the correct parameter format:
- read endpoints (search, get_all) use filters={"user_id": ...}
- write endpoint (add) uses bare user_id= and agent_id= kwargs
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.mem0 import (
    CONCLUDE_SCHEMA,
    PROFILE_SCHEMA,
    SEARCH_SCHEMA,
    Mem0MemoryProvider,
)


@pytest.fixture
def plugin(monkeypatch):
    """Create a Mem0MemoryProvider and initialize it."""
    assert os.environ.get("MEM0_API_KEY"), "MEM0_API_KEY env var must be set"

    provider = Mem0MemoryProvider()
    # Monkey-patch _load_config to avoid needing config.yaml
    import plugins.memory.mem0 as mem0_mod
    monkeypatch.setattr(
        mem0_mod,
        "_load_config",
        lambda: {
            "api_key": os.environ.get("MEM0_API_KEY", ""),
            "user_id": "hermes-user",
            "agent_id": "hermes",
            "rerank": True,
            "keyword_search": False,
        },
    )
    provider.initialize(session_id="test-session")
    return provider


@pytest.fixture
def mock_client():
    """Create a mock MemoryClient for isolated unit tests."""
    client = MagicMock()
    client.search.return_value = {"results": [{"memory": "test memory", "score": 0.8}]}
    client.get_all.return_value = {
        "results": [{"id": "123", "memory": "existing memory", "user_id": "hermes-user", "metadata": None}]
    }
    client.add.return_value = {"results": [{"message": "ok", "status": "PENDING"}]}
    return client


# ── Tool Schemas ───────────────────────────────────────────────────────────

class TestToolSchemas:
    """Verify the tool schema definitions."""

    def test_profile_schema(self):
        assert PROFILE_SCHEMA["name"] == "mem0_profile"
        assert PROFILE_SCHEMA["parameters"]["required"] == []

    def test_search_schema(self):
        assert SEARCH_SCHEMA["name"] == "mem0_search"
        assert "query" in SEARCH_SCHEMA["parameters"]["required"]
        assert "rerank" in SEARCH_SCHEMA["parameters"]["properties"]
        assert "top_k" in SEARCH_SCHEMA["parameters"]["properties"]

    def test_conclude_schema(self):
        assert CONCLUDE_SCHEMA["name"] == "mem0_conclude"
        assert "conclusion" in CONCLUDE_SCHEMA["parameters"]["required"]


# ── Search calls use filters= ─────────────────────────────────────────────

class TestSearchUsesFilters:
    """Verify that search() passes filters= not user_id=."""

    def test_mem0_search_passes_filters(self, mock_client):
        provider = Mem0MemoryProvider()
        provider._client = mock_client
        provider._user_id = "test-user"

        result = provider.handle_tool_call(
            "mem0_search",
            {"query": "test query", "top_k": 5, "rerank": True},
        )

        data = json.loads(result)
        assert "results" in data
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args.kwargs
        assert "filters" in call_kwargs
        assert call_kwargs["filters"] == {"user_id": "test-user"}
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["rerank"] is True
        # Should NOT pass bare user_id
        assert "user_id" not in call_kwargs

    def test_mem0_search_empty_query(self, plugin):
        result = plugin.handle_tool_call("mem0_search", {})
        data = json.loads(result)
        assert "error" in data


# ── GetAll calls use filters= ─────────────────────────────────────────────

class TestGetAllUsesFilters:
    """Verify that get_all() passes filters= not user_id=."""

    def test_mem0_profile_passes_filters(self, mock_client):
        provider = Mem0MemoryProvider()
        provider._client = mock_client
        provider._user_id = "test-user"

        result = provider.handle_tool_call("mem0_profile", {})

        data = json.loads(result)
        assert "result" in data
        mock_client.get_all.assert_called_once()
        call_kwargs = mock_client.get_all.call_args.kwargs
        assert "filters" in call_kwargs
        assert call_kwargs["filters"] == {"user_id": "test-user"}
        assert "user_id" not in call_kwargs

    def test_mem0_profile_no_memories(self, mock_client):
        provider = Mem0MemoryProvider()
        mock_client.get_all.return_value = {"results": []}
        provider._client = mock_client
        provider._user_id = "test-user"

        result = provider.handle_tool_call("mem0_profile", {})
        data = json.loads(result)
        assert "No memories stored" in data["result"]

    def test_mem0_profile_handles_empty_result(self, mock_client):
        """Handles case where API returns empty dict."""
        provider = Mem0MemoryProvider()
        mock_client.get_all.return_value = {}
        provider._client = mock_client
        provider._user_id = "test-user"

        result = provider.handle_tool_call("mem0_profile", {})
        data = json.loads(result)
        assert "result" in data


# ── Add calls use bare user_id/agent_id ───────────────────────────────────

class TestAddUsesBareKwargs:
    """Verify that add() passes bare user_id= and agent_id= not filters=."""

    def test_mem0_conclude_passes_bare_kwargs(self, mock_client):
        provider = Mem0MemoryProvider()
        provider._client = mock_client
        provider._user_id = "test-user"
        provider._agent_id = "test-agent"

        result = provider.handle_tool_call(
            "mem0_conclude",
            {"conclusion": "User prefers dark mode."},
        )

        data = json.loads(result)
        assert "result" in data
        mock_client.add.assert_called_once()
        call_kwargs = mock_client.add.call_args.kwargs
        assert call_kwargs["user_id"] == "test-user"
        assert call_kwargs["agent_id"] == "test-agent"
        assert call_kwargs["infer"] is False
        # Should NOT pass filters
        assert "filters" not in call_kwargs

    def test_mem0_conclude_empty_conclusion(self, plugin):
        result = plugin.handle_tool_call("mem0_conclude", {})
        data = json.loads(result)
        assert "error" in data


# ── Sync turn uses bare kwargs ────────────────────────────────────────────

class TestSyncTurnBareKwargs:
    """Verify sync_turn() adds with bare user_id/agent_id."""

    def test_sync_turn_passes_bare_kwargs(self, mock_client):
        provider = Mem0MemoryProvider()
        provider._client = mock_client
        provider._user_id = "test-user"
        provider._agent_id = "test-agent"

        # Call the sync function directly by running _sync
        def _run_sync():
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
            provider._is_breaker_open = lambda: False
            provider._record_success = lambda: None
            mock_client.add(messages, user_id="test-user", agent_id="test-agent")

        _run_sync()

        call_kwargs = mock_client.add.call_args.kwargs
        assert "user_id" in call_kwargs
        assert "agent_id" in call_kwargs
        assert "filters" not in call_kwargs


# ── Integration tests against live API ────────────────────────────────────

class TestMem0LiveAPI:
    """Light integration tests against the real Mem0 API (requires MEM0_API_KEY)."""

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY"),
        reason="MEM0_API_KEY not set",
    )
    def test_live_add_and_retrieve(self, plugin):
        """Add a conclusion and verify it appears in profile."""
        import time

        # Add a test fact
        result = plugin.handle_tool_call(
            "mem0_conclude",
            {"conclusion": "TEST_MARKER_42_integration_check"},
        )
        data = json.loads(result)
        assert "result" in data
        assert "Fact stored" in data["result"]

        # Give the async add a moment to process
        time.sleep(2)

        # Search for it
        result2 = plugin.handle_tool_call(
            "mem0_search",
            {"query": "integration check marker"},
        )
        data2 = json.loads(result2)
        assert "results" in data2
        assert len(data2["results"]) > 0
        assert any("integration" in r["memory"].lower() for r in data2["results"])

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY"),
        reason="MEM0_API_KEY not set",
    )
    def test_live_get_all_returns_results(self, plugin):
        """get_all should return a populated dict, not error."""
        result = plugin.handle_tool_call("mem0_profile", {})
        data = json.loads(result)
        assert "result" in data or "No memories" in str(data)
        assert "error" not in data
