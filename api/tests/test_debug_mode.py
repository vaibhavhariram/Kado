"""Tests for DEBUG mode metadata."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _mock_mode_with_debug(monkeypatch):
    """Enable mock mode and debug mode."""
    monkeypatch.setenv("MOCK_MODE", "1")
    monkeypatch.setenv("DEBUG", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def _mock_mode_no_debug(monkeypatch):
    """Enable mock mode but disable debug mode."""
    monkeypatch.setenv("MOCK_MODE", "1")
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _client():
    """Create a fresh test client with reloaded modules."""
    import importlib
    import main as main_mod
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


class TestDebugMode:
    def test_analyze_includes_debug_when_enabled(self, _mock_mode_with_debug):
        """POST /analyze should include debug metadata when DEBUG=1."""
        client = _client()
        dummy = b"fake video content"
        resp = client.post(
            "/analyze",
            files={"file": ("test.mp4", dummy, "video/mp4")},
        )
        assert resp.status_code == 200
        body = resp.json()
        
        # Verify debug field exists
        assert "debug" in body
        assert body["debug"] is not None
        
        # Verify debug structure
        debug = body["debug"]
        assert "num_segments" in debug
        assert "num_candidates" in debug
        assert "num_windows" in debug
        
        # Verify values are integers
        assert isinstance(debug["num_segments"], int)
        assert isinstance(debug["num_candidates"], int)
        assert isinstance(debug["num_windows"], int)
        
        # In mock mode, should have some segments
        assert debug["num_segments"] > 0

    def test_analyze_excludes_debug_when_disabled(self, _mock_mode_no_debug):
        """POST /analyze should NOT include debug metadata when DEBUG is unset."""
        client = _client()
        dummy = b"fake video content"
        resp = client.post(
            "/analyze",
            files={"file": ("test.mp4", dummy, "video/mp4")},
        )
        assert resp.status_code == 200
        body = resp.json()
        
        # Verify debug field is absent or None
        assert body.get("debug") is None
