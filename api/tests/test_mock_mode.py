"""Tests for MOCK_MODE and missing API key behavior."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _no_keys(monkeypatch):
    """Ensure OPENAI_API_KEY and MOCK_MODE are unset."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MOCK_MODE", raising=False)


@pytest.fixture
def _mock_mode(monkeypatch):
    """Enable mock mode."""
    monkeypatch.setenv("MOCK_MODE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _client():
    # Re-import to pick up env changes
    import importlib
    import main as main_mod
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


class TestMissingApiKey:
    def test_analyze_returns_501_without_key(self, _no_keys):
        """POST /analyze should return 501 JSON error when OPENAI_API_KEY is missing."""
        client = _client()
        dummy = b"fake video content"
        resp = client.post(
            "/analyze",
            files={"file": ("test.mp4", dummy, "video/mp4")},
        )
        assert resp.status_code == 501
        body = resp.json()
        assert "detail" in body
        assert "OPENAI_API_KEY" in body["detail"]


class TestMockMode:
    def test_analyze_works_in_mock_mode(self, _mock_mode):
        """POST /analyze should return valid JSON in MOCK_MODE without any API key."""
        client = _client()
        dummy = b"fake video content"
        resp = client.post(
            "/analyze",
            files={"file": ("test.mp4", dummy, "video/mp4")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "failures" in body
        assert isinstance(body["failures"], list)
        assert len(body["failures"]) > 0

        # Verify each failure has required fields
        for f in body["failures"]:
            assert "timestamp_seconds" in f
            assert "title" in f
            assert "expected" in f
            assert "actual" in f
            assert "evidence" in f
            assert "confidence" in f
            assert 0 <= f["confidence"] <= 1
