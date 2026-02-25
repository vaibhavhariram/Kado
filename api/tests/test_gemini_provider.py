"""Tests for Gemini extraction provider."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _gemini_mode_no_key(monkeypatch):
    """Set up Gemini extraction provider without API key."""
    monkeypatch.delenv("MOCK_MODE", raising=False)
    monkeypatch.setenv("EXTRACT_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Use local transcription to avoid needing OpenAI key
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "local")


def _client():
    """Create a fresh test client with reloaded modules."""
    import importlib
    import main as main_mod
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


class TestGeminiProvider:
    def test_missing_gemini_key_returns_501(self, _gemini_mode_no_key):
        """POST /analyze should return 501 JSON error when GEMINI_API_KEY is missing and EXTRACT_PROVIDER=gemini."""
        client = _client()
        dummy = b"fake video content"
        resp = client.post(
            "/analyze",
            files={"file": ("test.mp4", dummy, "video/mp4")},
        )
        
        # Should return 501, not 500 (stack trace)
        assert resp.status_code == 501, f"Expected 501, got {resp.status_code}: {resp.text}"
        
        # Should be JSON error
        body = resp.json()
        assert "detail" in body
        assert "GEMINI_API_KEY" in body["detail"]
