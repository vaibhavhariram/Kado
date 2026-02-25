"""Tests for whisper.cpp transcription provider."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _whispercpp_mode_no_model_path(monkeypatch):
    """Set up whispercpp transcription without model path."""
    monkeypatch.delenv("MOCK_MODE", raising=False)
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "whispercpp")
    monkeypatch.delenv("WHISPERCPP_MODEL_PATH", raising=False)
    monkeypatch.setenv("EXTRACT_PROVIDER", "mock")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _client():
    """Create a fresh test client with reloaded modules."""
    import importlib
    import main as main_mod
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


class TestWhisperCppProvider:
    def test_missing_model_path_returns_501(self, _whispercpp_mode_no_model_path):
        """POST /analyze should return 501 JSON error when WHISPERCPP_MODEL_PATH is missing."""
        client = _client()
        dummy = b"fake video content"
        resp = client.post(
            "/analyze",
            files={"file": ("test.mp4", dummy, "video/mp4")},
        )

        assert resp.status_code == 501, f"Expected 501, got {resp.status_code}: {resp.text}"

        body = resp.json()
        assert "detail" in body
        assert "WHISPERCPP_MODEL_PATH" in body["detail"]
