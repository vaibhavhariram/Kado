"""Tests for whisper.cpp transcription provider."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class _BlockFasterWhisperImport:
    """Meta path finder that blocks importing faster_whisper."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "faster_whisper":
            raise ImportError("faster_whisper is not installed (simulated)")


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

    def test_whispercpp_does_not_import_faster_whisper(self, monkeypatch, tmp_path):
        """TRANSCRIBE_PROVIDER=whispercpp must not import faster_whisper (no ImportError when it's missing)."""
        fake_model = tmp_path / "fake-model.bin"
        fake_model.write_bytes(b"x")
        monkeypatch.delenv("MOCK_MODE", raising=False)
        monkeypatch.setenv("TRANSCRIBE_PROVIDER", "whispercpp")
        monkeypatch.setenv("WHISPERCPP_MODEL_PATH", str(fake_model))
        monkeypatch.setenv("EXTRACT_PROVIDER", "mock")

        # Block faster_whisper so any import would raise
        sys.modules.pop("faster_whisper", None)
        blocker = _BlockFasterWhisperImport()
        sys.meta_path.insert(0, blocker)
        try:
            import importlib
            from stages import transcribe as transcribe_mod
            importlib.reload(transcribe_mod)

            mock_stdout = "[00:00.00 --> 00:01.00] test segment\n"
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=mock_stdout, stderr="")
                result = transcribe_mod.transcribe("/tmp/test.wav")

            assert len(result) == 1
            assert result[0].start == 0.0
            assert result[0].end == 1.0
            assert result[0].text == "test segment"
        finally:
            sys.meta_path.remove(blocker)
