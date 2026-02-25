"""Test for real mode with local transcription and mock extraction."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _real_mode_local_transcribe_mock_extract(monkeypatch):
    """Set up real mode with local transcription and mock extraction."""
    monkeypatch.delenv("MOCK_MODE", raising=False)
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "local")
    monkeypatch.setenv("EXTRACT_PROVIDER", "mock")
    # Don't need OpenAI key since we're using local+mock
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _client(monkeypatch):
    """Create a fresh test client with reloaded modules."""
    import importlib
    import main as main_mod
    import pipeline
    from stages import audio, transcribe
    from models import TranscriptSegment
    
    # Apply mocks before reload
    def mock_get_video_duration(path: str) -> float:
        return 10.0
    
    def mock_extract_audio(video_path: str) -> str:
        return "/tmp/mock_audio.wav"
    
    def mock_transcribe_local(wav_path: str):
        return [
            TranscriptSegment(start=0.0, end=2.0, text="Let me show you this feature"),
            TranscriptSegment(start=2.0, end=5.0, text="When I click here it doesn't work"),
            TranscriptSegment(start=5.0, end=8.0, text="This is clearly a bug"),
        ]
    
    # Patch at module level where they're imported
    monkeypatch.setattr("stages.audio.extract_audio", mock_extract_audio)
    monkeypatch.setattr("stages.transcribe._transcribe_local", mock_transcribe_local)
    
    importlib.reload(pipeline)
    importlib.reload(main_mod)
    
    monkeypatch.setattr(main_mod, "_get_video_duration", mock_get_video_duration)
    
    return TestClient(main_mod.app)


class TestRealModeLocalTranscription:
    def test_analyze_returns_200_with_mode_real(self, _real_mode_local_transcribe_mock_extract, monkeypatch):
        """POST /analyze should return 200 with mode='real' when using local transcription + mock extraction.
        
        This test verifies that:
        - Real mode (not MOCK_MODE) works without OpenAI API key
        - Local transcription provider can be used
        - Mock extraction provider provides fast, deterministic results
        - Response correctly indicates mode='real'
        """
        client = _client(monkeypatch)
        
        # Create a minimal valid video file (empty is fine for this test since
        # local transcription will be used, which we're assuming is installed)
        # Note: In reality, faster-whisper needs a valid audio file, but the test
        # framework should handle this or we'd need a fixture video
        dummy = b"fake video content"
        
        resp = client.post(
            "/analyze",
            files={"file": ("test.mp4", dummy, "video/mp4")},
        )
        
        # Should succeed with local transcription + mock extraction
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        
        body = resp.json()
        
        # Verify response structure
        assert "failures" in body
        assert "mode" in body
        assert isinstance(body["failures"], list)
        
        # Key assertion: mode should be 'real' not 'mock'
        assert body["mode"] == "real", f"Expected mode='real', got mode='{body['mode']}'"
        
        # Failures might be empty or have deterministic mock failures
        # (depends on whether local transcription finds candidates)
        # We just verify the structure is valid
        for f in body["failures"]:
            assert "timestamp_seconds" in f
            assert "title" in f
            assert "expected" in f
            assert "actual" in f
            assert "evidence" in f
            assert "confidence" in f
            assert 0 <= f["confidence"] <= 1
