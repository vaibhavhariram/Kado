"""Stage 2 — Transcribe audio via OpenAI Whisper API, local faster-whisper, or mock fixtures."""

import json
import logging
import os
from pathlib import Path

from models import TranscriptSegment

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _is_mock_mode() -> bool:
    return os.environ.get("MOCK_MODE", "").strip() in ("1", "true", "yes")


def _get_provider() -> str:
    """Get transcription provider from env, defaults to 'openai'."""
    return os.environ.get("TRANSCRIBE_PROVIDER", "openai").strip().lower()


def _mock_transcribe() -> list[TranscriptSegment]:
    """Return canned transcript segments from fixtures/transcript.json."""
    fixture_path = FIXTURES_DIR / "transcript.json"
    with open(fixture_path) as f:
        data = json.load(f)
    return [TranscriptSegment(**seg) for seg in data]


def _transcribe_local(wav_path: str) -> list[TranscriptSegment]:
    """Transcribe using local faster-whisper."""
    from faster_whisper import WhisperModel

    model_size = os.environ.get("WHISPER_MODEL", "base")
    logger.info("Loading faster-whisper model: %s", model_size)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    logger.info("Transcribing with faster-whisper")
    segments_iter, info = model.transcribe(wav_path, beam_size=5)

    logger.info("Detected language '%s' with probability %.2f", info.language, info.language_probability)

    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        segments.append(
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
            )
        )

    return segments


def _transcribe_openai(wav_path: str) -> list[TranscriptSegment]:
    """Transcribe using OpenAI Whisper API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(wav_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments: list[TranscriptSegment] = []
    for seg in response.segments:
        segments.append(
            TranscriptSegment(
                start=seg["start"] if isinstance(seg, dict) else seg.start,
                end=seg["end"] if isinstance(seg, dict) else seg.end,
                text=(seg["text"] if isinstance(seg, dict) else seg.text).strip(),
            )
        )

    return segments


def transcribe(wav_path: str) -> list[TranscriptSegment]:
    """Transcribe a WAV file into timestamped segments.

    Respects MOCK_MODE and TRANSCRIBE_PROVIDER environment variables.
    - MOCK_MODE=1: Returns canned fixtures
    - TRANSCRIBE_PROVIDER=openai (default): Uses OpenAI Whisper API
    - TRANSCRIBE_PROVIDER=local: Uses faster-whisper locally

    Args:
        wav_path: Path to a mono 16 kHz WAV file.

    Returns:
        Ordered list of TranscriptSegment objects.
    """
    if _is_mock_mode():
        return _mock_transcribe()

    provider = _get_provider()

    if provider == "local":
        logger.info("Using local faster-whisper transcription")
        return _transcribe_local(wav_path)
    elif provider == "openai":
        logger.info("Using OpenAI Whisper API transcription")
        return _transcribe_openai(wav_path)
    else:
        raise ValueError(f"Unknown TRANSCRIBE_PROVIDER: {provider}. Must be 'openai' or 'local'.")
