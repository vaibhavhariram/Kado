"""Stage 2 — Transcribe audio via OpenAI Whisper API (or mock fixtures)."""

import json
import os
from pathlib import Path

from models import TranscriptSegment

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _is_mock_mode() -> bool:
    return os.environ.get("MOCK_MODE", "").strip() in ("1", "true", "yes")


def _mock_transcribe() -> list[TranscriptSegment]:
    """Return canned transcript segments from fixtures/transcript.json."""
    fixture_path = FIXTURES_DIR / "transcript.json"
    with open(fixture_path) as f:
        data = json.load(f)
    return [TranscriptSegment(**seg) for seg in data]


def transcribe(wav_path: str) -> list[TranscriptSegment]:
    """Transcribe a WAV file into timestamped segments.

    If MOCK_MODE=1, returns canned fixtures instead of calling Whisper.

    Args:
        wav_path: Path to a mono 16 kHz WAV file.

    Returns:
        Ordered list of TranscriptSegment objects.
    """
    if _is_mock_mode():
        return _mock_transcribe()

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
