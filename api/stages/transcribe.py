"""Stage 2 — Transcribe audio via OpenAI Whisper API."""

import os

from openai import OpenAI

from models import TranscriptSegment


def transcribe(wav_path: str) -> list[TranscriptSegment]:
    """Transcribe a WAV file into timestamped segments using Whisper API.

    Args:
        wav_path: Path to a mono 16 kHz WAV file.

    Returns:
        Ordered list of TranscriptSegment objects.
    """
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
