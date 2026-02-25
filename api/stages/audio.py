"""Stage 1 — Extract audio from video using ffmpeg."""

import subprocess
import tempfile
from pathlib import Path


def extract_audio(video_path: str) -> str:
    """Convert video to mono WAV at 16 kHz using ffmpeg.

    Args:
        video_path: Path to the uploaded video file.

    Returns:
        Path to the generated WAV file (caller must clean up).

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    wav_path = tempfile.mktemp(suffix=".wav")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz
        "-f", "wav",
        "-y",                # overwrite
        wav_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")

    if not Path(wav_path).exists():
        raise RuntimeError("ffmpeg produced no output file")

    return wav_path
