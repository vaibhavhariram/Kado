"""Kado v0 — Pipeline orchestrator.

Runs the full analysis pipeline synchronously:
  video → audio → transcript → candidates → LLM extraction → dedupe → result
"""

import logging
import os
from pathlib import Path

from models import FailureEvent, TranscriptSegment
from stages.audio import extract_audio
from stages.transcribe import transcribe
from stages.candidates import detect_candidates, build_windows
from stages.extract import extract_failures
from stages.dedupe import merge_and_dedupe

logger = logging.getLogger(__name__)


def run_pipeline(video_path: str) -> list[FailureEvent]:
    """Run the full Kado analysis pipeline on a video file.

    Args:
        video_path: Path to the uploaded video.

    Returns:
        Sorted list of deduplicated FailureEvent objects.
    """
    wav_path: str | None = None

    try:
        # Stage 1: Extract audio
        logger.info("Stage 1: Extracting audio from %s", video_path)
        wav_path = extract_audio(video_path)
        logger.info("Audio extracted to %s", wav_path)

        # Stage 2: Transcribe
        logger.info("Stage 2: Transcribing audio")
        segments: list[TranscriptSegment] = transcribe(wav_path)
        logger.info("Got %d transcript segments", len(segments))

        if not segments:
            logger.warning("No transcript segments found — returning empty results")
            return []

        # Stage 3: Candidate detection
        logger.info("Stage 3: Detecting candidates")
        candidate_indices = detect_candidates(segments)
        logger.info("Found %d candidate segments", len(candidate_indices))

        if not candidate_indices:
            logger.info("No candidate segments — returning empty results")
            return []

        # Stage 4: Build windows
        logger.info("Stage 4: Building context windows")
        windows = build_windows(segments, candidate_indices)
        logger.info("Built %d windows", len(windows))

        # Stage 5: LLM extraction per window
        logger.info("Stage 5: Running LLM extraction on each window")
        all_failures: list[FailureEvent] = []
        for i, window in enumerate(windows):
            logger.info("  Processing window %d/%d", i + 1, len(windows))
            failures = extract_failures(window)
            all_failures.extend(failures)
            logger.info("  → %d failures found", len(failures))

        logger.info("Total raw failures: %d", len(all_failures))

        if not all_failures:
            return []

        # Stage 6: Merge/dedupe
        logger.info("Stage 6: Merging and deduplicating")
        result = merge_and_dedupe(all_failures)
        logger.info("Final failures after dedupe: %d", len(result))

        return result

    finally:
        # Cleanup WAV temp file
        if wav_path and Path(wav_path).exists():
            try:
                os.unlink(wav_path)
                logger.info("Cleaned up temp WAV: %s", wav_path)
            except OSError:
                logger.warning("Failed to clean up %s", wav_path)
