"""Kado v0 — Pipeline orchestrator.

Runs the full analysis pipeline synchronously:
  video → audio → transcript → candidates → LLM extraction → dedupe → result
"""

import logging
import os
from pathlib import Path
from typing import Union

from models import FailureEvent, TranscriptSegment
from stages.audio import extract_audio
from stages.transcribe import transcribe
from stages.candidates import detect_candidates, build_windows
from stages.extract import extract_failures
from stages.dedupe import merge_and_dedupe

logger = logging.getLogger(__name__)


def _is_mock_mode() -> bool:
    return os.environ.get("MOCK_MODE", "").strip() in ("1", "true", "yes")


def _is_debug_mode() -> bool:
    return os.environ.get("DEBUG", "").strip() in ("1", "true", "yes")


def run_pipeline(video_path: str, debug: bool = False) -> Union[list[FailureEvent], tuple[list[FailureEvent], dict]]:
    """Run the full Kado analysis pipeline on a video file.

    Args:
        video_path: Path to the uploaded video.
        debug: If True, returns (failures, debug_info) tuple with pipeline stats.

    Returns:
        If debug=False: Sorted list of deduplicated FailureEvent objects.
        If debug=True: Tuple of (failures, debug_dict) where debug_dict contains
                       num_segments, num_candidates, num_windows.
    """
    mock = _is_mock_mode()
    wav_path: str | None = None
    
    # Debug tracking
    debug_info = {"num_segments": 0, "num_candidates": 0, "num_windows": 0}

    try:
        # Stage 1: Extract audio (skip in mock mode — transcribe uses fixtures)
        if mock:
            logger.info("Stage 1: MOCK — skipping audio extraction")
        else:
            logger.info("Stage 1: Extracting audio from %s", video_path)
            wav_path = extract_audio(video_path)
            logger.info("Audio extracted to %s", wav_path)

        # Stage 2: Transcribe
        logger.info("Stage 2: Transcribing audio%s", " (MOCK)" if mock else "")
        segments: list[TranscriptSegment] = transcribe(wav_path or "")
        logger.info("Got %d transcript segments", len(segments))
        
        debug_info["num_segments"] = len(segments)

        if not segments:
            logger.warning("No transcript segments found — returning empty results")
            return ([], debug_info) if debug else []

        # Stage 3: Candidate detection
        logger.info("Stage 3: Detecting candidates")
        candidate_indices = detect_candidates(segments)
        logger.info("Found %d candidate segments", len(candidate_indices))
        
        debug_info["num_candidates"] = len(candidate_indices)

        if not candidate_indices:
            logger.info("No candidate segments — returning empty results")
            return ([], debug_info) if debug else []

        # Stage 4: Build windows
        logger.info("Stage 4: Building context windows")
        windows = build_windows(segments, candidate_indices)
        logger.info("Built %d windows", len(windows))
        
        debug_info["num_windows"] = len(windows)

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
            return ([], debug_info) if debug else []

        # Stage 6: Merge/dedupe
        logger.info("Stage 6: Merging and deduplicating")
        result = merge_and_dedupe(all_failures)
        logger.info("Final failures after dedupe: %d", len(result))

        return (result, debug_info) if debug else result

    finally:
        # Cleanup WAV temp file
        if wav_path and Path(wav_path).exists():
            try:
                os.unlink(wav_path)
                logger.info("Cleaned up temp WAV: %s", wav_path)
            except OSError:
                logger.warning("Failed to clean up %s", wav_path)
