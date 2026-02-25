"""Stage 3 — Candidate detection and window building."""

from models import TranscriptSegment

# Keywords that signal a potential failure in narrated video
FAILURE_KEYWORDS: list[str] = [
    "doesn't",
    "does not",
    "nothing happens",
    "broken",
    "error",
    "bug",
    "fails",
    "wrong",
    "stuck",
    "crash",
    "not working",
    "issue",
    "problem",
]


def detect_candidates(segments: list[TranscriptSegment]) -> list[int]:
    """Return indices of segments whose text contains any failure keyword.

    Case-insensitive matching.
    """
    candidates: list[int] = []
    for i, seg in enumerate(segments):
        lower_text = seg.text.lower()
        if any(kw in lower_text for kw in FAILURE_KEYWORDS):
            candidates.append(i)
    return candidates


def build_windows(
    segments: list[TranscriptSegment],
    candidate_indices: list[int],
    radius: int = 2,
) -> list[list[TranscriptSegment]]:
    """Build context windows of ±radius segments around each candidate.

    Overlapping windows are NOT merged — the LLM sees each candidate
    in its own context. Deduplication happens later.
    """
    windows: list[list[TranscriptSegment]] = []
    for idx in candidate_indices:
        start = max(0, idx - radius)
        end = min(len(segments), idx + radius + 1)
        windows.append(segments[start:end])
    return windows
