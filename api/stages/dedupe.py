"""Stage 5 — Merge and deduplicate failure events."""

from models import FailureEvent


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _are_duplicates(a: FailureEvent, b: FailureEvent) -> bool:
    """Two events are duplicates if timestamps are within 30s AND titles are similar."""
    time_close = abs(a.timestamp_seconds - b.timestamp_seconds) <= 30.0
    title_similar = _jaccard_similarity(a.title, b.title) > 0.5
    return time_close and title_similar


def _merge_events(keep: FailureEvent, drop: FailureEvent) -> FailureEvent:
    """Merge two duplicate events, keeping the higher-confidence one and combining evidence."""
    if drop.confidence > keep.confidence:
        keep, drop = drop, keep

    # Merge evidence if the drop has unique content
    merged_evidence = keep.evidence
    if drop.evidence and drop.evidence not in keep.evidence:
        merged_evidence = f"{keep.evidence} | {drop.evidence}"

    return FailureEvent(
        timestamp_seconds=keep.timestamp_seconds,
        title=keep.title,
        expected=keep.expected,
        actual=keep.actual,
        evidence=merged_evidence,
        confidence=keep.confidence,
    )


def merge_and_dedupe(events: list[FailureEvent]) -> list[FailureEvent]:
    """Merge duplicate failure events and sort by timestamp ascending.

    Duplicates: timestamps within 30s AND titles have Jaccard similarity > 0.5.
    Keeps higher confidence, merges evidence strings.
    """
    if not events:
        return []

    # Sort by timestamp first for stable processing
    sorted_events = sorted(events, key=lambda e: e.timestamp_seconds)

    merged: list[FailureEvent] = []
    used: set[int] = set()

    for i, event_a in enumerate(sorted_events):
        if i in used:
            continue

        current = event_a
        for j in range(i + 1, len(sorted_events)):
            if j in used:
                continue
            if _are_duplicates(current, sorted_events[j]):
                current = _merge_events(current, sorted_events[j])
                used.add(j)

        merged.append(current)
        used.add(i)

    return sorted(merged, key=lambda e: e.timestamp_seconds)
