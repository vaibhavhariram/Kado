"""Tests for the merge/dedupe stage."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import FailureEvent
from stages.dedupe import merge_and_dedupe, _jaccard_similarity


def _event(ts: float, title: str, confidence: float = 0.8, evidence: str = "evidence") -> FailureEvent:
    return FailureEvent(
        timestamp_seconds=ts,
        title=title,
        expected="expected",
        actual="actual",
        evidence=evidence,
        confidence=confidence,
    )


class TestJaccardSimilarity:
    def test_identical(self):
        assert _jaccard_similarity("button click fails", "button click fails") == 1.0

    def test_similar(self):
        sim = _jaccard_similarity("button click fails", "click button fails")
        assert sim > 0.5

    def test_different(self):
        sim = _jaccard_similarity("button click fails", "form validation error")
        assert sim < 0.5

    def test_empty(self):
        assert _jaccard_similarity("", "") == 1.0
        assert _jaccard_similarity("hello", "") == 0.0


class TestMergeAndDedupe:
    def test_no_duplicates(self):
        events = [
            _event(10, "Button broken"),
            _event(60, "Form error"),
        ]
        result = merge_and_dedupe(events)
        assert len(result) == 2

    def test_timestamp_duplicates(self):
        events = [
            _event(10, "Button click fails", confidence=0.7, evidence="evidence A"),
            _event(15, "Button click fails", confidence=0.9, evidence="evidence B"),
        ]
        result = merge_and_dedupe(events)
        assert len(result) == 1
        assert result[0].confidence == 0.9
        assert "evidence B" in result[0].evidence
        assert "evidence A" in result[0].evidence

    def test_no_merge_different_timestamps(self):
        events = [
            _event(10, "Button click fails"),
            _event(100, "Button click fails"),
        ]
        result = merge_and_dedupe(events)
        assert len(result) == 2

    def test_no_merge_different_titles(self):
        events = [
            _event(10, "Button click fails"),
            _event(15, "Completely unrelated database error"),
        ]
        result = merge_and_dedupe(events)
        assert len(result) == 2

    def test_sorted_output(self):
        events = [
            _event(50, "Late event"),
            _event(10, "Early event"),
            _event(30, "Middle event"),
        ]
        result = merge_and_dedupe(events)
        assert [e.timestamp_seconds for e in result] == [10, 30, 50]

    def test_empty_input(self):
        assert merge_and_dedupe([]) == []
