"""Tests for the candidate detection and window building stage."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import TranscriptSegment
from stages.candidates import detect_candidates, build_windows, FAILURE_KEYWORDS


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


class TestDetectCandidates:
    def test_matches_keywords(self):
        segments = [
            _seg(0, 5, "So I click the button"),
            _seg(5, 10, "and nothing happens"),
            _seg(10, 15, "The form just sits there"),
            _seg(15, 20, "I see an error message"),
        ]
        result = detect_candidates(segments)
        assert result == [1, 3]  # "nothing happens" and "error"

    def test_case_insensitive(self):
        segments = [_seg(0, 5, "This is BROKEN")]
        result = detect_candidates(segments)
        assert result == [0]

    def test_no_candidates(self):
        segments = [
            _seg(0, 5, "Everything looks good"),
            _seg(5, 10, "The feature works well"),
        ]
        result = detect_candidates(segments)
        assert result == []

    def test_multiple_keywords_same_segment(self):
        segments = [_seg(0, 5, "It fails with an error")]
        result = detect_candidates(segments)
        assert result == [0]  # only listed once


class TestBuildWindows:
    def test_basic_window(self):
        segments = [_seg(i, i + 1, f"seg {i}") for i in range(10)]
        windows = build_windows(segments, [5], radius=2)
        assert len(windows) == 1
        assert len(windows[0]) == 5  # segments 3,4,5,6,7
        assert windows[0][0].text == "seg 3"
        assert windows[0][4].text == "seg 7"

    def test_window_at_start(self):
        segments = [_seg(i, i + 1, f"seg {i}") for i in range(10)]
        windows = build_windows(segments, [0], radius=2)
        assert len(windows) == 1
        assert len(windows[0]) == 3  # segments 0,1,2
        assert windows[0][0].text == "seg 0"

    def test_window_at_end(self):
        segments = [_seg(i, i + 1, f"seg {i}") for i in range(10)]
        windows = build_windows(segments, [9], radius=2)
        assert len(windows) == 1
        assert len(windows[0]) == 3  # segments 7,8,9
        assert windows[0][-1].text == "seg 9"

    def test_multiple_candidates(self):
        segments = [_seg(i, i + 1, f"seg {i}") for i in range(10)]
        windows = build_windows(segments, [2, 7], radius=2)
        assert len(windows) == 2
