"""Kado v0 — Pydantic data models."""

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A timestamped segment of transcribed speech."""

    start: float
    end: float
    text: str


class FailureEvent(BaseModel):
    """A detected failure event from narrated video."""

    timestamp_seconds: float = Field(description="Seconds into the video where the failure occurs")
    title: str = Field(description="Short title summarizing the failure")
    expected: str = Field(description="What should have happened")
    actual: str = Field(description="What actually happened")
    evidence: str = Field(description="Transcript evidence grounding this failure")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0..1")


class AnalyzeResponse(BaseModel):
    """Response from POST /analyze."""

    failures: list[FailureEvent]
