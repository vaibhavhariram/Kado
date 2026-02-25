"""Stage 4 — LLM extraction of failure events from transcript windows."""

import json
import os

from openai import OpenAI

from models import FailureEvent, TranscriptSegment

SYSTEM_PROMPT = """\
You are a QA analysis assistant. You are given a window of timestamped transcript \
segments from a narrated screen recording. The narrator is describing what they see \
on screen and may mention bugs, errors, or unexpected behavior.

Your job: identify any software failure events described in this transcript window.

For each failure, output a JSON object with these exact fields:
- timestamp_seconds (float): the start time of the segment where the failure is described
- title (string): a short title summarizing the failure (max 10 words)
- expected (string): what should have happened
- actual (string): what actually happened
- evidence (string): exact quote(s) from the transcript that support this failure
- confidence (float 0-1): how confident you are this is a real software failure

Rules:
- Output ONLY a JSON array of failure objects. No markdown, no explanation.
- If no failures are found, output an empty array: []
- The evidence field MUST contain actual text from the provided transcript segments.
- Do not invent failures not supported by the transcript.
- A failure is a software bug, UI issue, or unexpected behavior — NOT user confusion or feature requests.
"""

REPAIR_PROMPT = """\
Your previous response was not valid JSON. Please output ONLY a valid JSON array \
of failure event objects (or [] if none). No markdown fences, no explanation, \
just the raw JSON array.
"""


def _format_window(window: list[TranscriptSegment]) -> str:
    """Format a window of segments for the LLM prompt."""
    lines = []
    for seg in window:
        lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
    return "\n".join(lines)


def _parse_failures(text: str) -> list[FailureEvent]:
    """Parse LLM response text into FailureEvent objects.

    Raises ValueError if JSON is invalid.
    """
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array")

    return [FailureEvent(**item) for item in data]


def extract_failures(window: list[TranscriptSegment]) -> list[FailureEvent]:
    """Extract failure events from a transcript window using GPT-4o-mini.

    Includes one retry with a repair prompt on JSON parse failure.
    Returns [] if both attempts fail.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user_content = _format_window(window)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # First attempt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        max_tokens=2000,
    )
    first_reply = response.choices[0].message.content or ""

    try:
        return _parse_failures(first_reply)
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Retry with repair prompt
    messages.append({"role": "assistant", "content": first_reply})
    messages.append({"role": "user", "content": REPAIR_PROMPT})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=2000,
    )
    second_reply = response.choices[0].message.content or ""

    try:
        return _parse_failures(second_reply)
    except (json.JSONDecodeError, ValueError, KeyError):
        # Both attempts failed — skip this window
        return []
