# Kado v0 — Specification

## Goal
Upload narrated video (≤5 min) → return machine-readable failure events with timestamps as strict JSON.

## Non-Goals
- No auth, DB, payments, analytics, email
- No integrations (Jira/Linear/GitHub)
- No vision/OCR/screenshots/frame diff
- No background queues/workers; keep synchronous

## API Contract

### `POST /analyze` (multipart form)
- **file**: video (mp4/mov/webm), max 5 min

### Response `200 OK`
```json
{
  "failures": [
    {
      "timestamp_seconds": 12.5,
      "title": "Button click does nothing",
      "expected": "Clicking submit should save the form",
      "actual": "Nothing happens after clicking submit",
      "evidence": "At 0:12 the narrator says 'I click submit and nothing happens'",
      "confidence": 0.85
    }
  ]
}
```

## Data Models

```
TranscriptSegment = { start: float, end: float, text: string }

FailureEvent = {
  timestamp_seconds: float,
  title: string,
  expected: string,
  actual: string,
  evidence: string,
  confidence: float (0..1)
}
```

## Pipeline (single synchronous request)

1. Save upload to temp path
2. `ffmpeg` → mono WAV 16 kHz
3. Transcribe audio → `TranscriptSegment[]` (OpenAI Whisper API)
4. Candidate detection — mark segment if text contains any keyword:
   `["doesn't","does not","nothing happens","broken","error","bug","fails","wrong","stuck","crash","not working","issue","problem"]`
5. For each candidate, create window of ±2 neighboring segments
6. LLM per window → `FailureEvent[]` (OpenAI GPT-4o-mini)
7. JSON validation; retry once with repair prompt on failure; skip on second failure
8. Merge/dedupe: duplicate if timestamps within 30 s AND titles semantically similar
9. Sort by `timestamp_seconds` ascending
10. Cleanup temp files always

## Stack
- **API**: FastAPI + Docker (Python 3.12, ffmpeg)
- **Transcription**: OpenAI Whisper API
- **LLM**: OpenAI GPT-4o-mini
- **Web**: Next.js (App Router, TypeScript)
