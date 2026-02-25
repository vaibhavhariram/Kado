# Kado v0 — Tasks

## Backend (FastAPI)
- [ ] Project scaffold (requirements.txt, Dockerfile, main.py)
- [ ] Data models (models.py)
- [ ] Stage: audio extraction (ffmpeg → WAV)
- [ ] Stage: transcription (Whisper API)
- [ ] Stage: candidate detection (keyword heuristic)
- [ ] Stage: window building (±2 neighbors)
- [ ] Stage: LLM extraction (GPT-4o-mini)
- [ ] Stage: JSON validation + retry
- [ ] Stage: merge/dedupe
- [ ] Pipeline orchestrator
- [ ] POST /analyze endpoint wiring
- [ ] Unit tests

## Frontend (Next.js)
- [ ] Project scaffold
- [ ] Upload page (drag-and-drop)
- [ ] Spinner/progress state
- [ ] JSON result viewer
- [ ] Copy JSON button

## Integration
- [ ] Makefile (api-dev, web-dev, docker-build, demo)
- [ ] End-to-end local test
- [ ] Docker build verification
