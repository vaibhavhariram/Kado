"""Kado v0 — FastAPI application."""

import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from models import AnalyzeResponse, DebugInfo
from pipeline import run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Allowed video formats
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}
MAX_DURATION_SECONDS = 300  # 5 minutes


def _is_mock_mode() -> bool:
    return os.environ.get("MOCK_MODE", "").strip() in ("1", "true", "yes")


def _is_debug_mode() -> bool:
    return os.environ.get("DEBUG", "").strip() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — verify ffmpeg is available on startup."""
    if _is_mock_mode():
        logger.info("MOCK_MODE is ON — API keys not required")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY is not set — /analyze will fail unless MOCK_MODE=1")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        logger.info("ffmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg is NOT available — audio extraction will fail")
    yield


app = FastAPI(
    title="Kado API",
    description="Upload narrated video → get timestamped failure events as JSON",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "mock_mode": _is_mock_mode()}


def _get_video_duration(path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:300]}")
    return float(result.stdout.strip())


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    """Upload a narrated video and get back detected failure events.

    Accepts mp4, mov, or webm files up to 5 minutes long.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Gate: require API keys based on provider selection
    mock = _is_mock_mode()
    transcribe_provider = os.environ.get("TRANSCRIBE_PROVIDER", "openai").strip().lower()
    extract_provider = os.environ.get("EXTRACT_PROVIDER", "openai").strip().lower()
    
    if not mock:
        # Check OpenAI API key
        needs_openai = (transcribe_provider == "openai" or extract_provider == "openai")
        if needs_openai and not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=501,
                detail="OPENAI_API_KEY is not configured. Set the env var or enable MOCK_MODE=1.",
            )
        
        # Check Gemini API key
        if extract_provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(
                status_code=501,
                detail="GEMINI_API_KEY is not configured. Set the env var when using EXTRACT_PROVIDER=gemini.",
            )

    # Save to temp file
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        # Validate duration (skip in mock mode — any file works)
        duration = 0.0
        if not mock:
            try:
                duration = _get_video_duration(tmp_path)
                if duration > MAX_DURATION_SECONDS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Video is {duration:.0f}s — max allowed is {MAX_DURATION_SECONDS}s (5 min)",
                    )
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=f"Cannot read video metadata: {e}")

        # Run the pipeline
        debug_enabled = _is_debug_mode()
        logger.info("Starting pipeline for %s (%.1fs, mock=%s, debug=%s)", file.filename, duration, mock, debug_enabled)
        
        if debug_enabled:
            failures, debug_info = run_pipeline(tmp_path, debug=True)
            return AnalyzeResponse(
                failures=failures,
                mode="mock" if mock else "real",
                debug=DebugInfo(**debug_info)
            )
        else:
            failures = run_pipeline(tmp_path, debug=False)
            return AnalyzeResponse(failures=failures, mode="mock" if mock else "real")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Always clean up the uploaded temp file
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
                logger.info("Cleaned up temp upload: %s", tmp_path)
            except OSError:
                logger.warning("Failed to clean up %s", tmp_path)
