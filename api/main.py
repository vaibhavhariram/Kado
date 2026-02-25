"""Kado v0 — FastAPI application."""

import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from models import AnalyzeResponse
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — verify ffmpeg is available on startup."""
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
    return {"status": "ok"}


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

    # Save to temp file
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        # Validate duration
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
        logger.info("Starting pipeline for %s (%.1fs)", file.filename, duration)
        failures = run_pipeline(tmp_path)

        return AnalyzeResponse(failures=failures)

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
