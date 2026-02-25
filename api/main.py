"""Kado v0 — FastAPI application."""

import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Tuple, Union

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


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _is_mock_mode() -> bool:
    return _is_truthy_env("MOCK_MODE")


def _is_debug_mode() -> bool:
    return _is_truthy_env("DEBUG")


def _get_transcribe_provider() -> str:
    return os.environ.get("TRANSCRIBE_PROVIDER", "openai").strip().lower()


def _get_extract_provider() -> str:
    return os.environ.get("EXTRACT_PROVIDER", "openai").strip().lower()


def _require_provider_keys(mock: bool) -> None:
    """Require API keys only for the selected providers (unless mock mode)."""
    if mock:
        return

    transcribe_provider = _get_transcribe_provider()
    extract_provider = _get_extract_provider()

    # OpenAI key needed only if either stage uses OpenAI
    if transcribe_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=501,
            detail="OPENAI_API_KEY is not configured. Set the env var or switch TRANSCRIBE_PROVIDER.",
        )

    if extract_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=501,
            detail="OPENAI_API_KEY is not configured. Set the env var or switch EXTRACT_PROVIDER.",
        )

    # Gemini key needed only if extraction uses Gemini
    if extract_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=501,
            detail="GEMINI_API_KEY is not configured. Set the env var when using EXTRACT_PROVIDER=gemini.",
        )

    # whisper-cpp model path and binary required when transcription uses whispercpp
    if transcribe_provider == "whispercpp":
        if not os.getenv("WHISPERCPP_MODEL_PATH", "").strip():
            raise HTTPException(
                status_code=501,
                detail="WHISPERCPP_MODEL_PATH is required when using TRANSCRIBE_PROVIDER=whispercpp.",
            )
        from stages.transcribe import get_whispercpp_binary_path
        if not get_whispercpp_binary_path():
            raise HTTPException(
                status_code=501,
                detail="whisper-cpp binary not found. Install with: brew install whisper-cpp",
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — verify ffmpeg is available on startup."""
    mock = _is_mock_mode()
    transcribe_provider = _get_transcribe_provider()
    extract_provider = _get_extract_provider()

    if mock:
        logger.info("MOCK_MODE is ON — using fixtures; external API keys not required")
    else:
        # Don't hard-fail startup; just log what will be required.
        needs_openai = transcribe_provider == "openai" or extract_provider == "openai"
        needs_gemini = extract_provider == "gemini"

        if needs_openai and not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY is not set — /analyze will fail if OpenAI providers are selected")
        if needs_gemini and not os.getenv("GEMINI_API_KEY"):
            logger.warning("GEMINI_API_KEY is not set — /analyze will fail if EXTRACT_PROVIDER=gemini is selected")

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        logger.info("ffmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg is NOT available — audio extraction will fail in real mode")

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
    """Health check endpoint with effective runtime configuration."""
    mock = _is_mock_mode()
    return {
        "ok": True,
        "mode": "mock" if mock else "real",
        "mock_mode": mock,
        "debug": _is_debug_mode(),
        "transcribe_provider": _get_transcribe_provider(),
        "extract_provider": _get_extract_provider(),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "has_gemini_key": bool(os.getenv("GEMINI_API_KEY")),
    }


def _get_video_duration(path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:300]}")
    return float(result.stdout.strip())


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    """Upload a narrated video and get back detected failure events."""
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    mock = _is_mock_mode()
    debug_enabled = _is_debug_mode()

    # Require only the relevant keys for the chosen providers
    _require_provider_keys(mock=mock)

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

        logger.info(
            "Starting pipeline for %s (%.1fs, mode=%s, debug=%s, transcribe=%s, extract=%s)",
            file.filename,
            duration,
            "mock" if mock else "real",
            debug_enabled,
            _get_transcribe_provider(),
            _get_extract_provider(),
        )

        # Run pipeline
        if debug_enabled:
            failures, debug_info = run_pipeline(tmp_path, debug=True)
            return AnalyzeResponse(
                failures=failures,
                mode="mock" if mock else "real",
                debug=DebugInfo(**debug_info) if debug_info else None,
            )

        failures = run_pipeline(tmp_path, debug=False)
        return AnalyzeResponse(
            failures=failures,
            mode="mock" if mock else "real",
            debug=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
                logger.info("Cleaned up temp upload: %s", tmp_path)
            except OSError:
                logger.warning("Failed to clean up %s", tmp_path)