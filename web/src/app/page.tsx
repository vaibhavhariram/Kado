"use client";

import { useState, useCallback, useRef } from "react";

// --- Types ---
interface FailureEvent {
  timestamp_seconds: number;
  title: string;
  expected: string;
  actual: string;
  evidence: string;
  confidence: number;
}

interface AnalyzeResponse {
  failures: FailureEvent[];
}

type AppState = "idle" | "selected" | "analyzing" | "done" | "error";

// --- Constants ---
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const ACCEPTED_FORMATS = ".mp4,.mov,.webm";
const MAX_SIZE_MB = 500;

const PIPELINE_STEPS = [
  "Extracting audio from video",
  "Transcribing narration",
  "Scanning for failure signals",
  "Analyzing with AI",
  "Deduplicating results",
];

// --- Helpers ---
function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function confidenceClass(c: number): string {
  if (c >= 0.7) return "confidence-high";
  if (c >= 0.4) return "confidence-medium";
  return "confidence-low";
}

// --- Component ---
export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [copied, setCopied] = useState(false);
  const [showJson, setShowJson] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [dragover, setDragover] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // File selection
  const handleFile = useCallback((f: File) => {
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (!["mp4", "mov", "webm"].includes(ext || "")) {
      setError("Unsupported format. Please upload MP4, MOV, or WebM.");
      setState("error");
      return;
    }
    if (f.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`File too large (${formatFileSize(f.size)}). Max is ${MAX_SIZE_MB} MB.`);
      setState("error");
      return;
    }
    setFile(f);
    setState("selected");
    setError("");
    setResult(null);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragover(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragover(true);
  };

  const onDragLeave = () => setDragover(false);

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  // Analyze
  const analyze = async () => {
    if (!file) return;

    setState("analyzing");
    setActiveStep(0);
    setResult(null);
    setError("");

    // Simulate progress steps ticking forward
    const stepInterval = setInterval(() => {
      setActiveStep((prev) => Math.min(prev + 1, PIPELINE_STEPS.length - 1));
    }, 8000);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: form,
      });

      clearInterval(stepInterval);

      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }

      const data: AnalyzeResponse = await res.json();
      setResult(data);
      setState("done");
    } catch (err: unknown) {
      clearInterval(stepInterval);
      setError(err instanceof Error ? err.message : "Something went wrong");
      setState("error");
    }
  };

  // Copy JSON
  const copyJson = () => {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Reset
  const reset = () => {
    setFile(null);
    setResult(null);
    setError("");
    setState("idle");
    setCopied(false);
    setShowJson(false);
    setActiveStep(0);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">Kado</div>
        <h1>Video → Bug Reports</h1>
        <p>
          Upload a narrated screen recording and get structured failure events
          with timestamps, automatically.
        </p>
      </header>

      {/* Upload zone — shown in idle state */}
      {state === "idle" && (
        <div
          className={`upload-zone ${dragover ? "dragover" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          <span className="upload-icon">🎬</span>
          <div className="upload-title">Drop your video here</div>
          <div className="upload-subtitle">or click to browse</div>
          <div className="upload-formats">
            <span className="format-badge">MP4</span>
            <span className="format-badge">MOV</span>
            <span className="format-badge">WebM</span>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_FORMATS}
            onChange={onFileInput}
            style={{ display: "none" }}
          />
        </div>
      )}

      {/* File selected */}
      {state === "selected" && file && (
        <>
          <div className="file-info">
            <div className="file-details">
              <span className="file-icon">🎬</span>
              <div>
                <div className="file-name">{file.name}</div>
                <div className="file-size">{formatFileSize(file.size)}</div>
              </div>
            </div>
            <button className="file-remove" onClick={reset}>
              Remove
            </button>
          </div>
          <button className="analyze-btn" onClick={analyze}>
            Analyze Video
          </button>
        </>
      )}

      {/* Analyzing / progress */}
      {state === "analyzing" && (
        <div className="progress-container">
          <div className="spinner" />
          <div className="progress-title">Analyzing your video…</div>
          <div className="progress-subtitle">
            This may take 30–90 seconds depending on video length
          </div>
          <div className="progress-steps">
            {PIPELINE_STEPS.map((step, i) => (
              <div
                key={step}
                className={`progress-step ${i < activeStep ? "done" : i === activeStep ? "active" : ""
                  }`}
              >
                <span>
                  {i < activeStep ? "✓" : i === activeStep ? "⟳" : "○"}
                </span>
                {step}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {state === "error" && (
        <div className="error-container">
          <div className="error-title">Analysis Failed</div>
          <div className="error-message">{error}</div>
          <button className="error-retry" onClick={reset}>
            Try Again
          </button>
        </div>
      )}

      {/* Results */}
      {state === "done" && result && (
        <div className="results">
          <div className="results-header">
            <div>
              <h2 className="results-title">Results</h2>
              <span className="results-count">
                {result.failures.length} failure
                {result.failures.length !== 1 ? "s" : ""} detected
              </span>
            </div>
            <div className="results-actions">
              <button
                className={`copy-btn ${copied ? "copied" : ""}`}
                onClick={copyJson}
              >
                {copied ? "✓ Copied" : "📋 Copy JSON"}
              </button>
              <button className="reset-btn" onClick={reset}>
                ↻ New Upload
              </button>
            </div>
          </div>

          {/* No failures */}
          {result.failures.length === 0 && (
            <div className="no-failures">
              <div className="no-failures-icon">✅</div>
              <div className="no-failures-title">No failures detected</div>
              <p>
                The narration didn&apos;t contain any clear failure signals. Try a
                video where you narrate bugs or issues you encounter.
              </p>
            </div>
          )}

          {/* Failure cards */}
          <div className="failure-list">
            {result.failures.map((f, i) => (
              <div key={i} className="failure-card">
                <div className="failure-header">
                  <span className="failure-timestamp">
                    ⏱ {formatTimestamp(f.timestamp_seconds)}
                  </span>
                  <span
                    className={`failure-confidence ${confidenceClass(
                      f.confidence
                    )}`}
                  >
                    {Math.round(f.confidence * 100)}% confidence
                  </span>
                </div>
                <h3 className="failure-title">{f.title}</h3>
                <div className="failure-field">
                  <div className="failure-label">Expected</div>
                  <div className="failure-value">{f.expected}</div>
                </div>
                <div className="failure-field">
                  <div className="failure-label">Actual</div>
                  <div className="failure-value">{f.actual}</div>
                </div>
                <div className="failure-field">
                  <div className="failure-label">Evidence</div>
                  <div className="failure-evidence">{f.evidence}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Raw JSON toggle */}
          <div className="json-viewer">
            <button
              className="json-toggle"
              onClick={() => setShowJson(!showJson)}
            >
              {showJson ? "▾ Hide Raw JSON" : "▸ Show Raw JSON"}
            </button>
            {showJson && (
              <div className="json-content">
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
