# Kado
Kado turns screen recordings into bug reports. Upload a narrated Loom or demo video and Kado detects where something breaks, then generates timestamped, copy-paste-ready reports with steps to reproduce and expected vs actual behavior—so you don’t have to rewatch demos or manually write tickets.

## Run locally

- **API** (mock mode, no keys): `make mock-dev`
- **API** (whisper.cpp + Gemini, no OpenAI): see [whisper.cpp setup](#whispercpp-setup) below
- **Web**: `make web-dev`

### whisper.cpp setup (macOS-friendly, no PyAV/faster-whisper)

```bash
brew install whisper-cpp
# Download model (e.g. ggml-base.en.bin) to api/models/
# https://huggingface.co/ggerganov/whisper.cpp/tree/main

# Brew installs whisper-cli; set WHISPERCPP_PATH if your binary has a different name
TRANSCRIBE_PROVIDER=whispercpp \
  WHISPERCPP_MODEL_PATH=$(pwd)/api/models/ggml-base.en.bin \
  WHISPERCPP_PATH=whisper-cli \
  make api-dev
```
