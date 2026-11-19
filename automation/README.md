# Local GPU controller (Windows)

Runs MoneyPrinterTurbo fully on this PC, no cloud services:

| Part | Local engine | Port |
| --- | --- | --- |
| Script + shot planning | Ollama `qwen3:14b` (GPU) | 11434 |
| Voice | Kokoro-FastAPI, voice `af_heart` (CPU) | 8880 |
| Footage | `automation/local_image_server.py`: SDXL (GPU) + on-screen code/text cards | 7860 |
| Subtitles | faster-whisper large-v3 (CUDA) | - |
| Encoding | h264_nvenc | - |
| App | API / WebUI | 8080 / 8501 |

## Start (one script)

Double-click **`script\Start-MoneyPrinter.bat`** (or the desktop shortcut that points at
`script\start.ps1`). It starts Ollama if needed and every service above, waits until each one
answers, then opens http://127.0.0.1:8501 in Chrome. If everything is already running it only opens
Chrome; it never restarts services, so videos in progress are not killed.

In the WebUI: Video Source **OpenAI Compatible Text-to-Image**, tick **Match Visuals to Script Order**,
Voiceover Service **Kokoro TTS** / `af_heart`, then **Generate Video**.

Optional: `script\Make-Video.bat "topic"` makes a video without the WebUI;
`script\Stop-MoneyPrinter.bat` stops everything except Ollama.

## How footage matches the script

The app saves `storage\tasks\<id>\script.json` (script + ordered search terms) before it asks for
images. For each video the image server sends that script to the local LLM once, which returns per
shot an SDXL scene prompt and the on-screen text (code, formula, binary example, key phrase). SDXL
draws the scene (~8-12 s per image on an RTX 5070 Ti) and the text is drawn on top as a code card in
the upper third, clear of the subtitles. Ollama is unloaded before each image so both fit in 16 GB.

## config.toml settings used

```toml
listen_host = "127.0.0.1"
listen_port = 8080
# [app]
llm_provider = "ollama"
ollama_base_url = "http://localhost:11434/v1"
ollama_model_name = "qwen3:14b"
video_source = "openai_image"
openai_image_base_url = "http://127.0.0.1:7860/v1"
openai_image_model = "sdxl"
openai_image_size = "768x1344"
subtitle_provider = "whisper"
video_codec = "h264_nvenc"
tts_server = "kokoro"
voice_name = "kokoro:af_heart"
# [whisper]
device = "cuda"
compute_type = "float16"
# [kokoro]
base_url = "http://127.0.0.1:8880/v1"
# [ui]
language = "en"
video_language = "en-US"
```

## Install

Next to this repo (in the same parent folder):

```powershell
# App
$env:UV_LINK_MODE="copy"; uv sync --frozen
uv pip install --python .venv\Scripts\python.exe nvidia-cublas-cu12 nvidia-cudnn-cu12
# Whisper model -> models\whisper-large-v3 (Systran/faster-whisper-large-v3)

# Voice: ..\Kokoro-FastAPI (github.com/remsky/Kokoro-FastAPI) + eSpeak NG (winget eSpeak-NG.eSpeak-NG)
uv venv --python 3.12; uv pip install -e ".[gpu-cu128]"
uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0

# Footage: ..\local-image-server\.venv
uv venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe diffusers transformers accelerate safetensors fastapi uvicorn requests pydantic
# SDXL (stabilityai/stable-diffusion-xl-base-1.0, fp16) downloads on first start
```

## Own footage instead (video_source = "local")

```powershell
python automation\split_clips.py                # raw_recordings\*.mp4 -> local_clips\ (10 s pieces)
python automation\trigger_video.py --subject "Top 3 GTA 6 police escapes"
```

`trigger_video.py` copies `local_clips\` into `storage\local_videos\` (the only folder the app
reads local footage from), posts `/api/v1/videos`, polls `/api/v1/tasks/{id}` and prints the MP4 path.

Logs for every service are in `storage\logs\`.
