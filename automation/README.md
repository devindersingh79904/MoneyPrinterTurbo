# Local GPU controller (Windows)

Runs MoneyPrinterTurbo fully on this PC: Ollama for the script, faster-whisper on CUDA for
subtitles, h264_nvenc for encoding, and your own clips as footage (`video_source = "local"`).

## config.toml settings used

```toml
listen_host = "127.0.0.1"
listen_port = 8080
# [app]
llm_provider = "ollama"
ollama_base_url = "http://localhost:11434/v1"
ollama_model_name = "qwen3:14b"
video_source = "local"
subtitle_provider = "whisper"
video_codec = "h264_nvenc"
# [whisper]
device = "cuda"
compute_type = "float16"
# [ui]
language = "en"
video_language = "en-US"
```

Install: `$env:UV_LINK_MODE="copy"; uv sync --frozen` (copy mode is needed inside OneDrive), then
`uv pip install --python .venv\Scripts\python.exe nvidia-cublas-cu12 nvidia-cudnn-cu12`.

## Use

```powershell
.\automation\start_services.ps1                 # API http://127.0.0.1:8080/docs, WebUI http://127.0.0.1:8501
python automation\split_clips.py                # raw_recordings\*.mp4 -> local_clips\ (10 s pieces)
python automation\trigger_video.py --subject "Top 3 GTA 6 police escapes"
.\automation\stop_services.ps1
```

`trigger_video.py` copies `local_clips\` into `storage\local_videos\` (the only folder the app
reads local footage from), posts `/api/v1/videos`, polls `/api/v1/tasks/{id}` and prints the MP4 path.

Notes: Edge TTS (default voice) is an online Microsoft service; Kokoro/Chatterbox are local options.
With 16 GB VRAM, set `OLLAMA_KEEP_ALIVE=0` if Whisper runs out of memory while the LLM is still loaded.
