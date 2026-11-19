# Start the MoneyPrinterTurbo API (127.0.0.1:8080) and WebUI (127.0.0.1:8501) in the background.
# Puts the pip-installed CUDA libraries on PATH so faster-whisper can use the GPU.
# Stop them with:  .\automation\stop_services.ps1
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$nv = Join-Path $root ".venv\Lib\site-packages\nvidia"
$env:PATH = (Join-Path $nv "cublas\bin") + ";" + (Join-Path $nv "cudnn\bin") + ";" + $env:PATH
$env:PYTHONPATH = $root
New-Item -ItemType Directory -Force (Join-Path $root "storage\logs") | Out-Null
$log = Join-Path $root "storage\logs"

# Optional local voice (Kokoro-FastAPI, 127.0.0.1:8880) and image generator (SDXL, 127.0.0.1:7860),
# installed next to this repo in ..\Kokoro-FastAPI and ..\local-image-server.
$docs = Split-Path -Parent $root
$kokoro = Join-Path $docs "Kokoro-FastAPI"
$extra = @()
if (Test-Path "$kokoro\.venv\Scripts\python.exe") {
    $env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
    $env:PYTHONUTF8 = 1; $env:PROJECT_ROOT = $kokoro; $env:USE_GPU = "false"
    $env:MODEL_DIR = "src/models"; $env:VOICES_DIR = "src/voices/v1_0"; $env:WEB_PLAYER_PATH = "$kokoro/web"
    $env:PYTHONPATH = "$kokoro;$kokoro/api"
    $k = Start-Process -FilePath "$kokoro\.venv\Scripts\python.exe" -WorkingDirectory $kokoro -WindowStyle Hidden -PassThru `
        -ArgumentList "-m uvicorn api.src.main:app --host 127.0.0.1 --port 8880" `
        -RedirectStandardOutput "$log\kokoro.out.log" -RedirectStandardError "$log\kokoro.err.log"
    $extra += $k.Id; "Kokoro PID $($k.Id)  http://127.0.0.1:8880/v1"
}
$imgPy = Join-Path $docs "local-image-server\.venv\Scripts\python.exe"
if (Test-Path $imgPy) {
    $i = Start-Process -FilePath $imgPy -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -ArgumentList "automation\local_image_server.py --port 7860" `
        -RedirectStandardOutput "$log\images.out.log" -RedirectStandardError "$log\images.err.log"
    $extra += $i.Id; "Images PID $($i.Id)  http://127.0.0.1:7860/v1"
}
$env:PYTHONPATH = $root

$api = Start-Process -FilePath $py -ArgumentList "main.py" -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$log\api.out.log" -RedirectStandardError "$log\api.err.log"
$ui = Start-Process -FilePath $py -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList "-m streamlit run .\webui\Main.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true --browser.gatherUsageStats=False --client.toolbarMode=minimal --logger.hideWelcomeMessage=True --server.showEmailPrompt=False" `
    -RedirectStandardOutput "$log\webui.out.log" -RedirectStandardError "$log\webui.err.log"
"$($api.Id) $($ui.Id) $($extra -join ' ')" | Set-Content -Encoding ascii (Join-Path $root "storage\services.pid")
"API   PID $($api.Id)  http://127.0.0.1:8080/docs"
"WebUI PID $($ui.Id)  http://127.0.0.1:8501"
