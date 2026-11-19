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

$api = Start-Process -FilePath $py -ArgumentList "main.py" -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$log\api.out.log" -RedirectStandardError "$log\api.err.log"
$ui = Start-Process -FilePath $py -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList "-m streamlit run .\webui\Main.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true --browser.gatherUsageStats=False --client.toolbarMode=minimal --logger.hideWelcomeMessage=True --server.showEmailPrompt=False" `
    -RedirectStandardOutput "$log\webui.out.log" -RedirectStandardError "$log\webui.err.log"
"$($api.Id) $($ui.Id)" | Set-Content -Encoding ascii (Join-Path $root "storage\services.pid")
"API   PID $($api.Id)  http://127.0.0.1:8080/docs"
"WebUI PID $($ui.Id)  http://127.0.0.1:8501"
