# Start everything MoneyPrinterTurbo needs locally, wait until it is ready, then open the WebUI in Chrome.
#   Ollama (script, 11434) - Kokoro (voice, 8880) - SDXL image server (footage, 7860)
#   API (8080) - WebUI (8501)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Test-Url($url) {
    try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 3 | Out-Null; return $true } catch { return $false }
}

Write-Host "== MoneyPrinterTurbo (local) ==" -ForegroundColor Cyan

# 1. Ollama
if (-not (Test-Url "http://127.0.0.1:11434/api/tags")) {
    Write-Host "Starting Ollama..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
}

# 2. If everything is already running, leave it alone (restarting would kill videos in progress).
#    Otherwise clear half-started leftovers and start the app services.
$appUrls = "http://127.0.0.1:8080/docs", "http://127.0.0.1:8501", "http://127.0.0.1:8880/v1/audio/voices", "http://127.0.0.1:7860/v1/models"
if (@($appUrls | Where-Object { -not (Test-Url $_) }).Count -eq 0) {
    Write-Host "Already running - not restarting." -ForegroundColor Green
} else {
    & "$root\automation\stop_services.ps1" | Out-Null
    & "$root\automation\start_services.ps1"
}

# 3. Wait for every service.
$services = [ordered]@{
    "Ollama (script)"      = "http://127.0.0.1:11434/api/tags"
    "Kokoro (voice)"       = "http://127.0.0.1:8880/v1/audio/voices"
    "SDXL images (footage)" = "http://127.0.0.1:7860/v1/models"
    "API"                  = "http://127.0.0.1:8080/docs"
    "WebUI"                = "http://127.0.0.1:8501"
}
foreach ($name in $services.Keys) {
    Write-Host -NoNewline ("Waiting for {0,-22}" -f $name)
    $ok = $false
    for ($i = 0; $i -lt 90; $i++) {
        if (Test-Url $services[$name]) { $ok = $true; break }
        Start-Sleep 2
    }
    if ($ok) { Write-Host " ready" -ForegroundColor Green }
    else { Write-Host " NOT READY - see storage\logs" -ForegroundColor Red }
}

# 4. Open the WebUI.
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (Test-Path $chrome) { Start-Process $chrome "http://127.0.0.1:8501" } else { Start-Process "http://127.0.0.1:8501" }

Write-Host ""
Write-Host "WebUI: http://127.0.0.1:8501   (make a video there, or run script\Make-Video.bat)" -ForegroundColor Cyan
Write-Host "Stop everything with script\Stop-MoneyPrinter.bat"
