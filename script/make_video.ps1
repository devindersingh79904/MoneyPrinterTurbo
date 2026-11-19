# Make one video fully locally: asks for a topic, the local LLM writes the script and keywords,
# SDXL draws matching images, Kokoro speaks, Whisper adds subtitles, NVENC encodes.
# Usage:  script\Make-Video.bat              (asks for the topic)
#         script\Make-Video.bat "your topic"
param([string]$Subject = "")
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Test-Url($url) {
    try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 3 | Out-Null; return $true } catch { return $false }
}

if (-not (Test-Url "http://127.0.0.1:8080/docs") -or -not (Test-Url "http://127.0.0.1:7860/v1/models")) {
    Write-Host "Services are not running - starting them first..." -ForegroundColor Yellow
    & "$PSScriptRoot\start.ps1"
}

if (-not $Subject) { $Subject = Read-Host "Video topic (e.g. How to check if a number is a power of 2)" }
if (-not $Subject.Trim()) { Write-Host "No topic given."; exit 1 }

$job = @{
    video_subject             = $Subject
    video_script              = ""        # empty = local LLM writes it
    video_terms               = ""        # empty = local LLM picks the shots
    video_source              = "openai_image"
    match_materials_to_script = $true     # images follow the narration order
    video_aspect              = "9:16"
    video_language            = "en-US"
    video_clip_duration       = 5
    video_count               = 1
    voice_name                = "kokoro:af_heart"
    voice_rate                = 1.0
    voice_volume              = 1.0
    subtitle_enabled          = $true
    bgm_type                  = "random"
    bgm_volume                = 0.2
} | ConvertTo-Json

$resp = Invoke-RestMethod -Method Post "http://127.0.0.1:8080/api/v1/videos" -ContentType "application/json" -Body $job
$task = $resp.data.task_id
Write-Host "Task $task started. This takes a few minutes..." -ForegroundColor Cyan

$start = Get-Date
while ($true) {
    Start-Sleep 5
    $d = (Invoke-RestMethod "http://127.0.0.1:8080/api/v1/tasks/$task").data
    $elapsed = [int]((Get-Date) - $start).TotalSeconds
    Write-Host -NoNewline ("`r  progress {0,3}%   {1}s   " -f $d.progress, $elapsed)
    if ($d.state -eq 1) { break }
    if ($d.state -eq -1) {
        Write-Host ""
        Write-Host "FAILED at stage '$($d.failed_stage)': $($d.error)" -ForegroundColor Red
        Write-Host "Logs: storage\logs\api.err.log and storage\logs\images.out.log"
        exit 1
    }
    if ($elapsed -gt 1800) { Write-Host ""; Write-Host "Timed out after 30 min." -ForegroundColor Red; exit 1 }
}

$video = Join-Path $root ("storage" + ($d.videos[0] -replace "/", "\"))
Write-Host ""
Write-Host "Script:" -ForegroundColor Cyan
Write-Host $d.script
Write-Host ""
Write-Host "Done: $video" -ForegroundColor Green
Start-Process $video
