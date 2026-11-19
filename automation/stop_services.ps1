# Stop the API (port 8080) and WebUI (port 8501) started by start_services.ps1.
# The venv python.exe is a launcher that starts a child process, so stop both the saved PIDs
# and whatever is listening on the two ports.
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "storage\services.pid"
$ids = @()
if (Test-Path $pidFile) { $ids += (Get-Content $pidFile) -split ' ' | Where-Object { $_ } }
$ids += Get-NetTCPConnection -State Listen -LocalPort 8080, 8501 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess
foreach ($id in ($ids | Sort-Object -Unique)) {
    try { Stop-Process -Id ([int]$id) -Force -ErrorAction Stop; "stopped PID $id" } catch { "PID $id was not running" }
}
if (Test-Path $pidFile) { Remove-Item $pidFile -Confirm:$false }
