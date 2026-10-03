# Runs detached until 05:00, then:
#   1) stops the TTS training process (last periodic checkpoint stays on disk)
#   2) exports the newest checkpoint to piper_voice\mn_voice.onnx
#   3) restarts the API server with GPU STT (training done -> VRAM free)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "run_till_5am.log"

Add-Content $log "waiting until 05:00 local..."
while ((Get-Date) -lt (Get-Date "05:00")) { Start-Sleep 60 }
Add-Content $log "05:00 reached - stopping training"

$tp = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'piper\.train' }
if ($tp) {
    taskkill /PID $tp.ProcessId /T /F
    Add-Content $log "killed trainer pid $($tp.ProcessId)"
} else {
    Add-Content $log "no trainer process found (already stopped?)"
}
Start-Sleep 10

powershell -NoProfile -ExecutionPolicy Bypass -File EXPORT-VOICE.ps1 *>> $log

$sp = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'server\.py' }
if ($sp) {
    taskkill /PID $sp.ProcessId /T /F
    Add-Content $log "killed old server pid $($sp.ProcessId)"
    Start-Sleep 5
}
$env:MN_TTS_DEVICE = "auto"
$env:MN_STT_DEVICE = "auto"
Start-Process -FilePath ".\venv\Scripts\python.exe" -ArgumentList "server.py" `
    -WorkingDirectory $PSScriptRoot -WindowStyle Minimized `
    -RedirectStandardOutput server.log -RedirectStandardError server.err.log
Add-Content $log "server restarted with auto device (GPU STT) - done"
