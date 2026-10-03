# NOTE: Continue (not Stop) — Windows PowerShell 5.1 turns harmless native
# stderr warnings (e.g. uvicorn/torch warnings) into terminating errors under Stop.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# CPU mode = zero VRAM fights with training (recommended while training runs).
# After training finishes, delete these two lines to let STT use the GPU (faster).
$env:MN_TTS_DEVICE = "cpu"
$env:MN_STT_DEVICE = "cpu"

Write-Host "Starting Mongolian Voice API on http://127.0.0.1:8771 ..." -ForegroundColor Cyan
& .\venv\Scripts\python.exe server.py
