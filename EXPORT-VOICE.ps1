$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$ckpt = Get-ChildItem -Recurse -Path lightning_logs -Filter *.ckpt -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $ckpt) { throw "No checkpoint found - train first (START-TRAIN.ps1)" }

Write-Host "Exporting $($ckpt.FullName) -> piper_voice\mn_voice.onnx" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path piper_voice | Out-Null
.\venv\Scripts\python.exe -u -m piper.train.export_onnx --checkpoint $ckpt.FullName --output-file piper_voice\mn_voice.onnx
Copy-Item piper_data\voice_config.json piper_voice\mn_voice.onnx.json -Force
Write-Host "Done. Restart the server (START-SERVER.ps1) to serve the new voice." -ForegroundColor Green
