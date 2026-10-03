# START-ASSISTANT.ps1 - launch the live Mongolian voice assistant.
#
# ASCII-ONLY FILE ON PURPOSE: Windows PowerShell 5.1 parses a BOM-less .ps1 as
# ANSI (cp1252 on this box), so a Cyrillic literal in here renders as mojibake.
# Every Mongolian string - including the example commands - is printed by
# assistant.py, which runs with PYTHONIOENCODING=utf-8.
#
# NOTE: Continue (not Stop) - Windows PowerShell 5.1 turns harmless native
# stderr warnings (e.g. uvicorn/torch warnings) into terminating errors under Stop.
param(
    [switch]$NoServer
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# Cyrillic in the console needs UTF-8 on both ends: the child writes UTF-8
# bytes, the console must be told to read them as UTF-8.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$serverScript = Join-Path $PSScriptRoot "START-SERVER.ps1"
$assistantScript = Join-Path $PSScriptRoot "assistant.py"
$waitSeconds = 180

function Stop-With($lines) {
    $lines | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    exit 1
}

function Get-VoiceHealth {
    param([string]$Url, [int]$TimeoutSec = 3)
    try {
        $h = Invoke-RestMethod -Uri $Url -TimeoutSec $TimeoutSec -ErrorAction Stop
        if ($h -and $h.status -eq "ok") { return $h }
    } catch { }
    return $null
}

if (-not (Test-Path $python)) {
    Stop-With @(
        "ERROR: venv Python not found -> $python",
        "  Monggol: Virtual envegiin Python oldson-gui. Zaarah:",
        "           cd `"$PSScriptRoot`"",
        "           python -m venv venv",
        "           .\venv\Scripts\python.exe -m pip install -r requirements.txt")
}
if (-not (Test-Path $assistantScript)) {
    Stop-With @(
        "ERROR: assistant.py not found -> $assistantScript",
        "  Monggol: assistant.py oldson-gui - tu first uileljin CODE geesig.",
        "  Run the assistant agent first, then run this launcher again.")
}

# Keep the health probe honest when the caller overrides the server URL.
$serverUrl = "http://127.0.0.1:8771"
for ($i = 0; $i -lt ($args.Count - 1); $i++) {
    if ($args[$i] -eq "--server") { $serverUrl = $args[$i + 1] }
}
$serverUrl = $serverUrl.TrimEnd("/")
$healthUrl = "$serverUrl/health"

Write-Host ""
Write-Host "================================================================" -ForegroundColor DarkCyan
Write-Host " Mongolian Voice Assistant - mic -> STT -> local action -> spoken reply" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------" -ForegroundColor DarkCyan
Write-Host " Speak Mongolian; it transcribes, runs the action and answers aloud." -ForegroundColor Gray
Write-Host " STT + TTS live in the voice server, this script only starts both." -ForegroundColor Gray
Write-Host "----------------------------------------------------------------" -ForegroundColor DarkCyan
Write-Host " Say (Latin spelling here; the Cyrillic forms come from assistant.py):" -ForegroundColor Gray
Write-Host '   "Bloknot nee"  "Excel nee"  "Word-iig haa"  "Khotolgoch nee"' -ForegroundColor White
Write-Host '   "Taniltsuulga nee"  "Tusalamzh" (what can I say)' -ForegroundColor White
Write-Host " Ctrl+C stops.  -NoServer = never start the server.  Every other" -ForegroundColor DarkGray
Write-Host " argument goes straight to assistant.py (--dry-run --no-speak --verbose ...)." -ForegroundColor DarkGray
Write-Host "================================================================" -ForegroundColor DarkCyan

$health = Get-VoiceHealth $healthUrl 3
if ($health) {
    Write-Host " Voice server: already up at $serverUrl" -ForegroundColor Green
} elseif ($NoServer) {
    Stop-With @(
        "ERROR: no voice server at $healthUrl and -NoServer was given.",
        "  Monggol: Server ajarlaagu. Eesig awaaj START-SERVER.ps1 ajiluu.",
        "  Start START-SERVER.ps1 in another window, then run this again.")
} else {
    if (-not (Test-Path $serverScript)) {
        Stop-With @(
            "ERROR: START-SERVER.ps1 not found -> $serverScript",
            "  Monggol: Server ajar script alga bolloo - Server ajaralgui.",
            "  Start the voice server yourself, then run this again.")
    }
    Write-Host " Voice server not running - starting it in a new window ..." -ForegroundColor Yellow
    Start-Process -FilePath "powershell" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$serverScript`"" `
        -WorkingDirectory $PSScriptRoot -WindowStyle Normal | Out-Null
    Write-Host " waiting for the voice server (loading Whisper medium) ..." -ForegroundColor DarkGray
    $deadline = (Get-Date).AddSeconds($waitSeconds)
    $elapsed = 0
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        $elapsed += 3
        $health = Get-VoiceHealth $healthUrl 3
        if ($health) { break }
        if ($elapsed % 15 -eq 0) {
            Write-Host (" waiting for the voice server (loading Whisper medium)... {0}s" -f $elapsed) -ForegroundColor DarkGray
        }
    }
    if (-not $health) {
        Stop-With @(
            "ERROR: voice server did not answer /health after $waitSeconds s.",
            "  Monggol: Server $waitSeconds hor too kharsan-gui.",
            "  Watch the other window - check the log for the real reason.")
    }
    Write-Host " Voice server is up at $serverUrl" -ForegroundColor Green
}

$sttDev = $health.stt.device
Write-Host (" TTS {0} on {1} | STT {2} on {3} | free VRAM {4} GB" -f `
    $health.tts.backend, $health.tts.device, $health.stt.model, $sttDev, $health.gpu_free_gb) -ForegroundColor DarkGray
if ($sttDev -ne "cuda") {
    Write-Host " Note: STT is on CPU. START-SERVER.ps1 pins MN_STT_DEVICE=cpu;" -ForegroundColor Yellow
    Write-Host "       set it to cuda there for GPU fp16 once training is done." -ForegroundColor Yellow
}

Write-Host ""
& $python -u assistant.py @args
exit $LASTEXITCODE
