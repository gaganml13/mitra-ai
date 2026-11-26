Set-Location -Path $PSScriptRoot

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "         STARTING MITRA AI COMPANION" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

function Test-PyImports([string]$py, [string]$mods) {
    & $py -c "import $mods" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

# Hub interpreter: the project .venv if it has the hub packages, else system python
$py = "python"
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if ((Test-Path $venvPy) -and (Test-PyImports $venvPy "flask, groq")) { $py = $venvPy }
if (-not (Test-PyImports $py "flask, flask_cors, requests")) {
    Write-Host "Missing hub packages. Run:  `"$py`" -m pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

function Test-Hub {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:5000/ping" -UseBasicParsing -TimeoutSec 1 | Out-Null
        return $true
    } catch { return $false }
}

if (Test-Hub) {
    # A hub is already running: reuse it. Starting a second one would only fail with "port in use".
    Write-Host "[1/3] Mitra hub already running on port 5000 - reusing it." -ForegroundColor Yellow
} else {
    Write-Host "[1/3] Starting Mitra hub on port 5000 ($py)..." -ForegroundColor Yellow
    # Own window, independent of this script: the dashboard keeps working even if the
    # voice companion below exits. Close the "Mitra hub" window to stop the hub.
    $serverProcess = Start-Process $py -ArgumentList "mitra_server.py" -PassThru -WindowStyle Minimized

    Write-Host "       Waiting for the hub to answer /ping ..." -ForegroundColor Yellow
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        if ($serverProcess.HasExited) { break }
        if (Test-Hub) { $ready = $true; break }
        Start-Sleep -Milliseconds 750
    }
    if (-not $ready) {
        Write-Host "The hub did not start. Run '$py mitra_server.py' directly to see the error." -ForegroundColor Red
        exit 1
    }
}

Write-Host "[2/3] Opening dashboard: http://localhost:5000" -ForegroundColor Yellow
Start-Process "http://localhost:5000"

Write-Host "[3/3] Launching Mitra AI Voice Companion..." -ForegroundColor Green
Write-Host ""
Write-Host "Tips:" -ForegroundColor Magenta
Write-Host " - Say 'Mitra' to wake up!" -ForegroundColor Magenta
Write-Host " - Say 'Mitra stop' to interrupt mid-speech!" -ForegroundColor Magenta
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Companion needs whisper/sounddevice; use whichever interpreter has them
$cpy = $py
if (-not (Test-PyImports $cpy "whisper, sounddevice")) { $cpy = "python" }

& $cpy mitra_ultimate.py

# The hub is deliberately left running so the open dashboard tab keeps working.
Write-Host ""
Write-Host "Voice companion stopped. The dashboard is still live at http://localhost:5000" -ForegroundColor Cyan
Write-Host "(close the minimized Mitra hub window to stop it)." -ForegroundColor Cyan
