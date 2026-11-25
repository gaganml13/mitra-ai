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

Write-Host "[1/3] Starting Mitra hub on port 5000 ($py)..." -ForegroundColor Yellow
$serverProcess = Start-Process $py -ArgumentList "mitra_server.py" -PassThru -WindowStyle Minimized

Write-Host "       Waiting for the hub to answer /ping ..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 40; $i++) {
    if ($serverProcess.HasExited) { break }
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:5000/ping" -UseBasicParsing -TimeoutSec 1 | Out-Null
        $ready = $true; break
    } catch { Start-Sleep -Milliseconds 750 }
}
if (-not $ready) {
    Write-Host "The hub did not start. Run '$py mitra_server.py' directly to see the error." -ForegroundColor Red
    exit 1
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

try {
    & $cpy mitra_ultimate.py
} finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
