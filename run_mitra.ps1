Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "         STARTING MITRA AI COMPANION" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Starting Parent Dashboard Server (port 5000)..." -ForegroundColor Yellow
$serverProcess = Start-Process python -ArgumentList "mitra_server.py" -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 2

Write-Host "[2/3] Opening Parent Dashboard in browser..." -ForegroundColor Yellow
Start-Process "parent_dashboard.html"

Write-Host "[3/3] Launching Mitra AI Voice Companion..." -ForegroundColor Green
Write-Host ""
Write-Host "Tips:" -ForegroundColor Magenta
Write-Host " - Say 'Mitra' to wake up!" -ForegroundColor Magenta
Write-Host " - Say 'Mitra stop' to interrupt mid-speech!" -ForegroundColor Magenta
Write-Host " - Simulate crying or distress to trigger parent voice comfort!" -ForegroundColor Magenta
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

try {
    python mitra_ultimate.py
} finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
