# PowerShell Launcher for VisionInspect AI Web Dashboard
Set-Location -Path $PSScriptRoot
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting VisionInspect AI Web Dashboard (.venv)" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\app.py"
