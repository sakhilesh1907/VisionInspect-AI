# PowerShell Launcher for VisionInspect AI React Enterprise Dashboard
Set-Location -Path $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting VisionInspect AI React Enterprise App (:8000)" -ForegroundColor Green
Write-Host " URL: http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan

& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000
