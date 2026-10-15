# PowerShell Launcher for VisionInspect AI
param (
    [switch]$Gradio
)

Set-Location -Path $PSScriptRoot

if ($Gradio) {
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host " Starting VisionInspect AI Gradio UI (:7860)" -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Cyan
    & "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\app.py"
} else {
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host " Starting VisionInspect AI React Enterprise App (:8000)" -ForegroundColor Green
    Write-Host " URL: http://127.0.0.1:8000" -ForegroundColor Yellow
    Write-Host " (Tip: Use -Gradio flag to launch Gradio on :7860)" -ForegroundColor DarkGray
    Write-Host "========================================================" -ForegroundColor Cyan
    & "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000
}
