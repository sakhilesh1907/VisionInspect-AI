@echo off
cd /d "%~dp0"
echo ========================================================
echo  Starting VisionInspect AI Gradio UI (:7860)
echo  URL: http://127.0.0.1:7860
echo ========================================================
call .venv\Scripts\activate.bat
python app.py
pause
