@echo off
cd /d "%~dp0"
echo ========================================================
echo  Starting VisionInspect AI Web Dashboard in .venv
echo ========================================================
call .venv\Scripts\activate.bat
python app.py
pause
