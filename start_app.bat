@echo off
cd /d "%~dp0"
echo ========================================================
echo  Starting VisionInspect AI React Enterprise App (:8000)
echo  URL: http://127.0.0.1:8000
echo ========================================================
call .venv\Scripts\activate.bat
python -m uvicorn api:app --host 127.0.0.1 --port 8000
pause
