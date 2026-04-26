@echo off
cd /d "%~dp0"
set "DEEPFACE_HOME=%CD%"
set "PYTHONIOENCODING=utf-8"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "src\emotion_webcam_opencv.py" %*
) else (
    python "src\emotion_webcam_opencv.py" %*
)
