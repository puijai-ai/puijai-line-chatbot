@echo off
chcp 65001 >nul
title PuijaiRiskNLP Web Server
echo ========================================================
echo  PuijaiRiskNLP - ระบบตรวจจับจุดเสี่ยงแชทบอทปุยใจ
echo  กำลังเริ่มเซิร์ฟเวอร์ที่ http://127.0.0.1:8000/ ...
echo ========================================================
cd /d "%~dp0"

:: ตรวจสอบ Python ที่ใช้งานได้ (優先 .venv หากใช้งานได้ ถ้าไม่ได้ให้ใช้ system python)
set PYTHON_CMD=python
if exist ".\.venv\Scripts\python.exe" (
    ".\.venv\Scripts\python.exe" --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=.\.venv\Scripts\python.exe
    )
)

echo Using Python: %PYTHON_CMD%
start http://127.0.0.1:8000/
%PYTHON_CMD% -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
