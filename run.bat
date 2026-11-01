@echo off
chcp 65001 >nul
title PuijaiRiskNLP Web Server
echo ========================================================
echo  PuijaiRiskNLP - ระบบตรวจจับจุดเสี่ยงแชทบอทปุยใจ
echo  กำลังเริ่มเซิร์ฟเวอร์ที่ http://127.0.0.1:8000/ ...
echo ========================================================
cd /d "%~dp0"
start http://127.0.0.1:8000/
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause

