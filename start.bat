@echo off
title Gold Day Trade Assistant (IQ Option Forex)
chcp 65001 > nul
cd /d "%~dp0"

echo ========================================================
echo   🌟 Gold Day Trade Assistant (XAU/USD - IQ Option) 🌟
echo ========================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ไม่พบ Python ในเครื่อง กรุณาติดตั้ง Python 3.10 ขึ้นไป
    pause
    exit /b 1
)

echo [1/3] กำลังตรวจสอบความพร้อมของระบบ...
python -m pip install -r requirements.txt --quiet

echo [2/3] กำลังเปิดเว็บเบราว์เซอร์ไปที่ http://localhost:8000 ...
start "" http://localhost:8000

echo [3/3] กำลังเริ่มต้นระบบวิเคราะห์และแจ้งเตือน...
echo [INFO] ระบบจะวิเคราะห์อัตโนมัติรอบค่ำ 19:00 - 22:00 น. (1M Sniper) และ 15:00 น.
echo [INFO] กด Ctrl+C เพื่อหยุดการทำงาน
echo ========================================================
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
