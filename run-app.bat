@echo off
echo ==============================================
echo       Starting NanoRush Application Suite
echo ==============================================
echo.

echo [1/2] Launching PyTorch Backend (Port 8000)...
start "NanoRush Backend" cmd /c "cd nano-chat-web && ..\.venv\Scripts\python.exe -m uvicorn main:app --reload"

echo [2/2] Launching Next.js Frontend (Port 3000)...
start "NanoRush Frontend" cmd /c "cd nano-chat-web && npm run dev"

echo.
echo Both servers are spinning up in separate windows!
echo Please wait a few seconds, then open your browser to:
echo http://localhost:3000
echo.
echo You can close this script window.
pause
