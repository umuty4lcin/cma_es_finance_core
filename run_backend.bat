@echo off
REM FastAPI backend'i baslatir (port 8000)
cd /d %~dp0
venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
