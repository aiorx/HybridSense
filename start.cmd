@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment missing. Run setup.ps1 in PowerShell first.
    exit /b 1
)
if not exist "frontend\dist\index.html" (
    echo Frontend build missing. Run npm run build in the frontend directory.
    exit /b 1
)
set "HYBRIDSENSE_PORT=8000"
if not "%~1"=="" set "HYBRIDSENSE_PORT=%~1"
set "PYTHONUTF8=1"
echo HybridSense: http://127.0.0.1:%HYBRIDSENSE_PORT%
"%~dp0.venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port "%HYBRIDSENSE_PORT%"
exit /b %errorlevel%
