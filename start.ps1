param([ValidateRange(1,65535)][int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonExe)) { throw 'Run setup.ps1 first.' }
if (!(Test-Path -LiteralPath 'frontend\dist\index.html')) { throw 'Build the frontend with: cd frontend; npm run build' }
$env:PYTHONUTF8 = '1'
Write-Host "HybridSense: http://127.0.0.1:$Port" -ForegroundColor Green
& $pythonExe -m uvicorn backend.app:app --host 127.0.0.1 --port $Port
