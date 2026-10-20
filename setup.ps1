$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
$env:npm_config_cache = Join-Path $PSScriptRoot '.cache\npm'
if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment.' }
}
uv pip install --python .venv\Scripts\python.exe -r requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
& .venv\Scripts\python.exe scripts\prepare_models.py
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare CodeBERTa weights.' }
Push-Location frontend
try {
    npm ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
if (!(Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
Write-Host 'Ready. Run .\start.ps1' -ForegroundColor Green
