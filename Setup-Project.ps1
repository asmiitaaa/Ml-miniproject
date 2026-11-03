$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10+ is required. Install Python and retry.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-dev.txt -r requirements-agent.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; check your internet connection.' }
& '.\.venv\Scripts\python.exe' -m traffic_eval doctor
Write-Host 'Setup complete. Run: .\Start-Project.ps1 -Mode Test'
