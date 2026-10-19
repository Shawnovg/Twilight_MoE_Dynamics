Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$taskPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create .venv and install requirements first; see README.md.'
}
& $taskPython -m jupyter lab notebooks/00_environment_check.ipynb
