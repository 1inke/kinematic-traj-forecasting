param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$projectRoot = Split-Path (Split-Path $experimentRoot -Parent) -Parent
$python = Join-Path $projectRoot '.venv-governance\Scripts\python.exe'
$entry = Join-Path $experimentRoot 'run_stage_a.py'
if (-not $Execute) {
    Write-Output 'DRY-RUN only. Latest human instruction: write scripts, DO NOT START.'
    & $python -B $entry
} else {
    # A switch alone cannot bypass the saved human instruction.
    & $python -B $entry --execute
}
exit $LASTEXITCODE

