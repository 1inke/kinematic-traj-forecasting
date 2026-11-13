$ErrorActionPreference = 'Stop'
$runDir = $PSScriptRoot
$workspaceDir = Split-Path (Split-Path $runDir -Parent) -Parent
Set-Location -LiteralPath $workspaceDir
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& "$workspaceDir/.venv-governance/Scripts/python.exe" -B -u "$runDir/worker_20261106.py" 1> "$runDir/worker_stdout_20261106.log" 2> "$runDir/worker_stderr_20261106.log"
exit $LASTEXITCODE
