$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $experimentRoot)
$pythonExe = Join-Path $workspaceRoot '.venv-governance/Scripts/python.exe'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = ''
Set-Location -LiteralPath $experimentRoot
& $pythonExe -X utf8 -B (Join-Path $experimentRoot 'code/run_decel_diag.py') 1> (Join-Path $PSScriptRoot 'stdout.log') 2> (Join-Path $PSScriptRoot 'stderr.log')
$workerCode = $LASTEXITCODE
@{at=(Get-Date -Format o);exit_code=$workerCode;status=$(if ($workerCode -eq 0) {'SUCCESS'} else {'FAILURE'})} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'WORKER_EXIT.json') -Encoding UTF8
exit $workerCode
