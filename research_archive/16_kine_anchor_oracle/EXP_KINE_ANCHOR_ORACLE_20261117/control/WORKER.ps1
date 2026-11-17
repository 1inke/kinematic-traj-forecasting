$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonExe = Join-Path $workspaceRoot '.venv-governance/Scripts/python.exe'
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES=''
Set-Location -LiteralPath $PSScriptRoot
& $pythonExe -X utf8 -B (Join-Path $PSScriptRoot 'run_job.py') 1> (Join-Path $PSScriptRoot 'WORKER.stdout.log') 2> (Join-Path $PSScriptRoot 'WORKER.stderr.log')
$workerCode=$LASTEXITCODE
@{at=(Get-Date -Format o);status=$(if ($workerCode -eq 0) {'SUCCESS'} else {'FAILURE'});exit_code=$workerCode} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'WORKER_EXIT.json') -Encoding UTF8
exit $workerCode
