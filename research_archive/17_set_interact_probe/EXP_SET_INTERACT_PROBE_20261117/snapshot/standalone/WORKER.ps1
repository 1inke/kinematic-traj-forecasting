$ErrorActionPreference='Stop'
$experimentRoot=Split-Path -Parent $PSScriptRoot
$workspaceRoot=Split-Path -Parent (Split-Path -Parent $experimentRoot)
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:PYTHONDONTWRITEBYTECODE='1'
Set-Location -LiteralPath $experimentRoot
& (Join-Path $workspaceRoot '.venv-governance/Scripts/python.exe') -X utf8 -B (Join-Path $experimentRoot 'code/run_probe.py') 1> (Join-Path $PSScriptRoot 'stdout.log') 2> (Join-Path $PSScriptRoot 'stderr.log')
$code=$LASTEXITCODE
@{at=(Get-Date -Format o);status=$(if ($code -eq 0) {'SUCCESS'} else {'FAILURE'});exit_code=$code} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'WORKER_EXIT.json') -Encoding UTF8
exit $code
