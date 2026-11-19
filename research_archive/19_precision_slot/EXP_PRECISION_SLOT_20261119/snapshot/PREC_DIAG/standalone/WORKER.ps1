$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $root)
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$attempt = Join-Path $PSScriptRoot ('attempts/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $attempt | Out-Null
@{ at = (Get-Date -Format o); worker_pid = $PID; attempt = $attempt } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ACTIVE.json') -Encoding UTF8
$ErrorActionPreference = 'Continue'
& (Join-Path $workspace '.venv-governance/Scripts/python.exe') -X utf8 -B (Join-Path $root 'run_prec_diag.py') 1> (Join-Path $attempt 'stdout.log') 2> (Join-Path $attempt 'stderr.log')
$code = $LASTEXITCODE
$status = if ($code -eq 0 -and (Test-Path -LiteralPath (Join-Path $root 'DONE.json'))) { 'SUCCESS' } else { 'FAILURE' }
@{ at = (Get-Date -Format o); status = $status; exit_code = $code; attempt = $attempt } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'WORKER_EXIT.json') -Encoding UTF8
exit $code
