$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$workspace=Split-Path -Parent (Split-Path -Parent $root)
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:PYTHONDONTWRITEBYTECODE='1'
Set-Location -LiteralPath $root
$logs=Join-Path $PSScriptRoot ('attempts/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $logs | Out-Null
@{at=(Get-Date -Format o);pid=$PID;attempt=$logs} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ACTIVE.json') -Encoding UTF8
$ErrorActionPreference='Continue'
& (Join-Path $workspace '.venv-governance/Scripts/python.exe') -X utf8 -B (Join-Path $root 'code/run_dec_probe.py') 1> (Join-Path $logs 'stdout.log') 2> (Join-Path $logs 'stderr.log')
$code=$LASTEXITCODE
$ErrorActionPreference='Stop'
$status=if ($code -ne 0) {'FAILURE'} elseif (Test-Path -LiteralPath (Join-Path $root 'DONE.json')) {'SUCCESS'} else {'PAUSED'}
$record=@{at=(Get-Date -Format o);status=$status;exit_code=$code;attempt=$logs} | ConvertTo-Json
$record | Set-Content -LiteralPath (Join-Path $logs 'WORKER_EXIT.json') -Encoding UTF8
$record | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'WORKER_EXIT.json') -Encoding UTF8
exit $code
