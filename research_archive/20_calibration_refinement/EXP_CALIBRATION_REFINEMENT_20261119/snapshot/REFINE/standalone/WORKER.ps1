param([Parameter(Mandatory)][ValidateSet('Mech','Abl')][string]$Stage)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $root)
$python = Join-Path $workspace '.venv-governance/Scripts/python.exe'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$attempt = Join-Path $PSScriptRoot ('attempts/' + $Stage + '_' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $attempt | Out-Null
@{ at = (Get-Date -Format o); worker_pid = $PID; attempt = $attempt; stage = $Stage } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot "ACTIVE_$Stage.json") -Encoding UTF8
$ErrorActionPreference = 'Continue'
if ($Stage -eq 'Mech') {
    & $python -X utf8 -B (Join-Path $root 'run_mech.py') --device cuda:0 1> (Join-Path $attempt 'stdout.log') 2> (Join-Path $attempt 'stderr.log')
    $finished = Join-Path $root 'MECH_DIAG/DONE.json'
} else {
    & $python -X utf8 -B (Join-Path $root 'run_abl.py') 1> (Join-Path $attempt 'stdout.log') 2> (Join-Path $attempt 'stderr.log')
    $finished = Join-Path $root 'ABL/DONE.json'
}
$code = $LASTEXITCODE
$status = if ($code -eq 0 -and (Test-Path -LiteralPath $finished)) { 'SUCCESS' } elseif ($code -eq 0) { 'PAUSED' } else { 'FAILURE' }
@{ at = (Get-Date -Format o); status = $status; exit_code = $code; attempt = $attempt; stage = $Stage } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot "WORKER_EXIT_$Stage.json") -Encoding UTF8
exit $code
