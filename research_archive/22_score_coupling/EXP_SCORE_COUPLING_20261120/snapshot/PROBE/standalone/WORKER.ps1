$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $root)
$phaseOne = Join-Path (Split-Path -Parent $root) 'EXP_SCORE_COUPLING_DIAG_EXEC_20261120'
$python = Join-Path $workspace '.venv-governance/Scripts/python.exe'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$attempt = Join-Path $PSScriptRoot ('attempts/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $attempt | Out-Null
@{at=(Get-Date -Format o); worker_pid=$PID; attempt=$attempt} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ACTIVE.json') -Encoding UTF8
while (-not (Test-Path -LiteralPath (Join-Path $phaseOne 'DONE.json'))) {
    $diag = Get-ScheduledTask -TaskName 'Trajectory-Score-Coupling-Diag-20261120' -ErrorAction SilentlyContinue
    if ($diag.State -ne 'Running') {
        @{at=(Get-Date -Format o); status='PHASE_ONE_INCOMPLETE'; exit_code=1; attempt=$attempt} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'WORKER_EXIT.json') -Encoding UTF8
        exit 1
    }
    Start-Sleep -Seconds 3
}
$ErrorActionPreference = 'Continue'
& $python -X utf8 -B (Join-Path $root 'run_probe.py') 1> (Join-Path $attempt 'stdout.log') 2> (Join-Path $attempt 'stderr.log')
$code = $LASTEXITCODE
$stage = if (Test-Path -LiteralPath (Join-Path $root 'CURRENT.json')) { (Get-Content -LiteralPath (Join-Path $root 'CURRENT.json') -Raw | ConvertFrom-Json).phase }
$state = if ($code -ne 0) { 'FAILURE' } elseif (Test-Path -LiteralPath (Join-Path $root 'DONE.json')) { 'SUCCESS' } else { $stage }
@{at=(Get-Date -Format o); status=$state; exit_code=$code; attempt=$attempt} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'WORKER_EXIT.json') -Encoding UTF8
exit $code
