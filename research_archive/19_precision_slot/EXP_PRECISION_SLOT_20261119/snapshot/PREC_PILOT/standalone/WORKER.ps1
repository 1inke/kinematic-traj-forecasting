$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $root)
$python = Join-Path $workspace '.venv-governance/Scripts/python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$attempt = Join-Path $PSScriptRoot ('attempts/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $attempt | Out-Null
@{ at = (Get-Date -Format o); pid = $PID; attempt = $attempt } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ACTIVE.json') -Encoding UTF8
$ErrorActionPreference = 'Continue'
& $python -B -X utf8 (Join-Path $root 'launch_pilot.py') 1> (Join-Path $attempt 'stdout.log') 2> (Join-Path $attempt 'stderr.log')
$code = $LASTEXITCODE
$status = if ($code -ne 0) { 'FAILURE' } elseif (Test-Path -LiteralPath (Join-Path $root 'DONE.json')) { 'SUCCESS' } else { 'PAUSED' }
$record = @{ at = (Get-Date -Format o); status = $status; exit_code = $code; attempt = $attempt } | ConvertTo-Json
$record | Set-Content -LiteralPath (Join-Path $attempt 'WORKER_EXIT.json') -Encoding UTF8
$record | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'WORKER_EXIT.json') -Encoding UTF8
exit $code
