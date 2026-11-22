$ErrorActionPreference = 'Stop'
$python = 'C:/coding_toobox/envs/astra2/python.exe'
$attempt = Join-Path $PSScriptRoot ('attempts/' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $attempt -Force | Out-Null
$entry = Join-Path $PSScriptRoot 'run_local.py'
$argument = '-u -B "' + $entry + '"'
$child = Start-Process -FilePath $python -ArgumentList $argument -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $attempt 'stdout.log') -RedirectStandardError (Join-Path $attempt 'stderr.log')
@{ pid = $child.Id; log_directory = $attempt; started_at = (Get-Date).ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ACTIVE.json') -Encoding UTF8
$child.WaitForExit()
$child.Refresh()
@{ pid = $child.Id; exit_code = $child.ExitCode; finished_at = (Get-Date).ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $attempt 'EXIT.json') -Encoding UTF8
exit $child.ExitCode
