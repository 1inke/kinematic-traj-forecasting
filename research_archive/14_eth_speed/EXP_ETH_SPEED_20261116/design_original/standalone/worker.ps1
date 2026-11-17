param([ValidateSet('prepare','diag','aug')][string]$Phase)
$ErrorActionPreference = 'Stop'
$packageDir = Split-Path -Parent $PSScriptRoot
$cfg = Get-Content -LiteralPath (Join-Path $packageDir 'CONFIG.local.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$logDir = Join-Path $cfg.output_dir ('logs\' + (Get-Date -Format 'yyyyMMdd_HHmmss_ffff'))
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$script = Join-Path $packageDir 'code\run_eth_speed.py'
$argsText = '-X utf8 -B "' + $script + '" ' + $Phase
$child = Start-Process -FilePath $cfg.python -ArgumentList $argsText -WorkingDirectory $packageDir -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput (Join-Path $logDir 'stdout.log') -RedirectStandardError (Join-Path $logDir 'stderr.log')
exit $child.ExitCode
