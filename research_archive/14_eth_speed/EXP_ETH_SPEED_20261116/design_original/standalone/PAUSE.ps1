$ErrorActionPreference = 'Stop'
$packageDir = Split-Path -Parent $PSScriptRoot
$cfg = Get-Content -LiteralPath (Join-Path $packageDir 'CONFIG.local.json') -Raw -Encoding UTF8 | ConvertFrom-Json
New-Item -ItemType Directory -Path $cfg.output_dir -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $cfg.output_dir 'PAUSE.flag'), 'Pause at the next complete epoch/case boundary.')
Write-Output 'Pause requested. Wait for STATUS to show PAUSED and the scheduled task to stop before shutdown. START resumes from the latest complete epoch/case.'
