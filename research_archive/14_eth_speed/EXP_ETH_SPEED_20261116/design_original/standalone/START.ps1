param([ValidateSet('prepare','diag','aug')][string]$Phase = 'diag')
$ErrorActionPreference = 'Stop'
$packageDir = Split-Path -Parent $PSScriptRoot
$cfg = Get-Content -LiteralPath (Join-Path $packageDir 'CONFIG.local.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskName = 'Trajectory-ETH-Speed-20261116'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { throw 'This experiment is already running.' }
New-Item -ItemType Directory -Path $cfg.output_dir -Force | Out-Null
$pausePath = Join-Path $cfg.output_dir 'PAUSE.flag'
if (Test-Path -LiteralPath $pausePath) { Remove-Item -LiteralPath $pausePath }
$worker = Join-Path $PSScriptRoot 'worker.ps1'
$actionArgs = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $worker + '" -Phase ' + $Phase
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $actionArgs -WorkingDirectory $packageDir
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Description 'Manual ETH speed exploration; no automatic follow-up training' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Output "Started $Phase in Task Scheduler. It runs independently of Codex while Windows remains awake and logged in."
