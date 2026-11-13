$ErrorActionPreference = 'Stop'
$executionRoot = $PSScriptRoot
$projectRoot = Split-Path (Split-Path $executionRoot -Parent) -Parent
$taskName = 'Trajectory-E2E-DGeo-Pilots-20261102'
if (Test-Path -LiteralPath (Join-Path $executionRoot 'WORKER_START.json')) { throw 'Existing worker evidence; do not duplicate.' }
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Current task exists; do not replace.' }
$python = Join-Path $projectRoot '.venv-governance\Scripts\python.exe'
$worker = Join-Path $executionRoot 'worker.py'
$action = New-ScheduledTaskAction -Execute $python -Argument ('-B "' + $worker + '"') -WorkingDirectory $projectRoot
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$definition = New-ScheduledTask -Action $action -Principal $principal -Settings $settings
Register-ScheduledTask -TaskName $taskName -InputObject $definition | Out-Null
[xml]$xml = Export-ScheduledTask -TaskName $taskName
$triggers = $xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count
if ($triggers -ne 0) { throw 'Task must have zero triggers.' }
Start-ScheduledTask -TaskName $taskName
[PSCustomObject]@{task=$taskName;trigger_count=$triggers;dispatched_at=(Get-Date).ToString('o')} | ConvertTo-Json -Compress
