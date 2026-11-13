$ErrorActionPreference = 'Stop'
$executionRoot = $PSScriptRoot
$projectRoot = Split-Path (Split-Path $executionRoot -Parent) -Parent
$taskName = 'Trajectory-EndpointDiscreteUtility-T-20261101'
if (Test-Path -LiteralPath (Join-Path $executionRoot 'WORKER_START.json')) { throw 'Worker-start evidence exists; no duplicate start.' }
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Current task exists; refuse replacement or restart.' }
$python = Join-Path $projectRoot '.venv-governance\Scripts\python.exe'
$worker = Join-Path $executionRoot 'worker_discrete.py'
$action = New-ScheduledTaskAction -Execute $python -Argument ('-B "' + $worker + '"') -WorkingDirectory $projectRoot
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$definition = New-ScheduledTask -Action $action -Principal $principal -Settings $settings
Register-ScheduledTask -TaskName $taskName -InputObject $definition | Out-Null
[xml]$xml = Export-ScheduledTask -TaskName $taskName
$triggers = $xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count
if ($triggers -ne 0) { throw 'Not an on-demand-only zero-trigger task.' }
Start-ScheduledTask -TaskName $taskName
[PSCustomObject]@{ task=$taskName; action=$python; arguments=('-B "'+$worker+'"'); trigger_count=$triggers; dispatched_at=(Get-Date).ToString('o') } | ConvertTo-Json -Compress
