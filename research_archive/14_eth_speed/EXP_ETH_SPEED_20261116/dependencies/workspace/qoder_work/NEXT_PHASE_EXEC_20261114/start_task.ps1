$ErrorActionPreference = 'Stop'
$phaseRoot = $PSScriptRoot
$projectRoot = Split-Path (Split-Path $phaseRoot -Parent) -Parent
$phaseTaskName = 'Trajectory-NextPhase-E1E4-20261114'
if (Test-Path -LiteralPath (Join-Path $phaseRoot 'WORKER_START.json')) { throw 'Existing worker. Do not duplicate.' }
if (Get-ScheduledTask -TaskName $phaseTaskName -ErrorAction SilentlyContinue) { throw 'Existing task. Do not replace.' }
$phasePython = Join-Path $projectRoot '.venv-governance\Scripts\python.exe'
$phaseWorker = Join-Path $phaseRoot 'worker.py'
$action = New-ScheduledTaskAction -Execute $phasePython -Argument ('-B "' + $phaseWorker + '"') -WorkingDirectory $projectRoot
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$definition = New-ScheduledTask -Action $action -Principal $principal -Settings $settings
Register-ScheduledTask -TaskName $phaseTaskName -InputObject $definition | Out-Null
[xml]$phaseXml = Export-ScheduledTask -TaskName $phaseTaskName
$phaseTriggerCount = $phaseXml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count
if ($phaseTriggerCount -ne 0) { throw 'Only on-demand zero-trigger task is allowed.' }
Start-ScheduledTask -TaskName $phaseTaskName
[PSCustomObject]@{task=$phaseTaskName;trigger_count=$phaseTriggerCount;dispatched_at=(Get-Date).ToString('o')} | ConvertTo-Json
