$experimentRoot=Split-Path -Parent $PSScriptRoot
Get-ScheduledTask -TaskName 'Trajectory-Set-Interact-Probe-20261117' | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName 'Trajectory-Set-Interact-Probe-20261117' | Select-Object LastRunTime,LastTaskResult
Get-Content -LiteralPath (Join-Path $experimentRoot 'CURRENT.json')
