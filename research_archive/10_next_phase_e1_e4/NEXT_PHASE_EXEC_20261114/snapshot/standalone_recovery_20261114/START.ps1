param([switch]$Status)
$ErrorActionPreference = 'Stop'
$recoveryRoot = $PSScriptRoot
$projectRoot = Split-Path (Split-Path (Split-Path $recoveryRoot -Parent) -Parent) -Parent
$standaloneTaskName = 'Trajectory-NextPhase-E1E4-Standalone-20261114'
if ($Status) {
    Get-ScheduledTask -TaskName $standaloneTaskName -ErrorAction SilentlyContinue | Select-Object TaskName,State
    $statusPath = Join-Path $recoveryRoot 'CURRENT.json'
    if (Test-Path -LiteralPath $statusPath) { Get-Content -LiteralPath $statusPath -Raw }
    exit
}
$preflight = Get-Content -LiteralPath (Join-Path $recoveryRoot 'PREFLIGHT_RECOVERY.json') -Raw | ConvertFrom-Json
if ($preflight.status -ne 'PASS') { throw 'Actual recovery preflight required.' }
$python = Join-Path $projectRoot '.venv-governance\Scripts\python.exe'
$worker = Join-Path $recoveryRoot 'worker_standalone.py'
$arguments = '-B "' + $worker + '"'
$existing = Get-ScheduledTask -TaskName $standaloneTaskName -ErrorAction SilentlyContinue
if (-not $existing) {
    $action = New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $projectRoot
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $definition = New-ScheduledTask -Action $action -Principal $principal -Settings $settings
    Register-ScheduledTask -TaskName $standaloneTaskName -InputObject $definition | Out-Null
} elseif ($existing.Actions.Execute -ne $python -or $existing.Actions.Arguments -ne $arguments) {
    throw 'Existing task action differs; refusing to replace.'
}
[xml]$taskXml = Export-ScheduledTask -TaskName $standaloneTaskName
$triggerCount = $taskXml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count
if ($triggerCount -ne 0) { throw 'Standalone task must have no automatic triggers.' }
if ((Get-ScheduledTask -TaskName $standaloneTaskName).State -eq 'Running') { Write-Output 'Already running; no duplicate started.'; exit }
Start-ScheduledTask -TaskName $standaloneTaskName
[PSCustomObject]@{task=$standaloneTaskName;dispatched_at=(Get-Date).ToString('o');trigger_count=$triggerCount;Codex_dependency=$false} | ConvertTo-Json
