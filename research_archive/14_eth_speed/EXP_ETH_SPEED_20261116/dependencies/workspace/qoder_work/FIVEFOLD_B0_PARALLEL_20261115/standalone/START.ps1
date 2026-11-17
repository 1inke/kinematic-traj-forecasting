param([switch]$Status)
$ErrorActionPreference = 'Stop'
$parallelRoot = Split-Path $PSScriptRoot -Parent
$config = Get-Content -LiteralPath (Join-Path $parallelRoot 'PARALLEL_CONFIG.json') -Raw | ConvertFrom-Json
$taskName = 'Trajectory-Fivefold-B0-Parallel-20261115'
$oldTaskName = 'Trajectory-Fivefold-B0-Standalone-20261115'
if ($Status) {
    Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Select-Object TaskName,State
    foreach ($name in @('CURRENT.json','WORKER_EXIT.json')) {
        $path = Join-Path $PSScriptRoot $name
        if (Test-Path -LiteralPath $path) { Get-Content -LiteralPath $path -Raw }
    }
    exit
}
$proof = Get-Content -LiteralPath (Join-Path $parallelRoot 'preflight\PREFLIGHT_PASS.json') -Raw | ConvertFrom-Json
if ($proof.status -ne 'PASS') { throw 'Parallel real GPU checks required.' }
$oldTask = Get-ScheduledTask -TaskName $oldTaskName -ErrorAction SilentlyContinue
if ($oldTask.State -eq 'Running') { throw 'Serial controller is still running.' }
if ($oldTask -and $oldTask.State -ne 'Disabled') { Disable-ScheduledTask -TaskName $oldTaskName | Out-Null }
$python = $config.python
$arguments = '-B "' + (Join-Path $PSScriptRoot 'worker.py') + '"'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $existing) {
    $action = New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $config.workspace
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -InputObject (New-ScheduledTask -Action $action -Principal $principal -Settings $settings) | Out-Null
} elseif ($existing.Actions.Execute -ne $python -or $existing.Actions.Arguments -ne $arguments) {
    throw 'Existing parallel action differs.'
}
[xml]$taskXml = Export-ScheduledTask -TaskName $taskName
if ($taskXml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count -ne 0) { throw 'Unexpected automatic triggers.' }
if ((Get-ScheduledTask -TaskName $taskName).State -eq 'Running') { Write-Output 'Already running; no duplicate started.'; exit }
Start-ScheduledTask -TaskName $taskName
[PSCustomObject]@{task=$taskName;dispatched_at=(Get-Date).ToString('o');concurrency=2;Codex_dependency=$false} | ConvertTo-Json
