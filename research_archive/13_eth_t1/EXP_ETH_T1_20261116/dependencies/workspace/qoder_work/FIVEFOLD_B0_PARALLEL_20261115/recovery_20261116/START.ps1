param([switch]$Status)
$ErrorActionPreference = 'Stop'
$recoveryRoot = $PSScriptRoot
$parallelRoot = Split-Path $recoveryRoot -Parent
$config = Get-Content -LiteralPath (Join-Path $parallelRoot 'PARALLEL_CONFIG.json') -Raw | ConvertFrom-Json
$taskName = 'Trajectory-Fivefold-B0-Recovery-Parallel-20261116'
if ($Status) {
    Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Select-Object TaskName,State
    foreach ($name in @('CURRENT.json','WORKER_EXIT.json')) {
        $path = Join-Path $parallelRoot ('standalone\' + $name)
        if (Test-Path -LiteralPath $path) { Get-Content -LiteralPath $path -Raw }
    }
    exit
}
$proof = Get-Content -LiteralPath (Join-Path $recoveryRoot 'PREFLIGHT_PASS.json') -Raw | ConvertFrom-Json
if ($proof.status -ne 'PASS' -or $proof.max_concurrency -ne 2) { throw 'Verified two-slot recovery is required.' }
foreach ($otherName in @('Trajectory-Fivefold-B0-Parallel-20261115','Trajectory-Fivefold-B0-Single-20261115','Trajectory-Fivefold-B0-Standalone-20261115')) {
    $other = Get-ScheduledTask -TaskName $otherName -ErrorAction SilentlyContinue
    if ($other.State -eq 'Running') { throw ('Another controller still runs: ' + $otherName) }
    if ($other -and $other.State -ne 'Disabled') { Disable-ScheduledTask -TaskName $otherName | Out-Null }
}
$arguments = '-B "' + (Join-Path $recoveryRoot 'worker_recovery.py') + '"'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $existing) {
    $action = New-ScheduledTaskAction -Execute $config.python -Argument $arguments -WorkingDirectory $config.workspace
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
    Register-ScheduledTask -TaskName $taskName -InputObject (New-ScheduledTask -Action $action -Principal $principal -Settings $settings) | Out-Null
} elseif ($existing.Actions.Execute -ne $config.python -or $existing.Actions.Arguments -ne $arguments) {
    throw 'Existing recovery task action differs.'
}
[xml]$taskXml = Export-ScheduledTask -TaskName $taskName
if ($taskXml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count -ne 0) { throw 'Unexpected automatic triggers.' }
$current = Get-ScheduledTask -TaskName $taskName
if ($current.State -eq 'Running') { Write-Output 'Already running; no duplicate started.'; exit }
if ($current.State -eq 'Disabled') { Enable-ScheduledTask -TaskName $taskName | Out-Null }
Start-ScheduledTask -TaskName $taskName
[PSCustomObject]@{task=$taskName;dispatched_at=(Get-Date).ToString('o');concurrency=2;Codex_dependency=$false} | ConvertTo-Json

