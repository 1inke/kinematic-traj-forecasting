param([switch]$Status)
$ErrorActionPreference = 'Stop'
$studyRoot = Split-Path $PSScriptRoot -Parent
$config = Get-Content -LiteralPath (Join-Path $studyRoot 'STUDY_CONFIG.json') -Raw | ConvertFrom-Json
$taskName = 'Trajectory-Fivefold-B0-Standalone-20261115'
if ($Status) {
    Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Select-Object TaskName,State
    foreach ($name in @('CURRENT.json', 'WORKER_EXIT.json')) {
        $path = Join-Path $PSScriptRoot $name
        if (Test-Path -LiteralPath $path) { Get-Content -LiteralPath $path -Raw }
    }
    exit
}
$proof = Get-Content -LiteralPath (Join-Path $studyRoot 'preflight\PREFLIGHT_PASS.json') -Raw | ConvertFrom-Json
if ($proof.status -ne 'PASS') { throw 'Actual CUDA recovery preflight required.' }
$python = $config.python
$arguments = '-B "' + (Join-Path $PSScriptRoot 'worker.py') + '"'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $existing) {
    $action = New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $config.workspace
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $definition = New-ScheduledTask -Action $action -Principal $principal -Settings $settings
    Register-ScheduledTask -TaskName $taskName -InputObject $definition | Out-Null
} elseif ($existing.Actions.Execute -ne $python -or $existing.Actions.Arguments -ne $arguments) {
    throw 'Existing task action differs; refusing to replace.'
}
[xml]$taskXml = Export-ScheduledTask -TaskName $taskName
$triggerCount = $taskXml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count
if ($triggerCount -ne 0) { throw 'Unexpected automatic task triggers.' }
if ((Get-ScheduledTask -TaskName $taskName).State -eq 'Running') { Write-Output 'Already running; no duplicate started.'; exit }
Start-ScheduledTask -TaskName $taskName
[PSCustomObject]@{task=$taskName;dispatched_at=(Get-Date).ToString('o');trigger_count=0;Codex_dependency=$false} | ConvertTo-Json
