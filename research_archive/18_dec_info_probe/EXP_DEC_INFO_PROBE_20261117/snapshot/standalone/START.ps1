$ErrorActionPreference='Stop'
$taskName='Trajectory-Dec-Info-Probe-20261117'
$root=Split-Path -Parent $PSScriptRoot
if (Test-Path -LiteralPath (Join-Path $root 'DONE.json')) {
    Write-Output 'Scientific output already sealed. Check STATUS.ps1.'
    exit 0
}
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') {Write-Output 'Already running';exit 0}
if (-not $existing) {
    $workerPath=Join-Path $PSScriptRoot 'WORKER.ps1'
    $action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $workerPath + '"') -WorkingDirectory $PSScriptRoot
    $principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 4) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'C0-only frozen decoder information probe, P0 gate and at most 18 runs.' | Out-Null
} elseif ($existing.Actions.Arguments -notlike "*$PSScriptRoot*") {
    throw 'Existing task points to a different worker'
}
$marker=Join-Path $root 'PAUSE.request'
if (Test-Path -LiteralPath $marker) {Remove-Item -LiteralPath $marker}
Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'TASK.xml') -Encoding UTF8
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
