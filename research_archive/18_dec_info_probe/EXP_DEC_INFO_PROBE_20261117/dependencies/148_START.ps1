$ErrorActionPreference='Stop'
$taskName='Trajectory-Set-Interact-Probe-20261117'
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') {Write-Output 'Already running';exit 0}
if (-not $existing) {
    $workerPath=Join-Path $PSScriptRoot 'WORKER.ps1'
    $action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $workerPath + '"') -WorkingDirectory $PSScriptRoot
    $principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 4) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Frozen B0 cached-feature probe, Q1 gate then at most18 small refiner runs; no end-to-end training.' | Out-Null
}
$pausePath=Join-Path (Split-Path -Parent $PSScriptRoot) 'PAUSE.request'
if (Test-Path -LiteralPath $pausePath) {Remove-Item -LiteralPath $pausePath}
Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'TASK.xml') -Encoding UTF8
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
