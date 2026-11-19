$ErrorActionPreference = 'Stop'
$taskName = 'Trajectory-Prec-Diag-20261119'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') {
    Write-Output 'Already running'
    exit 0
}
$workerPath = Join-Path $PSScriptRoot 'WORKER.ps1'
if (-not $existing) {
    $argument = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $workerPath + '"'
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argument -WorkingDirectory $PSScriptRoot
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'CPU-only PREC-DIAG on saved C0 predictions' | Out-Null
} elseif ($existing.Actions.Arguments -notlike "*$workerPath*") {
    throw 'Existing task points to a different worker'
}
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
