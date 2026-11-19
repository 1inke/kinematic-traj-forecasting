$ErrorActionPreference = 'Stop'
$taskName = 'Trajectory-Prec-Pilot-20261119'
$root = Split-Path -Parent $PSScriptRoot
if (Test-Path -LiteralPath (Join-Path $root 'DONE.json')) { Write-Output 'Already complete'; exit 0 }
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { Write-Output 'Already running'; exit 0 }
$worker = Join-Path $PSScriptRoot 'WORKER.ps1'
if (-not $existing) {
    $argument = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $worker + '"'
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argument -WorkingDirectory $PSScriptRoot
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Two 32-epoch single-GPU PREC-PILOT runs' | Out-Null
} elseif ($existing.Actions.Arguments -notlike "*$worker*") {
    throw 'Existing task points to a different worker'
}
$pauseFile = Join-Path $root 'PAUSE.request'
if (Test-Path -LiteralPath $pauseFile) { Remove-Item -LiteralPath $pauseFile }
Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'TASK.xml') -Encoding UTF8
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
