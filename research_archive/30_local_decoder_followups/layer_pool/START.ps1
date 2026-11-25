param([switch] $Worker)
$ErrorActionPreference = 'Stop'
$task = 'Trajectory-Layer-Pool-20261124'

if ($Worker) {
    $attempt = Join-Path $PSScriptRoot ('attempts/' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
    New-Item -ItemType Directory -Path $attempt -Force | Out-Null
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $argument = '-u -B "' + (Join-Path $PSScriptRoot 'run.py') + '"'
    $child = Start-Process -FilePath 'C:/coding_toobox/envs/astra2/python.exe' -ArgumentList $argument `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput (Join-Path $attempt 'stdout.log') -RedirectStandardError (Join-Path $attempt 'stderr.log')
    @{ exit_code=$child.ExitCode; finished_at=(Get-Date).ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $attempt 'EXIT.json') -Encoding UTF8
    exit $child.ExitCode
}

$existing = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
if (-not $existing) {
    $argument = '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $PSCommandPath + '" -Worker'
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argument -WorkingDirectory $PSScriptRoot
    $principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -Hidden
    Register-ScheduledTask -TaskName $task -Action $action -Principal $principal -Settings $settings | Out-Null
}
if ($existing.State -ne 'Running') { Start-ScheduledTask -TaskName $task }
Get-ScheduledTask -TaskName $task | Select-Object TaskName, State
