$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-F16ContextComplementarity-S1-20261022'
$pre=Get-Content -LiteralPath (Join-Path $root 'PREFLIGHT.json') -Raw -Encoding UTF8|ConvertFrom-Json
$review=Get-Content -LiteralPath (Join-Path $root 'PRELAUNCH_REVIEW.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($pre.status -ne 'PASS' -or $review.status -ne 'PASS'){throw 'Actual current preflight/review required'}
$active=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('independent_worker.py')})
if($active.Count -or (Test-Path -LiteralPath (Join-Path $root 'worker_state.json')) -or (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)){throw 'Same worker/task exists; no duplicate launch'}
$python='<WORKSPACE_ROOT>\.venv-governance\Scripts\python.exe'
$action=New-ScheduledTaskAction -Execute $python -Argument ('-B -u "'+(Join-Path $root 'independent_worker.py')+'"') -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -Principal $principal|Out-Null
Start-ScheduledTask -TaskName $name
[pscustomobject]@{task=$name;requested_at=(Get-Date).ToString('o');triggers=0}|ConvertTo-Json
