$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-F16ContextTail-T1-20261023'
$pre=Get-Content -LiteralPath (Join-Path $root 'PREFLIGHT.json') -Raw -Encoding UTF8|ConvertFrom-Json
$review=Get-Content -LiteralPath (Join-Path $root 'PRELAUNCH_REVIEW.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($pre.status -ne 'PASS' -or $review.status -ne 'PASS' -or $pre.source_digest -ne $review.source_digest){throw 'Actual current preflight required'}
$active=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('worker.py')})
if($active.Count -or (Test-Path -LiteralPath (Join-Path $root 'worker_state.json')) -or (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)){throw 'Same task/worker exists; no duplicate'}
$action=New-ScheduledTaskAction -Execute '<WORKSPACE_ROOT>\.venv-governance\Scripts\python.exe' -Argument ('-B -u "'+(Join-Path $root 'worker.py')+'"') -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -Principal $principal|Out-Null
Start-ScheduledTask -TaskName $name
[pscustomobject]@{task=$name;requested_at=(Get-Date).ToString('o');triggers=0}|ConvertTo-Json
