$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$name='Trajectory-DecoderGeometry-Q0-20261024'
$pre=Get-Content -LiteralPath (Join-Path $root 'PREFLIGHT.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($pre.status -ne 'PASS_ACTUAL_SAVED_CPU8X128'){throw 'Actual saved preflight required'}
$same=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('worker.py')})
if($same.Count -or (Test-Path -LiteralPath (Join-Path $root 'worker_state.json')) -or (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)){throw 'Same worker/task exists; no duplicate'}
$cmd='-B -u "'+(Join-Path $root 'launch_worker.py')+'"'
$action=New-ScheduledTaskAction -Execute '<WORKSPACE_ROOT>\.venv-governance\Scripts\python.exe' -Argument $cmd -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -Principal $principal|Out-Null
Start-ScheduledTask -TaskName $name
[pscustomobject]@{task=$name;requested_at=(Get-Date).ToString('o');triggers=0}|ConvertTo-Json
