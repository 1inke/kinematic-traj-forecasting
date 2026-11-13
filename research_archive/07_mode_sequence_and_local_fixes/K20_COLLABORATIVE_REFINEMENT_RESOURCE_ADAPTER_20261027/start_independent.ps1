$ErrorActionPreference='Stop';$root=$PSScriptRoot;$name='Trajectory-K20CollaborativeRefinement-ResourceAdapter-20261027'
$pre=Get-Content -LiteralPath (Join-Path $root 'PRELAUNCH_REVIEW.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($pre.status -ne 'PASS'){throw 'Full own preflight required'}
$same=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.IndexOf($root,[StringComparison]::OrdinalIgnoreCase) -ge 0})
if($same.Count -or (Test-Path -LiteralPath (Join-Path $root 'worker_state.json')) -or (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)){throw 'Duplicate forbidden'}
$a=New-ScheduledTaskAction -Execute '<WORKSPACE_ROOT>\.venv-governance\Scripts\python.exe' -Argument ('-B -u "'+(Join-Path $root 'launch_worker.py')+'"') -WorkingDirectory $root
$s=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$p=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $a -Settings $s -Principal $p|Out-Null
Start-ScheduledTask -TaskName $name
[pscustomobject]@{task=$name;requested_at=(Get-Date).ToString('o');triggers=0}|ConvertTo-Json
