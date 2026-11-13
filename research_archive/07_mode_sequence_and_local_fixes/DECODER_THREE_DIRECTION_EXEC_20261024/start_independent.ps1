$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$name='Trajectory-DecoderThreeDirections-20261024'
$pre=Get-Content -LiteralPath (Join-Path $root 'PRELAUNCH.json') -Raw -Encoding UTF8|ConvertFrom-Json
$cpu=Get-Content -LiteralPath (Join-Path $root 'CODE_VERIFICATION.json') -Raw -Encoding UTF8|ConvertFrom-Json
$encoder=Get-Content -LiteralPath (Join-Path $root 'prelaunch_encoder/ENCODER_SMOKE.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($pre.status -ne 'PASS_ACTUAL_CPU_SOURCES_IDS_METADATA_SAVED128' -or $cpu.status -ne 'PASS_CODE_AND_SYNTHETIC_ONLY' -or $encoder.status -ne 'PASS'){throw 'Actual source/CPU/synthetic/frozen-encoder prelaunch required'}
$active=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('series_worker.py')})
if($active.Count -or (Test-Path -LiteralPath (Join-Path $root 'worker_state.json')) -or (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)){throw 'Same worker/task exists; no duplicate'}
$action=New-ScheduledTaskAction -Execute '<WORKSPACE_ROOT>\.venv-governance\Scripts\python.exe' -Argument ('-B -u "'+(Join-Path $root 'series_worker.py')+'"') -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -Principal $principal|Out-Null
Start-ScheduledTask -TaskName $name
[pscustomobject]@{task=$name;requested_at=(Get-Date).ToString('o');triggers=0}|ConvertTo-Json
