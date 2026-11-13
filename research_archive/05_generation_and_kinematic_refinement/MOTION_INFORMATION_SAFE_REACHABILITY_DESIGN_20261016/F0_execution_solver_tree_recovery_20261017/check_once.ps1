$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-MotionInformation-F0-SolverTreeRecovery-20261017'
$task=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
$processes=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('independent_worker.py')}|Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine)
$z=[ordered]@{checked_at=(Get-Date).ToString('o');task=$name;task_state=$null;triggers=$null;processes=$processes;worker=$null;exit=$null;progress=$null;stderr_bytes=0;completed_jobs=0;histories=@()}
if($task){$z.task_state=[string]$task.State;$xml=[xml](Export-ScheduledTask -TaskName $name);$z.triggers=@($xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*')).Count}
foreach($pair in @(@('worker','worker_state.json'),@('exit','exit_status.json'),@('progress','progress.json'))){$file=Join-Path $root $pair[1];if(Test-Path -LiteralPath $file){$z[$pair[0]]=Get-Content -LiteralPath $file -Raw -Encoding UTF8|ConvertFrom-Json}}
$file=Join-Path $root 'worker_stderr.log';if(Test-Path -LiteralPath $file){$z.stderr_bytes=(Get-Item -LiteralPath $file).Length}
$file=Join-Path $root 'JOBS.json';if(Test-Path -LiteralPath $file){$z.completed_jobs=@((Get-Content -LiteralPath $file -Raw -Encoding UTF8|ConvertFrom-Json).PSObject.Properties).Count}
$z|ConvertTo-Json -Depth 8
