$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-F16ContextComplementarity-S1-20261022'
$task=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
$processes=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('independent_worker.py')}|Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine)
$z=[ordered]@{checked_at=(Get-Date).ToString('o');task=$name;task_state=$null;triggers=$null;processes=$processes;worker=$null;exit=$null;progress=$null;stderr_bytes=0;histories=@();completed_scientific_jobs=0}
if($task){$z.task_state=[string]$task.State;$xml=[xml](Export-ScheduledTask -TaskName $name);$z.triggers=@($xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*')).Count}
foreach($pair in @(@('worker','worker_state.json'),@('exit','exit_status.json'),@('progress','runtime_state.json'))){$file=Join-Path $root $pair[1];if(Test-Path -LiteralPath $file){$z[$pair[0]]=Get-Content -LiteralPath $file -Raw -Encoding UTF8|ConvertFrom-Json}}
$file=Join-Path $root 'worker_stderr.log';if(Test-Path -LiteralPath $file){$z.stderr_bytes=(Get-Item -LiteralPath $file).Length}
foreach($seed in @(42,43,44)){$file=Join-Path $root "runs/A_seed$seed/history.json";$epochs=0;if(Test-Path -LiteralPath $file){$epochs=@(Get-Content -LiteralPath $file -Raw -Encoding UTF8|ConvertFrom-Json).Count};$z.histories+=@{arm='A';seed=$seed;epochs=$epochs}}
$z.completed_scientific_jobs=@(Get-ChildItem -LiteralPath (Join-Path $root 'runs') -Filter 'audit_*.json' -Recurse -ErrorAction SilentlyContinue).Count
$z|ConvertTo-Json -Depth 10
