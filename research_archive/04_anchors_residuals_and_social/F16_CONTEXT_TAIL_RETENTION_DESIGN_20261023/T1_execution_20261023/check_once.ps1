$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-F16ContextTail-T1-20261023'
$task=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
$ps=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('worker.py')}|Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine)
$z=[ordered]@{checked_at=(Get-Date).ToString('o');task=$name;task_state=$null;triggers=$null;processes=$ps;worker=$null;exit=$null;progress=$null;stderr_bytes=0;completed_runs=0;histories=@()}
if($task){$z.task_state=[string]$task.State;$xml=[xml](Export-ScheduledTask -TaskName $name);$z.triggers=@($xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*')).Count}
foreach($pair in @(@('worker','worker_state.json'),@('exit','exit_status.json'),@('progress','runtime_state.json'))){$file=Join-Path $root $pair[1];if(Test-Path -LiteralPath $file){$z[$pair[0]]=Get-Content -LiteralPath $file -Raw -Encoding UTF8|ConvertFrom-Json}}
$file=Join-Path $root 'worker_stderr.log';if(Test-Path -LiteralPath $file){$z.stderr_bytes=(Get-Item -LiteralPath $file).Length}
foreach($arm in @('A_replay','P_relative')){foreach($seed in @(42,43,44)){$run=Join-Path $root "runs/${arm}_seed$seed";$file=Join-Path $run 'history.json';$epochs=0;if(Test-Path -LiteralPath $file){$epochs=@(Get-Content -LiteralPath $file -Raw -Encoding UTF8|ConvertFrom-Json).Count};$z.histories+=@{arm=$arm;seed=$seed;epochs=$epochs};if(Test-Path -LiteralPath (Join-Path $run 'done.json')){$z.completed_runs++}}}
$z|ConvertTo-Json -Depth 12
