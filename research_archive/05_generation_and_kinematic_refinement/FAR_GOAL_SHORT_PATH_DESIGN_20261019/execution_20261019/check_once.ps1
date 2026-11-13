$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-FarGoalShortPath-H1H2-20261019'
$worker=Get-Content -LiteralPath (Join-Path $root 'worker_state.json') -Raw -Encoding UTF8|ConvertFrom-Json
$proc=Get-CimInstance Win32_Process -Filter ('ProcessId='+$worker.pid)
$parent=if($proc){Get-CimInstance Win32_Process -Filter ('ProcessId='+$proc.ParentProcessId)}else{$null}
$task=Get-ScheduledTask -TaskName $name
[xml]$xml=Export-ScheduledTask -TaskName $name
$triggers=@($xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*')).Count
$exit=if(Test-Path -LiteralPath (Join-Path $root 'exit_status.json')){Get-Content -LiteralPath (Join-Path $root 'exit_status.json') -Raw -Encoding UTF8|ConvertFrom-Json}else{$null}
$progress=if(Test-Path -LiteralPath (Join-Path $root 'progress.json')){Get-Content -LiteralPath (Join-Path $root 'progress.json') -Raw -Encoding UTF8|ConvertFrom-Json}else{$null}
$histories=@(Get-ChildItem -LiteralPath (Join-Path $root 'runs') -Recurse -Filter history.json -ErrorAction SilentlyContinue|ForEach-Object{$history=@(Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8|ConvertFrom-Json);[pscustomobject]@{run=$_.Directory.Name;epochs=$history.Count}})
$jobs=if(Test-Path -LiteralPath (Join-Path $root 'JOBS.json')){Get-Content -LiteralPath (Join-Path $root 'JOBS.json') -Raw -Encoding UTF8|ConvertFrom-Json}else{$null}
[pscustomobject]@{at=(Get-Date).ToString('o');worker=$worker;exit=$exit;process=if($proc){[pscustomobject]@{pid=$proc.ProcessId;parent=$proc.ParentProcessId;created=$proc.CreationDate.ToString('o');commandline=$proc.CommandLine}}else{$null};parent_name=if($parent){$parent.Name}else{$null};task_state=$task.State.ToString();task_triggers=$triggers;progress=$progress;histories=$histories;completed_jobs=if($jobs){@($jobs.PSObject.Properties).Count}else{0};stderr_bytes=(Get-Item -LiteralPath (Join-Path $root 'worker_stderr.log')).Length}|ConvertTo-Json -Depth 12
