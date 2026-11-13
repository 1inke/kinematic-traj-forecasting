$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-K20SingleEdit-E0-AuditRecovery-20261025'
function ReadJson($n){$p=Join-Path $root $n;if(Test-Path -LiteralPath $p){Get-Content -LiteralPath $p -Raw -Encoding UTF8|ConvertFrom-Json}}
$task=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
$count=$null;if($task){[xml]$xml=Export-ScheduledTask -TaskName $name;$count=@($xml.SelectNodes("/*[local-name()='Task']/*[local-name()='Triggers']/*")).Count}
$processes=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.IndexOf($root,[StringComparison]::OrdinalIgnoreCase) -ge 0}|Select-Object ProcessId,ParentProcessId,CommandLine,CreationDate)
[pscustomobject]@{checked_at=(Get-Date).ToString('o');task=$name;task_state=if($task){[string]$task.State}else{$null};triggers=$count;processes=$processes;worker=(ReadJson 'worker_state.json');exit=(ReadJson 'exit_status.json');progress=(ReadJson 'runtime_state.json');stderr_bytes=if(Test-Path -LiteralPath (Join-Path $root 'worker_stderr.log')){(Get-Item -LiteralPath (Join-Path $root 'worker_stderr.log')).Length}else{0};result_present=Test-Path -LiteralPath (Join-Path $root 'RESULT.json')}|ConvertTo-Json -Depth 12
