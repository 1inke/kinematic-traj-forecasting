$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-DecoderGeometry-Q0-AuditRecovery-20261024'
function Read-Optional($p){if(Test-Path -LiteralPath $p){Get-Content -LiteralPath $p -Raw -Encoding UTF8|ConvertFrom-Json}else{$null}}
$task=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
$triggers=$null;if($task){[xml]$xml=Export-ScheduledTask -TaskName $name;$triggers=$xml.SelectNodes("//*[local-name()='Triggers']/*").Count}
$processes=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.IndexOf($root,[StringComparison]::OrdinalIgnoreCase) -ge 0}|Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine)
$stderr=Join-Path $root 'worker_stderr.log'
[ordered]@{checked_at=(Get-Date).ToString('o');task=$name;task_state=if($task){[string]$task.State}else{$null};triggers=$triggers;processes=$processes;worker=(Read-Optional (Join-Path $root 'worker_state.json'));exit=(Read-Optional (Join-Path $root 'exit_status.json'));progress=(Read-Optional (Join-Path $root 'runtime_state.json'));stderr_bytes=if(Test-Path -LiteralPath $stderr){(Get-Item -LiteralPath $stderr).Length}else{0};result_present=(Test-Path -LiteralPath (Join-Path $root 'RESULT.json'))}|ConvertTo-Json -Depth 20
