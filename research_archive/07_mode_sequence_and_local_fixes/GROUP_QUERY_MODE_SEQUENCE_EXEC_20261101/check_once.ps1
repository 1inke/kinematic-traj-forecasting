$ErrorActionPreference = 'Stop'
$executionRoot = $PSScriptRoot
$start = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $executionRoot 'WORKER_START.json') | ConvertFrom-Json
$taskName = 'Trajectory-GroupQuery-Q-20261101'
$task = Get-ScheduledTask -TaskName $taskName
[xml]$xml = Export-ScheduledTask -TaskName $taskName
$actual = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$start.pid)
$chain = @()
if ($actual) {
  if ($actual.CommandLine.IndexOf('GROUP_QUERY_MODE_SEQUENCE_EXEC_20261101\worker.py',[StringComparison]::OrdinalIgnoreCase) -lt 0) { throw 'Actual PID CLI mismatch; no restart.' }
  $chain += $actual | Select-Object ProcessId,ParentProcessId,CreationDate,Name,CommandLine
  $next = [int]$actual.ParentProcessId
  for ($k=0;$k -lt 3 -and $next -gt 0;$k++) {
    $parent = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $next)
    if (-not $parent) { break }
    $chain += $parent | Select-Object ProcessId,ParentProcessId,CreationDate,Name,CommandLine
    if ($parent.Name -eq 'svchost.exe') { break }
    $next = [int]$parent.ParentProcessId
  }
}
$state = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $executionRoot 'WORKER_STATE.json') | ConvertFrom-Json
$exitPath = Join-Path $executionRoot 'WORKER_EXIT.json'
$exitValue = if(Test-Path -LiteralPath $exitPath){Get-Content -Raw -Encoding UTF8 -LiteralPath $exitPath | ConvertFrom-Json}else{$null}
$progressPath = Join-Path $executionRoot 'progress.json'
$progress = if(Test-Path -LiteralPath $progressPath){Get-Content -Raw -Encoding UTF8 -LiteralPath $progressPath | ConvertFrom-Json}else{$null}
$stderr = @()
foreach($name in @('SCIENCE','REVIEW')){
 $p = Join-Path $executionRoot ('logs\'+$name+'_stderr.log')
 if(Test-Path -LiteralPath $p){$stderr += [PSCustomObject]@{stage=$name;bytes=(Get-Item -LiteralPath $p).Length;tail=@(Get-Content -LiteralPath $p -Tail 8)}}
}
$runs = @()
foreach($arm in @('Q1','Q3')){foreach($seed in @(42,43,44)){
 $run = Join-Path $executionRoot ('runs\'+$arm+'_'+$seed)
 $hist = Join-Path $run 'history.json'
 $epochs = @(Get-ChildItem -LiteralPath $run -Filter 'history_epoch*.json' -ErrorAction SilentlyContinue).Count
 $runs += [PSCustomObject]@{arm=$arm;seed=$seed;epochs=$epochs;history_final=(Test-Path -LiteralPath $hist);done=(Test-Path -LiteralPath (Join-Path $run 'done.json'));FIT_done=(Test-Path -LiteralPath (Join-Path $run 'FIT\done.json'));inner_done=(Test-Path -LiteralPath (Join-Path $run 'inner\done.json'))}
}}
[PSCustomObject]@{checked_at=(Get-Date).ToString('o');task_state=[string]$task.State;XML_XPath_trigger_count=$xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count;worker_start=$start;actual_chain=$chain;worker_state=$state;worker_exit=$exitValue;progress=$progress;stderr=$stderr;runs=$runs} | ConvertTo-Json -Depth 10 -Compress
