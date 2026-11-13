$ErrorActionPreference = 'Stop'
$executionRoot = $PSScriptRoot
$start = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $executionRoot 'WORKER_START.json') | ConvertFrom-Json
$taskName = 'Trajectory-EndpointDiscreteUtility-T-20261101'
$task = Get-ScheduledTask -TaskName $taskName
[xml]$xml = Export-ScheduledTask -TaskName $taskName
$actual = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$start.pid)
$chain = @()
if ($actual) {
  if ($actual.CommandLine.IndexOf('ENDPOINT_DISCRETE_SET_UTILITY_EXEC_20261101\worker_discrete.py',[StringComparison]::OrdinalIgnoreCase) -lt 0) { throw 'Actual worker PID command-line mismatch, no restart.' }
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
$exitValue = if (Test-Path -LiteralPath $exitPath) { Get-Content -Raw -Encoding UTF8 -LiteralPath $exitPath | ConvertFrom-Json } else { $null }
$stderr = @()
foreach ($name in @('TINY','REVIEW')) {
  $p = Join-Path $executionRoot ('logs\'+$name+'_stderr.log')
  if (Test-Path -LiteralPath $p) { $stderr += [PSCustomObject]@{ stage=$name; bytes=(Get-Item -LiteralPath $p).Length; tail=@(Get-Content -LiteralPath $p -Tail 8) } }
}
$stdoutPath = Join-Path $executionRoot ('logs\'+$state.stage+'_stdout.log')
$progress = if (Test-Path -LiteralPath $stdoutPath) { @(Get-Content -LiteralPath $stdoutPath -Tail 2) } else { @() }
$runs = @()
$tiny = Join-Path $executionRoot 'tiny'
if (Test-Path -LiteralPath $tiny) {
 foreach ($run in Get-ChildItem -LiteralPath $tiny -Directory) {
  $hist = Join-Path $run.FullName 'history.json'
  $runs += [PSCustomObject]@{ run=$run.Name; history_rows=if(Test-Path -LiteralPath $hist){@(Get-Content -Raw -Encoding UTF8 -LiteralPath $hist | ConvertFrom-Json).Count}else{0}; done=(Test-Path -LiteralPath (Join-Path $run.FullName 'done.json')); saved_points=@(Get-ChildItem -LiteralPath $run.FullName -Directory -Filter 'point_*').Count }
 }
}
[PSCustomObject]@{ checked_at=(Get-Date).ToString('o'); task=$taskName; task_state=[string]$task.State; XML_XPath_trigger_count=$xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count; worker_start=$start; actual_chain=$chain; worker_state=$state; worker_exit=$exitValue; stderr=$stderr; actual_progress=$progress; runs=$runs } | ConvertTo-Json -Depth 10 -Compress
