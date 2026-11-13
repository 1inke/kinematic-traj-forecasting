$ErrorActionPreference = 'Stop'
$executionRoot = $PSScriptRoot
$taskName = 'Trajectory-E2E-DGeoV2-Pilots-20261103'
$record = Get-Content -LiteralPath (Join-Path $executionRoot 'WORKER_START.json') -Encoding UTF8 -Raw | ConvertFrom-Json
$workerPid = [int]$record.pid
$process = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $workerPid)
$chain = @()
if ($process) {
  if ($process.CommandLine.IndexOf((Join-Path $executionRoot 'worker.py'),[System.StringComparison]::OrdinalIgnoreCase) -lt 0) { throw 'Actual worker CLI mismatch' }
  $chain += $process | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine
  $parentId = [int]$process.ParentProcessId
  for ($level=0; $level -lt 3; $level++) {
    $parentProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $parentId)
    if (-not $parentProcess) { break }
    $chain += $parentProcess | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine
    if ($parentProcess.Name -eq 'svchost.exe') { break }
    $parentId = [int]$parentProcess.ParentProcessId
  }
}
$task = Get-ScheduledTask -TaskName $taskName
[xml]$xml = Export-ScheduledTask -TaskName $taskName
$triggers = $xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count
$state = Get-Content -LiteralPath (Join-Path $executionRoot 'WORKER_STATE.json') -Encoding UTF8 -Raw | ConvertFrom-Json
$exitResult = $null
if (Test-Path -LiteralPath (Join-Path $executionRoot 'WORKER_EXIT.json')) { $exitResult = Get-Content -LiteralPath (Join-Path $executionRoot 'WORKER_EXIT.json') -Encoding UTF8 -Raw | ConvertFrom-Json }
$latestProgress = Get-ChildItem -LiteralPath (Join-Path $executionRoot 'progress') -Filter '*.json' -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
$progressResult = $null
if ($latestProgress) { $progressResult = Get-Content -LiteralPath $latestProgress.FullName -Encoding UTF8 -Raw | ConvertFrom-Json }
$errors = @(Get-ChildItem -LiteralPath (Join-Path $executionRoot 'logs') -Filter '*stderr.log' -ErrorAction SilentlyContinue | Select-Object Name,Length)
[PSCustomObject]@{checked_at=(Get-Date).ToString('o');actual_pid=$workerPid;process_chain=$chain;task_state=[string]$task.State;XML_trigger_count=$triggers;worker=$state;exit=$exitResult;progress=$progressResult;stderr=$errors} | ConvertTo-Json -Depth 15 -Compress

