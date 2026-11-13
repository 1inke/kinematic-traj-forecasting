$ErrorActionPreference = 'Stop'
$executionRoot = $PSScriptRoot
$workerStart = Get-Content -Raw -LiteralPath (Join-Path $executionRoot 'p0_os_start.json') | ConvertFrom-Json
$taskName = 'Trajectory-EndpointTemporal-P0-20261029'
$task = Get-ScheduledTask -TaskName $taskName
[xml]$xml = Export-ScheduledTask -TaskName $taskName
$actual = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$workerStart.pid)
$parent = $null
if ($actual) {
  if ($actual.CommandLine.IndexOf('ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029\p0_os_worker.py', [StringComparison]::OrdinalIgnoreCase) -lt 0) { throw 'Actual PID command-line mismatch; do not restart.' }
  $parent = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$actual.ParentProcessId)
}
$exitPath = Join-Path $executionRoot 'p0_os_exit.json'
$successPath = Join-Path $executionRoot 'P0_WORKER_SUCCESS.json'
$failurePath = Join-Path $executionRoot 'P0_RESUME_FAILURE.json'
$reviewPath = Join-Path $executionRoot 'p0_completion_review\REVIEW.json'
$exitValue = if (Test-Path -LiteralPath $exitPath) { Get-Content -Raw -LiteralPath $exitPath | ConvertFrom-Json } else { $null }
$successValue = if (Test-Path -LiteralPath $successPath) { Get-Content -Raw -LiteralPath $successPath | ConvertFrom-Json } else { $null }
$failureValue = if (Test-Path -LiteralPath $failurePath) { Get-Content -Raw -LiteralPath $failurePath | ConvertFrom-Json } else { $null }
$reviewValue = if (Test-Path -LiteralPath $reviewPath) { (Get-Content -Raw -LiteralPath $reviewPath | ConvertFrom-Json).status } else { $null }
[PSCustomObject]@{
  checked_at = (Get-Date).ToString('o')
  task = $taskName
  task_state = [string]$task.State
  XML_XPath_trigger_count = $xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count
  actual = if ($actual) { $actual | Select-Object ProcessId,ParentProcessId,CreationDate,Name,CommandLine } else { $null }
  parent = if ($parent) { $parent | Select-Object ProcessId,ParentProcessId,CreationDate,Name,CommandLine } else { $null }
  worker_start = $workerStart.started_at
  worker_success = $successValue
  worker_exit = $exitValue
  failure = $failureValue
  review_status = $reviewValue
  stderr_bytes = (Get-Item -LiteralPath (Join-Path $executionRoot 'p0_worker_stderr.log')).Length
  stderr = Get-Content -LiteralPath (Join-Path $executionRoot 'p0_worker_stderr.log') -Tail 15
  actual_progress = Get-Content -LiteralPath (Join-Path $executionRoot 'p0_worker_stdout.log') -Tail 2
} | ConvertTo-Json -Depth 9 -Compress
