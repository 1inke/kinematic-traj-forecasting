$ErrorActionPreference = 'Stop'
$parallelRoot = Split-Path $PSScriptRoot -Parent
$config = Get-Content -LiteralPath (Join-Path $parallelRoot 'PARALLEL_CONFIG.json') -Raw | ConvertFrom-Json
$task = Get-ScheduledTask -TaskName 'Trajectory-Fivefold-B0-Recovery-Parallel-20261116' -ErrorAction SilentlyContinue
if ($task.State -eq 'Running') { Stop-ScheduledTask -TaskName $task.TaskName }
$owned = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like ('*' + $parallelRoot + '*') })
foreach ($candidate in ($owned | Sort-Object ProcessId -Descending)) {
    $fresh = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $candidate.ProcessId)
    if ($fresh -and $fresh.CreationDate -eq $candidate.CreationDate -and $fresh.CommandLine -like ('*' + $parallelRoot + '*')) { Stop-Process -Id $candidate.ProcessId -Force }
}
$remaining = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like ('*' + $parallelRoot + '*') })
if ($remaining.Count -ne 0) { throw 'Experiment children still alive; pause not confirmed.' }
& $config.python -B (Join-Path $PSScriptRoot 'record_pause_recovery.py')
exit $LASTEXITCODE
