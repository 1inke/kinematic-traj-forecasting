$ErrorActionPreference = 'Stop'
$parallelRoot = Split-Path $PSScriptRoot -Parent
$config = Get-Content -LiteralPath (Join-Path $parallelRoot 'PARALLEL_CONFIG.json') -Raw | ConvertFrom-Json
$taskName = 'Trajectory-Fivefold-B0-Parallel-20261115'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task.State -eq 'Running') { Stop-ScheduledTask -TaskName $taskName }
$owned = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like ('*' + $parallelRoot + '*') })
foreach ($candidate in ($owned | Sort-Object ProcessId -Descending)) {
    $fresh = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $candidate.ProcessId)
    if ($fresh -and $fresh.CreationDate -eq $candidate.CreationDate -and $fresh.CommandLine -like ('*' + $parallelRoot + '*')) { Stop-Process -Id $candidate.ProcessId -Force }
}
$remaining = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like ('*' + $parallelRoot + '*') })
if ($remaining.Count -ne 0) { throw 'Experiment children still alive; pause not confirmed.' }
& $config.python -B (Join-Path $parallelRoot 'code\record_pause.py')
exit $LASTEXITCODE
