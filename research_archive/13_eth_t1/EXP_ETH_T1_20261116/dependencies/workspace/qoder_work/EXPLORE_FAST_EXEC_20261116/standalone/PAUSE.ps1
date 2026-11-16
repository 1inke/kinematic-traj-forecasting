$ErrorActionPreference='Stop'
$exploreRoot=Split-Path $PSScriptRoot -Parent
$config=Get-Content -LiteralPath (Join-Path $exploreRoot 'CONFIG.json') -Raw | ConvertFrom-Json
$task=Get-ScheduledTask -TaskName 'Trajectory-Explore-Fast-20261116' -ErrorAction SilentlyContinue
if($task.State -eq 'Running'){Stop-ScheduledTask -TaskName $task.TaskName}
$owned=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like ('*'+$exploreRoot+'*')})
foreach($candidate in ($owned | Sort-Object ProcessId -Descending)){
 $fresh=Get-CimInstance Win32_Process -Filter ('ProcessId='+$candidate.ProcessId)
 if($fresh -and $fresh.CreationDate -eq $candidate.CreationDate -and $fresh.CommandLine -like ('*'+$exploreRoot+'*')){Stop-Process -Id $candidate.ProcessId -Force}
}
$remaining=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like ('*'+$exploreRoot+'*')})
if($remaining.Count){throw 'Exploration children still alive.'}
& $config.python -X utf8 -B (Join-Path $exploreRoot 'code\record_pause.py')
exit $LASTEXITCODE

