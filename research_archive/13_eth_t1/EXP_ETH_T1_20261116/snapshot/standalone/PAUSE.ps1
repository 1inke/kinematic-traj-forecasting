$ErrorActionPreference='Stop'
$ethRoot=Split-Path $PSScriptRoot -Parent
$config=Get-Content -LiteralPath (Join-Path $ethRoot 'CONFIG.json') -Raw | ConvertFrom-Json
$task=Get-ScheduledTask -TaskName $config.task -ErrorAction SilentlyContinue
if($task.State -eq 'Running'){Stop-ScheduledTask -TaskName $task.TaskName}
$owned=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like ('*'+$ethRoot+'*')})
foreach($candidate in ($owned | Sort-Object ProcessId -Descending)){
 $fresh=Get-CimInstance Win32_Process -Filter ('ProcessId='+$candidate.ProcessId)
 if($fresh -and $fresh.CreationDate -eq $candidate.CreationDate -and $fresh.CommandLine -like ('*'+$ethRoot+'*')){Stop-Process -Id $candidate.ProcessId -Force}
}
$remaining=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like ('*'+$ethRoot+'*')})
if($remaining.Count){throw 'ETH-T1 children still alive.'}
& $config.python -X utf8 -B (Join-Path $ethRoot 'adapter\record_pause.py')
exit $LASTEXITCODE
