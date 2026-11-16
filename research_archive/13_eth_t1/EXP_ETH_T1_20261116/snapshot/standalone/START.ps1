param([switch]$Status)
$ErrorActionPreference='Stop'
$ethRoot=Split-Path $PSScriptRoot -Parent
$config=Get-Content -LiteralPath (Join-Path $ethRoot 'CONFIG.json') -Raw | ConvertFrom-Json
$taskName=$config.task
if($Status){
 Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Select-Object TaskName,State
 foreach($name in @('CURRENT.json','WORKER_EXIT.json')){$p=Join-Path $PSScriptRoot $name;if(Test-Path -LiteralPath $p){Get-Content -LiteralPath $p -Raw}}
 exit
}
$proof=Get-Content -LiteralPath (Join-Path $ethRoot 'PREFLIGHT_PASS.json') -Raw | ConvertFrom-Json
if($proof.status -ne 'PASS'){throw 'Real CUDA preflight required.'}
foreach($otherName in @('Trajectory-Explore-Fast-20261116','Trajectory-Fivefold-B0-Recovery-Parallel-20261116','Trajectory-Fivefold-B0-Parallel-20261115','Trajectory-Fivefold-B0-Single-20261115','Trajectory-Fivefold-B0-Standalone-20261115')){
 $other=Get-ScheduledTask -TaskName $otherName -ErrorAction SilentlyContinue
 if($other.State -eq 'Running'){throw ('Another experiment still running: '+$otherName)}
 if($other -and $other.State -ne 'Disabled'){Disable-ScheduledTask -TaskName $otherName | Out-Null}
}
$arguments='-X utf8 -B "'+(Join-Path $PSScriptRoot 'worker.py')+'"'
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if(-not $existing){
 $action=New-ScheduledTaskAction -Execute $config.python -Argument $arguments -WorkingDirectory $config.workspace
 $principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
 $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
 Register-ScheduledTask -TaskName $taskName -InputObject (New-ScheduledTask -Action $action -Principal $principal -Settings $settings) | Out-Null
}elseif($existing.Actions.Execute -ne $config.python -or $existing.Actions.Arguments -ne $arguments){throw 'Existing task action differs.'}
[xml]$xml=Export-ScheduledTask -TaskName $taskName
if($xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*').Count -ne 0){throw 'Unexpected automatic trigger.'}
$state=(Get-ScheduledTask -TaskName $taskName).State
if($state -eq 'Running'){Write-Output 'Already running; no duplicate started.';exit}
if($state -eq 'Disabled'){Enable-ScheduledTask -TaskName $taskName | Out-Null}
Start-ScheduledTask -TaskName $taskName
[pscustomobject]@{task=$taskName;at=(Get-Date).ToString('o');max_concurrency=2;new_training_runs=1;seed=42;epochs=64;COMBO_started=$false} | ConvertTo-Json
