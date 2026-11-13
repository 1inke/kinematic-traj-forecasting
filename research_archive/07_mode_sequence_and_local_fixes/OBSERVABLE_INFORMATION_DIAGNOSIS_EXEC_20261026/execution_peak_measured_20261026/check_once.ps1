$ErrorActionPreference='Stop';$root=$PSScriptRoot;$name='Trajectory-ObservableInformation-Steps12-MeasuredPeak-20261026'
$task=Get-ScheduledTask -TaskName $name;$xml=[xml](Export-ScheduledTask -TaskName $name)
$process=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object{$_.CommandLine -and $_.CommandLine.IndexOf($root,[StringComparison]::OrdinalIgnoreCase) -ge 0}|Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine)
$files=@{};foreach($n in @('worker_state.json','exit_status.json','runtime_state.json','progress.json','current_child.json','RESULT.json')){$p=Join-Path $root $n;if(Test-Path -LiteralPath $p){$files[$n]=Get-Content -LiteralPath $p -Raw -Encoding UTF8|ConvertFrom-Json}}
$stderr=Get-Item -LiteralPath (Join-Path $root 'worker.stderr.log') -ErrorAction SilentlyContinue
[pscustomobject]@{at=(Get-Date).ToString('o');taskstate=$task.State.ToString();triggercount=@($xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*')).Count;processes=$process;files=$files;worker_stderr_bytes=if($stderr){$stderr.Length}else{0}}|ConvertTo-Json -Depth 30


