$ErrorActionPreference='SilentlyContinue'
$root='<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018'
$log=Join-Path $root 'logs\overnight_chain.log'
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss+09:00'), $m) }
L 'chain watcher start (ps1)'
try { (Get-Process -Id $PID).PriorityClass='BelowNormal' } catch {}
while($true){
  $k0=Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_k0\.py' }
  if(-not $k0){ break }
  L ("waiting K0 PIDs=" + (($k0|ForEach-Object ProcessId) -join ','))
  Start-Sleep -Seconds 180
}
$rf=Join-Path $root 'results\seed44\FIT_cal\result_fields.json'
if(-not (Test-Path $rf)){ L 'K0 exited but FIT_cal result_fields missing'; exit 3 }
if(Test-Path (Join-Path $root 'k1_exec\runtime_state.json')){ L 'K1 already complete'; exit 0 }
if(Test-Path (Join-Path $root 'logs\k1_overnight.started')){ L 'K1 start marker exists'; exit 0 }
Set-Content (Join-Path $root 'logs\k1_overnight.started') (Get-Date -Format o)
L 'starting K1'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; $env:OPENBLAS_NUM_THREADS='2'; $env:NUMEXPR_NUM_THREADS='2'; $env:VECLIB_MAXIMUM_THREADS='2'
$env:JOINT_FEAS_ROOT='<WORKSPACE_ROOT>'
$env:JOINT_FEAS_EXEC=$root
$env:JOINT_FEAS_K1=(Join-Path $root 'k1_exec')
New-Item -ItemType Directory -Force -Path $env:JOINT_FEAS_K1 | Out-Null
$py=Join-Path $root '.venv-k0\Scripts\python.exe'
$k1log=Join-Path $root 'logs\k1_overnight.log'
$p=Start-Process -FilePath $py -ArgumentList @('-u',(Join-Path $root 'run_k1.py')) -WorkingDirectory $root -RedirectStandardOutput $k1log -RedirectStandardError (Join-Path $root 'logs\k1_overnight.err') -WindowStyle Hidden -PassThru
Start-Sleep 2
try { (Get-Process -Id $p.Id).PriorityClass='BelowNormal' } catch {}
Get-Process python* | ForEach-Object { try{ $_.PriorityClass='BelowNormal' }catch{} }
L ("K1 launched pid=$($p.Id)")
Wait-Process -Id $p.Id
L ("K1 finished exit check runtime_state")
if(Test-Path (Join-Path $root 'k1_exec\GO_K2_REGISTER.json')){ L 'GO_K2_REGISTER present' }
L 'chain done'
