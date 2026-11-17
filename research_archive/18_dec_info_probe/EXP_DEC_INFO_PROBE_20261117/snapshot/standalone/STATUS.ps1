$root=Split-Path -Parent $PSScriptRoot
Get-ScheduledTask -TaskName 'Trajectory-Dec-Info-Probe-20261117' -ErrorAction SilentlyContinue | Select-Object TaskName,State
foreach ($relative in @('CURRENT.json','standalone/ACTIVE.json','standalone/WORKER_EXIT.json')){
    $p=Join-Path $root $relative
    if (Test-Path -LiteralPath $p) {Get-Content -LiteralPath $p}
}
