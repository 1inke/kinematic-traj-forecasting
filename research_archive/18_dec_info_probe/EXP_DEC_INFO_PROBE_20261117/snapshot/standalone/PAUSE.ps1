$root=Split-Path -Parent $PSScriptRoot
if (Test-Path -LiteralPath (Join-Path $root 'DONE.json')) {Write-Output 'Already complete';exit 0}
(Get-Date -Format o) | Set-Content -LiteralPath (Join-Path $root 'PAUSE.request') -Encoding UTF8
Write-Output 'Pause requested. Check STATUS.ps1 for the completed epoch and task exit before shutdown.'
