$experimentRoot=Split-Path -Parent $PSScriptRoot
Set-Content -LiteralPath (Join-Path $experimentRoot 'PAUSE.request') -Value (Get-Date -Format o) -Encoding UTF8
Write-Output 'Pause requested; worker stops after the current complete epoch checkpoint.'
