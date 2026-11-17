$packageDir = Split-Path -Parent $PSScriptRoot
$cfg = Get-Content -LiteralPath (Join-Path $packageDir 'CONFIG.local.json') -Raw -Encoding UTF8 | ConvertFrom-Json
Get-ScheduledTask -TaskName 'Trajectory-ETH-Speed-20261116' -ErrorAction SilentlyContinue | Select-Object TaskName,State
$progress = Join-Path $cfg.output_dir 'PROGRESS.json'
$lastExit = Join-Path $cfg.output_dir 'LAST_EXIT.json'
if (Test-Path -LiteralPath $progress) { Get-Content -LiteralPath $progress -Raw -Encoding UTF8 }
if (Test-Path -LiteralPath $lastExit) { Get-Content -LiteralPath $lastExit -Raw -Encoding UTF8 }
$mirror = Join-Path $cfg.output_dir 'status_mirror'
if (Test-Path -LiteralPath $mirror) {
    Get-ChildItem -LiteralPath $mirror -Filter '*.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 }
}
