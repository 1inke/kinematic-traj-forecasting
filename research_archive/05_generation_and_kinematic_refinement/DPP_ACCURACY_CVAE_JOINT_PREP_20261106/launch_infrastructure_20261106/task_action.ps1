param([Parameter(Mandatory=$true)][string]$ExecutionRoot)
$ErrorActionPreference='Stop'
$launchDir=$PSScriptRoot
$workspaceDir='<WORKSPACE_ROOT>'
Set-Location -LiteralPath $workspaceDir
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& "$workspaceDir/.venv-governance/Scripts/python.exe" -B -u "$launchDir/worker_admission.py" --root $ExecutionRoot
exit $LASTEXITCODE
