param(
    [ValidateSet("Preflight", "Run", "All", "Status")]
    [string]$Phase = "All",
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$Python = "<PYTHON_ENV_ROOT>\envs\astra2\python.exe"
$Script = "<WORKSPACE_ROOT>\DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES\directional_regime_pilot.py"

if ($Phase -eq "Status") {
    & $Python $Script --action status --device $Device
    exit $LASTEXITCODE
}

if ($Phase -eq "Preflight" -or $Phase -eq "All") {
    & $Python $Script --action preflight --device $Device
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Phase -eq "Run" -or $Phase -eq "All") {
    & $Python $Script --action run --device $Device
    exit $LASTEXITCODE
}
