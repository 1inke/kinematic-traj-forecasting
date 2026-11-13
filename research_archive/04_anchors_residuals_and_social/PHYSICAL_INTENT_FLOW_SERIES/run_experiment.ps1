param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Audit", "Pilot", "Analyze")]
    [string]$Phase,
    [string]$Device = "cuda:0",
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Python = "<PYTHON_ENV_ROOT>\envs\astra2\python.exe"
$Runner = "<WORKSPACE_ROOT>\PHYSICAL_INTENT_FLOW_SERIES\run_pif1.py"
$Stage = $Phase.ToLowerInvariant()
$Arguments = @("-u", $Runner, "--stage", $Stage, "--device", $Device)
if ($DryRun) { $Arguments += "--dry-run" }
if ($Force) { $Arguments += "--force" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "PIF1 phase failed with exit code $LASTEXITCODE" }

