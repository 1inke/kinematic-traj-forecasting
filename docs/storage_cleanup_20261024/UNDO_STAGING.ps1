# Remove only extra staging data created by this review; preserve original NPZs.
$ErrorActionPreference = 'Stop'
$cleanupDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$workspaceDirectory = [System.IO.Path]::GetFullPath((Join-Path $cleanupDirectory '..\..'))
$workspacePrefix = $workspaceDirectory.TrimEnd('\') + '\'
$batches = @(Get-ChildItem -LiteralPath $cleanupDirectory -Filter 'DELETE_BATCH_*.json' -File)
foreach ($batch in $batches) {
    $spec = Get-Content -LiteralPath $batch.FullName -Raw -Encoding utf8 | ConvertFrom-Json
    foreach ($entry in $spec.files) {
        $source = [System.IO.Path]::GetFullPath([string]$entry.absolute_path)
        if (-not $source.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Source outside workspace' }
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw 'An original NPZ is missing: archive must be kept' }
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw 'An original NPZ changed: archive must be kept' }
    }
}
$testFile = Join-Path $cleanupDirectory 'restore_check\MODE_PRESERVING_CONDITIONAL_REFINEMENT_DESIGN_20261017\P1_execution_final_20261018\native\FIT_paths.npz'
if (Test-Path -LiteralPath $testFile -PathType Leaf) {
    if ((Get-FileHash -LiteralPath $testFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne '5a28d7b48fa663c5a2f06cf598c764a4c12aac23fe41267bb0dd27293eb2e738') { throw 'Restore test changed; retaining it' }
    Remove-Item -LiteralPath $testFile -Force
}
# Only these newly created data files, no recursive deletion and no research files.
foreach ($name in @('payloads.bin','index.sqlite','index.sqlite-journal')) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $cleanupDirectory ('npz_archive\'+$name)))
    if (-not $target.StartsWith($cleanupDirectory+'\npz_archive\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid staging target' }
    if (Test-Path -LiteralPath $target -PathType Leaf) { Remove-Item -LiteralPath $target -Force }
}
@{status='UNDONE_ORIGINALS_VERIFIED_PRESENT';utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $cleanupDirectory 'STAGING_UNDONE.json') -Encoding utf8
'Extra staging data removed; all original experiment NPZ files retained.'
