# Prepared for direct human execution after the agent's deletion tools were blocked.
# Never starts training or edits scientific code/results. Restorable payloads retained.
$ErrorActionPreference = 'Stop'
$cleanupDirectory = $PSScriptRoot
$archiveScript = Join-Path $cleanupDirectory 'archive.py'
if (Test-Path -LiteralPath (Join-Path $cleanupDirectory 'STAGING_UNDONE.json')) { throw 'Staging was undone. Do not use old deletion batches; rebuild/review the plan first.' }
$freeBefore = (Get-PSDrive C).Free
Push-Location (Join-Path $cleanupDirectory '..\..')
try {
    python $archiveScript reconcile
    if ($LASTEXITCODE -ne 0) { throw 'Archive/source reconciliation failed' }
    python $archiveScript verify
    if ($LASTEXITCODE -ne 0) { throw 'Existing recovery archive verification failed' }
    foreach ($batch in @(Get-ChildItem -LiteralPath $cleanupDirectory -Filter 'DELETE_BATCH_*.json' -File | Sort-Object Name)) {
        & (Join-Path $cleanupDirectory 'remove_verified.ps1') -Batch $batch.FullName
    }
    while ($true) {
        python $archiveScript stage --batch-gib 2
        if ($LASTEXITCODE -ne 0) { throw 'Staging or exact-byte verification failed; original files retained' }
        $latest = Get-ChildItem -LiteralPath $cleanupDirectory -Filter 'DELETE_BATCH_*.json' -File | Sort-Object Name -Descending | Select-Object -First 1
        $spec = Get-Content -LiteralPath $latest.FullName -Raw -Encoding utf8 | ConvertFrom-Json
        if (@($spec.files).Count -eq 0) { break }
        & (Join-Path $cleanupDirectory 'remove_verified.ps1') -Batch $latest.FullName
    }
    python $archiveScript reconcile
    if ($LASTEXITCODE -ne 0) { throw 'Final reconciliation failed' }
    python $archiveScript verify
    if ($LASTEXITCODE -ne 0) { throw 'Final archive verification failed' }
    $testFile = Join-Path $cleanupDirectory 'restore_check\MODE_PRESERVING_CONDITIONAL_REFINEMENT_DESIGN_20261017\P1_execution_final_20261018\native\FIT_paths.npz'
    if (Test-Path -LiteralPath $testFile -PathType Leaf) {
        if ((Get-FileHash -LiteralPath $testFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne '5a28d7b48fa663c5a2f06cf598c764a4c12aac23fe41267bb0dd27293eb2e738') { throw 'Restore test changed; retaining it' }
        Remove-Item -LiteralPath $testFile -Force
    }
    [pscustomobject]@{status='COMPLETE';freeGiB=(Get-PSDrive C).Free/1GB;changeSinceManualStartGiB=((Get-PSDrive C).Free-$freeBefore)/1GB} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $cleanupDirectory 'MANUAL_COMPLETION.json') -Encoding utf8
} finally { Pop-Location }
