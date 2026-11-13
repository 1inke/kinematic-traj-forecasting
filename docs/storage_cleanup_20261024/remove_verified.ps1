param([Parameter(Mandatory=$true)][string]$Batch)
$ErrorActionPreference = 'Stop'
$cleanupDirectory = $PSScriptRoot
$workspaceDirectory = [System.IO.Path]::GetFullPath((Join-Path $cleanupDirectory '..\..'))
$workspacePrefix = $workspaceDirectory.TrimEnd('\') + '\'
$batchPath = (Resolve-Path -LiteralPath $Batch).Path
if (-not $batchPath.StartsWith($cleanupDirectory + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Batch is outside cleanup directory' }
$spec = Get-Content -LiteralPath $batchPath -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $spec.verified -or $spec.format -ne 'VERIFIED_LOSSLESS_NPZ_CLEANUP_V1' -or $spec.root -ne $workspaceDirectory) { throw 'Invalid verified batch' }
if (-not (Test-Path -LiteralPath (Join-Path $cleanupDirectory 'npz_archive\payloads.bin') -PathType Leaf) -or -not (Test-Path -LiteralPath (Join-Path $cleanupDirectory 'npz_archive\index.sqlite') -PathType Leaf)) { throw 'Recovery archive missing; no source deletion permitted' }
$receipt = [System.Collections.Generic.List[object]]::new()
$receiptPath = Join-Path $cleanupDirectory ('DELETE_RECEIPT_' + [System.IO.Path]::GetFileName($batchPath))
$previousReceipt = @()
if (Test-Path -LiteralPath $receiptPath) { $previousReceipt = @(Get-Content -LiteralPath $receiptPath -Raw -Encoding utf8 | ConvertFrom-Json) }
try {
    foreach ($entry in $spec.files) {
        $target = [System.IO.Path]::GetFullPath([string]$entry.absolute_path)
        if (-not $target.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase) -or [System.IO.Path]::GetExtension($target) -ne '.npz') { throw "Invalid source target: $target" }
        $manifestPath = [System.IO.Path]::GetFullPath([string]$entry.manifest)
        if (-not $manifestPath.StartsWith($cleanupDirectory + '\npz_archive\manifests\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid manifest path' }
        if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.manifest_sha256) { throw 'Manifest changed' }
        $stored = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
        if ($stored.sha256 -ne $entry.sha256 -or $stored.bytes -ne $entry.bytes -or $stored.unique_data_discarded -ne $false) { throw 'Manifest/source mismatch' }
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
            $old = @($previousReceipt | Where-Object { $_.path -eq $target -and $_.sha256 -eq $entry.sha256 })
            if ($old.Count -ne 1) { throw "Source disappeared without earlier receipt: $target" }
            $receipt.Add($old[0]); continue
        }
        $item = Get-Item -LiteralPath $target -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -or $item.Length -ne $entry.bytes) { throw 'Source shape changed' }
        $sourceHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($sourceHash -ne $entry.sha256) { throw "Source changed: $target" }
        # Native PowerShell, exact validated source file only; no recursive removal.
        Remove-Item -LiteralPath $target -Force
        if (Test-Path -LiteralPath $target) { throw "Removal did not finish: $target" }
        $receipt.Add([pscustomobject]@{path=$target;bytes=$entry.bytes;sha256=$sourceHash;status='REMOVED_AFTER_LOSSLESS_BYTE_VERIFICATION';utc=[DateTime]::UtcNow.ToString('o')})
    }
} finally {
    ConvertTo-Json -InputObject @($receipt.ToArray()) -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
}
[pscustomobject]@{removed=$receipt.Count;logicalGiB=(($receipt | Measure-Object -Property bytes -Sum).Sum/1GB);receipt=$receiptPath} | ConvertTo-Json -Compress
