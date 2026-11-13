# Neighbor swap 6802 (no retrain)
- Model: F16 IR1 `general` seed1 (encoder+decoder forward only)
- Drop stationary: ped **[111, 70]** (path_len ~0)
- Insert BR: **151, 150** (pad frames 759–762 = hold first xy @763)

## Metrics (heading-normalized frame; rotation-invariant ADE/FDE)

| | minADE20 | minFDE20 | top1 ADE | top1 FDE |
|---|---|---|---|---|
| baseline | 0.4264 | 0.5698 | 1.3782 | 2.6326 |
| patched | 0.4264 | 0.5698 | 1.3782 | 2.6326 |
| delta (p-b) | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

Overnight cached native (original neighbors, world): minADE=0.4427 minFDE=1.0591 top1ADE=1.1122

Plot: `compare_6802_neighbor_swap.png`

## Why delta = 0 (important)

Model social graph: **Top-K=4**, **radius=6 m** (last-obs distances).

| ped | dist@766 | in Top-4? | in radius? |
|-----|----------|-----------|------------|
| 148 | 0.84 | yes | yes |
| 131 | 1.03 | yes | yes |
| 146 | 1.26 | yes | yes |
| 147 | 2.60 | yes | yes |
| 144 | 2.74 | no | yes |
| 145 | 3.20 | no | yes |
| 70 (dropped) | 3.71 | no | yes |
| 111 (dropped) | 3.73 | no | yes |
| 151 (inserted) | 8.41 | no | **no** |
| 150 (inserted) | 8.58 | no | **no** |

So the swap did not change any edge the focal agent actually attends to. Output ADE/FDE/logits are bit-identical.

Overnight cached native (original neighbors): minADE=0.4427 minFDE=1.0591 top1ADE=1.1122 (different pipeline/cache; listed for reference only).
