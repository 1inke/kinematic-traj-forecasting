# GRAPH_FORCE 6802

F16 IR1 same weights. Neighbor pack includes 151/150; focal social edges forced to close+BR.

| condition | minADE20 | minFDE20 | top1 ADE | top1 FDE |
|---|---|---|---|---|
| baseline | 0.4264 | 0.5698 | 1.3782 | 2.6326 |
| swap_only | 0.4264 | 0.5698 | 1.3782 | 2.6326 |
| graph_force | 0.3424 | 0.3852 | 1.3389 | 2.5522 |

Delta force−baseline: minADE -0.0841, minFDE -0.1846, top1ADE -0.0392, top1FDE -0.0804

Cand traj max|Δ| force vs baseline: 0.180445 m; best-end shift 0.2019 m; top1-end shift 0.0867 m

Forced edges: agents [1, 2, 3, 4, 8, 7] = peds [148, 131, 146, 147, 151, 150]

Plot: `GRAPH_FORCE_compare.png`
