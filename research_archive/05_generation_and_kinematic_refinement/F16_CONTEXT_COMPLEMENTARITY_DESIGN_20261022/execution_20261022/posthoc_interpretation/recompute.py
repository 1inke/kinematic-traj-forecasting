"""Read-only inner-data interpretation. Prints JSON; writes no output files."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
with np.load(root / 'native_inner.npz', allow_pickle=False) as z:
    native = {k: z[k] for k in z.files}

def quality(paths):
    distance = np.linalg.norm(paths.astype(np.float64) - native['gt'][:, None], axis=-1)
    ade, fde = distance.mean(-1), distance[:, :, -1]
    return ade, fde, ((ade <= .2) & (fde <= .4)).any(1)

def acceleration(paths, mask):
    p = paths[mask].astype(np.float64)
    v = np.diff(np.concatenate([np.zeros_like(p[:, :, :1]), p], axis=2), axis=2) / .4
    return np.linalg.norm(np.diff(v, axis=2) / .4, axis=-1).mean((0, 1)).tolist()

na, nf, ng = quality(native['paths'])
change = native['groups'] == 2
out = {'scope': 'posthoc_saved_inner_only_no_model', 'native_acceleration_by_time_change': acceleration(native['paths'], change), 'records': []}
for seed in (42, 43, 44):
    directory = root / f'runs/A_seed{seed}'
    with np.load(directory / 'predictions_inner.npz', allow_pickle=False) as z:
        for key in ('sample_id', 'gt', 'native', 'groups', 'track'):
            np.testing.assert_array_equal(z[key], native[key])
        paths = z['paths']
    aa, af, ag = quality(paths)
    with np.load(directory / 'sample_metrics_inner.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(aa, z['ADE'])
        np.testing.assert_array_equal(af, z['FDE'])
        np.testing.assert_array_equal(ag, z['coverage'])
    for group, mask in [('overall', np.ones(len(ng), dtype=bool)), ('change', change)]:
        n = int(mask.sum())
        row = {'seed': seed, 'group': group, 'n': n, 'parts': {}}
        for name, subset in [('retained', ng & ag), ('rescue', ~ng & ag), ('lost', ng & ~ag), ('neither', ~ng & ~ag)]:
            subset = subset & mask
            row['parts'][name] = {'n': int(subset.sum()), 'ADE_contribution': float((aa.min(1) - na.min(1))[subset].sum() / n), 'FDE_contribution': float((af.min(1) - nf.min(1))[subset].sum() / n)}
        assert sum(v['n'] for v in row['parts'].values()) == n
        for label, delta in [('ADE', aa.min(1)-na.min(1)), ('FDE', af.min(1)-nf.min(1))]:
            row['delta_' + label] = float(delta[mask].mean())
            np.testing.assert_allclose(sum(v[label + '_contribution'] for v in row['parts'].values()), row['delta_' + label], rtol=0, atol=1e-14)
        if group == 'change':
            row['acceleration_by_time'] = acceleration(paths, mask)
        out['records'].append(row)
print(json.dumps(out, ensure_ascii=False, indent=2))
