"""Verify the incomplete archive and reconstruct aggregates, with NumPy only.

No training, inference, GPU use, raw-data loading, file writing or automatic resume.
Per-window outputs support aggregate reanalysis; recomputing trajectory errors from
coordinates additionally requires the omitted predictions and original targets.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT / 'snapshot/scientific'
sys.path.insert(0, str(SCI / 'code'))
from fivefold_metrics import aggregate, decide_budget


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def compare(actual, expected, context):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (context, 'keys')
        for k, v in expected.items(): compare(actual[k], v, f'{context}/{k}')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12), (context, actual, expected)
    else:
        assert actual == expected, (context, actual, expected)


def evaluate_saved(folder, split, expected):
    with np.load(folder / f'{split}_window_metrics.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    ids = arrays.pop('ids'); groups = arrays.pop('group')
    assert ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer)
    assert len(np.unique(ids)) == len(ids)
    assert len(ids) == expected['overall']['n']
    assert groups.shape == ids.shape and np.isin(groups, [0, 1, 2]).all()
    for k, a in arrays.items():
        assert a.shape == ids.shape and np.isfinite(a).all(), (folder, k)
    actual = aggregate(arrays, groups)
    compare(actual, expected, str(folder.relative_to(ROOT)))
    return actual


def main():
    manifest = read(ROOT / 'ARTIFACT_MANIFEST.json')['files']
    for rel, expected in manifest.items():
        p = ROOT / rel
        assert p.resolve().is_relative_to(ROOT.resolve())
        assert sha(p) == expected, rel
    status = read(ROOT / 'STATUS_INCOMPLETE.json')
    assert status['status'] == 'PAUSED_INCOMPLETE' and not status['automatic_resume']
    assert status['final_fivefold_three_seed_summary_available'] is False
    current = read(ROOT / 'snapshot/orchestration/standalone/CURRENT.json')
    assert current['status'] == 'PAUSED_BY_USER'
    with (ROOT / 'RUN_STATUS.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 25
    assert len({r['run'] for r in rows}) == 25
    counts = {(phase, state): sum(r['phase'] == phase and r['status'] == state for r in rows)
              for phase in ('budget', 'final') for state in ('COMPLETE', 'PAUSED_PARTIAL', 'NOT_STARTED')}
    assert counts == {('budget','COMPLETE'):10, ('budget','PAUSED_PARTIAL'):0, ('budget','NOT_STARTED'):0,
                      ('final','COMPLETE'):2, ('final','PAUSED_PARTIAL'):2, ('final','NOT_STARTED'):11}
    results, budget_rows = {}, {}
    for row in rows:
        run = SCI / 'FIVEFOLD_RUNS/runs' / row['run']
        if row['status'] == 'COMPLETE':
            assert (run / 'DONE.json').is_file()
            m = read(run / 'metrics.json'); split = row['evaluation_split']
            c = m['contract']
            assert c['phase'] == row['phase'] and c['fold'] == row['fold']
            assert c['seed'] == int(row['seed']) and c['epochs'] == int(row['epochs_planned'])
            assert m['train']['updates'] == int(row['updates_saved'])
            result = evaluate_saved(run, split, m[split])
            results[row['run']] = result['overall']
            for key in ('minADE', 'minFDE'):
                compare(result['overall'][key], float(row[key]), row['run'] + '/' + key)
            if split == 'val':
                budget_rows.setdefault(row['fold'], {})[int(row['epochs_planned'])] = {
                    k: result['overall'][k] for k in ('minADE', 'minFDE')}
        else:
            assert not (run / 'DONE.json').exists() and not (run / 'metrics.json').exists()
            assert not row['minADE'] and not row['minFDE']
            if row['status'] == 'PAUSED_PARTIAL':
                ck = read(run / f"ckpt_epoch{int(row['epochs_saved']):03d}.json")
                assert ck['sha256'] == row['checkpoint_sha256']
                assert ck['updates'] == int(row['updates_saved'])
                assert ck['epoch'] == int(row['epochs_saved'])
    decision = read(SCI / 'FIVEFOLD_RUNS/BUDGET_DECISION.json')
    recalculated = decide_budget(budget_rows)
    compare(recalculated, {k:decision[k] for k in recalculated}, 'budget decision')
    assert recalculated['E_star'] == 128
    seal = read(SCI / 'FIVEFOLD_RUNS/DECISION_SEAL.json')
    assert sha(SCI / 'FIVEFOLD_RUNS/BUDGET_DECISION.json') == seal['decision_sha256']
    for name, expected in seal['budget_done_sha256'].items():
        assert sha(SCI / 'FIVEFOLD_RUNS/runs' / name / 'DONE.json') == expected
    # Revalidate every included file referenced by original DONE receipts.
    mapped = {x['source']:x['archive_path'] for x in read(ROOT/'COPY_PROVENANCE.json')['copies']}
    for original, rel in mapped.items():
        if Path(rel).name == 'SOURCE_MANIFEST.json':
            for name, expected in read(ROOT/rel)['files'].items():
                if name in mapped: assert sha(ROOT/mapped[name]) == expected, name
    for run in (SCI/'FIVEFOLD_RUNS/runs').glob('*'):
        if not (run/'DONE.json').exists(): continue
        for original, expected in read(run/'DONE.json')['files'].items():
            if original in mapped: assert sha(ROOT/mapped[original]) == expected
            else:
                assert original.endswith(('.pt', '_predictions.npz')), ('unexpected omission', original)
    ref = read(ROOT / 'reference_analysis/BUDGET_REFERENCE.json')
    reference = {}
    for row in ref['rows']:
        fold = row['fold']
        assert row['train_fraction'] == .95 and row['seed'] == 42 and row['epochs'] == 128
        result = evaluate_saved(ROOT/'reference_analysis/budget_reference'/fold, 'test', row['test'])
        reference[fold] = result['overall']
        for original, expected in row['files'].items():
            if original in mapped: assert sha(ROOT/mapped[original]) == expected
            else: assert original.endswith('_predictions.npz')
    equal = {k:float(np.mean([r[k] for r in reference.values()])) for k in ('minADE','minFDE')}
    compare(equal, ref['equal_weight'], 'reference equal-weight mean')
    assert len(reference) == 5 and ref['not_the_unfinished_three_seed_result'] is True
    print(json.dumps(dict(status='PASS', study_status='PAUSED_INCOMPLETE',
          artifact_hashes_verified=len(manifest), metric_populations_verified=len(results)+len(reference),
          budget_complete='10/10', final_complete='2/15', final_partial=2, final_not_started=11,
          E_star=128, relative_validation_ADE_gain=recalculated['mean_relative_minADE_gain_64_to_128'],
          exploratory_95pct_single_seed_reference=equal,
          final_fivefold_three_seed_summary=None,
          limitation='Aggregate reconstruction from saved per-window metrics; no raw-coordinate error recomputation.'), indent=2))


if __name__ == '__main__': main()
