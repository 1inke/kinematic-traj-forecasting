"""Paired G/H saved comparisons; explicit execution only, no model loading."""
import argparse
from pathlib import Path
import numpy as np
from io_contract import (checked_seal, load_npz, validate_observed, npz_new,
                         json_new, sha, seal, finish_stage)
from bootstrap import make_draws, paired_ci


def compare(observed, g_dirs, h_dirs, out, check=lambda: None):
    x = load_npz(observed)
    validate_observed(x)
    arms, bindings = [], []
    for arm, folders in zip(('G', 'H'), (g_dirs, h_dirs)):
        if len(folders) != 3:
            raise ValueError('exactly 3 paired seed42/43/44 directories required')
        samples = []
        for seed, folder in zip((42, 43, 44), folders):
            receipt = checked_seal(folder)
            if receipt['observed_sha'] != sha(observed):
                raise ValueError('population binding mismatch')
            contract = receipt.get('model_contract')
            if not contract or contract['seed'] != seed or contract['arm'] != arm:
                raise ValueError('require paired G/H seeds42,43,44 in order; no duplicate/baseline seeds')
            bindings.append({k: contract[k] for k in
                             ('code_digest', 'stats_sha', 'split_sha', 'preparation_done_sha',
                              'branch', 'source_lock_sha', 'teacher_done_sha')})
            z = load_npz(Path(folder)/'guarded_samples.npz')
            if not np.array_equal(z['sample_id'], x['sample_id']):
                raise ValueError('sample ID mismatch')
            samples.append(z)
        arms.append({key: np.stack([z[key] for z in samples]) for key in
                     ('parameter_mse', 'regret', 'covered', 'native_covered', 'identity_regret')})
    if any(b != bindings[0] for b in bindings):
        raise ValueError('G/H model source/statistics/teacher/branch contracts differ')
    g, h = arms
    if not np.array_equal(g['native_covered'], h['native_covered']):
        raise ValueError('Native reference mismatch')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    draws = make_draws(x['track_key'])
    npz_new(out/'bootstrap_indices.npz', **draws)
    contrasts = dict(parameter_mse=h['parameter_mse']-g['parameter_mse'],
                     regret=h['regret']-g['regret'],
                     coverage=h['covered'].astype(float)-h['native_covered'])
    reports, files = {}, [out/'bootstrap_indices.npz']
    for group, mask in [('overall', np.ones(len(x['sample_id']), bool))]+[
            (name, x['group'] == k) for k, name in enumerate(('stable', 'transition', 'change'))]:
        reports[group] = {}
        for metric, difference in contrasts.items():
            check()
            report, samples = paired_ci(difference, x['track_key'], mask, draws, family=6)
            reports[group][metric] = report
            name = out/f'{group}_{metric}_draws.npz'
            npz_new(name, **samples)
            files.append(name)
    json_new(out/'CI.json', reports)
    files.append(out/'CI.json')
    for folder in list(g_dirs)+list(h_dirs):
        checked_seal(folder)
    # This is statistical output, not an automatic scientific PASS/final release.
    seal(out, files, stage='PAIRED_EXPLORATORY_CI_REQUIRES_GATE_REVIEW',
         observed_sha=sha(observed), paired_directory_order='42,43,44',
         input_metrics={str(Path(f).resolve()): sha(Path(f)/'done.json') for f in list(g_dirs)+list(h_dirs)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observed', required=True)
    p.add_argument('--g', nargs=3, required=True)
    p.add_argument('--h', nargs=3, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--run', action='store_true')
    p.add_argument('--guard', required=True)
    p.add_argument('--inventory-bytes', type=int, required=True)
    a = p.parse_args()
    if not a.run:
        print('PLAN_ONLY')
        return
    from run_stage import resource_guard
    g = resource_guard(a.guard, a.inventory_bytes)
    compare(a.observed, a.g, a.h, a.out, g.check)
    json_new(Path(a.out)/'RESOURCE.json', dict(final=g.check(force=True), observations=g.observations))
    finish_stage(a.out, 'RESOURCE.json')


if __name__ == '__main__':
    main()
