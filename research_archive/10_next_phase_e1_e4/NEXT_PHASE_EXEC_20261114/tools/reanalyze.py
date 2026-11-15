"""CPU-only verification of published primary metrics and all bootstrap intervals.

Requires only NumPy. No model, labels, original drive paths, torch or GPU is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
GROUPS = ('overall', 'stable', 'transition', 'change')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def population(name, split):
    with np.load(PACKAGE / 'analysis_inputs' / name / f'{split}.npz', allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def masks(groups):
    return {name: np.ones(len(groups), bool) if index == 0 else groups == index - 1
            for index, name in enumerate(GROUPS)}


def assert_close(actual, expected, where):
    if not np.allclose(actual, expected, rtol=0, atol=2e-12):
        raise ValueError('Published analysis mismatch: ' + where)


def bootstrap(method, reference):
    ids = method[0]['ids']
    track = np.array(['|'.join(str(value).split('|')[:2]) for value in ids])
    unique, codes = np.unique(track, return_inverse=True)
    nt = len(unique)
    draws = 20000
    rng = np.random.default_rng(20261114)
    ti = rng.integers(0, nt, (draws, nt))
    si = rng.integers(0, 3, (draws, 3))
    output = {}
    for group, mask in masks(method[0]['groups']).items():
        counts = np.bincount(codes, weights=mask, minlength=nt)
        den = counts[ti].sum(1)
        valid = den > 0
        if len(np.unique(track[mask])) < 20 or valid.mean() < .95:
            raise ValueError('BOOTSTRAP_INSUFFICIENT_TRACKS')
        output[group] = {}
        for key in ('coverage', 'minADE', 'minFDE', 'effective_good'):
            delta = np.array([p[key] for p in method], float) - np.array([p[key] for p in reference], float)
            sums = np.array([np.bincount(codes, weights=value * mask, minlength=nt) for value in delta])
            selected = sums[:, ti].sum(2).T
            boot = selected[np.arange(draws)[:, None], si].mean(1)[valid] / den[valid]
            # A direct expanded-window formula checks a different arithmetic route.
            valid_indices = np.flatnonzero(valid)
            for index in valid_indices[:8]:
                multip = np.bincount(ti[index], minlength=nt)[codes] * mask
                direct = np.mean([(delta[seed] * multip).sum() / multip.sum() for seed in si[index]])
                assert_close(direct, boot[np.searchsorted(valid_indices, index)], 'independent bootstrap arithmetic')
            output[group][key] = dict(
                point=float(delta[:, mask].mean()), CI95=np.quantile(boot, [.025, .975]).tolist(),
                family176_CI=np.quantile(boot, [.025 / 176, 1 - .025 / 176]).tolist(),
                valid_draws=int(valid.sum()), tracks=int(len(np.unique(track[mask]))), draws=draws,
                seed=20261114, independent_expanded_window_checks=8)
    return output


def verify_e1():
    budget = read(PACKAGE / 'snapshot/E1/BUDGET_DECISION.json')
    for epoch in (8, 16, 32, 64):
        data = [population(f'E1B_E{epoch}_s{seed}', 'calib') for seed in (42, 43)]
        ade = np.mean([value['minADE'].mean() for value in data])
        coverage = np.mean([value['coverage'][value['groups'] == 2].mean() for value in data])
        assert_close(ade, budget['rows'][str(epoch)]['overall_minADE'], 'E1/calibration/ADE')
        assert_close(coverage, budget['rows'][str(epoch)]['change_coverage'], 'E1/calibration/coverage')
    noise = read(PACKAGE / 'snapshot/E1/E1A_NULL.json')
    contract = read(PACKAGE / 'snapshot/E2E_V2_CONTRACT.json')
    for split in ('fit', 'dev'):
        for label, names, expected in [
            ('E1A', noise[split]['reference_names'], noise[split]['summary']),
            ('E1C', contract['null'][split]['reference_names'], contract['null'][split]['summary']),
        ]:
            data = [population(name, split) for name in names]
            for group, mask in masks(data[0]['groups']).items():
                coverage, lost = [], []
                for i, reference in enumerate(data):
                    for j, method in enumerate(data):
                        if i == j:
                            continue
                        coverage.append(100 * (method['coverage'][mask].mean() - reference['coverage'][mask].mean()))
                        lost.append(float((reference['coverage'] & ~method['coverage'] & mask).sum() /
                                          (reference['coverage'] & mask).sum()))
                for key, values in [('coverage_diff_pp', coverage), ('lost_rate', lost)]:
                    assert_close(np.mean(values), expected[group][key]['mean'], label + '/' + split + '/' + group + '/' + key + '/mean')
                    assert_close(np.std(values, ddof=1), expected[group][key]['sd'], label + '/' + split + '/' + group + '/' + key + '/sd')
    return dict(calibration_budgets=4, calibration_populations=8, noise_reference_families=2,
                paired_noise_splits=2, E_star=budget['E_star'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify package hashes and full 20,000-draw bootstrap.')
    parser.add_argument('--output', type=Path, help='Write validation report to a new file.')
    args = parser.parse_args()
    manifest_path = PACKAGE / 'ARTIFACT_MANIFEST.json'
    hashes = 0
    if args.verify and manifest_path.exists():
        for relative, expected in read(manifest_path)['files'].items():
            path = (PACKAGE / relative).resolve()
            if not path.is_relative_to(PACKAGE.resolve()):
                raise ValueError('Publication path escapes package')
            if sha(path) != expected['sha256']:
                raise ValueError('Publication checksum mismatch: ' + relative)
            hashes += 1
    index = read(PACKAGE / 'analysis_inputs/INDEX.json')
    for entry in index['populations']:
        if sha(PACKAGE / entry['path']) != entry['sha256']:
            raise ValueError('Analysis input checksum mismatch: ' + entry['path'])
        value = population(entry['run'], entry['split'])
        aggregate_path = (PACKAGE / 'snapshot/legacy_reference/OLD42' / entry['split'] / 'metrics.json'
                          if entry['run'] == 'OLD42' else
                          PACKAGE / 'snapshot/runs' / entry['run'] / entry['split'] /
                          ('summary_v2.json' if entry['split'] == 'calib' else 'metrics.json'))
        aggregate = read(aggregate_path)
        for group, mask in masks(value['groups']).items():
            for key in ('coverage', 'minADE', 'minFDE'):
                assert_close(value[key][mask].mean(), aggregate[group][key], entry['path'] + '/' + group + '/' + key)
            if 'effective_good' in value:
                assert_close(value['effective_good'][mask].mean(), aggregate[group]['effective_good'], entry['path'] + '/effective_good')
    e1_validation = verify_e1()
    report = read(PACKAGE / 'snapshot/RESULT_ANALYSIS.json')
    checks = 0
    rows = []
    for comparison, result in report['comparisons'].items():
        for split in ('fit', 'dev'):
            data = result[split]
            method = [population(name, split) for name in data['method_names']]
            reference = [population(name, split) for name in data['reference_names']]
            for value in method + reference:
                if not np.array_equal(value['ids'], method[0]['ids']) or not np.array_equal(value['groups'], method[0]['groups']):
                    raise ValueError('Paired ID/group mismatch')
            for group, mask in masks(method[0]['groups']).items():
                expected = data['groups'][group]
                diffs = [100 * (float(m['coverage'][mask].mean()) - float(b['coverage'][mask].mean()))
                         for m, b in zip(method, reference)]
                assert_close(diffs, expected['coverage']['per_seed'], comparison + '/' + split + '/' + group + '/coverage')
                for key in ('minADE', 'minFDE'):
                    diffs = [float(m[key][mask].mean() - b[key][mask].mean()) for m, b in zip(method, reference)]
                    assert_close(diffs, expected[key + '_per_seed'], comparison + '/' + split + '/' + group + '/' + key)
                for m, b, exchange in zip(method, reference, expected['exchange']):
                    rescue = int((m['coverage'] & ~b['coverage'] & mask).sum())
                    lost = int((~m['coverage'] & b['coverage'] & mask).sum())
                    if rescue != exchange['rescue'] or lost != exchange['lost']:
                        raise ValueError('Rescue/lost mismatch: ' + comparison)
                checks += 1
            if split == 'dev':
                rows.append(dict(comparison=comparison, change_delta_pp=data['groups']['change']['coverage']['mean'],
                                 overall_delta_pp=data['groups']['overall']['coverage']['mean'],
                                 harm_groups=data['harm_groups'], verdict=result.get('verdict')))
                if args.verify:
                    rebuilt = bootstrap(method, reference)
                    for group in GROUPS:
                        for key in ('coverage', 'minADE', 'minFDE', 'effective_good'):
                            for field in ('point', 'CI95', 'family176_CI'):
                                assert_close(rebuilt[group][key][field], result['bootstrap_dev'][group][key][field],
                                             comparison + '/' + group + '/' + key + '/' + field)
                    checks += 16
        print(json.dumps(dict(verified=comparison), ensure_ascii=False), flush=True)
    validation = dict(status='PASS', populations=len(index['populations']), comparisons=len(rows),
                      E1=e1_validation,
                      group_endpoint_checks=checks, manifest_files_verified=hashes,
                      bootstrap_draws=20000 if args.verify else 0, bootstrap_seed=20261114,
                      tolerance=2e-12, rows=rows,
                      scope='Primary point metrics, per-seed rescue/lost and all saved bootstrap point/CI endpoints. '
                            'Motion/tail audits are archived original evidence, not recomputed from compact inputs.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(validation, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    main()
