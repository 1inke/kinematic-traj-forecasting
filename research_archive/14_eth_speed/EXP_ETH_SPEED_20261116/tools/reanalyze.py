"""Portable NumPy-only audit of published errors/labels; no model inference.

Coordinate errors and evaluation maneuver labels were independently recomputed
locally before publication. Their raw coordinates are intentionally not included.
"""
from pathlib import Path, PureWindowsPath
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT/'snapshot'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def equal(actual, expected, label='root'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (label, 'keys')
        for k, v in expected.items(): equal(actual[k], v, label+'/'+k)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for i, v in enumerate(expected): equal(actual[i], v, label+'/'+str(i))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, atol=1e-12, rtol=1e-12), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def gain(b, a): return (b-a)/b if b is not None and a is not None and b > 0 else None


def block(w, mask):
    n = int(mask.sum())
    if not n: return dict(n=0, small_group=True, minADE=None, minFDE=None, coverage=None)
    result = {k: float(w[k][mask].mean()) for k in
              ('minADE', 'minFDE', 'coverage', 'top1_ADE', 'top1_FDE', 'top1_good')}
    result.update(n=n, small_group=n < 20, ADE_p95=float(np.quantile(w['minADE'][mask], .95)),
                  FDE_p95=float(np.quantile(w['minFDE'][mask], .95)))
    return result


def summary(w, labels):
    speed = labels['speed_bin']; maneuver = labels['maneuver']; group = labels['group']
    maneuvers = ('stationary', 'start', 'stop', 'turn_and_speed', 'turn', 'acceleration', 'deceleration', 'steady')
    return dict(overall=block(w, np.ones(len(group), bool)),
        speed={name: block(w, speed == i) for i, name in enumerate(('low', 'middle', 'high'))},
        maneuver={m: block(w, maneuver == m) for m in maneuvers},
        original_group={name: block(w, group == i) for i, name in enumerate(('stable', 'transition', 'change'))},
        speed_by_maneuver={name: {m: block(w, (speed == i) & (maneuver == m)) for m in maneuvers}
                           for i, name in enumerate(('low', 'middle', 'high'))})


def paired(b, a, mask):
    count = int(mask.sum())
    if not count: return dict(n=0)
    gb, ga = b['coverage'], a['coverage']
    result = dict(n=count, B0_covered=int(gb[mask].sum()), AUG_covered=int(ga[mask].sum()),
                  rescued=int((mask & ~gb & ga).sum()), lost=int((mask & gb & ~ga).sum()),
                  top1_B0_good=int(b['top1_good'][mask].sum()), top1_AUG_good=int(a['top1_good'][mask].sum()))
    for k in ('minADE', 'minFDE', 'top1_ADE', 'top1_FDE', 'same_path_FDE'):
        before, after = b[k][mask], a[k][mask]; delta = after-before
        result[k] = dict(B0=float(before.mean()), AUG=float(after.mean()),
            relative_gain=gain(float(before.mean()), float(after.mean())), delta_sum=float(delta.sum()),
            improved_windows=int((delta < -1e-9).sum()), worsened_windows=int((delta > 1e-9).sum()),
            tied_windows=int((np.abs(delta) <= 1e-9).sum()),
            B0_p95=float(np.quantile(before, .95)), AUG_p95=float(np.quantile(after, .95)),
            B0_p99=float(np.quantile(before, .99)), AUG_p99=float(np.quantile(after, .99)))
    return result


def main():
    manifest = read(ROOT/'ARTIFACT_MANIFEST.json')['files']
    for rel, digest in manifest.items():
        p = (ROOT/rel).resolve(); assert p.is_relative_to(ROOT.resolve()); assert sha(p) == digest, rel
    copies = read(ROOT/'COPY_PROVENANCE.json')['copies']
    norm = lambda x: x.replace('\\', '/').casefold()
    mapping = {norm(row['source']): row['archive_path'] for row in copies}
    for row in copies: assert sha(ROOT/row['archive_path']) == row['sha256']
    def source_path(original): return ROOT/mapping[norm(original)]
    inputs = read(SNAP/'INPUT_CONTRACT.json'); input_sha = sha(SNAP/'INPUT_CONTRACT.json')
    for original, digest in inputs['files'].items():
        if norm(original) in mapping: assert sha(source_path(original)) == digest
    published_seal_entries = 0
    for seal in (SNAP/'DIAG/DONE.json', SNAP/'SPEED_AUG/B0_wide_eth_E64_s42/DONE.json'):
        rec = read(seal); assert rec['input_contract_sha256'] == input_sha
        for original, digest in rec['files'].items():
            if norm(original) in mapping:
                assert sha(source_path(original)) == digest; published_seal_entries += 1
            else:
                assert original.endswith(('.pt', '/predictions.npz', '\\predictions.npz')), original
    status = read(ROOT/'STATUS.json'); analysis = read(SNAP/'ANALYSIS_20261116.json')
    diag = read(SNAP/'DIAG/RESULTS.json'); decision = read(SNAP/'DIAG/DECISION.json')
    final = read(SNAP/'SPEED_AUG/B0_wide_eth_E64_s42/RESULTS.json')
    assert status['execution_status'] == 'COMPLETE'
    assert read(SNAP/'LAST_EXIT.json')['exit_code'] == 0
    assert read(SNAP/'LAST_EXIT.json')['status'] == 'SUCCESS'
    assert not status['baseline_upgraded'] and not status['extra_seeds_executed']
    assert not status['old_fivefold_resumed'] and not status['COMBO_executed']
    assert final['contract']['input_contract_sha256'] == input_sha

    labels = {key: arrays(ROOT/f'analysis_inputs/{key}_labels.npz') for key in ('eth_test', 'hotel_test', 'eth_val')}
    metrics = {}; rebuilt = {}; populations = 0
    for fold in ('eth', 'hotel'):
        label = labels[fold+'_test']; profile = diag['folds'][fold]['profile']
        training_speed = label['train_observed_speed_mps']
        fit = dict(n=len(training_speed), p50=float(np.quantile(training_speed, .5)),
                   p90=float(np.quantile(training_speed, .9)), p95=float(np.quantile(training_speed, .95)),
                   maximum=float(training_speed.max()))
        equal(fit, profile, fold+'/training_profile')
        bins = np.searchsorted([fit['p50'], fit['p90']], label['observed_speed_mps'], side='right')
        np.testing.assert_array_equal(bins, label['speed_bin'])
        scale = np.clip(fit['p90']/np.maximum(label['observed_speed_mps'], fit['p90']), .5, 1.)
        scale_input = arrays(SNAP/f'DIAG/{fold}/scale_inputs.npz')
        np.testing.assert_allclose(scale, scale_input['scale'], atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(scale_input['ids'], label['ids'])
        rebuilt[fold] = dict(cases={})
        for case, expected in diag['folds'][fold]['cases'].items():
            w = arrays(SNAP/f'DIAG/{fold}/{case}/window_metrics.npz')
            for field in ('ids', 'group', 'speed_bin'): np.testing.assert_array_equal(w[field], label[field])
            actual = summary(w, label); equal(actual, expected, fold+'/'+case)
            metrics[fold+'/'+case] = w; rebuilt[fold]['cases'][case] = actual; populations += 1
        rebuilt[fold].update(baseline_replay_passed=diag['folds'][fold]['baseline_replay_passed'],
                             low_speed_identity_passed=diag['folds'][fold]['low_speed_identity_passed'])
    aug = {}
    for split in ('test', 'val'):
        w = arrays(SNAP/f'SPEED_AUG/B0_wide_eth_E64_s42/{split}/window_metrics.npz')
        label = labels['eth_'+split]
        for field in ('ids','group','speed_bin'): np.testing.assert_array_equal(w[field], label[field])
        actual = summary(w, label); equal(actual, final['outputs'][split], 'AUG/'+split)
        aug[split] = w; populations += 1
    vb = arrays(ROOT/'references/B0_eth_E64_s42/val_window_metrics.npz')
    equal(summary(vb, labels['eth_val']), analysis['val']['B0'], 'baseline_validation')
    equal(summary(aug['val'], labels['eth_val']), analysis['val']['AUG'], 'aug_validation')

    b = metrics['eth/identity']; a = aug['test']; lab = labels['eth_test']; all_mask = np.ones(364, bool)
    results = dict(overall=paired(b, a, all_mask),
       speed={name: paired(b, a, lab['speed_bin'] == i) for i, name in enumerate(('low','middle','high'))},
       maneuver={name: paired(b, a, lab['maneuver'] == name) for name in sorted(set(lab['maneuver']))},
       original_group={name: paired(b, a, lab['group'] == i) for i, name in enumerate(('stable','transition','change'))},
       speed_by_maneuver={name: {m: paired(b, a, (lab['speed_bin'] == i) & (lab['maneuver'] == m)) for m in sorted(set(lab['maneuver']))}
                          for i, name in enumerate(('low','middle','high'))})
    for family in ('speed','maneuver','original_group'):
        for row in results[family].values():
            if row['n']: row['share_of_overall_ADE_delta'] = row['minADE']['delta_sum']/results['overall']['minADE']['delta_sum']
    equal(results, analysis['paired'], 'all_paired_groups')
    gains = {k: gain(float(b[k].mean()),float(a[k].mean())) for k in ('minADE','minFDE')}
    verdict = 'PROMISING_SINGLE_SEED' if all(v >= .02 for v in gains.values()) else 'NO_CLEAR_JOINT_GAIN'
    assert verdict == final['comparisons']['verdict'] == status['original_fixed_verdict']
    equal(gains['minADE'], final['comparisons']['ADE_gain']); equal(gains['minFDE'], final['comparisons']['FDE_gain'])
    equal(gains['minADE'], status['relative_ADE_improvement']); equal(gains['minFDE'], status['relative_FDE_improvement'])

    # Rebuild the fixed diagnostic triage from reconstructed metric summaries.
    sys_path = str(ROOT/'design_original/code')
    import sys
    sys.path.insert(0, sys_path)
    from eth_speed_math import diagnostic_decision
    rebuilt_decision = diagnostic_decision(rebuilt, read(SNAP/'DIAG/PREPROCESSING_AUDIT.json'))
    equal(rebuilt_decision, {k:v for k,v in decision.items() if k != 'input_contract_sha256'}, 'diag_triage')

    endpoint = {}
    for label, mask in [('all', all_mask), ('deceleration', lab['maneuver'] == 'deceleration'),
                        ('steady', lab['maneuver'] == 'steady'), ('change', lab['group'] == 2)]:
        mask = mask & lab['heading_valid']; rec = dict(n=int(mask.sum()))
        for arm in ('B0','AUG'):
            for mode in ('Top1','oracle_FDE'):
                values = lab[arm+'_'+mode+'_longitudinal_error'][mask]
                rec[arm+'_'+mode] = dict(mean_signed_longitudinal_error=float(values.mean()),
                                         positive_error_fraction=float((values > 0).mean()))
        endpoint[label] = rec
    equal(endpoint, analysis['endpoint_diagnostics'], 'derived_endpoint_errors')
    original = inputs['config']['baselines']['eth']['receipt']
    receipt = read(source_path(original)); olddir = PureWindowsPath(original).parent
    times = []
    for i in range(1,65):
        row = read(SNAP/f'SPEED_AUG/B0_wide_eth_E64_s42/ckpt_epoch{i:03d}.json')
        old = read(source_path(str(olddir/f'ckpt_epoch{i:03d}.json')))
        assert row['epoch'] == i and row['updates'] == 450*i and row['order_hash'] == old['order_hash']
        times.append(row['epoch_wall_seconds'])
    assert row['order_hash'] == receipt['train']['order_hash'] == final['order_hash']
    assert final['contract']['train_index_sha256'] == receipt['contract']['train_index_sha']
    equal(row['session_elapsed_seconds'], analysis['timing']['training_session_seconds'])
    equal(float(np.median(times)), analysis['timing']['epoch_seconds_median'])
    print(json.dumps(dict(status='PASS', execution='COMPLETE', fixed_verdict=verdict,
        artifact_hashes_verified=len(manifest), published_sealed_outputs_verified=published_seal_entries,
        metric_populations_rebuilt=populations, baseline_validation_rebuilt=True,
        all_64_epoch_orders_match=True, new_training_runs=1, relative_ADE_improvement=gains['minADE'],
        relative_FDE_improvement=gains['minFDE'], coverage_rescued=results['overall']['rescued'],
        coverage_lost=results['overall']['lost'], baseline_upgraded=False,
        limitation='Saved-error and derived-label reanalysis only. Raw coordinate errors, maneuver labels and endpoint geometry cannot be rederived without omitted local targets/predictions.'),indent=2))


if __name__ == '__main__': main()
