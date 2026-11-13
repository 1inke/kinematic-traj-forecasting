"""C0: reconstruct causal neighbors; seal train-only empirical motion references."""
from collections import Counter
from itertools import combinations
import hashlib
import time
import numpy as np
from common import HERE, ROOT, ASTRA, sha, write_json, write_csv, seal, verify, now
from causal_dataset import RECORDINGS, load_recording, windows, past_input, path_hash, groups, motion


def summary(a):
    a = np.asarray(a, dtype=np.float64)
    return {'count': int(a.size), 'mean': float(a.mean()),
            **{f'p{q}': float(np.quantile(a, q / 100, method='linear'))
               for q in (95, 99, 99.5)}}


def main():
    started = time.perf_counter()
    out = HERE / 'C0'
    out.mkdir(exist_ok=True)
    if (out / 'EVALUATION_CONTRACT.json').exists():
        raise RuntimeError('C0_ALREADY_SEALED_DO_NOT_RECOMPUTE')
    label_root = ASTRA / 'datasets/eth_ucy/labels'
    data_root = ASTRA / 'datasets/eth_ucy/processed_data_diverse'
    source_paths = [label_root / 'eth' / (r + '_train.txt') for r in RECORDINGS]
    source_paths += [data_root / 'eth_data_train.npy', data_root / 'eth_num_train.npy']
    source_paths += [HERE / n for n in ('common.py', 'causal_dataset.py', 'data_lineage.py')]
    source_paths += list((ROOT / 'docs/E2E_MOTION_RESEARCH_RESET_20261102').glob('*.md'))
    source_paths += [ROOT / 'REGIME_CHANGE_DETECTOR_SERIES/run_detector_experiment.py',
                     ROOT / 'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/formal_config.json',
                     ASTRA / 'data/eth_preprocessor.py', ASTRA / 'data/process_eth.py',
                     ASTRA / 'data/eth.py']
    source = seal(source_paths)
    write_json(out / 'SOURCE_MANIFEST.json', {'files': source, 'at': now()})
    catalog, records, fingerprints = {}, [], {}
    raw_counter = Counter()
    for name, scene in RECORDINGS.items():
        frames, tracks, lo, hi = load_recording(label_root / 'eth' / (name + '_train.txt'))
        catalog[name] = (frames, tracks, lo, hi)
        cutoff = lo + .8 * (hi - lo)
        fp = {}
        for ped, points in tracks.items():
            t = sorted(points)
            value = np.stack([points[k] for k in t]).astype('<f4')
            fp[ped] = hashlib.sha256(np.asarray(t, '<i8').tobytes() + value.tobytes()).hexdigest()
        fingerprints[name] = fp
        for ped, start, path in windows(tracks):
            raw_counter[path_hash(scene, path)] += 1
            split = 'fit' if start + 19 < cutoff else ('dev' if start > cutoff else 'boundary_purged')
            records.append({'id': f'{name}|{ped}|{start}', 'recording': name,
                            'target': ped, 'start': start, 't0': start + 7, 'end': start + 19,
                            'cutoff': cutoff, 'split': split})
        print('RAW_RECORDING', name, 'target_windows', sum(r['recording'] == name for r in records), flush=True)
    # No scientific access to official ETH test contents. All-fold filenames/SHA inventory only.
    inventory = []
    for fold in ('eth', 'hotel', 'univ', 'zara01', 'zara02'):
        for p in sorted((label_root / fold).glob('*.txt')):
            inventory.append({'fold': fold, 'filename': p.name, 'sha256': sha(p),
                              'size': p.stat().st_size, 'access': 'BYTES_SHA_ONLY_NO_GT_PARSE'
                              if p not in source_paths else 'OFFICIAL_ETH_TRAIN_PARSED'})
    write_csv(out / 'ALL_FOLD_FILE_INVENTORY.csv', inventory)
    old = np.load(data_root / 'eth_data_train.npy', mmap_mode='r')
    old_counter = Counter(path_hash(int(row[0, 0, 0]), row[0, :, -2:]) for row in old)
    equivalence = {'processed_count': len(old), 'raw_count': len(records),
                   'processed_missing_in_raw': sum((old_counter - raw_counter).values()),
                   'raw_extra_vs_processed': sum((raw_counter - old_counter).values()),
                   'multiset_exact': old_counter == raw_counter}
    write_json(out / 'OFFICIAL_POOL_EQUIVALENCE.json', equivalence)
    if not equivalence['multiset_exact']:
        raise ValueError('OFFICIAL_TARGET_POOL_LINEAGE_UNRESOLVED')
    dev_tracks = {(r['recording'], r['target']) for r in records if r['split'] == 'dev'}
    for r in records:
        if r['split'] == 'fit' and (r['recording'], r['target']) in dev_tracks:
            r['split'] = 'dev_target_purged'
    # Exact cross-recording coordinate+time signatures can expose aliases; absence is not proof.
    aliases = []
    for x, y in combinations(RECORDINGS, 2):
        tx, ty = catalog[x][1], catalog[y][1]
        fx, fy = fingerprints[x], fingerprints[y]
        exact = sorted(set(fx.values()) & set(fy.values()))
        pointsx = {(t, *map(float, p)) for tr in tx.values() for t, p in tr.items()}
        pointsy = {(t, *map(float, p)) for tr in ty.values() for t, p in tr.items()}
        aliases.append({'recording_a': x, 'recording_b': y,
                        'exact_full_track_signatures': len(exact),
                        'exact_time_xy_points': len(pointsx & pointsy),
                        'same_footage_verified': False,
                        'relation': 'UNIVERSITY_SISTER_UNRESOLVED' if x.startswith(('students', 'uni_')) and y.startswith(('students', 'uni_'))
                        else 'ZARA_SISTER_UNRESOLVED' if 'zara' in x and 'zara' in y else 'NO_EXACT_ALIAS_FOUND_NOT_PHYSICAL_INDEPENDENCE_PROOF'})
    write_json(out / 'RECORDING_ALIAS_AUDIT.json', {'pairs': aliases,
        'policy': 'Exact duplicate tracks would STOP; sister-recording footage timing remains UNKNOWN; no claim of independent whole-scene generalization.'})
    if any(a['exact_full_track_signatures'] for a in aliases):
        raise ValueError('EXACT_RECORDING_ALIAS_REQUIRES_RESOLUTION')
    datasets = {}
    neighbor_rows, audit_rows = [], []
    for split in ('fit', 'dev'):
        selected = [r for r in records if r['split'] == split]
        if not selected:
            raise ValueError('EMPTY_PREREGISTERED_SPLIT')
        pasts, futures, ids, ns, neighbor_ids, groups_all, scores = [], [], [], [], [], [], []
        for r in selected:
            frames, tracks, _, _ = catalog[r['recording']]
            p, ni, n, eligible = past_input(frames, r['target'], r['t0'])
            f = np.stack([tracks[r['target']][j] for j in range(r['t0'] + 1, r['end'] + 1)])
            pasts.append(p); futures.append(f); ids.append(r['id']); ns.append(n); neighbor_ids.append(ni)
            neighbor_rows.append({'id': r['id'], 'split': split, 'eligible_past_neighbors': eligible,
                                  'used_neighbors': n - 1, 'truncated': max(eligible - 8, 0),
                                  'neighbor_ids': '|'.join(map(str, ni[:n]))})
        p, f = np.stack(pasts), np.stack(futures)
        g, score = groups(p[:, 0], f)
        data = dict(past=p, future=f, ids=np.asarray(ids), num_valid=np.asarray(ns, np.int64),
                    neighbor_ids=np.stack(neighbor_ids), group=g, change_score=score)
        path = out / (split + '.npz')
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **data)
        datasets[split] = data
        audit_rows.append({'split': split, 'windows': len(ids),
                           'target_tracks': len({(r['recording'], r['target']) for r in selected}),
                           **{n: int((g == i).sum()) for i, n in enumerate(('stable', 'transition', 'change'))}})
    write_csv(out / 'WINDOW_REGISTRY.csv', records)
    write_csv(out / 'NEIGHBOR_REGISTRY.csv', neighbor_rows)
    # Future deletion and corruption happen ONLY in memory audit copies.
    chosen = sorted([r for r in records if r['split'] in ('fit', 'dev')],
                    key=lambda r: hashlib.sha256(('causal-c0-v1|' + r['id']).encode()).hexdigest())[:128]
    for r in chosen:
        frames = catalog[r['recording']][0]
        expected = past_input(frames, r['target'], r['t0'])
        truncated = {t: v for t, v in frames.items() if t <= r['t0']}
        corrupted = dict(truncated)
        corrupted[r['t0'] + 1] = {987654321: np.array([1e9, -1e9], np.float32)}
        for modified in (truncated, corrupted):
            actual = past_input(modified, r['target'], r['t0'])
            if not all(np.array_equal(a, b) for a, b in zip(expected, actual)):
                raise ValueError('FUTURE_DEPENDENT_INPUT')
    counts = Counter(r['split'] for r in records)
    time_audit = []
    for name in RECORDINGS:
        fit = [r for r in records if r['recording'] == name and r['split'] == 'fit']
        dev = [r for r in records if r['recording'] == name and r['split'] == 'dev']
        fitend = max((r['end'] for r in fit), default=None)
        devstart = min((r['start'] for r in dev), default=None)
        overlap = len({r['target'] for r in fit} & {r['target'] for r in dev})
        if overlap or (fit and dev and fitend >= devstart):
            raise ValueError('WITHIN_RECORDING_SPLIT_OVERLAP')
        time_audit.append({'recording': name, 'min': catalog[name][2], 'max': catalog[name][3],
                           'cutoff': catalog[name][2] + .8 * (catalog[name][3] - catalog[name][2]),
                           'fit': len(fit), 'dev': len(dev), 'fit_max_end': fitend,
                           'dev_min_start': devstart, 'same_target_overlap': overlap,
                           'same_frame_or_future_interval_overlap': False})
    fit = datasets['fit']
    m = motion(fit['past'][:, 0], fit['future'])
    speed = np.linalg.norm(np.diff(fit['past'][:, 0].astype(np.float64), axis=1) / .4, axis=-1).mean(1)
    bins = np.quantile(speed, [.25, .5, .75], method='linear')
    strata = np.searchsorted(bins, speed, side='right')
    per_step = {k: [summary(v[:, i]) for i in range(v.shape[1])] for k, v in m.items()}
    # Presealed sensitivity only: independent 1cm coordinate noise on PAST, fixed RNG.
    noise = np.random.default_rng(20261102).normal(0, .01, fit['past'][:, 0].shape)
    mn = motion(fit['past'][:, 0].astype(np.float64) + noise, fit['future'])
    reference = {'scope': 'FIT_RAW_GT_ONLY', 'dt_seconds': .4,
                 'quantile_method': 'numpy_linear', 'windows': len(fit['ids']),
                 'target_tracks': audit_rows[0]['target_tracks'], 'raw': {k: summary(v) for k, v in m.items()},
                 'per_step': per_step, 'speed_quartile_edges': bins.tolist(),
                 'speed_strata': {str(i): {'windows': int((strata == i).sum()),
                    'metrics': {k: summary(v[strata == i]) if (strata == i).any() else None for k, v in m.items()}} for i in range(4)},
                 'past_noise_sensitivity': {'sigma_m': .01, 'seed': 20261102,
                    'GT_future_unchanged': True, 'model_input_unchanged': True,
                    'metrics': {k: summary(v) for k, v in mn.items()}},
                 'interpretation': 'Empirical, window-weighted and correlated; not physiological limits or independent samples.'}
    write_json(out / 'MOTION_REFERENCE.json', reference)
    write_json(out / 'DATA_LINEAGE.json', {'status': 'PASS_OFFICIAL_TARGET_MULTISET_CAUSAL_REBUILD',
        'pool_equivalence': equivalence, 'recordings': list(RECORDINGS), 'test_access': 'FILE_SHA_ONLY',
        'time_semantics': 'Official label sampling indices, contiguous step1; dt .4 from official frequency2.5 config; NOT verified video frame numbers.',
        'all_fold_inventory': 'ALL_FOLD_FILE_INVENTORY.csv', 'cross_recording_footage_independence': 'UNKNOWN_SISTER_RECORDINGS_NOT_CLAIMED',
        'old_scene_id_collapses_recordings': True, 'source_files': source})
    write_json(out / 'SPLIT_AUDIT.json', {'status': 'PASS_WITHIN_RECORDING_TIME_AND_TARGET_DISJOINT',
        'rule': 'last20% raw official TRAIN recording time; full20 touching boundary purged; fit removes dev target tracks',
        'counts': dict(counts), 'populations': audit_rows, 'recordings': time_audit,
        'cross_recording_physical_context_independence': 'UNVERIFIED_NOT_CLAIMED',
        'no_data_files_deleted': True})
    write_json(out / 'CAUSAL_NEIGHBOR_AUDIT.json', {'status': 'PASS_PAST_ONLY_128_FUTURE_TRUNCATION_AND_CORRUPTION_EXACT',
        'old_preprocessing_future_eligibility': True, 'policy': 'Full8 observed, nearest t0 distance then stable numericID; focal+8; no future eligibility; padded agents num_valid mask',
        'tested': [r['id'] for r in chosen], 'windows': len(neighbor_rows),
        'truncated_windows': sum(r['truncated'] > 0 for r in neighbor_rows),
        'neighbor_inputs_future_reads': 0, 'disk_data_deletions': 0})
    write_json(out / 'EVALUATION_CONTRACT.json', {'status': 'SEALED_BEFORE_B0_SCIENTIFIC_OUTPUT',
        'ledger': ['E2E_V1_EXPLORATORY', 'LEGACY_SHADOW'], 'pilot': {'seed': 42, 'epochs': 8, 'batch': 64, 'selection': 'FIXED_FINAL_EMA_0.995'},
        'main_future_candidate_quality': 'J=max(ADE/.2,FDE/.4), good if J<=1 same physical slot',
        'groups': {'stable': 'score<.25', 'transition': '.25<=score<.5', 'change': 'score>=.5',
                   'source': 'historical future_change_score formula; float64 world evaluation; never model input'},
        'cluster': 'SciPy complete-link D=max(meanpath/.10,endpoint/.20), cut1, full20',
        'motion': '12v11a10j float64 dt.4; pooled five: speedp95/accmean/accp95/jerkmean/jerkp95; junction separately',
        'tail_thresholds': {k: v['p99.5'] for k, v in reference['raw'].items()},
        'tail_noninferiority_later_vs_fresh_B0': {'exceedance_rate_max_increase_pp': 1, 'positive_exceedance_magnitude_p95_max_relative_harm': .01,
            'zero_reference_or_less_than20_tracks': 'NOT_ESTIMABLE_NO_FLOOR'},
        'continuous_later_pilot_screen': {'overall_or_change_ADE_improves': True, 'other_ADE_max_harm': .01, 'overall_stable_transition_FDE_max_harm': .01},
        'legacy_old_native_new_source': 'NOT_COMPARABLE_NO_MATCHED_OLD_NATIVE_OUTPUT; do not rename CV to old native',
        'single_seed_significance': 'NOT_PERMITTED', 'later_family2': 'overallADE UDelta-GRU and DDiff-DGate corrected97.5% wholetrack+seed',
        'budget': 'full conservative pilot <=1800 seconds; no automatic multi-day formal runs'})
    verify(source)
    outputs = seal([p for p in out.iterdir() if p.is_file()])
    write_json(out / 'C0_RESULT.json', {'status': 'COMPLETE_C0_ALLOW_B0_OWN_PREFLIGHT',
        'elapsed_seconds': time.perf_counter() - started, 'at': now(), 'outputs': outputs,
        'source_final_unchanged': True, 'training_updates': 0,
        'qualification': 'Within-recording clean exposure; sister-footage identity unresolved explicitly, not a blind whole-scene generalization certificate.'})
    print('C0_COMPLETE', audit_rows, flush=True)


if __name__ == '__main__':
    main()
