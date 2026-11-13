"""CAL v1.1 proposal builder. READ-ONLY on all project inputs.

Rule (variant B in qoder_work/C3_C6_划分与来源问题分析_20261020.md)
-----------------------------------------------------------------
CAL_v1.1 = original_CAL
           minus every window whose track appears in FIT / inner / outer
           minus every window whose [start, start+19] interval overlaps a FIT
           interval of the same sequence (implemented as a per-sequence cutoff).

Why both clauses:
  * track clause  -> removes same-person dependence (PROTOCOL 2.3 / 2.4).
  * time clause   -> removes the 19-frame boundary band where FIT futures and
                     CAL futures are the same physical trajectory (C3).

This directory is the only write target. cache / metadata / splits are opened
read-only and their SHA-256 is re-checked after the build.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent


def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / 'src' / 'f16').is_dir() and (p / 'config' / 'data_manifest.v1.json').is_file():
            return p
    raise RuntimeError('project root not found from ' + str(start))


ROOT = find_root(OUT)

CACHE = ROOT / 'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
META = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
SPLITS = ROOT / 'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'

EXPECT = dict(windows=1620, change=443, tracks=77, change_tracks=50)
HORIZON = 20  # observed 8 + predicted 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load() -> dict:
    with np.load(CACHE, allow_pickle=False) as z:
        ids = z['sample_id'].copy()
        groups = z['change_group'].astype(int)
    with np.load(META, allow_pickle=False) as z:
        meta = {k: z[k].copy() for k in z.files}
    order = np.argsort(meta['sample_id'])
    pos = np.searchsorted(meta['sample_id'][order], ids)
    if np.any(pos >= len(order)):
        raise ValueError('metadata does not cover every cached sample_id')
    loc = order[pos]
    if not np.array_equal(meta['sample_id'][loc], ids):
        raise ValueError('metadata alignment failed')
    md = {k: v[loc] for k, v in meta.items()}

    with np.load(SPLITS, allow_pickle=False) as z:
        if not np.array_equal(z['sample_id'], ids[:25912]):
            raise ValueError('split registry / cache identity mismatch')
        split = {
            'FIT': z['grouped_train_index'].copy(),
            'inner': z['inner_cal_index'].copy(),
            'outer': z['outer_eval_index'].copy(),
        }
    split['original_CAL'] = np.arange(25912, 28791)

    track = np.array([f'{s}::{int(p)}' for s, p in zip(md['sequence'], md['pedestrian_id'])])
    return dict(ids=ids, groups=groups, md=md, split=split, track=track,
                start=md['start_frame'].astype(np.int64),
                unknown=md['ambiguous_track_time'].astype(bool) | (md['start_frame'] < 0),
                seq=md['sequence'])


def interval_overlap(a: np.ndarray, b: np.ndarray, start: np.ndarray,
                     unknown: np.ndarray, seq: np.ndarray) -> int:
    """Count windows in `a` whose [start, start+19] intersects any window in `b`
    **of the same sequence** (frame indices are only comparable within a video).

    Windows with unknown time never count as overlapping; they are reported
    separately by the caller.
    """
    hits = 0
    for i in a:
        if unknown[i]:
            continue
        t, end = int(start[i]), int(start[i]) + HORIZON - 1
        ok = b[(seq[b] == seq[i]) & ~unknown[b]]
        if len(ok) and np.any((start[ok] <= end) & (start[ok] + HORIZON - 1 >= t)):
            hits += 1
    return hits


def main() -> None:
    data = load()
    ids, groups, track, start = data['ids'], data['groups'], data['track'], data['start']
    seq, unknown, split = data['seq'], data['unknown'], data['split']

    cal = split['original_CAL']
    others = ['FIT', 'inner', 'outer']
    other_tracks = np.unique(np.concatenate([track[split[n]] for n in others]))
    excluded_tracks = np.intersect1d(track[cal], other_tracks)

    # Per-sequence time cutoff: first frame with no FIT interval of that sequence alive.
    cutoff = {}
    for s in np.unique(seq[cal]):
        fit_s = split['FIT'][(seq[split['FIT']] == s) & ~unknown[split['FIT']]]
        cutoff[str(s)] = -1 if len(fit_s) == 0 else int(start[fit_s].max()) + HORIZON

    keep_track = ~np.isin(track[cal], excluded_tracks)
    keep_time = np.array([int(start[i]) >= cutoff[str(seq[i])] for i in cal])
    keep = keep_track & keep_time
    cal11 = np.sort(cal[keep])

    got = dict(windows=len(cal11), change=int((groups[cal11] == 2).sum()),
               tracks=len(set(track[cal11])),
               change_tracks=len(set(track[cal11][groups[cal11] == 2])))
    if got != EXPECT:
        raise RuntimeError(f'CAL v1.1 population {got} != expected {EXPECT}')

    # ---- invariants -------------------------------------------------------
    checks = {}
    checks['C1_sample_id'] = {n: len(np.intersect1d(ids[cal11], ids[split[n]])) for n in others}
    checks['C2_track'] = {n: len(np.intersect1d(track[cal11], track[split[n]])) for n in others}
    checks['C3_interval'] = {n: interval_overlap(cal11, split[n], start, unknown, seq) for n in others}
    fit_keys = {(str(seq[i]), int(start[i])) for i in split['FIT'] if not unknown[i]}
    checks['C7_same_frame'] = {n: int(sum((str(seq[i]), int(start[i])) in fit_keys for i in cal11))
                               for n in ['FIT']}
    checks['subset_of_original_CAL'] = bool(np.all(np.isin(cal11, cal)))
    checks['unknown_time_windows'] = int(unknown[cal11].sum())
    checks['duplicate_ids'] = int(len(cal11) - len(np.unique(ids[cal11])))

    bad = {k: v for k, v in checks.items()
           if isinstance(v, dict) and any(v.values())}
    if bad or not checks['subset_of_original_CAL'] or checks['unknown_time_windows'] \
            or checks['duplicate_ids']:
        raise RuntimeError('invariant failure: ' + json.dumps({k: checks[k] for k in bad}))

    # ---- unassigned population (C8) --------------------------------------
    used = np.zeros(len(ids), bool)
    for ix in split.values():
        used[ix] = True
    unassigned = np.flatnonzero(~used)

    # ---- write -----------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / 'cal_v11_index.npy'
    np.save(index_path, cal11)

    with open(OUT / 'cal_v11_windows.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'track', 'sequence', 'start_frame', 'end_frame', 'change'])
        for i in cal11:
            w.writerow([int(ids[i]), track[i], seq[i], int(start[i]),
                        int(start[i]) + HORIZON - 1, int(groups[i] == 2)])

    with open(OUT / 'excluded_tracks.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['track', 'source_split', 'n_windows_in_cal', 'n_change_in_cal'])
        for t in excluded_tracks:
            in_cal = cal[track[cal] == t]
            src = [n for n in others if np.any(track[split[n]] == t)]
            w.writerow([t, '+'.join(src), len(in_cal), int((groups[in_cal] == 2).sum())])

    dropped = cal[~np.isin(cal, cal11)]
    inputs = {p.relative_to(ROOT).as_posix(): sha256(p) for p in [CACHE, META, SPLITS]}
    outputs = {p.name: sha256(p) for p in [index_path, OUT / 'cal_v11_windows.csv',
                                           OUT / 'excluded_tracks.csv']}

    report = dict(
        split_version='1.1',
        state='PROPOSAL_NOT_AUTHOR_FROZEN',
        base='original_CAL = registry indices 25912..28790',
        rule=dict(
            drop_tracks='tracks shared with FIT/inner/outer',
            drop_time='start < per-sequence FIT cutoff (max FIT end + 1)',
            per_sequence_cutoff=cutoff,
        ),
        population=dict(
            original_CAL=dict(windows=len(cal), change=int((groups[cal] == 2).sum()),
                              tracks=len(set(track[cal])),
                              change_tracks=len(set(track[cal][groups[cal] == 2]))),
            cal_v11=got,
            dropped=dict(windows=len(dropped), change=int((groups[dropped] == 2).sum()),
                         tracks=len(excluded_tracks)),
            untouched={n: dict(windows=len(split[n]), change=int((groups[split[n]] == 2).sum()),
                               tracks=len(set(track[split[n]]))) for n in others},
        ),
        checks=checks,
        unassigned_registry=dict(windows=len(unassigned),
                                 change=int((groups[unassigned] == 2).sum()),
                                 rule='excluded from every train / eval population in v1.1'),
        inputs_sha256=inputs,
        outputs_sha256=outputs,
        script_sha256=sha256(Path(__file__).resolve()),
        inputs_unchanged={k: inputs[k] == sha256(ROOT / k) for k in inputs},
    )
    report['inputs_unchanged'] = all(report['inputs_unchanged'].values())
    if not report['inputs_unchanged']:
        raise RuntimeError('an input changed during the build')
    (OUT / 'SPLIT_V1_1.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(cal_v11=got, dropped=report['population']['dropped'],
                          checks={k: checks[k] for k in ['C1_sample_id', 'C2_track',
                                                         'C3_interval', 'C7_same_frame']},
                          inputs_unchanged=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
