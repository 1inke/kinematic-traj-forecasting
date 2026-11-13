"""Reproduce the C3 / C6 / C7 / C8 evidence used in the v1.1 analysis.

Read-only. Recomputes from the frozen cache / metadata / split registry and the
native training manifest, and writes CONTEXT_AUDIT.json next to this script.
Nothing outside this directory is written.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent


def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / 'src' / 'f16').is_dir() and (p / 'config' / 'data_manifest.v1.json').is_file():
            return p
    raise RuntimeError('project root not found')


ROOT = find_root(OUT)
CACHE = ROOT / 'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
META = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
SPLITS = ROOT / 'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'
NATIVE = ROOT / ('OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/'
                 'results/eth/h3/f16_temporal_social_cross/seed_1/val_scene_-1/split_indices.npz')

HORIZON = 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load():
    with np.load(CACHE, allow_pickle=False) as z:
        ids = z['sample_id'].copy()
        groups = z['change_group'].astype(int)
    with np.load(META, allow_pickle=False) as z:
        meta = {k: z[k].copy() for k in z.files}
    pos = np.searchsorted(meta['sample_id'][np.argsort(meta['sample_id'])], ids)
    order = np.argsort(meta['sample_id'])
    loc = order[pos]
    if not np.array_equal(meta['sample_id'][loc], ids):
        raise ValueError('metadata alignment failed')
    md = {k: v[loc] for k, v in meta.items()}
    with np.load(SPLITS, allow_pickle=False) as z:
        split = {'FIT': z['grouped_train_index'].copy(),
                 'inner': z['inner_cal_index'].copy(),
                 'outer': z['outer_eval_index'].copy()}
    split['original_CAL'] = np.arange(25912, 28791)
    track = np.array([f'{s}::{int(p)}' for s, p in zip(md['sequence'], md['pedestrian_id'])])
    return ids, groups, md, split, track


def c3(fit, ev, track, start, unknown):
    by = {}
    for i in fit:
        if not unknown[i] and start[i] >= 0:
            by.setdefault(track[i], []).append((int(start[i]), int(start[i]) + HORIZON - 1))
    for v in by.values():
        v.sort()
    full = future = contained = same = unknown_ev = 0
    for i in ev:
        if track[i] in by:
            same += 1
        if unknown[i] or start[i] < 0:
            unknown_ev += 1
            continue
        t, end = int(start[i]), int(start[i]) + HORIZON - 1
        iv = by.get(track[i], ())
        if any(a <= end and b >= t for a, b in iv):
            full += 1
        if any(a <= end and b >= t + 8 for a, b in iv):
            future += 1
        if any(a <= t + 8 and b >= end for a, b in iv):
            contained += 1
    return dict(n=len(ev), same_track=same, full_interval_overlap=full,
                future_intersection=future, entire_future12_contained=contained,
                unknown_eval=unknown_ev)


def main():
    ids, groups, md, split, track = load()
    seq = md['sequence']
    start = md['start_frame'].astype(np.int64)
    unknown = md['ambiguous_track_time'].astype(bool) | (start < 0)

    cal11 = np.load(OUT / 'cal_v11_index.npy')
    pops = dict(split)
    pops['cal_v11'] = cal11

    # C7: same (sequence, start_frame) co-occurrence with FIT, different track.
    fit_keys = {}
    for i in split['FIT']:
        if not unknown[i]:
            fit_keys.setdefault((str(seq[i]), int(start[i])), set()).add(track[i])
    c7 = {}
    for name, ix in pops.items():
        if name == 'FIT':
            continue
        hit = sum(1 for i in ix
                  if (str(seq[i]), int(start[i])) in fit_keys
                  and track[i] not in fit_keys[(str(seq[i]), int(start[i]))])
        c7[name] = dict(windows_with_same_frame_fit_neighbour=hit, n=len(ix),
                        change=int(sum(1 for i in ix
                                       if (str(seq[i]), int(start[i])) in fit_keys
                                       and track[i] not in fit_keys[(str(seq[i]), int(start[i]))]
                                       and groups[i] == 2)))

    # C3 recheck on the frozen splits (must reproduce the audit's 457 / 295 / 0).
    c3_out = {}
    for name in ['inner', 'outer', 'original_CAL']:
        ix = split[name]
        c3_out[name] = dict(
            overall=c3(split['FIT'], ix, track, start, unknown),
            change=c3(split['FIT'], ix[groups[ix] == 2], track, start, unknown))

    # C8: registered but unassigned windows.
    used = np.zeros(len(ids), bool)
    for ix in split.values():
        used[ix] = True
    un = np.flatnonzero(~used)
    c8 = dict(windows=len(un), change=int((groups[un] == 2).sum()),
              unknown_time=int(unknown[un].sum()),
              by_sequence=dict(Counter(str(s) for s in seq[un]).most_common()),
              assigned=len(ids) - len(un), registry=len(ids))

    # C6: native training manifest coverage.
    with np.load(NATIVE, allow_pickle=False) as z:
        train = set(int(x) for x in z['train'])
        val = np.array([int(x) for x in z['validation']], dtype=np.int64)
    c6 = dict(artifact=NATIVE.relative_to(ROOT).as_posix(),
              train_unique=len(train), validation=len(val),
              outer_in_train=int(sum(int(ids[i]) in train for i in split['outer'])),
              outer_n=len(split['outer']),
              cal_in_train=int(sum(int(ids[i]) in train for i in split['original_CAL'])),
              cal_n=len(split['original_CAL']),
              fit_in_train=int(sum(int(ids[i]) in train for i in split['FIT'])),
              fit_n=len(split['FIT']),
              validation_in_train=int(sum(int(x) in train for x in val)))

    # native validation tail composition (separate pool, never in gradient per manifest)
    with np.load(META, allow_pickle=False) as z:
        allseq = z['sequence']
        allid = z['sample_id']
    look = {int(s): i for i, s in enumerate(allid)}
    val_seq = [str(allseq[look[int(x)]]) for x in val]
    c6['validation_sequences'] = dict(Counter(val_seq).most_common())

    inputs = {p.relative_to(ROOT).as_posix(): sha256(p) for p in [CACHE, META, SPLITS, NATIVE]}
    report = dict(scope='READ_ONLY_REPRODUCTION_OF_ANALYSIS_CLAIMS',
                  C7_same_frame_cooccurrence=c7,
                  C3_frozen_splits=c3_out,
                  C8_unassigned=c8,
                  C6_native_exposure=c6,
                  inputs_sha256=inputs,
                  script_sha256=sha256(Path(__file__).resolve()),
                  inputs_unchanged=all(inputs[k] == sha256(ROOT / k) for k in inputs))
    if not report['inputs_unchanged']:
        raise RuntimeError('an input changed during the audit')
    (OUT / 'CONTEXT_AUDIT.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(C3={k: v['overall']['full_interval_overlap'] for k, v in c3_out.items()},
                          CAL_future=c3_out['original_CAL']['overall']['future_intersection'],
                          CAL_contained=c3_out['original_CAL']['overall']['entire_future12_contained'],
                          C7=c7, C8=dict(windows=c8['windows'], change=c8['change']),
                          C6=dict(outer=f"{c6['outer_in_train']}/{c6['outer_n']}",
                                  cal=f"{c6['cal_in_train']}/{c6['cal_n']}")),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
