"""Explicit C0 completion extension. No training or model imports."""
import csv
from itertools import combinations
import hashlib
import numpy as np
from common import *
from causal_dataset import RECORDINGS, load_recording


def main():
    out = HERE / 'C0_completion_review'
    out.mkdir(exist_ok=False)
    base = read_json(HERE / 'C0/C0_RESULT.json')
    verify(base['outputs'])
    verify(read_json(HERE / 'C0/SOURCE_MANIFEST.json')['files'])
    with (HERE / 'C0/WINDOW_REGISTRY.csv').open(encoding='utf-8-sig') as f:
        registry = list(csv.DictReader(f))
    fit, dev = [dict(np.load(HERE / 'C0' / (s + '.npz'))) for s in ('fit', 'dev')]
    if set(fit['ids']) & set(dev['ids']):
        raise ValueError('ID_OVERLAP')
    def tracks(d):
        return {'|'.join(x.split('|')[:2]) for x in d['ids']}
    if tracks(fit) & tracks(dev):
        raise ValueError('TARGET_OVERLAP')
    # Probe possible track aliases even if label time starts at different offsets.
    signatures = {}
    for name in RECORDINGS:
        _, tt, _, _ = load_recording(ASTRA / 'datasets/eth_ucy/labels/eth' / (name + '_train.txt'))
        signatures[name] = set()
        for ped, tr in tt.items():
            times = np.array(sorted(tr), dtype='<i8')
            xy = np.stack([tr[t] for t in times]).astype('<f4')
            signatures[name].add(hashlib.sha256((times - times[0]).tobytes() + xy.tobytes()).hexdigest())
    alias = [{'a': a, 'b': b, 'exact_track_ignoring_time_origin': len(signatures[a] & signatures[b])}
             for a, b in combinations(RECORDINGS, 2)]
    if any(r['exact_track_ignoring_time_origin'] for r in alias):
        raise ValueError('OFFSET_ALIAS_UNRESOLVED')
    p = fit['past'][:, 0].astype(float); f = fit['future'].astype(float)
    v = (f - np.concatenate([p[:, -1:], f[:, :-1]], 1)) / .4
    a = (v[:, 1:] - v[:, :-1]) / .4
    j = (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) / .16
    reference = read_json(HERE / 'C0/MOTION_REFERENCE.json')
    max_error = 0.
    for key, value in [('speed', v), ('acceleration', a), ('jerk', j)]:
        norm = np.sqrt((value * value).sum(-1))
        for label, actual in [('mean', norm.mean()), ('p95', np.quantile(norm, .95)),
                              ('p99', np.quantile(norm, .99)), ('p99.5', np.quantile(norm, .995))]:
            max_error = max(max_error, abs(float(actual) - reference['raw'][key][label]))
    if max_error > 2e-11:
        raise ValueError('FIT_MOTION_REFERENCE_REVIEW_MISMATCH')
    write_json(out / 'REVIEW.json', {'status': 'PASS_C0_INDEPENDENT_SPLIT_TARGET_MOTION_AND_OFFSET_ALIAS_CHECK',
        'at': now(), 'fit': len(fit['ids']), 'dev': len(dev['ids']), 'max_error': max_error,
        'time_offset_aliases': alias, 'footage_identity_still_unverified': True,
        'same_assistant_second_implementation_not_external_review': True,
        'extension_source': seal([Path(__file__)]), 'C0_files': base['outputs']})
    print('C0_REVIEW_PASS', max_error, flush=True)


if __name__ == '__main__':
    main()
