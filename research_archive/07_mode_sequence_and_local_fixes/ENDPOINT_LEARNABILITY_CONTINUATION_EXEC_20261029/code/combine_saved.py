"""P3 saved-only composition; requires reviewed scientific admission, not training."""
import argparse
from pathlib import Path
import numpy as np
from io_contract import (load_npz, validate_observed, verify_manifest, checked_seal,
                         code_digest, sha, npz_new, json_new, seal, finish_stage)
from inference import combine


def run(observed, endpoint_dir, time_dir, admission, out, check=lambda: None):
    gate = verify_manifest(admission)
    if (gate.get('status') != 'PASS_BOTH_BRANCHES_SIGNAL_AND_DEPLOYMENT' or
            gate.get('code_digest') != code_digest()):
        raise ValueError('P3 NOT_TRIGGERED: verified scientific gate review required')
    x = load_npz(observed)
    validate_observed(x)
    saved, receipts = [], []
    for branch, folder in (('END', endpoint_dir), ('TIME', time_dir)):
        r = checked_seal(folder)
        if r['observed_sha'] != sha(observed) or r['branch'] != branch:
            raise ValueError('P3 branch/input mismatch')
        if r.get('model_contract', {}).get('arm') != 'H':
            raise ValueError('P3 requires H models, not a baseline or G')
        z = load_npz(Path(folder)/'predictions.npz')
        if not np.array_equal(z['sample_id'], x['sample_id']):
            raise ValueError('P3 sample order mismatch')
        saved.append(z)
        receipts.append(r)
    if receipts[0]['model_contract']['seed'] != receipts[1]['model_contract']['seed']:
        raise ValueError('P3 must pair the same seed')
    # Admission must bind the exact immutable E/T outputs being composed.
    expected = gate.get('prediction_done_sha', {})
    for branch, folder in (('END', endpoint_dir), ('TIME', time_dir)):
        if expected.get(branch) != sha(Path(folder)/'done.json'):
            raise ValueError('P3 gate did not admit these exact saved outputs')
    result = combine(x['past'], x['native'], saved[0]['executed_action_id'],
                     saved[1]['executed_action_id'], check)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    npz_new(out/'joint.npz', sample_id=x['sample_id'],
            endpoint_executed_id=saved[0]['executed_action_id'],
            time_executed_id=saved[1]['executed_action_id'], **result)
    seal(out, [out/'joint.npz'], stage='P3_SAVED_COMPOSITION_REQUIRES_FULL_REVIEW',
         observed_sha=sha(observed), admission_sha=sha(admission), no_training=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('observed', 'endpoint-dir', 'time-dir', 'admission', 'out', 'guard'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--inventory-bytes', type=int, required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    if a.run:
        from run_stage import resource_guard
        g = resource_guard(a.guard, a.inventory_bytes)
        run(a.observed, a.endpoint_dir, a.time_dir, a.admission, a.out, g.check)
        json_new(Path(a.out)/'RESOURCE.json', dict(final=g.check(force=True), observations=g.observations))
        finish_stage(a.out, 'RESOURCE.json')
    else:
        print('PLAN_ONLY')
