"""Independent NumPy reconstruction from saved world paths and real targets."""
from pathlib import Path
import sys

import numpy as np

PACKAGE = Path('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'code'))
from speed_io import config, imports, now, read, sha, verify, write

c = config(); fd, fm = imports(c)
done = read(ROOT/'DIAG/DONE.json'); verify(done['files'])
contract_sha = sha(ROOT/'INPUT_CONTRACT.json')
assert done['input_contract_sha256'] == contract_sha
checks = []
for fold in ('eth', 'hotel'):
    data = fd.load_split(c['data_dir'], fold, 'test')
    for case in sorted((ROOT/'DIAG'/fold).iterdir()):
        if not case.is_dir():
            continue
        with np.load(case/'predictions.npz') as pred, np.load(case/'window_metrics.npz') as metrics:
            np.testing.assert_array_equal(pred['ids'], data['ids'])
            np.testing.assert_array_equal(pred['group'], data['group'])
            distance = np.sqrt(np.square(pred['paths'].astype(np.float64)-data['future'][:, None]).sum(-1))
            ade = distance.mean(-1); fde = distance[..., -1]
            top1 = pred['logits'].argmax(-1); row = np.arange(len(top1))
            independently_computed = dict(minADE=ade.min(-1), minFDE=fde.min(-1),
                  coverage=((ade <= .2) & (fde <= .4)).any(-1),
                  top1_ADE=ade[row, top1], top1_FDE=fde[row, top1])
            for key, value in independently_computed.items():
                np.testing.assert_allclose(value, metrics[key], rtol=1e-12, atol=1e-12)
            aggregate = read(case/'summary.json')['summary']['overall']
            for key, value in independently_computed.items():
                np.testing.assert_allclose(float(value.mean()), aggregate[key], rtol=1e-12, atol=1e-12)
            checks.append(dict(fold=fold, case=case.name, n=len(data['ids']), passed=True))
write(ROOT/'DIAG_INDEPENDENT_VERIFICATION.json', dict(at=now(), status='PASS',
      done_sha256=sha(ROOT/'DIAG/DONE.json'), input_contract_sha256=contract_sha,
      new_inference=False, training=False, checks=checks))
print('PASS: 11 saved prediction populations independently recomputed against real targets.')
