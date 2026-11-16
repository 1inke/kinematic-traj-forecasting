"""Check every scheduling entry point is clamped to one, including explicit limit=2."""
from types import SimpleNamespace
from pathlib import Path
import worker_single

calls = []
writes = []
worker = SimpleNamespace(pool=lambda specs, stage, limit=2: calls.append((specs, stage, limit)),
                         rt=SimpleNamespace(atomic=lambda path, value: writes.append((path, value))))
worker_single.install_single_mode(worker, 'fixture_source')
for stage in ('PARALLEL_BUDGET','PARALLEL_FINAL','BUDGET_DECISION','SUMMARY'):
    worker.pool(['job1','job2'], stage)
    worker.pool(['job1','job2'], stage, limit=2)
assert len(calls) == 8 and all(c[2] == 1 for c in calls)
worker.rt.atomic(Path('CURRENT.json'), dict(controller='PARALLEL',concurrency_limit=2,stage='PARALLEL_BUDGET'))
assert writes[0][1] == dict(controller='SINGLE',concurrency_limit=1,stage='SINGLE_BUDGET',execution_adapter_source_sha256='fixture_source')
worker.rt.atomic(Path('immutable_receipt.json'), dict(files={'source':'sha'}))
assert writes[1][1] == dict(files={'source':'sha'})
print('PASS: all eight dispatch paths clamp to one; metadata reflects SINGLE; source receipts unchanged.')
