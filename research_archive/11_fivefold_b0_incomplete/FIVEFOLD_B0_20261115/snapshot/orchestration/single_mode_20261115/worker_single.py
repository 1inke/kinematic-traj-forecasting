"""User-requested single concurrency adapter; all sealed science is reused unchanged."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARALLEL = HERE.parent


def load_worker():
    spec = importlib.util.spec_from_file_location('sealed_fivefold_controller', PARALLEL / 'standalone/worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_single_mode(worker, adapter_sha):
    original_pool = worker.pool
    original_atomic = worker.rt.atomic

    def one_at_a_time(specs, stage, limit=1):
        return original_pool(specs, stage.replace('PARALLEL_', 'SINGLE_'), limit=1)

    def single_atomic(path, value):
        value = dict(value)
        if value.get('controller') == 'PARALLEL':
            value['controller'] = 'SINGLE'
        if 'concurrency_limit' in value:
            value['concurrency_limit'] = 1
        if isinstance(value.get('stage'), str):
            value['stage'] = value['stage'].replace('PARALLEL_', 'SINGLE_')
        if Path(path).name in ('ACTIVE_WORKER.json', 'CURRENT.json', 'CHILD_STATE.json',
                               'RESULT.json', 'FAILURE.json', 'WORKER_EXIT.json', 'RECEIPT.json'):
            value.update(execution_adapter_source_sha256=adapter_sha, concurrency_limit=1)
        return original_atomic(path, value)

    worker.pool = one_at_a_time
    worker.rt.atomic = single_atomic
    if hasattr(worker, 'finish'):
        original_finish = worker.finish

        def verified_finish():
            worker.rt.verify_files(worker.read_json(HERE / 'SOURCE_MANIFEST.json')['files'])
            return original_finish()

        worker.finish = verified_finish


def main():
    worker = load_worker()
    manifest = worker.read_json(HERE / 'SOURCE_MANIFEST.json')
    worker.rt.verify_files(manifest['files'])
    policy = worker.read_json(HERE / 'SINGLE_POLICY.json')
    if policy['max_concurrency'] != 1:
        raise ValueError('SINGLE_CONCURRENCY_POLICY_REQUIRED')
    install_single_mode(worker, worker.sha256(HERE / 'SOURCE_MANIFEST.json'))
    worker.main()


if __name__ == '__main__':
    main()
