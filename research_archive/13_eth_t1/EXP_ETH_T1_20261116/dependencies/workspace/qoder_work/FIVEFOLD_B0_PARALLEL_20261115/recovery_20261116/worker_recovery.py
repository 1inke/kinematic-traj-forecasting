"""Resume the sealed two-slot controller with resilient monitor publication."""
import importlib.util
from pathlib import Path
import safe_status

HERE = Path(__file__).resolve().parent
PARALLEL = HERE.parent


def load_worker():
    spec = importlib.util.spec_from_file_location('sealed_parallel_recovery', PARALLEL / 'standalone/worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def main():
    worker = load_worker()
    worker.rt.verify_files(worker.read_json(HERE / 'SOURCE_MANIFEST.json')['files'])
    if worker.read_json(HERE / 'PREFLIGHT_PASS.json')['status'] != 'PASS':
        raise RuntimeError('RECOVERY_PREFLIGHT_REQUIRED')
    safe_status.install(worker.rt)
    import os
    worker.rt.atomic(HERE / 'ACTIVE_RECOVERY.json', dict(
        at=worker.now(), pid=os.getpid(), max_concurrency=2,
        adapter_source_sha256=worker.sha256(HERE / 'SOURCE_MANIFEST.json'),
        original_parallel_source_sha256=worker.sha256(PARALLEL / 'SOURCE_MANIFEST.json'),
        scientific_contracts_unchanged=True, dependency_on_Codex_API_or_app=False))
    original_job = worker.ChildJob

    class RecoveryJob(original_job):
        def popen(self, command, *args, **kwargs):
            if Path(command[2]).resolve() != (PARALLEL / 'code/parallel_child.py').resolve():
                raise ValueError('UNEXPECTED_CHILD_ENTRYPOINT')
            command[2] = str(HERE / 'child_recovery.py')  # Mutate recorded command too.
            return super().popen(command, *args, **kwargs)

    worker.ChildJob = RecoveryJob
    original_finish = worker.finish

    def finish():
        worker.rt.verify_files(worker.read_json(HERE / 'SOURCE_MANIFEST.json')['files'])
        original_finish()
        worker.rt.atomic(HERE / 'RECOVERY_SUCCESS.json', dict(
            at=worker.now(), status='SUCCESS',
            adapter_source_sha256=worker.sha256(HERE / 'SOURCE_MANIFEST.json'),
            result_sha256=worker.sha256(PARALLEL / 'standalone/RESULT.json')))

    worker.finish = finish
    worker.main()  # Original max-two schedule, mutex, job ownership and science.


if __name__ == '__main__':
    main()
