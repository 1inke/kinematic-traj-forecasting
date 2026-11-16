"""Exercise actual Windows sharing locks and verify all retained scientific state."""
import ctypes
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import safe_status as io
import worker_recovery

HERE = Path(__file__).resolve().parent
worker = worker_recovery.load_worker()
rt = worker.rt


def held(path):
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint,
                               ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    api.CreateFileW.restype = ctypes.c_void_p
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = api.CreateFileW(str(path), 0x80000000, 3, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return api, handle


def main():
    config = worker.read_json(worker.ROOT / 'PARALLEL_CONFIG.json')
    rt.verify_files(worker.read_json(worker.ROOT / 'SOURCE_MANIFEST.json')['files'])
    output = Path(config['output'])
    seal = worker.read_json(output / 'DECISION_SEAL.json')
    assert worker.sha256(output / 'BUDGET_DECISION.json') == seal['decision_sha256']
    assert worker.read_json(output / 'BUDGET_DECISION.json')['E_star'] == 128
    for name, expected in seal['budget_done_sha256'].items():
        assert worker.sha256(output / 'runs' / name / 'DONE.json') == expected
        rt.validate_done(output / 'runs' / name)
    checkpoints = {}
    for seed in (42, 43):
        directory = output / 'runs' / f'final_eth_E128_s{seed}'
        for path in sorted(directory.glob('ckpt_epoch*.pt')):
            side = worker.read_json(path.with_suffix('.json'))
            assert worker.sha256(path) == side['sha256']
            checkpoints[directory.name] = side
        assert checkpoints[directory.name]['epoch'] == 63
    tests = []
    test_root = worker.ROOT / 'standalone' / 'recovery_io_tests'
    test_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=test_root) as tmp:
        folder = Path(tmp)
        critical = folder / 'critical_receipt.json'
        io.replace_json(critical, {'v': 0})
        api, handle = held(critical)
        timer = threading.Timer(.22, lambda: api.CloseHandle(handle))
        timer.start()
        try:
            io.atomic(critical, {'v': 1})
            assert worker.read_json(critical) == {'v': 1}
        finally:
            timer.join()
        tests.append('ACTUAL_WINDOWS_TRANSIENT_LOCK_RETRIED')
        api, handle = held(critical)
        try:
            try:
                io.atomic(critical, {'v': 2})
            except OSError as exc:
                assert io.transient(exc)
            else:
                raise AssertionError('CRITICAL_FAILURE_WAS_SWALLOWED')
            assert worker.read_json(critical) == {'v': 1}
        finally:
            api.CloseHandle(handle)
        tests.append('CRITICAL_PERSISTENT_LOCK_FAILS_CLOSED')
        monitor = folder / 'CURRENT.json'
        io.atomic(monitor, {'v': 0})
        api, handle = held(monitor)
        try:
            io.atomic(monitor, {'v': 3})
            assert worker.read_json(monitor) == {'v': 0}
            digest = io.hashlib.sha256(str(monitor.resolve()).encode('utf-8')).hexdigest()[:20]
            mirror = io.HERE / 'status_mirror' / (digest + '_CURRENT.json')
            assert worker.read_json(mirror)['value'] == {'v': 3}
        finally:
            api.CloseHandle(handle)
        tests.append('MONITOR_PERSISTENT_LOCK_MIRRORED_NONFATAL')
        io.atomic(monitor, {'v': 4})
        assert worker.read_json(monitor) == {'v': 4}
        tests.append('MONITOR_RECOVERS_AFTER_RELEASE')
        try:
            io.atomic(critical, {'bad': float('nan')})
        except ValueError:
            pass
        else:
            raise AssertionError('INVALID_SCIENTIFIC_JSON_WAS_SWALLOWED')
        assert not list(folder.glob('*.tmp'))
        tests.append('INVALID_JSON_FATAL_AND_TEMP_FILES_CLEANED')
        shared = folder / 'concurrent.json'
        failures = []
        def writer(index):
            try:
                for serial in range(10):
                    io.atomic(shared, dict(writer=index, serial=serial))
            except BaseException as exc:
                failures.append(repr(exc))
        threads = [threading.Thread(target=writer, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not failures, failures
        assert worker.read_json(shared)['serial'] == 9
        assert not list(folder.glob('*.tmp'))
        tests.append('CONCURRENT_WRITERS_UNIQUE_TEMPS_VALID_JSON')
    sys.path.insert(0, config['exec_dir'])
    import resource_guard as rg
    estimate = worker.read_json(Path(config['exec_dir']) / 'PREFLIGHT_E1.json')['estimated_peak_bytes']
    admission = rg.decide(rg.snapshot(), estimate, 'startup')
    assert admission['status'] == 'PASS', admission
    import shutil
    disk_free = shutil.disk_usage(config['workspace']).free
    assert disk_free >= 20 * 1024**3, disk_free
    value = dict(at=worker.now(), status='PASS', max_concurrency=2, tests=tests,
                 sealed_source_files_verified=len(worker.read_json(worker.ROOT / 'SOURCE_MANIFEST.json')['files']),
                 budget_done_verified=len(seal['budget_done_sha256']),
                 decision_sha256=seal['decision_sha256'], E_star=128,
                 checkpoints=checkpoints, admission=admission, disk_free_bytes=disk_free,
                 scientific_code_and_contracts_unchanged=True)
    io.replace_json(HERE / 'PREFLIGHT_PASS.json', value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
