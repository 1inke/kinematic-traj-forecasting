"""Seal parallel execution extension after all real preflight checks."""
import ast
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT.parent / 'FIVEFOLD_B0_EXEC_20261115'
sys.path.insert(0, str(OLD / 'code'))
import runtime as rt
from fivefold_common import now, read_json, sha256


def main():
    if (ROOT / 'SOURCE_MANIFEST.json').exists():
        raise RuntimeError('ALREADY_SEALED_NO_RESEAL')
    benchmark = read_json(ROOT / 'preflight/BENCHMARK_PASS.json')
    smoke = read_json(ROOT / 'preflight/PARALLEL_SMOKE_PASS.json')
    if benchmark['status'] != 'PASS' or smoke['status'] != 'PASS' or benchmark['compute_speedup'] < 1.1:
        raise RuntimeError('PARALLEL_NOT_VERIFIED_OR_NO_SPEEDUP')
    old_config = read_json(OLD / 'STUDY_CONFIG.json')
    for path in list((ROOT / 'code').glob('*.py')) + list(HERE.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    assert '6 passed' in (ROOT / 'logs/new_numpy_test.log').read_text(encoding='utf-8').lower() or 'PASS' in (ROOT / 'logs/new_numpy_test.log').read_text(encoding='utf-8')
    assert 'checks passed' in (ROOT / 'logs/new_fake_test.log').read_text(encoding='utf-8').lower()
    pause = OLD / 'standalone/pause_events/20261115_164854/PAUSE_COMPLETE.json'
    paused = read_json(pause)
    assert paused['actual_processes_verified_zero']
    assert sha256(paused['durable_checkpoint']) == paused['checkpoint_sha256']
    rt.validate_done(Path(old_config['output']) / 'runs/budget_eth_E64_s42')
    config = dict(old_config, task='Trajectory-Fivefold-B0-Parallel-20261115', max_concurrency=2,
                  serial_execution_root=str(OLD), scientific_contracts_unchanged=True,
                  human_authorization='Prepare, verify, then directly start parallel execution',
                  parallel_source_location=str(ROOT / 'design_original'))
    rt.save_once(ROOT / 'PARALLEL_CONFIG.json', config)
    rt.save_once(ROOT / 'REUSE_RECEIPT.json', dict(at=now(), existing_DONE_sha256=sha256(Path(config['output']) / 'runs/budget_eth_E64_s42/DONE.json'),
                 paused_checkpoint_sha256=paused['checkpoint_sha256'], paused_epoch=paused['epoch'], paused_updates=paused['updates'],
                 old_source_manifest_sha256=sha256(OLD / 'SOURCE_MANIFEST.json'),
                 runtime_contract_must_remain_identical=True, ongoing_checkpoint_may_roll_off_after_two_epochs=True))
    rt.atomic(ROOT / 'preflight/PREFLIGHT_PASS.json', dict(at=now(), status='PASS',
              benchmark_sha256=sha256(ROOT / 'preflight/BENCHMARK_PASS.json'),
              real_parallel_smoke_sha256=sha256(ROOT / 'preflight/PARALLEL_SMOKE_PASS.json'),
              new_numpy_and_fake_parallel_tests=True, old_DONE_and_resume_SHA_verified=True,
              max_concurrency=2, same_science=True, owner_exit_kills_actual_child=True))
    original = read_json(OLD / 'SOURCE_MANIFEST.json')
    files = dict(original['files'])
    rt.verify_files(files)
    additions = [OLD / 'SOURCE_MANIFEST.json', ROOT / 'PARALLEL_CONFIG.json', ROOT / 'REUSE_RECEIPT.json',
                 ROOT / 'DESIGN_COPY_MANIFEST.json', ROOT / 'README_EXECUTION.md', pause]
    additions += list((ROOT / 'code').glob('*.py')) + [p for p in HERE.iterdir() if p.suffix in ('.py','.ps1','.cmd')]
    additions += [p for p in (ROOT / 'design_original').rglob('*') if p.is_file()]
    additions += [p for p in (ROOT / 'preflight').rglob('*') if p.is_file()]
    additions += list((ROOT / 'logs').glob('*.log'))
    files.update({str(p.resolve()): sha256(p) for p in additions})
    files.update(smoke['files'])
    rt.verify_files(files)
    rt.save_once(ROOT / 'SOURCE_MANIFEST.json', dict(at=now(), status='SEALED_PARALLEL_EXTENSION_BEFORE_RESUME',
                 previous_source_sha256=sha256(OLD / 'SOURCE_MANIFEST.json'), files=files))
    disk = shutil.disk_usage(ROOT).free
    if disk < 10 * 1024**3:
        raise RuntimeError('CURRENT_OUTPUT_DISK_SHORTAGE')
    print('PARALLEL_READY_TO_START source=' + sha256(ROOT / 'SOURCE_MANIFEST.json') + ' files=' + str(len(files)))


if __name__ == '__main__':
    main()
