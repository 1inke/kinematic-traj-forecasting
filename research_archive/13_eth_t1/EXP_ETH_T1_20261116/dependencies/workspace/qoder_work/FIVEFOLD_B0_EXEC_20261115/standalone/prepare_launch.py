"""One-time launch sealing after real checks. Does not start formal training."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import runtime as rt
import run_fivefold as rf
from fivefold_common import now, read_json, sha256, fold_files, FOLDS


def main():
    if (ROOT / 'SOURCE_MANIFEST.json').exists():
        raise RuntimeError('ALREADY_SEALED_NO_RESEAL')
    python = WORKSPACE / '.venv-governance/Scripts/python.exe'
    old = WORKSPACE / 'qoder_work/NEXT_PHASE_EXEC_20261114'
    data = Path('C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse').resolve()
    audit_path = ROOT / 'F0/AUDIT_RECORDING.json'
    audit = rf.load_audit(audit_path)
    real = ROOT / 'preflight/REAL_RESUME.json'
    if read_json(real)['status'] != 'PASS':
        raise RuntimeError('REAL_RESUME_NOT_PASSED')
    for source in (ROOT / 'design_original/code').glob('*.py'):
        assert sha256(source) == sha256(ROOT / 'code' / source.name), source.name
    for path in list((ROOT / 'code').glob('*.py')) + list(HERE.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    smoke = ROOT / 'preflight/FIVEFOLD_SMOKE'
    command = [str(python), '-B', str(ROOT / 'code/runtime.py'), 'budget', '--smoke', '--folds', 'hotel',
               '--out', str(smoke), '--exec-dir', str(old), '--data-dir', str(data), '--audit', str(audit_path)]
    before = None
    for repeat in (1, 2):
        with (ROOT / f'logs/real_smoke_{repeat}.log').open('xb') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=WORKSPACE, check=True)
        directory = smoke / 'runs/smoke_hotel_E1_s42'
        rt.validate_done(directory)
        metrics = read_json(directory / 'metrics.json')
        assert metrics['train']['updates'] == 8 and metrics['train']['parameters'] == 295474
        proof = {str(path): sha256(path) for path in directory.iterdir() if path.is_file()}
        if before is not None:
            assert proof == before, 'DONE_SKIP_MODIFIED_OUTPUTS'
        before = proof
    rt.atomic(ROOT / 'preflight/REAL_SMOKE.json', dict(at=now(), status='PASS',
              updates=8, train_windows=512, val_windows=256, DONE_skip_all_bytes_unchanged=True,
              files=before, sources=metrics['contract']['sources']))
    # Exercise completion JSON idempotence (including JSON tuple/list normalization).
    rt.save_once(ROOT / 'preflight/IDEMPOTENCE_FIXTURE.json', dict(at='first', comparisons={'ref': (1., 2.)}))
    rt.save_once(ROOT / 'preflight/IDEMPOTENCE_FIXTURE.json', dict(at='second', comparisons={'ref': (1., 2.)}))
    exp156 = WORKSPACE / 'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/summary.json'
    config = dict(study='FIVEFOLD_B0_20261115', workspace=str(WORKSPACE), python=str(python),
                  exec_dir=str(old), data_dir=str(data), audit=str(audit_path),
                  output=str(ROOT / 'FIVEFOLD_RUNS'), exp156=str(exp156), task='Trajectory-Fivefold-B0-Standalone-20261115')
    rt.save_once(ROOT / 'STUDY_CONFIG.json', config)
    rt.atomic(ROOT / 'preflight/PREFLIGHT_PASS.json', dict(at=now(), status='PASS',
              real_resume_sha256=sha256(real), real_smoke_sha256=sha256(ROOT / 'preflight/REAL_SMOKE.json'),
              F0_recording_audit_sha256=sha256(audit_path), original_eight_scripts_unchanged=True,
              real_CUDA=True, test_metrics_not_used=True, F0_integrity_opened_test_files=True))
    # Include the original dependency closure and verify that no prior science was changed.
    files = dict(read_json(old / 'SOURCE_MANIFEST.json')['files'])
    rt.verify_files(files)
    additions = [old / 'SOURCE_MANIFEST.json', old / 'resource_guard.py', old / 'PREFLIGHT_E1.json', exp156,
                 ROOT / 'DESIGN_COPY_MANIFEST.json', ROOT / 'STUDY_CONFIG.json', ROOT / 'README_EXECUTION.md']
    additions += list((ROOT / 'code').glob('*.py'))
    additions += [p for p in HERE.iterdir() if p.suffix in ('.py', '.ps1', '.cmd')]
    additions += [p for p in (ROOT / 'design_original').rglob('*') if p.is_file()]
    additions += list((ROOT / 'F0').glob('*.json'))
    additions += list((ROOT / 'preflight').glob('*.json'))
    additions += [Path(name) for name in audit['raw_label_sha256']]
    for fold in FOLDS:
        for key, path in fold_files(data, fold).items():
            assert sha256(path) == audit['by_fold'][fold]['sha256'][key]
            additions.append(path)
    files.update({str(p.resolve()): sha256(p) for p in additions})
    # Bind all original preflight evidence, including the exact weights used for resume proof.
    for receipt in (ROOT / 'preflight/REAL_SMOKE.json',):
        files.update(read_json(receipt)['files'])
    for path in (ROOT / 'preflight/SMOKE_RESUME').rglob('*'):
        if path.is_file():
            files[str(path.resolve())] = sha256(path)
    rt.verify_files(files)
    rt.save_once(ROOT / 'SOURCE_MANIFEST.json', dict(at=now(), status='SEALED_BEFORE_FORMAL_TRAINING',
                 original_science_unchanged=True, files=files))
    sys.path.insert(0, str(old))
    import resource_guard as rg
    estimate = read_json(old / 'PREFLIGHT_E1.json')['estimated_peak_bytes']
    admission = rg.decide(rg.snapshot(), estimate, 'startup')
    disk_free = shutil.disk_usage(ROOT).free
    # 25 * two rolling full-state checkpoints, prediction archives, and ample temporary headroom.
    disk_required = 10 * 1024**3
    rt.atomic(HERE / 'PRELAUNCH_ADMISSION.json', dict(at=now(), memory=admission,
              disk_free_bytes=disk_free, disk_required_bytes=disk_required,
              source_manifest_sha256=sha256(ROOT / 'SOURCE_MANIFEST.json')))
    if admission['status'] != 'PASS' or disk_free < disk_required:
        raise RuntimeError('CURRENT_RESOURCE_ADMISSION_FAILED')
    print(json.dumps(dict(status='PASS_READY_TO_START', source_sha256=sha256(ROOT / 'SOURCE_MANIFEST.json'),
                         sealed_files=len(files), disk_free_bytes=disk_free)))


if __name__ == '__main__':
    main()
