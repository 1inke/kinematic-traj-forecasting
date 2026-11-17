"""Seal supplementary verification without changing frozen research outputs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from probe_io import read, write, verify, seal, sha, now


def main(mode):
    done = read(ROOT / 'DONE.json')
    assert done['status'] == 'SUCCESS' and done['completed_runs'] == 18
    verify(done['files'])
    verify(read(ROOT / 'SOURCE_MANIFEST.json')['files'])
    paths = [ROOT / p for p in ('SOURCE_MANIFEST.json', 'DONE.json',
             'audit_results.py', 'supplement_control.py', 'cache/DONE.json',
             'A0/DONE.json')]
    paths.extend(Path(p) for p in done['files'])
    for folder in sorted((ROOT / 'runs').iterdir()):
        receipt = read(folder / 'DONE.json')
        verify(receipt['files'])
        paths.extend(Path(p) for p in receipt['files'])
    paths = list(dict.fromkeys(paths))
    contract = ROOT / 'VERIFICATION_INPUT_CONTRACT.json'
    if mode == 'prepare':
        assert not contract.exists(), 'Do not silently replace an existing audit input contract'
        write(contract, dict(at=now(), purpose='Independent post-run audit; no new training or gate changes', files=seal(paths)))
        print('AUDIT_INPUTS_SEALED', len(paths))
        return
    assert mode == 'finalize'
    verify(read(contract)['files'])
    audit = read(ROOT / 'INDEPENDENT_VERIFICATION.json')
    assert audit['status'] == 'PASS' and audit['completed_runs'] == 18
    assert audit['script_sha256'] == sha(ROOT / 'audit_results.py')
    worker = read(ROOT / 'standalone/WORKER_EXIT.json')
    assert worker['status'] == 'SUCCESS' and worker['exit_code'] == 0
    checkpoints = list((ROOT / 'runs').glob('*/checkpoints/epoch*.json'))
    assert len(checkpoints) == 360
    for path in checkpoints:
        verify(read(path)['files'])
    extra = [contract, ROOT / 'INDEPENDENT_VERIFICATION.json',
             ROOT / 'ADDITIONAL_DESCRIPTIVE.json', ROOT / 'RESULT_INTERPRETATION_ZH.md',
             ROOT / 'standalone/WORKER_EXIT.json', ROOT / 'standalone/PROCESS_ANCESTRY.json']
    write(ROOT / 'COMPLETION_VERIFIED.json', dict(at=now(), status='PASS',
          completed_runs=18, complete_epochs=360, optimizer_updates=30600,
          checkpoint_receipts_verified=len(checkpoints), independent_audit='PASS',
          raw_verdict=audit['raw_verdict'], protection_status=audit['protection_status'],
          B_stage_started=False, published=False, files=seal(paths + extra)))
    write(ROOT / 'CURRENT.json', dict(at=now(), phase='COMPLETE_VERIFIED',
          completed_runs=18, verdict=audit['raw_verdict'], protection_status=audit['protection_status']))
    print('COMPLETE_VERIFIED', audit['raw_verdict'], audit['protection_status'])


if __name__ == '__main__':
    main(sys.argv[1])
