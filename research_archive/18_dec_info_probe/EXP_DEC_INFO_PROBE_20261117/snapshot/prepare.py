import ast
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
from dec_io import ROOT, OLD, PREVIOUS, HELPER, ARMS, SEEDS, read, write, verify, seal, sha, now

assert not (ROOT / 'SOURCE_MANIFEST.json').exists()
old_manifest = PREVIOUS / 'SOURCE_MANIFEST.json'
assert sha(old_manifest) == 'cd6c0b430cb7562c3eedd28ccb13f70288a99bc5e3f08e227d7baa188a073a01'
previous_cache = PREVIOUS / 'cache/DONE.json'
verify(read(previous_cache)['files'])
files = dict(read(old_manifest)['files'])
files.update(read(previous_cache)['files'])
prior_results = [PREVIOUS / 'D0.json', *(PREVIOUS / f'runs/A1_s{s}/RESULT.json' for s in SEEDS)]
completion = read(PREVIOUS / 'COMPLETION_VERIFIED.json')
assert completion['status'] == 'PASS'
for p in prior_results:
    assert sha(p) == completion['files'][str(p)]
files.update(seal([old_manifest, previous_cache, PREVIOUS / 'COMPLETION_VERIFIED.json', *prior_results]))
verify(files)
base = OLD / 'runs/E1C_E64_s42/final.pt'
assert sha(base) == '6a1781eb53f4a2811a4a1afd532d7f567c66c016935a9ab5b6c5c365ec226b29'
assert sha(ROOT / 'design_original/EXP_DEC_INFO_PROBE_20261117.md') == sha(Path('C:/Users/linke/Downloads/1/EXP_DEC_INFO_PROBE_20261117.md'))
log = ROOT / 'preflight/UNIT.txt'
unit = log.read_text(encoding='utf-8-sig')
assert 'Ran 8 tests' in unit and 'OK' in unit
write(ROOT / 'CONFIG.json', dict(study='EXP_DEC_INFO_PROBE_20261117', device='cuda:0', arms=list(ARMS),
      base_seed=42, refiner_seeds=list(SEEDS), epochs=20, batch=256, optim='AdamW', lr=.001,
      weight_decay=.0001, risk_k=3, c4='DISTANCE_SHARED_UP_TO_THREE_USER_CONFIRMED',
      only_C0_fit_dev=True, no_augmentation=True, no_automatic_B_stage=True))
write(ROOT / 'PREFLIGHT_INPUTS.json', dict(at=now(), status='PASS', previous_frozen_inputs_verified=len(files),
      base_checkpoint_sha256=sha(base), CPU_tests=8, original_design_byte_exact=True, real_cache_preflight_in_worker=True))
for p in ROOT.rglob('*'):
    if p.is_file() and (p.suffix in ('.py','.md','.ps1','.cmd') or p.name in ('CONFIG.json','PREFLIGHT_INPUTS.json','UNIT.txt')):
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
        files[str(p)] = sha(p)
write(ROOT / 'SOURCE_MANIFEST.json', dict(at=now(), files=files))
print('SOURCE_FROZEN', len(files), sha(ROOT / 'SOURCE_MANIFEST.json'), flush=True)
