from pathlib import Path
import ast,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
from probe_io import ROOT,OLD,W,read,write,verify,sha,seal,now
old_source=OLD/'SOURCE_MANIFEST.json';assert sha(old_source)=='45433a9918b278db938d0fdf1a0af692e937a631b8e167ba0dcac5b58ba27cf6';verify(read(old_source)['files'])
lock=OLD/'runs/E1C_E64_s42/FINAL_LOCK.json';verify(read(lock)['files'])
expected=read(OLD/'FINAL_OUTPUT_SEAL.json')['files'];selected=[OLD/f'C0/{s}.npz' for s in ('fit','dev')]+[OLD/f'runs/E1C_E64_s42/{s}/predictions.npz' for s in ('fit','dev')]
for p in selected:assert sha(p)==expected[str(p)],p
assert 'Ran 8 tests' in (ROOT/'preflight/UNIT.stderr.log').read_text(encoding='utf-8-sig') and 'OK' in (ROOT/'preflight/UNIT.stderr.log').read_text(encoding='utf-8-sig')
helper=Path('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116/code/eth_speed_math.py');helper_expected=read(W/'qoder_work/EXP_DECEL_DIAG_EXEC_20261117/SOURCE_MANIFEST.json')['files'][str(helper)];assert sha(helper)==helper_expected
config=dict(study='EXP_SET_INTERACT_PROBE_20261117',device='cuda:0',base='E1C_E64_s42/fixed_final_EMA',epochs=20,batch=256,lr=.001,weight_decay=.0001,seeds=[0,1,2],max_runs=18,head_and_base_frozen=True,strict_pointwise_tokens=True,protection_reporting='separate_status_user_confirmed',effective_endpoint_clustering='complete_link_0.2m',sigma='sample_SD_relative_A1_vs_A0_gains_ddof1',user_confirmed_at='2026-11-17',numpy_loss_verification_source=str(OLD/'loss_independent.py'))
write(ROOT/'CONFIG.json',config)
write(ROOT/'PREFLIGHT_INPUTS.json',dict(at=now(),status='PASS',old_source_files=len(read(old_source)['files']),base_final_sha256=sha(OLD/'runs/E1C_E64_s42/final.pt'),data_and_saved_predictions=seal(selected),unit_tests=8,user_decisions=['strict_pointwise_tokens_all_arms','retain_mechanism_separate_protection_failure']))
files=dict(read(old_source)['files']);files.update(read(lock)['files']);files.update(seal([*selected,old_source,lock,OLD/'FINAL_OUTPUT_SEAL.json',helper]))
for p in ROOT.rglob('*'):
    if p.is_file() and (p.suffix in ('.py','.md','.ps1','.cmd') or p.name in ('CONFIG.json','PREFLIGHT_INPUTS.json')):
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
        files[str(p.resolve())]=sha(p)
write(ROOT/'SOURCE_MANIFEST.json',dict(at=now(),files=files))
print(json.dumps(dict(status='SOURCE_FROZEN',files=len(files),device=config['device'],research_runs_max=18),ensure_ascii=False))
