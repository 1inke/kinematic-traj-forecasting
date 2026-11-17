"""Validate selected historical sealed inputs and freeze this CPU analysis."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
W=ROOT.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
speed=W/'qoder_work/EXP_ETH_SPEED_EXEC_20261116'
old=W/'qoder_work/NEXT_PHASE_EXEC_20261114'
cfg=read('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116/CONFIG.local.json')
seals=[speed/'INPUT_CONTRACT.json',speed/'DIAG/DONE.json',speed/'SPEED_AUG/B0_wide_eth_E64_s42/DONE.json',old/'FINAL_OUTPUT_SEAL.json']
expected={}
for p in seals:
    for k,v in read(p)['files'].items():expected[str(Path(k).resolve()).lower()]=v
helper=Path('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116/code/eth_speed_math.py')
pop=[dict(dataset='ETH',model=m,prediction=str(speed/p)) for m,p in [
 ('B0_s42','DIAG/eth/identity/predictions.npz'),
 ('AUG_s42','SPEED_AUG/B0_wide_eth_E64_s42/test/predictions.npz'),
 ('T1_s42','DIAG/eth/T1_saved_descriptive_only/predictions.npz')]]
pop.extend(dict(dataset='C0_dev',model=f'B0_s{s}',prediction=str(old/f'runs/E1C_E64_s{s}/dev/predictions.npz')) for s in range(42,47))
data=Path(cfg['data_dir']);raw=[data/'eth_data_test.npy',data/'eth_num_test.npy']
if not raw[1].exists():raw[1]=data/'num_test.npy'
selected=[helper,old/'C0/dev.npz',*raw,*[Path(p['prediction']) for p in pop]]
checked=[]
for p in selected:
    digest=sha(p);key=str(p.resolve()).lower()
    if key not in expected:raise ValueError('NOT_IN_PRIOR_SEAL '+str(p))
    if digest!=expected[key]:raise ValueError('PRIOR_SEAL_MISMATCH '+str(p))
    checked.append(dict(path=str(p),sha256=digest))
config=dict(study='EXP_DECEL_DIAG_20261117',maneuver_source=str(helper),fivefold_code=cfg['fivefold_code'],data_dir=str(data),C0_dev=str(old/'C0/dev.npz'),populations=pop)
write(ROOT/'CONFIG.json',config)
write(ROOT/'RULES.json',dict(dt=.4,direction='mean(last 3 observed displacements), norm > .03m',history_edges=[.85,1.15],speed_denominator_floor=.1,decel_ratio_max=.8,speed_ratio_match_tolerance=.15,longitudinal_reach_m=.4,top1_overshoot_m=.4,calibration_edges=[0,.2,.4,.6,.8,1],small_group_n_lt=20,A='all_overshoot >= .30 OR reach < .60',B='reach >= .70 AND top1_overshoot >= .50',B1='history-decel top1 overshoot >= history-steady OR unconditional history-decel actual-minus-predicted >= .20',B2='>50% of deceleration overshoots are historical steady AND all nonempty calibration bin gaps < .10',unresolved_B='B_UNRESOLVED',decision_population='ETH/AUG_s42, maneuver deceleration, valid direction',notes='IMPLEMENTATION_NOTES.md is fixed before formal computation'))
write(ROOT/'INPUT_VALIDATION.json',dict(status='PASS',selected_prior_sealed_files=checked,source_seals={str(p):sha(p) for p in seals}))
files=[*selected,*seals,Path(cfg['fivefold_code'])/'fivefold_data.py',Path(cfg['fivefold_code'])/'fivefold_common.py']
files.extend(p for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ('.py','.md','.ps1','.cmd') or p.name in ('CONFIG.json','RULES.json','INPUT_VALIDATION.json')))
write(ROOT/'SOURCE_MANIFEST.json',dict(study='EXP_DECEL_DIAG_20261117',files={str(p.resolve()):sha(p) for p in files}))
print(json.dumps(dict(status='FROZEN',prior_sealed_files=len(selected),manifest_files=len(set(files)),populations=len(pop))))
