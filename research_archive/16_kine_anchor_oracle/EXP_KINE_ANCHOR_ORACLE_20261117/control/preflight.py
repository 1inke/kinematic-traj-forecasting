from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;W=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'design_original/code'))
import kine_common as kc
import kine_repr as kr
import run_kine_oracle as run
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
data=Path('C:/Users/linke/Desktop/DRSE/ASTRA/datasets/eth_ucy/processed_data_diverse')
five=W/'qoder_work/FIVEFOLD_B0_EXEC_20261115';speed=W/'qoder_work/EXP_ETH_SPEED_EXEC_20261116'
sys.path.insert(0,str(five/'code'));old=imported('old_fivefold_data',five/'code/fivefold_data.py')
helper=imported('old_eth_speed_math',Path('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116/code/eth_speed_math.py'))
audit=five/'F0/AUDIT_RECORDING.json';assert read(audit)['passed']
seals=[five/'SOURCE_MANIFEST.json',audit,speed/'INPUT_CONTRACT.json',speed/'DIAG/DONE.json',speed/'SPEED_AUG/B0_wide_eth_E64_s42/DONE.json',W/'qoder_work/EXP_ETH_T1_EXEC_20261116/SOURCE_MANIFEST.json']
expected={}
for path in seals:
    doc=read(path)
    for k,v in doc.get('files',{}).items():expected[str(Path(k).resolve()).lower()]=v
references=[('eth','B0_s42',speed/'DIAG/eth/identity/predictions.npz'),('eth','AUG_s42',speed/'SPEED_AUG/B0_wide_eth_E64_s42/test/predictions.npz')]
selected=[audit,five/'code/fivefold_data.py',five/'code/fivefold_common.py',Path(helper.__file__),W/'FRONT_ENCODER_SERIES/train_front_variant.py',W/'HIERARCHICAL_DECODER_SERIES/hierarchical_decoder_models.py']
checks=[]
for fold in kc.FOLDS:
    selected.extend(kc.fold_files(data,fold).values())
    train=kc.load_split(data,fold,'train');test=kc.load_split(data,fold,'test')
    fi,vi=kc.scene_tail_split(train['scene']);oldfi,oldvi=old.scene_tail_split(train['scene'])
    np.testing.assert_array_equal(fi,oldfi);np.testing.assert_array_equal(vi,oldvi)
    for d in (train,test):
        group,score=old.change_groups(d['past'][:,0],d['future'])
        np.testing.assert_array_equal(d['group'],group);np.testing.assert_array_equal(d['change_score'],score)
        np.testing.assert_array_equal(kc.maneuver_labels(d['past'],d['future']),helper.maneuver_labels(d['past'],d['future']))
        np.testing.assert_array_equal(kc.observed_speed(d['past']),helper.observed_speed(d['past']))
        # Independent B0 row-vector rotation; no torch imported.
        p=d['past'][:,0].astype(float);disp=p[:,-1]-p[:,-4];norm=np.linalg.norm(disp,axis=-1)
        heading=np.zeros_like(disp);heading[:,0]=1;valid=norm>.03;heading[valid]=disp[valid]/norm[valid,None]
        matrix=np.stack([heading[:,0],-heading[:,1],heading[:,1],heading[:,0]],axis=-1).reshape(-1,2,2)
        fl,vr,hv=kc.focal_frame(d['past'],d['future'])
        np.testing.assert_allclose(fl,np.einsum('ntc,ncd->ntd',d['future'].astype(float)-p[:,-1,None],matrix),atol=1e-12,rtol=1e-12)
        np.testing.assert_array_equal(hv,valid)
    checks.append(dict(fold=fold,train=len(train['ids']),fit=len(fi),val=len(vi),test=len(test['ids']),split_groups_labels_rotation='PASS'))
    if fold=='eth':
        for _,name,path in references:
            selected.append(path)
            with np.load(path,allow_pickle=False) as pred:
                np.testing.assert_array_equal(pred['ids'],test['ids'])
                if 'future' in pred:np.testing.assert_array_equal(pred['future'],test['future'])
                if 'scene' in pred:np.testing.assert_array_equal(pred['scene'],test['scene'])
                for key in ('group','groups'):
                    if key in pred:np.testing.assert_array_equal(pred[key],test['group'])
            a,f=run.reference_errors(path,test);assert np.isfinite(a).all() and np.isfinite(f).all()
input_receipts=[]
for path in selected:
    h=sha(path);key=str(path.resolve()).lower()
    assert key in expected,'Not in prior seal: '+str(path)
    assert h==expected[key],'Prior seal mismatch: '+str(path)
    input_receipts.append(dict(path=str(path.resolve()),sha256=h,bytes=path.stat().st_size))
for path in (ROOT/'design_original').rglob('*'):
    if path.is_file():assert sha(path)==sha(Path('C:/Users/linke/Downloads/1/EXP_KINE_ANCHOR_ORACLE_20261117')/path.relative_to(ROOT/'design_original'))
assert 'ALL TESTS PASSED' in (ROOT/'SELF_TEST.stdout.log').read_text(encoding='utf-8-sig')
assert 'negative control verdict: NOT_SUPPORTED' in (ROOT/'SELF_TEST.stdout.log').read_text(encoding='utf-8-sig')
assert 'torch' not in sys.modules
config=dict(study='EXP_KINE_ANCHOR_ORACLE_20261117',data_dir=str(data),output_dir=str(W/'qoder_work/EXP_KINE_ANCHOR_ORACLE_EXEC_20261117'),folds=list(kc.FOLDS),seeds=[0,1,2],fits=150,references=[f'{f}:{n}:{p}' for f,n,p in references],threads=2)
write(ROOT/'CONFIG.json',config)
write(ROOT/'PREFLIGHT.json',dict(status='PASS',at=datetime.now(timezone(timedelta(hours=8))).isoformat(),self_tests=10,negative_control='NOT_SUPPORTED',folds=checks,original_design_byte_match=True,prior_sealed_inputs=input_receipts,recording_isolation_reused_audit_sha256=sha(audit),no_torch=True))
files=[*selected,*seals,*[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.ps1','.cmd')],ROOT/'CONFIG.json',ROOT/'PREFLIGHT.json']
write(ROOT/'SOURCE_MANIFEST.json',dict(files={str(p.resolve()):sha(p) for p in files}))
print(json.dumps(dict(status='PREFLIGHT_FROZEN',input_files=len(selected),source_files=len(set(files)),folds=checks),ensure_ascii=False))
