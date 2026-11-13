"""Read-only NumPy review; no torch, model execution or training."""
from pathlib import Path
import json, csv, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
seen = {}
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def source(rel):
    p = ROOT / rel
    seen[str(p.relative_to(ROOT))] = digest(p)
    return p
def read_json(rel):
    return json.loads(source(rel).read_text(encoding='utf-8-sig'))
def good(p, gt):
    e = np.linalg.norm(p.astype(np.float64)-gt[:,None], axis=-1)
    return (e.mean(-1)<=.2)&(e[:,:,-1]<=.4)

e0dir='qoder_work/E0_CALV11_ORACLE_OPPORTUNITY_20261020/'
e1dir='qoder_work/E1_STUDENT_DISTILL_SCREENING_20261020/'
e0=read_json(e0dir+'RESULT.json')
with np.load(source('REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz')) as z:
    native=z['candidate'][...,:24].reshape(-1,20,12,2).astype(np.float64)
    groups=z['change_group'].copy()
    ids=z['sample_id'].copy()
with np.load(source('F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz')) as z:
    assert np.array_equal(ids,z['sample_id'])
    gt=z['future'].astype(np.float64)
with np.load(source('GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz')) as z:
    fit=z['grouped_train_index'].copy()
clean=np.load(source('qoder_work/v1_1/cal_v11_index.npy'))
with source('qoder_work/v1_1/cal_v11_windows.csv').open(encoding='utf-8-sig') as f:
    tracks=np.array([r['track'] for r in csv.DictReader(f)])
result={'scope':'Independent saved coverage, teacher occupancy, JSON aggregation and exploratory shared-track bootstrap; no model, full motion, clusters or W-gate replay.', 'E0':[], 'teachers':[], 'E1':[]}
ds=[]
for seed in (42,43,44):
    kinpath=Path(e0['results'][str(seed)]['kin_src'])
    kin=np.load(source(kinpath.relative_to(ROOT)))
    for name,ix in [('clean',clean),('dropped',np.setdiff1d(np.arange(25912,28791),clean))]:
        idx=np.load(source(e0dir+f'k0c_indices_{name}_seed{seed}.npy'))
        assert idx.shape==(len(ix),20) and np.all((idx>=0)&(idx<40))
        assert all(len(np.unique(row))==20 for row in idx)
        pool=np.concatenate([native[ix],kin[ix-25912]],axis=1)
        pred=pool[np.arange(len(ix))[:,None],idx]
        ng=good(native[ix],gt[ix]).any(1); pg=good(pred,gt[ix]).any(1)
        ch=groups[ix]==2
        delta=(pg[ch].astype(float)-ng[ch].astype(float))
        saved=np.load(source(e0dir+f'delta_change_{name}_seed{seed}.npy'))
        assert np.array_equal(delta,saved)
        result['E0'].append(dict(seed=seed,pop=name,N=int(ch.sum()),native=int(ng[ch].sum()),oracle=int(pg[ch].sum()),rescue=int((~ng[ch]&pg[ch]).sum()),lost=int((ng[ch]&~pg[ch]).sum())))
        if name=='clean': ds.append(delta)
    fk=np.load(source(f'UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_FIT_seed{seed}.npy'))
    ng=good(native[fit],gt[fit]); kg=good(fk,gt[fit]); fail=~ng.any(1)
    teach=fail&kg.any(1)
    result['teachers'].append(dict(seed=seed,failed_windows=int(fail.sum()),teachable_failed_windows=int(teach.sum()),teachable_fraction=float(teach.sum()/fail.sum()),complement_path_count=int(kg[fail].sum()),mean_paths_per_failed=float(kg[fail].sum()/fail.sum()),change_failed=int((fail&(groups[fit]==2)).sum()),change_teachable=int((teach&(groups[fit]==2)).sum())))
    j=read_json(e1dir+('RESULT.json' if seed==42 else f'RESULT_seed{seed}.json'))
    for arm,r in j['results'].items():
        result['E1'].append(dict(seed=seed,arm=arm,eval=r['eval'],history_last_two=r['history'][-2:]))

# Exploratory correction: same track draw across all seeds, optional paired seed resampling.
d=np.stack(ds); tr=tracks[groups[clean]==2]; uniq=np.unique(tr)
counts=np.array([(tr==u).sum() for u in uniq])
sums=np.stack([d[:,tr==u].sum(1) for u in uniq],axis=1)
rng=np.random.default_rng(20261020)
draw=rng.integers(len(uniq),size=(20000,len(uniq)))
means=sums[:,draw].sum(-1)/counts[draw].sum(-1)[None,:]
fixed=means.mean(0)
sd=rng.integers(3,size=(20000,3))
hier=means[sd,np.arange(20000)[:,None]].mean(1)
result['exploratory_bootstrap']={'point_pp':float(d.mean()*100),'shared_track_fixed_three_seed_ci_pp':(np.quantile(fixed,[.025,.975])*100).tolist(),'shared_track_resampled_seed_ci_pp':(np.quantile(hier,[.025,.975])*100).tolist(),'note':'Post-hoc sensitivity, not preregistered confirmation; 50 tracks / 3 seeds, no causal exposure conclusion.'}
result['inputs_sha_before']=seen.copy()
result['inputs_unchanged']=all(digest(ROOT/p)==sha for p,sha in seen.items())
result['review_script_sha']=digest(Path(__file__))
(OUT/'SAVED_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k in ('E0','teachers','exploratory_bootstrap','inputs_unchanged')},ensure_ascii=False,indent=2))
