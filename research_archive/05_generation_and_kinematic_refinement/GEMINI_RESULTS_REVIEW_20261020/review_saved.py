"""Read-only NumPy audit of saved outer arrays; no training/import of old runners.

Writes only this review directory. Not a source-lock reconstruction, full-cluster
audit, checkpoint replay or confirmatory analysis. Review CIs are post-hoc.
"""
from pathlib import Path
import hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
EXPS = ['D1_TIMED_MULTI_HORIZON_PLAN_20261020',
        'D2_MOTION_PRETRAINED_REPRESENTATION_20261020',
        'EXP03_KINEMATIC_DAMPING_FACTORIAL',
        'D3_FUTURE_INTERACTION_CONDITIONED_20261020']
sources = {}
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def watch(p):
    sources[str(p)] = sha(p)
    return p
def readj(p): return json.loads(watch(p).read_text(encoding='utf-8-sig'))
def npz(p): return np.load(watch(p), allow_pickle=False)
def errors(p, gt):
    e = np.sqrt(((p.astype(np.float64)-gt[:,None])**2).sum(-1))
    a,f = e.mean(-1),e[:,:,-1]
    return np.stack((((a<=.2)&(f<=.4)).any(1),a.min(1),f.min(1),np.maximum(a/.2,f/.4).min(1)),axis=1)
def motion(p):
    v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4
    a=np.diff(v,axis=2)/.4; j=np.diff(a,axis=2)/.4
    v,a,j=[np.sqrt((x*x).sum(-1)) for x in (v,a,j)]
    return dict(speed_p95=float(np.quantile(v,.95)),accel_mean=float(a.mean()),accel_p95=float(np.quantile(a,.95)),jerk_mean=float(j.mean()),jerk_p95=float(np.quantile(j,.95)))

with npz(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz') as z:
    ids=z['sample_id']; groups=z['change_group']
    native=z['candidate'][...,:24].reshape(-1,20,12,2).astype(np.float64)
with npz(ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz') as z:
    ix=z['outer_eval_index']; assert np.array_equal(z['sample_id'],ids[:25912])
with npz(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz') as z:
    assert np.array_equal(z['sample_id'],ids); gt=z['future'].astype(np.float64)
with npz(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz') as z:
    order=np.argsort(z['sample_id']); loc=order[np.searchsorted(z['sample_id'][order],ids)]
    assert np.array_equal(z['sample_id'][loc],ids)
    tr=np.array([f'{s}::{int(p)}' for s,p in zip(z['sequence'][loc],z['pedestrian_id'][loc])])
ids,groups,native,gt,tr=ids[ix],groups[ix],native[ix],gt[ix],tr[ix]
base=errors(native,gt); change=groups==2
out={'scope':'24 outer saved outputs: all-group quality/counts/pooled motion, history; no clusters, model or original CI replay', 'experiments':{}}

def ci_review(delta):
    # delta: paired seed x window x metrics. Sample complete outer tracks,
    # then subset change; sample paired model-seeds. Supplement, not prereg.
    tracks,inv=np.unique(tr,return_inverse=True); nt=len(tracks)
    denom=np.bincount(inv[change],minlength=nt)
    sums=np.zeros((3,nt,delta.shape[-1]))
    for s in range(3): np.add.at(sums[s],inv[change],delta[s,change])
    rng=np.random.default_rng(20261020); draws=[]
    for _ in range(40):
        w=rng.multinomial(nt,np.full(nt,1/nt),size=500)
        sw=rng.multinomial(3,[1/3]*3,size=500)/3
        den=w@denom
        num=np.einsum('bt,stm,bs->bm',w,sums,sw)
        draws.append(num[den>0]/den[den>0,None])
    d=np.concatenate(draws)
    return {'point':delta[:,change].mean((0,1)).tolist(),'ci95':np.quantile(d,[.025,.975],axis=0).tolist(),'simultaneous48':np.quantile(d,[.05/(2*48),1-.05/(2*48)],axis=0).tolist(),'n_draws':len(d),'note':'Post-hoc seed+track sensitivity, full outer track bootstrap, family48 conservative reference; only ~10 tail draws.'}

for name in EXPS:
    folder=ROOT/'GEMINI_EXP'/name; result=readj(folder/'RESULT.json')
    for p in folder.glob('*.py'): watch(p)
    readj(folder/'ARITHMETIC_AUDIT.json')
    record={'runs':{},'reported_bootstrap':result['bootstrap_summary']}; vec={}; maxerr=0.
    for arm in ['A','B']:
        arr=[]
        for seed in [42,43,44]:
            key=f'{arm}_seed{seed}'; rf=folder/'runs'/key
            with npz(rf/'outer_predictions.npz') as z:
                assert np.array_equal(z['sample_id'],ids)
                assert np.array_equal(z['native'],native) and np.array_equal(z['gt'],gt)
                p=z['paths'].astype(np.float64); assert p.shape==(1035,20,12,2) and np.isfinite(p).all()
            v=errors(p,gt); arr.append(v)
            history=readj(rf/'history.json')
            hist=history['generator'] if isinstance(history,dict) else history
            assert len(hist)==25
            if isinstance(history,dict): assert len(history['pretrain'])==20
            rr={}
            for gn,mask in [('overall',np.ones(len(ids),bool)),('stable',groups==0),('transition',groups==1),('change',change)]:
                b=base[:,0].astype(bool); g=v[:,0].astype(bool)
                m=motion(p[mask]); bm=motion(native[mask]); saved=result['runs'][key]['outer'][gn]
                vals=dict(good=int(g[mask].sum()),rescue=int((g&~b&mask).sum()),lost=int((~g&b&mask).sum()),ADE=float(v[mask,1].mean()),FDE=float(v[mask,2].mean()),J=float(v[mask,3].mean()))
                for k,value in vals.items(): maxerr=max(maxerr,abs(value-saved[k]))
                for k,value in m.items(): maxerr=max(maxerr,abs(value-saved['motion'][k]))
                rr[gn]={**vals,'motion_ratios':{k:m[k]/bm[k] if bm[k] else None for k in m},'motion_pass':all(m[k]<=1.01*bm[k] for k in m)}
            record['runs'][key]=rr
        vec[arm]=np.stack(arr)
    record['max_outer_numeric_error']=maxerr
    record['paired_B_minus_A_review_CI']=ci_review(vec['B']-vec['A'])
    record['B_minus_native_review_CI']=ci_review(vec['B']-base[None])
    record['secondary_reported_change']={s:{k:result['runs'][k][s]['change']['good'] for k in result['runs']} for s in ['FIT','inner','original_CAL']}
    assert maxerr < 1e-10, (name,maxerr)
    out['experiments'][name]=record
    print(name,'saved outer PASS',maxerr,flush=True)

# Existing encoded past was heading-normalized before D2 masks were applied.
with npz(ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz') as z:
    past=z['past'].astype(np.float64)
    heading=past[:,-1]-past[:,-4]
    moving=np.linalg.norm(heading,axis=1)>.03
    out['D2_cached_heading']={'moving_windows':int(moving.sum()),'max_abs_lateral_recent_displacement':float(abs(heading[moving,1]).max()),'note':'Cache construction rotates using last minus fourth-from-last observed point; masking last two after this transform retains their heading information.'}

assert all(sha(Path(p))==v for p,v in sources.items()),'Source changed during review'
out['source_sha_before_after_unchanged']=sources
out['original_audit_status_not_validated']=True
(HERE/'SAVED_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('SAVED_REVIEW.json written; original files unchanged',flush=True)
