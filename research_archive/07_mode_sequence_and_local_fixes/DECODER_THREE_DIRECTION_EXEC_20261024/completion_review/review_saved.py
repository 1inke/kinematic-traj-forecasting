"""Completion supplement: read-only saved data; no model, optimizer or forward.

Full SciPy clustering was independently audited during production and is SHA
bound here. This supplement independently repeats geometry, motion, identities,
frozen populations, histogram eligibility and every registered 20k bootstrap.
"""
import sys,json,hashlib,csv,gc
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from f16.resource import Guard
OUT=Path(__file__).resolve().parent
OLD=ROOT/'qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022'
PACK={'A':('direction_A_time_basis/train_execution_20261024',('A_cv_free','A_cv_basis')),
      'C1':('direction_C_set_mechanisms/C1_execution_20261024',('C1_hard','C1_awta')),
      'C2':('direction_C_set_mechanisms/C2_execution_20261024',('C2_isomlp','C2_setmix'))}
GROUPS=('overall','stable','transition','change');SEEDS=(42,43,44)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x))
def write(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=plain),encoding='utf-8')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def table(n,rows):
    with (OUT/n).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
MAX=0.;checks=0
def close(a,b,label,tol=2e-12):
    global MAX,checks
    aa=np.asarray(a);bb=np.asarray(b)
    assert aa.shape==bb.shape,(label,aa.shape,bb.shape)
    if aa.dtype.kind in 'USOb' or bb.dtype.kind in 'USOb':assert np.array_equal(aa,bb),label
    else:
        error=float(np.max(np.abs(aa.astype(float)-bb.astype(float)))) if aa.size else 0.
        assert error<=tol,(label,error,tol)
        MAX=max(MAX,error)
    checks+=1
def geometry(p,y):
    dist=np.sqrt(np.square(p.astype(float)-y[:,None].astype(float)).sum(-1))
    a=dist.mean(-1);f=dist[:,:,-1];j=np.maximum(a/.2,f/.4)
    return dict(ADE=a.min(1),FDE=f.min(1),J=j.min(1),coverage=((a<=.2)&(f<=.4)).any(1),
                a=a,f=f,j=j,dist=dist,q=(a+.5*f).argmin(1),jw=j.argmin(1))
def physical(p):
    v=np.concatenate((p[:,:,:1].astype(float),np.diff(p.astype(float),axis=2)),axis=2)/.4
    a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
    return [np.sqrt(np.square(z).sum(-1)) for z in (v,a,j)]
def pooled(arr,mask):
    s,a,j=arr
    return dict(speed_p95=float(np.quantile(s[mask],.95)),accel_mean=float(a[mask].mean()),
                accel_p95=float(np.quantile(a[mask],.95)),jerk_mean=float(j[mask].mean()),jerk_p95=float(np.quantile(j[mask],.95)))
def sources():
    manifests=[HERE/'series_source_manifest.json']+[HERE/v[0]/'source_manifest.json' for v in PACK.values()]
    manifests += [HERE/'direction_A_time_basis/screen_execution_20261024/source_manifest.json',
                  HERE/'direction_B_scene_prior/probe_execution_20261024/source_manifest.json']
    reg={}
    for p in manifests:
        m=read(p);d={k:v['sha256'] for k,v in m['files'].items()}
        assert hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()==m['digest']
        for v in m['files'].values():
            assert v['path'] not in reg or reg[v['path']]==v['sha256'];reg[v['path']]=v['sha256']
    for p,h in reg.items():assert sha(p)==h,('SOURCE_CHANGED',p)
    return reg
def bootstrap(v,track,expected,fixed=False):
    """Independent expansion: bootstrap multiplicities weight original windows."""
    v=np.atleast_2d(v).astype(float);u,inv=np.unique(track,return_inverse=True);rng=np.random.default_rng(20261023)
    res=np.empty(20000)
    for begin in range(0,20000,128):
        end=min(begin+128,20000);weights=[];ss=[]
        for i in range(begin,end):
            sd=np.arange(len(v)) if fixed else rng.integers(len(v),size=len(v))
            td=rng.integers(len(u),size=len(u));weights.append(np.bincount(td,minlength=len(u))[inv]);ss.append(sd)
        weights=np.asarray(weights,float);sums=weights@v.T/weights.sum(1)[:,None]
        res[begin:end]=np.take_along_axis(sums,np.asarray(ss),axis=1).mean(1)
    got=dict(point=float(v.mean()),CI95=np.quantile(res,[.025,.975]).tolist(),valid=20000,tracks=len(u))
    for k in ('point','CI95','valid','tracks'):close(got[k],expected[k],'bootstrap '+k)
    return got,res
def review_b(ref):
    root=HERE/'direction_B_scene_prior';p=root/'probe_execution_20261024';r=read(p/'RESULT.json');z=load(p/'probe_arrays.npz')
    bundle=read(root/'records_20261024.json');assert sha(root/'records_20261024.json')==r['records_SHA'];assert sha(p/'probe_arrays.npz')==r['arrays_SHA']
    rows=bundle['records'];byid={x['sample_id']:x for x in rows};bank=[byid[int(i)] for i in z['FIT_source_registry']]
    col={k:np.asarray([x[k] for x in bank]) for k in bank[0] if k not in ('split','endpoint')}
    cell=col['cell'];ends=np.asarray([x['endpoint'] for x in bank]);outside=(np.abs(ends)>6).any(1)
    ix=np.minimum(np.floor((ends+6)/.75).astype(int),15);bucket=ix[:,0]*16+ix[:,1];bucket[outside]=256
    maps={};rng=np.random.default_rng(20261023)
    for scene,h in sorted(set(zip(col['scene'],col['heading']))):
        cells,counts=np.unique(cell[(col['scene']==scene)&(col['heading']==h)],axis=0,return_counts=True)
        supported=[tuple(x) for x in cells[counts>5]];perm=rng.permutation(len(supported))
        maps[(scene,h)]=dict(zip(supported,[supported[x] for x in perm]))
    armnames=z['arms'].tolist();checked=0
    for i,sid in enumerate(z['sample_id']):
        if i%128==0:guard.check();print('B eligibility',i,flush=True)
        q=byid[int(sid)];base=col['track']!=q['track'];h=q['heading'];scene=q['scene'];seq=q['sequence']
        for j,arm in enumerate(armnames):
            valid=True;cold=False
            if arm=='M0_motion_iso':chosen=base;level='M0'
            elif arm.startswith('XSCENE'):
                chosen=base&(col['scene']!=scene)&((col['heading']==h) if h>=0 else True)
                level='cross_scene_heading' if h>=0 else 'cross_scene_no_heading'
            elif arm=='M1_heading_global':
                chosen=base&(col['scene']==scene)&(col['heading']==h) if h>=0 else np.zeros(len(base),bool)
                if chosen.sum()>5:level='scene_heading'
                else:chosen=base;level='M0'
            else:
                causal=arm.endswith('causal');valid=not causal or q['verified_time'];rs=base&(col['scene']==scene)
                if not valid:close(False,z['valid'][j,i],'invalid time');continue
                if causal:rs &= col['verified_time']&(col['sequence']==seq)&(col['end_index']<=q['t0'])
                target=tuple(q['cell'])
                if arm.startswith('SHUF') and h>=0:
                    if causal:
                        supported=[tuple(x) for x in np.unique(cell[rs&(col['heading']==h)],axis=0)]
                        sd=int.from_bytes(hashlib.sha256(json.dumps([int(sid),20261023],separators=(',',':')).encode()).digest()[:8],'little')
                        perm=np.random.default_rng(sd).permutation(len(supported));mapping=dict(zip(supported,[supported[x] for x in perm]))
                    else:mapping=maps.get((scene,h),{})
                    target=mapping.get(target,target)
                cm=(cell==target).all(1);choices=[]
                if h>=0:choices.append((rs&cm&(col['heading']==h),'cell_heading'))
                choices.append((rs&cm,'cell'))
                if h>=0:choices.append((rs&(col['heading']==h),'scene_heading'))
                chosen=None
                for m,label in choices:
                    if m.sum()>5:chosen=m;level=label;break
                if chosen is None:
                    chosen=base&(col['sequence']!=seq) if causal else base
                    level='M0_nonsequence' if causal else 'M0';cold=causal
            close(valid,z['valid'][j,i],'valid');close(cold,z['cold_start'][j,i],'cold');close(level,z['levels'][j,i],'level')
            saved=np.unpackbits(z['packed_query_source_membership'][j,i],bitorder='little')[:len(base)]
            close(saved,chosen,'all source memberships')
            counts=np.bincount(bucket[chosen],minlength=257);close(counts,z['counts'][j,i],'histogram');close(counts.sum(),z['support'][j,i],'support')
            y=ref['gt'][i,-1];b=256 if np.any(np.abs(y)>6) else int(np.minimum(np.floor((y+6)/.75).astype(int),15)@np.array([16,1]))
            nll=-np.log((counts[b]+.1)/(counts.sum()+25.7));close(nll,z['NLL'][j,i],'NLL')
            top=np.argsort(-counts[:256],kind='stable')[:4];cent=np.stack((-6+(top//16+.5)*.75,-6+(top%16+.5)*.75),1)
            close(np.linalg.norm(cent-y,axis=1).min()<=.4,z['opportunity'][j,i],'Opp4')
            close(np.linalg.norm(cent[:,None]-ref['native'][i,None,:,-1],axis=2).min(1),z['endpoint_redundancy'][j,i],'redundancy')
            checked+=1
    boot={}
    change=ref['groups']==2
    for mode in ('static','causal'):
        mask=change&z['valid'][armnames.index('PRIOR_'+mode)]
        delta=z['NLL'][armnames.index('PRIOR_'+mode),mask]-z['NLL'][0,mask]
        boot[mode],draws=bootstrap(delta,ref['track'][mask],r['decisions'][mode]['paired_track_CI'])
        np.savez_compressed(OUT/('B_'+mode+'_draws.npz'),values=draws)
    result=dict(status='PASS',eligible_queries=checked,invalid_queries=int((~z['valid']).sum()),
                full_eligibility_independent_vectorized=True,all_memberships_counts_NLL_Opp4_exact=True,
                bootstrap=boot,original_result=r)
    write('B_PROBE_REVIEW.json',result);return result
def main():
    source_before=sources();print('SOURCE PASS',len(source_before),flush=True)
    refs={s:load(OLD/f'native_{s}.npz') for s in ('FIT','inner')}
    ne={s:geometry(d['native'],d['gt']) for s,d in refs.items()};nm={s:physical(d['native']) for s,d in refs.items()}
    rows=[];motionrows=[];frozenrows=[];output_sha={};samples={};history_summary=[];audit_count=0
    for package,(relative,arms) in PACK.items():
        e=HERE/relative;samples[package]={};stats=load(e/'statistics.npz')
        for arm in arms:
            samples[package][arm]=[]
            for seed in SEEDS:
                guard.check();run=e/f'{arm}_seed{seed}';done=read(run/'done.json');assert done['status']=='SUCCESS' and done['eval_buffers_exact']
                for name,digest in done['outputs'].items():assert sha(run/name)==digest,(run,name);output_sha[str(run/name)]=digest
                hist=read(run/'history.json');assert len(hist)==8 and sum(h['steps'] for h in hist)==592
                assert [h['epoch'] for h in hist]==list(range(1,9))
                for h in hist:assert sum(h['hard_histogram'])==18909 and sum(x['n'] for x in h['batches'])==18909
                history_summary.append(dict(package=package,arm=arm,seed=seed,epoch8loss=hist[-1]['loss'],
                    epoch8entropy=hist[-1]['entropy'],epoch8active=hist[-1]['hard_active'],epoch8switch=hist[-1]['winner_switch_rate']))
                for split,d in refs.items():
                    print(package,arm,seed,split,flush=True);z=load(run/f'predictions_{split}.npz');p=z['paths'];assert p.shape==(len(d['gt']),20,12,2) and np.isfinite(p).all()
                    for k in ('sample_id','gt','native','groups','track'):close(z[k],d[k],k,0)
                    assert len(np.unique(z['sample_id']))==len(p)
                    a=read(run/f'audit_{split}.json');assert a['status']=='PASS' and not a['mismatches'];audit_count+=1
                    m=read(run/f'metrics_{split}.json');ee=geometry(p,d['gt']);ph=physical(p);base=ne[split]
                    with np.load(OLD/f'runs/A_seed{seed}/predictions_{split}.npz') as oz:old=geometry(oz['paths'],d['gt']);oldp=oz['paths'].copy() if arm=='C1_hard' else None
                    if arm=='C1_hard':close(p,oldp,'C1 hard bitwise old A',0)
                    sub=read(run/f'frozen_old_subsets_{split}.json')
                    for gi,g in enumerate(GROUPS):
                        take=np.ones(len(p),bool) if gi==0 else d['groups']==gi-1;r=m[g];cov=ee['coverage'];nc=base['coverage'];ac=old['coverage']
                        fresh=dict(n=int(take.sum()),tracks=len(np.unique(d['track'][take])),good=int(cov[take].sum()),native_good=int(nc[take].sum()),
                          rescue=int((take&cov&~nc).sum()),lost=int((take&~cov&nc).sum()),retained=int((take&cov&nc).sum()),coverage=float(cov[take].mean()))
                        for k in ('ADE','FDE','J'):fresh[k]=float(ee[k][take].mean())
                        for k,v in fresh.items():close(v,r[k],k)
                        for event,mask in [('rescue',take&cov&~nc),('lost',take&~cov&nc)]:close(d['sample_id'][mask],r[event+'_IDs'],event+' IDs',0)
                        ids=np.flatnonzero(take);q=ee['q'][ids];jw=ee['jw'][ids]
                        close(ee['a'][ids,q].mean(),r['Q_winner_ADE'],'Q_ADE');close(ee['f'][ids,q].mean(),r['Q_winner_FDE'],'Q_FDE')
                        for key,slots in [('Q',q),('J',jw)]:close(ee['dist'][ids,slots].reshape(-1,3,4).mean((0,2)),r[key+'_segments'],key+' segments')
                        mot=pooled(ph,take);nat=pooled(nm[split],take)
                        for k,v in mot.items():close(v,r['motion'][k],k);close(nat[k],r['native_motion'][k],'native '+k)
                        mp=all(mot[k]<=1.01*nat[k] for k in mot);close(mp,r['motion_pass'],'motion gate')
                        rows.append(dict(package=package,arm=arm,seed=seed,split=split,group=g,**fresh,effective_good=r['effective_good'],distinct=r['distinct'],G_base=r['G_base'],motion_pass=mp))
                        for k,v in mot.items():motionrows.append(dict(package=package,arm=arm,seed=seed,split=split,group=g,metric=k,value=v,native=nat[k],relative=v/nat[k]-1,passed=v<=1.01*nat[k]))
                        masks={'retained':ac&nc,'rescued':ac&~nc,'lost':~ac&nc,'neither':~ac&~nc}
                        for label,mask in masks.items():
                            mask &= take;rr=sub[g][label];close(int(mask.sum()),rr['n'],'subset n');close(d['sample_id'][mask],rr['IDs'],'subset IDs',0);close(int(cov[mask].sum()),rr['new_good'],'subset coverage')
                            if mask.any():
                                for k in ('ADE','FDE','J'):
                                    delta=ee[k]-old[k];close(delta[mask].mean(),rr[k+'_delta_mean'],'subset mean');close(delta[mask].sum()/take.sum(),rr[k+'_whole_group_signed_contribution'],'signed contribution')
                            frozenrows.append(dict(package=package,arm=arm,seed=seed,split=split,group=g,subset=label,n=rr['n'],new_good=rr['new_good']))
                    if split=='inner':samples[package][arm].append({k:ee[k].copy() for k in ('coverage','ADE','FDE')}|{'accel':ph[1].mean((1,2))})
                    del z,p,ph,ee,old;gc.collect()
    table('ALL_GROUPS.csv',rows);table('ALL_MOTION.csv',motionrows);table('FROZEN_OLD_SUBSETS.csv',frozenrows);write('HISTORY_SUMMARY.json',history_summary)
    boot={};change=refs['inner']['groups']==2;track=refs['inner']['track'][change]
    for package,(relative,arms) in PACK.items():
        dec=read(HERE/relative/'DECISION.json');boot[package]={};save={}
        for k in ('coverage','ADE','FDE','accel'):
            values=np.array([samples[package][arms[1]][i][k][change].astype(float)-samples[package][arms[0]][i][k][change].astype(float) for i in range(3)])
            a,draws=bootstrap(values,track,dec['primary_and_secondary_unadjusted_CI'][k]);b,fd=bootstrap(values,track,dec['fixed_seed_sensitivity'][k],True)
            boot[package][k]=dict(paired=a,fixed=b);save[k]=draws;save[k+'_fixed']=fd
        values=np.array([s['coverage'][change].astype(float)-ne['inner']['coverage'][change] for s in samples[package][arms[1]]])
        boot[package]['native_coverage'],save['native_coverage']=bootstrap(values,track,dec['native_coverage_CI'])
        np.savez_compressed(OUT/(package+'_bootstrap_draws.npz'),**save)
    write('BOOTSTRAP_REVIEW.json',dict(status='PASS',implementation='raw-window multiplicity expansion, not production track-sum code',draws=20000,results=boot))
    b=review_b(refs['inner']);source_after=sources();assert source_before==source_after
    for p,h in output_sha.items():assert sha(p)==h
    write('SOURCE_OUTPUT_SHA.json',dict(status='PASS',sources=source_after,completed_outputs=output_sha))
    write('SUMMARY.json',dict(status='PASS_SAVED_DATA_COMPLETION_SUPPLEMENT',formal_experiments=3,runs=18,epochs=144,steps=10656,
        source_count=len(source_after),completed_output_count=len(output_sha),scientific_saved_populations=36,
        bound_independent_full_cluster_audits=audit_count,independent_geometry_motion_group_rows=len(rows),
        motion_checks=len(motionrows),frozen_subset_rows=len(frozenrows),comparison_count=checks,max_abs_error=MAX,
        bootstrap=boot,model_forward=False,optimizer_loaded=False,full_cluster_third_replay=False,external_review=False))
    guard.check();write('RESOURCE.json',dict(estimate=6*2**30,inventory='CPU torch import/checkpoint metadata up to3GiB + saved FIT paths/native/geometry and physical scratch2GiB +1GiB IO/bootstrap reserve; not fixed resource threshold',observations=guard.observations))
    print('COMPLETION SUPPLEMENT PASS',MAX,checks,flush=True)
if __name__=='__main__':
    guard=Guard(6*2**30);guard.check('startup');main()
