"""CPU checkpoint, A-screen and stored resource contract review; no model."""
import sys,gc
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from review_saved import *
import torch
def exact(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b
def run():
    g=Guard(6*2**30);g.check('startup');sources_before=sources();modelrows=[];checkpoint_sha={}
    for package,(relative,arms) in PACK.items():
        e=HERE/relative;stats=load(e/'statistics.npz');pre=read(e/'PREFLIGHT.json');assert pre['status']=='PASS'
        assert pre['source_digest']==read(e/'source_manifest.json')['digest']
        for n,h in pre['execution_inputs'].items():assert sha(e/n)==h
        for arm in arms:
            for seed in SEEDS:
                g.check();run=e/f'{arm}_seed{seed}';done=read(run/'done.json');b=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
                resume=torch.load(run/'resume.pt',map_location='cpu',weights_only=False);assert exact(b,resume)
                assert b['step']==592 and len(b['history'])==8 and b['contract']==done['contract']
                assert b['contract']['source_digest']==pre['source_digest'] and b['contract']['stats']==sha(e/'statistics.npz')
                for k in ('model','optimizer','rng','assignment','history'):assert k in b
                assert len(b['assignment'])==18909 and ((b['assignment']>=0)&(b['assignment']<20)).all()
                st=b['model'];close(st['mean'].numpy(),stats['mean'],'scaler mean',0);close(st['std'].numpy(),stats['std'],'scaler std',0)
                assert all(torch.isfinite(t).all() for t in st.values())
                params={k:t for k,t in st.items() if k not in ('mean','std','projection')};nparam=sum(t.numel() for t in params.values())
                assert nparam==(98552 if package=='C2' else 82008)
                if package=='A':close(st['projection'].numpy(),np.eye(12,dtype=np.float32) if arm=='A_cv_free' else load(e/'projection.npy') if False else np.load(e/'projection.npy').astype(np.float32),'projection buffer',0)
                opt=b['optimizer']['state'];assert len(opt)==len(params)
                assert all(float(v['step'])==592 for v in opt.values())
                assert all(torch.isfinite(v[k]).all() for v in opt.values() for k in ('exp_avg','exp_avg_sq'))
                checkpoint_sha[str(run/'checkpoint.pt')]=sha(run/'checkpoint.pt');checkpoint_sha[str(run/'resume.pt')]=sha(run/'resume.pt')
                modelrows.append(dict(package=package,arm=arm,seed=seed,parameters=nparam,epochrows=8,steps=592,checkpoint_resume_allstate_exact=True,RNG_keys=list(b['rng'])))
                del b,resume;gc.collect()
    write('CHECKPOINT_CONTRACTS.json',dict(status='PASS',runs=modelrows,files=checkpoint_sha,no_model_or_optimizer_instantiated=True))
    screen=HERE/'direction_A_time_basis/screen_execution_20261024';result=read(screen/'RESULT.json');basis=load(screen/'bases.npz');manifest=read(screen/'basis_manifest.json')
    assert sha(screen/'bases.npz')==result['bases_SHA']==manifest['sha256'];refs={s:load(OLD/f'native_{s}.npz') for s in ('FIT','inner')}
    # Read previously seen FIT-only past; never open outer/CAL arrays.
    encoder=ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
    with np.load(encoder) as enc:
        lookup={int(s):i for i,s in enumerate(enc['sample_id'])};allpast=enc['past']
        past={s:allpast[[lookup[int(i)] for i in d['sample_id']]].copy() for s,d in refs.items()}
    fit=refs['FIT'];v=(past['FIT'][:,-1]-past['FIT'][:,-2]).astype(float)/.4
    cv=v[:,None]*np.arange(1,13)[None,:,None]*.4;matrix=(fit['gt']-cv).transpose(0,2,1).reshape(-1,12)
    _,sv,vh=np.linalg.svd(matrix,full_matrices=False);psi=vh[:4].copy();psi*=np.sign(psi[:,-1,None]);close(psi,basis['svd4'],'FIT SVD only')
    close(sv,basis['singular_values_FIT'],'FIT singular values')
    screen_rows=[];hashes={};audits=0
    for name in ('knot3','svd4'):
        psi=basis[name];P=psi.T@np.linalg.solve(psi@psi.T,psi);close(P,manifest['bases'][name]['projector'],'basis projection')
        for split,d in refs.items():
            g.check();v=(past[split][:,-1]-past[split][:,-2]).astype(float)/.4;cv=v[:,None]*np.arange(1,13)[None,:,None]*.4
            oracle=load(screen/f'{name}_oracle_{split}.npz');op=np.einsum('tu,bud->btd',P,d['gt']-cv)+cv
            close(op,oracle['paths'],'oracle paths');close(oracle['sample_id'],d['sample_id'],'oracle IDs',0)
            dist=np.linalg.norm(op-d['gt'],axis=-1);close(dist.mean(1),oracle['ADE'],'oracle ADE');close(dist[:,-1],oracle['FDE'],'oracle FDE')
            hashes[str(screen/f'{name}_oracle_{split}.npz')]=sha(screen/f'{name}_oracle_{split}.npz')
            for seed in SEEDS:
                p=screen/f'{name}_seed{seed}_{split}.npz';r=result['records'][name][split]['seeds'][str(seed)];assert sha(p)==r['paths_SHA']
                z=load(p);old=load(OLD/f'runs/A_seed{seed}/predictions_{split}.npz')['paths'];y=np.einsum('tu,bkud->bktd',P,old-cv[:,None])+cv[:,None]
                close(y,z['paths'],'projected paths');ee=geometry(y,d['gt']);m=read(p.with_name(p.stem+'_metrics.json'));audit=read(p.with_name(p.stem+'_audit.json'));assert audit['status']=='PASS' and not audit['mismatches'];audits+=1
                for gi,label in enumerate(GROUPS):
                    take=np.ones(len(y),bool) if gi==0 else d['groups']==gi-1;close(int(ee['coverage'][take].sum()),m[label]['good'],'screen good')
                    screen_rows.append(dict(basis=name,split=split,seed=seed,group=label,n=int(take.sum()),good=m[label]['good'],old_good=int(geometry(old[take],d['gt'][take])['coverage'].sum()),accel_mean=m[label]['motion']['accel_mean']))
                hashes[str(p)]=sha(p);del z,y,old,ee;gc.collect()
    table('A_SCREEN_ALL_GROUPS.csv',screen_rows);write('A_SCREEN_REVIEW.json',dict(status='PASS',projected_saved_populations=12,oracle_populations=4,bound_full_cluster_motion_audits=audits,all_saved_geometry_reconstructed=True,FIT_only_SVD_recomputed=True,files=hashes))
    resources={}
    for p in [HERE/v[0]/'RESOURCE.json' for v in PACK.values()]+[screen/'RESOURCE.json',HERE/'direction_B_scene_prior/probe_execution_20261024/RESOURCE.json']:
        r=read(p);obs=r['observations'];E=r['estimate'];ss=[o['snapshot'] for o in obs];assert all(o['status']=='PASS' for o in obs)
        for o,s in zip(obs,ss):
            assert max(s['rss'],s['private'],s['peak_rss'],s['peak_private'])<=1.3*E
            assert s['physical']>=o['required_physical'] and s['commit']>=o['required_commit']
        resources[str(p)]=dict(E=E,max_private=max(s['peak_private'] for s in ss),max_RSS=max(s['peak_rss'] for s in ss),min_physical=min(s['physical'] for s in ss),min_commit=min(s['commit'] for s in ss),observations=len(obs))
    write('PRODUCTION_RESOURCE_REVIEW.json',dict(status='PASS',stages=resources,snapshot_observation_time_not_peak_event_time=True))
    assert sources()==sources_before
    for p,h in checkpoint_sha.items():assert sha(p)==h
    g.check();write('CONTRACT_REVIEW.json',dict(status='PASS',checkpoint_runs=18,epochs=144,steps=10656,source_count=len(sources_before),resource_observations=g.observations,
        no_model_forward=True,no_optimizer_instantiated=True,profile_supplement_separate_limited_forward_disclosed=True))
    print('CHECKPOINT SCREEN RESOURCE CONTRACT PASS',flush=True)
if __name__=='__main__':run()
