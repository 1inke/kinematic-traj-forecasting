"""All401 inference and saved CPU second-implementation geometry/cluster audit."""
from base import *
from src.f16.eval import evaluate,errors,motion,clusters
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,fcluster
fb=module('_observable_original_features',OLD/'feature_builder.py')

def build_features(z,obs,lo):
    assert np.array_equal(z['sample_id'],obs['sample_id'][lo:lo+len(z['sample_id'])])
    return np.array([fb.build(p,t,n,k,a) for p,t,n,k,a in zip(obs['past'][lo:],obs['tokens'][lo:],z['native'],z['kin'],z['motion'])],np.float32)

def edit(z,s):
    assert s.shape==z['eligible'].shape and np.isfinite(s).all()
    chosen=np.where(z['eligible'],s,-np.inf).argmax(1);paths=z['native'].copy()
    for i,a in enumerate(chosen):
        if a:r,l=divmod(int(a)-1,20);paths[i,r]=z['kin'][i,l]
    return chosen,paths

def assignment(z,chosen):
    row=np.arange(len(chosen));b=z['B'].any(1);hitB=z['B'][row,chosen];hitT=z['target'][row,chosen];native=errors(z['native'],z['gt'])['coverage']
    q=z['quality'];actual=q[row,chosen];ref=q[:,0];harm=(actual>ref).any(1);opportunity=(z['B']&(q[:,:,2]<=1)).any(1)&~native;rescued=(actual[:,2]<=1)&~native
    return dict(n=len(chosen),B_nonempty=int(b.sum()),B_empty=int((~b).sum()),hit_T=float(hitT.mean()),hit_B=float(hitB[b].mean()) if b.any() else None,empty_B_STOP=float((chosen[~b]==0).mean()) if (~b).any() else None,error_IDs=z['sample_id'][~hitT].tolist(),STOP=int((chosen==0).sum()),executed=int((chosen!=0).sum()),B_chosen=int(hitB.sum()),eligible_nonpositive_execution=int(((chosen!=0)&~hitB).sum()),native_good=int(native.sum()),original_good_with_B=int((native&b).sum()),original_failure_with_B=int((~native&b).sum()),true_rescue_opportunities=int(opportunity.sum()),true_rescue_utilized=int((opportunity&rescued).sum()),rescue_utilization=float((opportunity&rescued).sum()/opportunity.sum()) if opportunity.any() else None,harmful_execution=int(((chosen!=0)&harm).sum()),ineffective_nonB_execution=int(((chosen!=0)&~hitB&~harm).sum()),STOP_missed_B=int(((chosen==0)&b).sum()),B_original_good_continuous_actions=int((z['B']&native[:,None]).sum()),B_original_failure_still_failure_actions=int((z['B']&~native[:,None]&(q[:,:,2]>1)).sum()),B_true_rescue_actions=int((z['B']&~native[:,None]&(q[:,:,2]<=1)).sum()),chosen_quality_delta_mean=(actual-ref).mean(0).tolist())

def save_population(arm,seed,split,score,g):
    assert (HERE/'ALL_PROBE_FINALS_LOCK.json').exists(),'No held effects before ALL final lock'
    out=HERE/'populations'/f'{arm}_seed{seed}'/split;out.mkdir(parents=True,exist_ok=True);c=dict(source=source(),arm=arm,seed=seed,split=split,finals=sha256(HERE/'ALL_PROBE_FINALS_LOCK.json'))
    if (out/'done.json').exists():completed(out,c);return
    if list(out.iterdir()):raise FileExistsError('Preserve partial population')
    with np.load(HERE/f'prepared/observations_{split}.npz') as q:obs={k:q[k] for k in q.files}
    lo=0;chunks=[]
    for j,z in enumerate(aux_chunks(seed,split)):
        x=build_features(z,obs,lo);s=np.asarray(score(x),np.float64);chosen,paths=edit(z,s);name=f'chunk{j:04d}.npz'
        save_npz(out/name,scores=s,chosen=chosen,paths=paths,sample_id=z['sample_id'],eligible=z['eligible'],B=z['B'],target=z['target']);chunks.append(name);lo+=len(s)
        g.check();mark('PROBE_SAVE_ALL401',arm=arm,seed=seed,split=split,saved_windows=lo)
    assert lo==len(obs['sample_id']);resource_report(out,g);finish(out,c,chunks=chunks,n=lo)

def independent_check(z,s,chosen,paths):
    choice=np.array([int(np.flatnonzero(z['eligible'][i])[np.argmax(s[i,z['eligible'][i]])]) for i in range(len(s))]);assert np.array_equal(choice,chosen)
    for i,a in enumerate(chosen):
        if not a:assert np.array_equal(paths[i],z['native'][i]);continue
        r,l=divmod(int(a)-1,20);assert np.array_equal(paths[i,r],z['kin'][i,l]);assert np.array_equal(np.delete(paths[i],r,axis=0),np.delete(z['native'][i],r,axis=0))
    e=errors(paths,z['gt']);cc=clusters(paths,e['good']);review=[]
    for i,(p,y) in enumerate(zip(paths,z['gt'])):
        dist=np.sqrt(((p-y)**2).sum(-1));a=dist.sum(-1)/12;f=dist[:,-1];good=(a<=.2)&(f<=.4)
        assert np.max(np.abs(a-e['ADE'][i]))<=2e-12 and np.max(np.abs(f-e['FDE'][i]))<=2e-12 and np.array_equal(good,e['good'][i])
        d=np.maximum(pdist(p.reshape(20,24),metric=lambda u,v:np.sqrt(((u.reshape(12,2)-v.reshape(12,2))**2).sum(-1)).mean())/.1,pdist(p[:,-1])/.2)
        labels=fcluster(linkage(d,method='complete'),1,criterion='distance');review.append([len(set(labels)),len(set(labels[good]))])
    assert np.array_equal(cc,np.array(review));return e,cc

def independent_motion(paths):
    v=np.stack([(paths[:,:,t]-(paths[:,:,t-1] if t else 0))/.4 for t in range(12)],2)
    a=np.stack([(v[:,:,t+1]-v[:,:,t])/.4 for t in range(11)],2);j=np.stack([(a[:,:,t+1]-a[:,:,t])/.4 for t in range(10)],2)
    speed,acc,jerk=[np.sqrt((x*x).sum(-1)) for x in (v,a,j)]
    return dict(speed_p95=float(np.quantile(speed,.95,method='linear')),accel_mean=float(acc.mean()),accel_p95=float(np.quantile(acc,.95,method='linear')),jerk_mean=float(jerk.mean()),jerk_p95=float(np.quantile(jerk,.95,method='linear')))

def audit(arm,seed,split,tiny=False,reading=512):
    g=guard('AUDIT')
    if tiny:
        folder=HERE/'step1'/f'{arm}_seed{seed}';out=folder/f'audit{reading:03d}';out.mkdir(exist_ok=True);inputfile=folder/f'reading{reading:03d}.npz'
        with np.load(HERE/'prepared/tiny.npz') as q:z={k:q[k] for k in q.files if k not in ('features','opt_row')}
        with np.load(inputfile) as q:s=q['scores']
        chosen,paths=edit(z,s);blocks=[(z,s,chosen,paths)]
    else:
        folder=HERE/'populations'/f'{arm}_seed{seed}'/split;d=read_json(folder/'done.json');completed(folder,d['contract']);out=folder/'audit';out.mkdir(exist_ok=True);inputfile=folder/'done.json';blocks=[]
        for name,z in zip(d['chunks'],aux_chunks(seed,split)):
            with np.load(folder/name) as q:
                for k in ('sample_id','eligible','B','target'):assert np.array_equal(q[k],z[k])
                blocks.append((z,q['scores'],q['chosen'],q['paths']))
    c=dict(source=source(),input_sha=sha256(inputfile),independent_pdist=True)
    if (out/'done.json').exists():completed(out,c);return
    parts=[];allcc=[]
    for z,s,ch,p in blocks:
        e,cc=independent_check(z,s,ch,p);allcc.append(cc);parts.append((z,ch,p,e));g.check()
    full={k:np.concatenate([z[k] for z,_,_,_ in parts]) for k in ('sample_id','track_key','group','native','gt','quality','eligible','B','target')};paths=np.concatenate([p for _,_,p,_ in parts]);choice=np.concatenate([ch for _,ch,_,_ in parts]);rows,e=evaluate(paths,full['gt'],full['group'],full['native'],full['sample_id'],full['track_key'],include_clusters=False);cc=np.concatenate(allcc)
    for i,name in enumerate(('overall','stable','transition','change')):
        mask=np.ones(len(paths),bool) if i==0 else full['group']==i-1
        if mask.any():rows[name].update(distinct=float(cc[mask,0].mean()),effective_good=float(cc[mask,1].mean()))
        if mask.any():
            for p,key in ((paths,'motion'),(full['native'],'native_motion')):
                check=independent_motion(p[mask]);assert all(abs(check[k]-rows[name][key][k])<=2e-12 for k in check)
    assignments={}
    for i,name in enumerate(('overall','stable','transition','change')):
        m=np.ones(len(paths),bool) if i==0 else full['group']==i-1
        assignments[name]=assignment({k:v[m] for k,v in full.items()},choice[m]) if m.any() else dict(n=0)
    native=errors(full['native'],full['gt'])['coverage'];good=e['coverage'];save_npz(out/'sample_metrics.npz',sample_id=full['sample_id'],track_key=full['track_key'],group=full['group'],good=good,native_good=native,chosen=choice,ADE=e['ADE'],FDE=e['FDE'],J=e['J'],Q_slot=e['Q_winner'],J_slot=e['J_winner'],Q_segments=e['Q_segments'],J_segments=e['J_segments'])
    write_json(out/'METRICS.json',rows);write_json(out/'ASSIGNMENT.json',assignments);resource_report(out,g);finish(out,c,n=len(paths),all401_argmax_fixed19_STOP=True,full_samecandidate_quality=True,full_independent_complete_link=True,pooled_all5=True)

if __name__=='__main__':
    try:audit(sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[1]=='tiny',int(sys.argv[5]) if len(sys.argv)>5 else 512)
    except BaseException as e:fail(e);raise
