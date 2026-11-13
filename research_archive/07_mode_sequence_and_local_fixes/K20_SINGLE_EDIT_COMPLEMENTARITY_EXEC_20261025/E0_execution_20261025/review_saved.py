"""Independent direct-reconstruction reviewer; no production label/motion/quality calls."""
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist,squareform
from common import *
def direct(n,k,y):
    paths=np.broadcast_to(np.asarray(n,np.float64),(401,20,12,2)).copy()
    for a in range(1,401):r,l=divmod(a-1,20);paths[a,r]=k[l]
    # Explicit observed-origin concat, not production diff(prepend) implementation.
    origin=np.concatenate((np.zeros((401,20,1,2)),paths),axis=2)
    velocity=(origin[:,:,1:]-origin[:,:,:-1])/.4
    acceleration=(velocity[:,:,1:]-velocity[:,:,:-1])/.4
    jerk=(acceleration[:,:,1:]-acceleration[:,:,:-1])/.4
    mm=[np.sqrt(np.sum(v*v,axis=-1)).reshape(401,-1) for v in (velocity,acceleration,jerk)]
    m=np.column_stack((np.quantile(mm[0],.95,axis=1),mm[1].mean(1),np.quantile(mm[1],.95,axis=1),mm[2].mean(1),np.quantile(mm[2],.95,axis=1)))
    d=np.sqrt(np.sum((paths-y[None,None])**2,axis=-1));a=d.mean(-1);f=d[:,:,-1];jj=np.maximum(a/.2,f/.4)
    q=np.column_stack((a.min(1),f.min(1),jj.min(1)))
    rejects=m>m[0]*1.01;rejects[0]=False;eligible=~rejects.any(1)
    B=np.zeros(401,bool)
    for i in range(1,401):B[i]=eligible[i] and all(q[i]<=q[0]) and any(q[i]<q[0]) and (q[0,2]>1 or q[i,2]<=1)
    target=B.copy()
    if not B.any():target[0]=True
    choices=np.flatnonzero(B);act=min(choices,key=lambda i:(q[i,2],q[i,0],q[i,1],(i-1)//20,(i-1)%20)) if len(choices) else 0
    return dict(motion=m,quality=q,rejects=rejects,eligible=eligible,B=B,target=target,margin=q[0]-q,action=int(act)),paths[int(act)]
def compare(saved,reference):
    maximum=0.
    for name in ('motion','quality','margin'):
        err=float(np.max(np.abs(saved[name]-reference[name])));maximum=max(maximum,err)
        if err>2e-12:raise ValueError(f'Independent arithmetic {name}: {err}')
    for name in ('eligible','rejects','B','target'):
        if not np.array_equal(saved[name],reference[name]):raise ValueError('Literal classification mismatch '+name)
    if saved['action']!=reference['action']:raise ValueError('Oracle action mismatch')
    return maximum
def independent_population(paths,gt,groups,native,ids,tracks):
    from src.f16.eval import evaluate
    # Canonical global metrics and separate pdist complete-link certificate.
    rows,err=evaluate(paths,gt,groups,native,ids,tracks,include_clusters=True)
    independent=[]
    for p,g in zip(paths,err['good']):
        # pdist Euclidean at each time, then same complete-link max metric.
        dist=np.stack([pdist(p[:,t,:]) for t in range(12)])
        condensed=np.maximum(dist.mean(0)/.1,dist[-1]/.2)
        label=fcluster(linkage(condensed,method='complete'),1,criterion='distance')
        independent.append([len(set(label)),len(set(label[g]))])
    independent=np.asarray(independent)
    dp=np.sqrt(np.sum((np.asarray(paths,np.float64)-np.asarray(gt,np.float64)[:,None])**2,axis=-1))
    dn=np.sqrt(np.sum((np.asarray(native,np.float64)-np.asarray(gt,np.float64)[:,None])**2,axis=-1))
    pa,pf=dp.mean(-1),dp[:,:,-1];na,nf=dn.mean(-1),dn[:,:,-1]
    pg=((pa<=.2)&(pf<=.4)).any(1);ng=((na<=.2)&(nf<=.4)).any(1)
    for gi,name in enumerate(('overall','stable','transition','change')):
        mask=np.ones(len(paths),bool) if gi==0 else groups==gi-1
        if not mask.any():continue
        if rows[name]['distinct']!=float(independent[mask,0].mean()) or rows[name]['effective_good']!=float(independent[mask,1].mean()):raise ValueError('Independent complete-link classification')
        counts=dict(n=int(mask.sum()),good=int(pg[mask].sum()),native_good=int(ng[mask].sum()),rescue=int((mask&pg&~ng).sum()),lost=int((mask&~pg&ng).sum()),retained=int((mask&pg&ng).sum()))
        if any(rows[name][k]!=v for k,v in counts.items()):raise ValueError('Independent population coverage/event denominator')
        vals=(pa[mask].min(1).mean(),pf[mask].min(1).mean(),np.maximum(pa[mask]/.2,pf[mask]/.4).min(1).mean())
        if max(abs(rows[name][k]-v) for k,v in zip(('ADE','FDE','J'),vals))>2e-12:raise ValueError('Independent population minADE/minFDE/same-slotJ')
        if rows[name]['rescue_IDs']!=ids[mask&pg&~ng].tolist() or rows[name]['lost_IDs']!=ids[mask&~pg&ng].tolist():raise ValueError('Independent event IDs')
        # Flatten ALL windows/candidates before five statistics, including speed of observation->first prediction.
        p=np.asarray(paths[mask],np.float64);v=(np.concatenate((np.zeros_like(p[:,:,:1]),p),axis=2)[:,:,1:]-np.concatenate((np.zeros_like(p[:,:,:1]),p),axis=2)[:,:,:-1])/.4
        a=(v[:,:,1:]-v[:,:,:-1])/.4;j=(a[:,:,1:]-a[:,:,:-1])/.4
        s,aa,jj=[np.sqrt(np.sum(x*x,axis=-1)).ravel() for x in (v,a,j)]
        vals=(np.quantile(s,.95),aa.mean(),np.quantile(aa,.95),jj.mean(),np.quantile(jj,.95))
        if max(abs(rows[name]['motion'][k]-x) for k,x in zip(NAMES,vals))>2e-12:raise ValueError('Independent pooled physics')
    return rows,independent
def audit_seed(seed,data,source,guard):
    from data_contract import kin_opt
    folder=HERE/f'seed{seed}';done=validate_done(folder/'done.json',source);kin=kin_opt(seed,data['opt_row'])
    maximum=0.;count=0;oracle_paths=[]
    for filename in done['chunks']:
        with np.load(folder/filename,allow_pickle=False) as z:
            start=int(z['start']);end=start+len(z['sample_id'])
            assert np.array_equal(z['sample_id'],data['ids'][start:end])
            for i in range(end-start):
                row=start+i;ref,path=direct(data['native'][row],kin[row],data['gt'][row])
                saved={name:z[name][i] for name in ('motion','quality','rejects','eligible','B','target','margin','action')}
                maximum=max(maximum,compare(saved,ref));oracle_paths.append(path);count+=400
        guard.check();stage('E0_INDEPENDENT_ALL_ACTION_AUDIT',seed=seed,audited_windows=end,audited_nonSTOP=count)
    op=np.asarray(oracle_paths)
    with np.load(folder/'oracle.npz',allow_pickle=False) as z:
        if not np.array_equal(z['paths'],op) or not np.array_equal(z['sample_id'],data['ids']):raise ValueError('Oracle saved path/ID')
    rows,cc=independent_population(op,data['gt'],data['groups'],data['native'],data['ids'],data['tracks'])
    production=read_json(folder/'POPULATION_METRICS.json')
    if digest(rows)!=digest(production):raise ValueError('Population metrics source/review disagreement')
    validate_done(folder/'done.json',source)
    result=dict(status='PASS_INDEPENDENT_ALL_ACTIONS_AND_ORACLE',seed=seed,windows=len(op),nonSTOP_actions=count,max_abs_error=maximum,full_K20_clusters=True,all_group_pooled_motion=True,saved_arithmetic_not_model_replay=True)
    write_json(folder/'ARITHMETIC_AUDIT.json',result);return result
