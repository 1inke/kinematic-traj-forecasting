"""Second implementation of full saved-array arithmetic; no model or training imports."""
import hashlib
import time
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist
import attribution_core as c

def equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.dtype.kind in 'bUiS' or b.dtype.kind in 'bUiS':assert np.array_equal(a,b)
    else:assert np.allclose(a,b,rtol=0,atol=2e-12,equal_nan=True),float(np.nanmax(np.abs(a-b)))

def scalar_quality(p,gt):
    e=np.sqrt(np.sum((p-gt[None])**2,axis=-1));a=np.sum(e,axis=1)/12;f=e[:,11]
    return a,f,np.maximum(a/.2,f/.4),(a<=.2)&(f<=.4)

def physics(p,past):
    v=(p-np.concatenate((np.zeros((20,1,2)),p[:,:11]),axis=1))/.4
    a=(v[:,1:]-v[:,:11])/.4;dv=v[:,1:]-v[:,:11];j=(dv[:,1:]-dv[:,:10])/(.4**2)
    vn,an,jn=[np.sqrt((aa*aa).sum(-1)).ravel() for aa in (v,a,j)]
    junction=np.sqrt(((p[:,0]/.4-(past[-1]-past[-2])/.4)**2).sum(1))/.4
    return np.array([np.quantile(vn,.95),an.mean(),np.quantile(an,.95),jn.mean(),np.quantile(jn,.95),junction.mean(),np.quantile(junction,.95)])

def clusters(p,good):
    # Second distance construction: independent pdist perpoint rather than square matrix.
    ds=np.stack([pdist(p[:,t,:]) for t in range(12)],axis=1)
    labs=fcluster(linkage(np.maximum(ds.mean(1)/.10,ds[:,11]/.20),method='complete'),1,criterion='distance')
    return len(set(labs.tolist())),len(set(labs[good].tolist()))

def review_maps(z,ids,track,sequence,maps,check):
    a=z.reshape(-1,40).astype(float);wid=np.repeat(ids,20);sl=np.tile(np.arange(20),128)
    tr=np.repeat(track,20);seq=np.repeat(sequence,20);unique=np.unique(tr)
    for q in range(2560):
        if q%64==0:check()
        # Separate by track, scalar block distances, ordered Python tie arbitration.
        choices={scope:{arm:[] for arm in ('G','H')} for scope in ('main','all')}
        ts={'main':[],'all':[]}
        for t in unique:
            if t==tr[q]:continue
            ix=np.flatnonzero(tr==t)
            dp=np.sum((a[ix,16:]-a[q,16:])**2,axis=1)/24
            dh=np.sum((a[ix,:16]-a[q,:16])**2,axis=1)/16
            for scope in ('main','all'):
                if scope=='main' and seq[ix[0]]!=seq[q]:continue
                ts[scope].append(t)
                for arm,dd in [('G',dp),('H',.5*(dp+dh))]:
                    pos=min(range(len(ix)),key=lambda j:(dd[j],int(wid[ix[j]]),int(sl[ix[j]])))
                    choices[scope][arm].append((dd[pos],int(wid[ix[pos]]),int(sl[ix[pos]]),int(ix[pos])))
        for scope in ('main','all'):
            for arm in ('G','H'):
                chosen=sorted(choices[scope][arm])[:5];expected=[v[-1] for v in chosen]+[-1]*(5-len(chosen))
                equal(expected,maps[arm+'_'+scope][q])
            tt=sorted(ts[scope],key=lambda t:hashlib.sha256(f'endpoint-support-attribution-v1|{wid[q]}|{sl[q]}|{t}'.encode()).hexdigest())[:5]
            hh=[]
            for t in tt:
                ix=np.flatnonzero(tr==t)
                hh.append(int(min(ix,key=lambda j:hashlib.sha256(f'endpoint-support-attribution-candidate-v1|{wid[q]}|{sl[q]}|{wid[j]}|{sl[j]}'.encode()).hexdigest())))
            equal(hh+[-1]*(5-len(hh)),maps['hash_'+scope][q])

def main():
    g=c.guard();g.check(mode='startup',force=True);t=time.perf_counter()
    source=c.json_read(c.ROOT/'SOURCE_EXTENSION.json')['files'];c.verify_files(source,g.check)
    z=c.load_npz(c.ROOT/'derived_ports.npz');obs=c.load_npz(c.ROOT/'observed_inputs.npz')
    saved=c.load_npz(c.ROOT/'port_window_metrics.npz');cf=c.load_npz(c.ROOT/'counterfactual_metrics.npz')
    paths,gt=z['paths'],z['gt'];a0=c.space('END')
    for s,p in enumerate(paths):
        for i in range(128):
            g.check();a,f,j,good=scalar_quality(p[i],gt[i])
            for nm,val in [('ADE',a),('FDE',f),('J',j),('good',good)]:equal(val,z[nm][s,i]);equal(val,saved[c.PORTS[s]+'_'+nm][i])
            dis,eff=clusters(p[i],good);equal(dis,saved[c.PORTS[s]+'_distinct'][i]);equal(eff,saved[c.PORTS[s]+'_effective_good'][i])
            equal(good.any(),saved[c.PORTS[s]+'_covered'][i])
            # Guard evaluated on exact production paths, not near-equal reconstructed paths.
            if s in (2,3,4):
                for k in range(20):
                    n=z['native'][i,k]
                    pp=p[i,k];v=(pp-np.vstack((np.zeros((1,2)),pp[:-1])))/.4
                    vv=(n-np.vstack((np.zeros((1,2)),n[:-1])))/.4
                    aa=np.diff(v,axis=0)/.4;an=np.diff(vv,axis=0)/.4
                    jj=np.diff(v,n=2,axis=0)/(.4**2);jn=np.diff(vv,n=2,axis=0)/(.4**2)
                    vn,aan,jjn,vo,ao,jo=[np.linalg.norm(t,axis=-1) for t in (v,aa,jj,vv,an,jn)]
                    met=np.array([np.quantile(vn,.95),aan.mean(),np.quantile(aan,.95),jjn.mean(),np.quantile(jjn,.95)])
                    base=np.array([np.quantile(vo,.95),ao.mean(),np.quantile(ao,.95),jo.mean(),np.quantile(jo,.95)])
                    vo0=(z['past'][i,-1]-z['past'][i,-2])/.4
                    failed=np.r_[~(met<=base*1.01),not(np.linalg.norm(v[0]-vo0)/.4<=1.01*np.linalg.norm(vv[0]-vo0)/.4),not np.isfinite(pp).all()]
                    if np.array_equal(n,pp):failed[:]=False
                    equal(failed,z['failed'][s,i,k])
        print('review port',c.PORTS[s],flush=True)
    # Independently reconstruct continuous and all actually selected END paths.
    for i in range(128):
        for k in range(20):
            for s in (0,2,3,4):
                action=int(z['action_id'][s,i,k]);native=z['native'][i,k]
                out=native.copy()
                if action!=a0.identity:
                    for tt in range(12):out[tt]=native[tt]+((tt+1)/12)*a0.params[action]
                equal(out,paths[s,i,k])
            equal(z['native'][i,k]+(np.arange(1,13)/12)[:,None]*(.4*z['continuous_param'][i,k]),paths[1,i,k])
    for mode in ('restore','edit'):
        for i in range(128):
            g.check()
            for k in range(20):
                pp=np.array(paths[3 if mode=='restore' else 0,i],copy=True)
                pp[k]=paths[0 if mode=='restore' else 3,i,k]
                _,_,j,good=scalar_quality(pp,gt[i]);dis,eff=clusters(pp,good)
                for nm,val in [('covered',good.any()),('minJ',j.min()),('distinct',dis),('effective_good',eff)]:equal(val,cf[mode+'_'+nm][i,k])
                ph=physics(pp,z['past'][i])
                for nm,v in zip(('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95','junction_mean','junction_p95'),ph):equal(v,cf[mode+'_'+nm][i,k])
        print('review all2560 counterfactual sets',mode,flush=True)
    maps=c.load_npz(c.ROOT/'neighbor_map.npz');seal=c.json_read(c.ROOT/'NEIGHBOR_MAP_SEAL.json')
    assert seal['sha']==c.sha(c.ROOT/'neighbor_map.npz')
    review_maps(obs['normalized'],z['sample_id'],z['track_key'],z['sequence'],maps,g.check)
    # All cross-applied actions use each query's own safe mask / J table.
    ut=c.load_npz(c.ROOT/'neighbor_utility_metrics.npz');acts=z['teacher_action'].ravel();jm=z['J_matrix'].reshape(2560,49)
    sf=z['safe'].reshape(2560,49);base=z['J'][0].ravel();target=z['target'].reshape(2560,2)
    for key,mp in maps.items():
        if 'distance_squared' in key:continue
        for q in range(2560):
            for r,j in enumerate(mp[q]):
                if j<0:continue
                act=acts[j];exe=act if sf[q,act] else a0.identity
                equal(ut[key+'_executed_gap'][q,r],jm[q,exe]-jm[q,acts[q]])
                equal(ut[key+'_proposal_gap'][q,r],jm[q,act]-jm[q,acts[q]])
                equal(ut[key+'_proposal_safe'][q,r],float(sf[q,act]))
                equal(ut[key+'_target_distance'][q,r],np.linalg.norm(target[q]-target[j]))
    db=c.json_read(c.ROOT/'exact_input_conflicts.json')
    for arm,zz in [('G',obs['normalized'][...,16:]),('H',obs['normalized'])]:
        dic={};zz=zz.reshape(2560,-1).copy();zz[zz==0]=0
        for i,row in enumerate(zz):dic.setdefault(row.tobytes(),[]).append(i)
        total=0.
        for ix in dic.values():
            yy=target[ix];mu=sum(yy)/len(ix);total+=sum(float(np.dot(v-mu,v-mu)) for v in yy)
        equal(total/2/2560,db[arm]['empirical_MSE_lower_bound']);equal(len(dic),db[arm]['groups'])
    draws=c.load_npz(c.ROOT/'BOOTSTRAP_DRAWS.npz');tr=z['track_key'];cov=z['good'].any(2);m=z['good'][0].sum(1)
    dj=z['J'][0]-z['J'][3]
    hm=ut['H_main_executed_gap'].mean(1).reshape(128,20).mean(1);hs=ut['hash_main_executed_gap'].mean(1).reshape(128,20).mean(1)
    source_values=[cov[3].astype(float)-cov[0],(cov[0]&~cov[3]).astype(float),z['J'][3].min(1)-z['J'][0].min(1),
        np.where(z['good'][0],dj,0).sum(1),dj.mean(1),hm-hs]
    masks=[np.ones(128,bool),cov[0],np.ones(128,bool),m==1,m==0,np.isfinite(hm-hs)]
    rng=np.random.default_rng(20261030);equal(rng.integers(0,97,(5000,97)),draws['track_draws'])
    for b,dr in enumerate(draws['track_draws']):
        if b%128==0:g.check()
        ix=np.concatenate([np.flatnonzero(tr==draws['tracks'][j]) for j in dr])
        for j,(v,mask) in enumerate(zip(source_values,masks)):
            pick=ix[mask[ix]];value=float(np.mean(v[pick])) if len(pick) else np.nan
            equal(value,draws['statistics'][j,b])
    # MSE reconciliation and fixedstage algebra, support strata are Native-frozen.
    equal((z['J'][1]-z['J'][4])+(z['J'][2]-z['J'][1])+(z['J'][3]-z['J'][2]),z['J'][3]-z['J'][4])
    old=c.json_read(c.PARENT/'tiny_v2/METRICS.json')
    active=np.any(z['target']!=0,axis=-1)
    equal(z['initial_error'][active].mean(),old['initial']['active_candidate_MSE'])
    equal(z['final_error'][active].mean(),old['final']['active_candidate_MSE'])
    g.check(force=True);c.verify_files(source,g.check)
    out=c.ROOT/'completion_review';out.mkdir(exist_ok=False)
    c.write_json(out/'RESOURCE.json',dict(policy='warning_first_remaining_v3',estimate=g.estimate,observations=g.observations))
    c.write_json(out/'REVIEW.json',dict(status='PASS_INDEPENDENT_SAVED_ATTRIBUTION',
        window_ports=640,candidate_ports=12800,counterfactual_sets=5120,neighbor_queries=2560,
        all_neighbor_scopes_and_hash_ties=True,all5000draws6statistics=True,all_samecandidate_quality=True,
        all_complete_link_pdist=True,all_counterfactual_physics=True,exact_duplicate_bound=True,
        arithmetic_atol=2e-12,scientific_epsilon=0,source_initial_final_unchanged=True,
        no_Torch_or_checkpoint_deserialization=True,same_assistant_second_implementation_not_external=True,
        elapsed_seconds=time.perf_counter()-t))
    print('PASS independent full attribution review',flush=True)

if __name__=='__main__':main()
