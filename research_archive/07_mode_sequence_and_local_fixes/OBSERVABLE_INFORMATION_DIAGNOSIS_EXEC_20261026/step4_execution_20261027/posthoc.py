"""Labels are admitted ONLY after all3 GT-free geometry/selection locks."""
from core import *
from matching import action_map
def reconstruct_quality(native,kin,gt):
    values=[]
    for p in (native,kin):
        d=np.sqrt(((p-gt)**2).sum(-1));a=d.mean(-1);f=d[:,-1];values.append(np.stack((a,f,np.maximum(a/.2,f/.4)),-1))
    n,k=values;out=np.empty((401,3));out[0]=n.min(0)
    for r in range(20):out[1+20*r:1+20*(r+1)]=np.minimum(np.concatenate((n[:r],n[r+1:])).min(0),k)
    return out
def positives(q,e):
    b=e&(q<=q[0]).all(-1)&(q<q[0]).any(-1)&((q[0,2]>1)|(q[:,2]<=1));b[0]=False;return b
def compare(qe,pe,qb,pb,qquality,pquality,mapping):
    domain=qe&pe[mapping];assert domain[0] and mapping[0]==0
    bq=qb&domain;bp=pb[mapping]&domain;both=bool(bq.any() and bp.any());overlap=bq&bp;union=bq|bp
    if both:event='CONFLICT' if not overlap.any() else 'COMMON_POSITIVE'
    elif not qb.any() and not pb.any():event='BOTH_ORIGINAL_NO_OPPORTUNITY'
    elif qb.any() and pb.any():event='COMMON_DOMAIN_OPPORTUNITY_REMOVAL'
    else:event='ONE_ORIGINAL_NO_OPPORTUNITY'
    jq=qquality[:,2];jp=pquality[mapping,2];bqj=float(jq[domain].min());bpj=float(jp[domain].min())
    regret=float(np.maximum(jq-bqj,jp-bpj)[domain].min())
    return dict(event=event,domain_count=int(domain.sum()),q_original_B=int(qb.sum()),p_original_B=int(pb.sum()),q_common_B=int(bq.sum()),p_common_B=int(bp.sum()),
        intersection=int(overlap.sum()),union=int(union.sum()),both_positive=both,conflict=bool(not overlap.any()) if both else None,
        overlap=float(overlap.sum()/union.sum()) if both else None,regret=regret,
        q_native_good=bool(qquality[0,2]<=1),p_native_good=bool(pquality[0,2]<=1),
        q_rescue_opportunity=bool(qquality[0,2]>1 and (qb&(qquality[:,2]<=1)).any()),
        p_rescue_opportunity=bool(pquality[0,2]>1 and (pb&(pquality[:,2]<=1)).any())),dict(domain=domain,q_positive=bq,p_positive=bp,qJ=jq,pJ=jp)
def independent_compare(qe,pe,qb,pb,qquality,pquality,nmap,kmap):
    # Literal401-action loop, no reuse of production action_map/common-domain formula.
    domain=[];qpos=[];ppos=[];qj=[];pj=[]
    for a in range(401):
        if a:r,l=divmod(a-1,20);b=1+20*int(nmap[r])+int(kmap[l])
        else:b=0
        if qe[a] and pe[b]:domain.append(a);qpos.append(bool(qb[a]));ppos.append(bool(pb[b]));qj.append(float(qquality[a,2]));pj.append(float(pquality[b,2]))
    assert domain[0]==0
    nq=sum(qpos);np_=sum(ppos);inter=sum(x and y for x,y in zip(qpos,ppos));union=sum(x or y for x,y in zip(qpos,ppos))
    bestq=min(qj);bestp=min(pj);regret=min(max(x-bestq,y-bestp) for x,y in zip(qj,pj))
    return dict(domain_count=len(domain),q_common_B=nq,p_common_B=np_,intersection=inter,union=union,
        both_positive=bool(nq and np_),conflict=bool(not inter) if nq and np_ else None,overlap=inter/union if nq and np_ else None,regret=regret)
def main(seed):
    lock=read_json(HERE/'ALL_SELECTIONS_LOCK.json');folder=HERE/f'selections/seed{seed}';assert sha256(folder/'done.json')==lock['done_SHA'][str(seed)]
    pack=load_npz(folder/'MATCHES.npz');selected=load_npz(folder/'SELECTED.npz');pools={s:load_npz(HERE/f'prepared/pools_seed{seed}_{s}.npz') for s in ('opt','hold')}
    labels={s:old_columns(seed,s,('sample_id','quality','B','gt','group')) for s in ('opt','hold')}
    for s in labels:assert np.array_equal(labels[s]['sample_id'],pools[s]['sample_id'])
    g=guard();records=[];arrays=[];checked=set();timing=time.perf_counter()
    for mode in ('primary','reverse'):
        for i in selected[mode]:
            i=int(i);qr=int(pack['query_row'][i]);pr=int(pack['library_row'][i])
            for split,row in (('hold',qr),('opt',pr)):
                key=(split,row)
                if key not in checked:
                    q=reconstruct_quality(pools[split]['native'][row],pools[split]['kin'][row],labels[split]['gt'][row]);saved=labels[split]['quality'][row]
                    assert np.max(abs(q-saved))<=2e-12
                    assert np.array_equal(positives(q,pools[split]['eligible'][row]),labels[split]['B'][row])
                    checked.add(key)
            prefix='' if mode=='primary' else 'reverse_';nm=pack[prefix+'native_map'][i];km=pack[prefix+'kin_map'][i]
            mapping=action_map(nm,km);args=(pools['hold']['eligible'][qr],pools['opt']['eligible'][pr],labels['hold']['B'][qr],labels['opt']['B'][pr],labels['hold']['quality'][qr],labels['opt']['quality'][pr])
            r,a=compare(*args,mapping);check=independent_compare(*args,nm,km)
            for k,v in check.items():
                if isinstance(v,float):assert abs(r[k]-v)<=2e-12,k
                else:assert r[k]==v,k
            delta=labels['hold']['gt'][qr]-labels['opt']['gt'][pr]
            r.update(seed=seed,mode=mode,pair_index=i,query_row=qr,library_row=pr,query_ID=int(pools['hold']['sample_id'][qr]),library_ID=int(pools['opt']['sample_id'][pr]),
                group=int(labels['hold']['group'][qr]),dH=float(pack['dH'][i]),dP=float(pack['dP' if mode=='primary' else 'reverse_dP'][i]),d=float(pack['d' if mode=='primary' else 'reverse_d'][i]),
                future_mean_distance=float(np.linalg.norm(delta,axis=-1).mean()),future_endpoint_distance=float(np.linalg.norm(delta[-1])),
                numerically_ambiguous=pack['numerically_ambiguous'][i].tolist(),numerical_uniqueness_gap=pack['alternative_gap'][i].tolist())
            records.append(r);arrays.append(a)
            if len(records)%128==0:g.check()
    # Alignment conclusions on the SAME primary-supported pairs, independent of
    # whether reverse sensitivity selected exactly the same five neighbors.
    sensitivity=[]
    for i in selected['primary']:
        i=int(i);qr=int(pack['query_row'][i]);pr=int(pack['library_row'][i]);args=(pools['hold']['eligible'][qr],pools['opt']['eligible'][pr],labels['hold']['B'][qr],labels['opt']['B'][pr],labels['hold']['quality'][qr],labels['opt']['quality'][pr])
        a,_=compare(*args,action_map(pack['native_map'][i],pack['kin_map'][i]));b,_=compare(*args,action_map(pack['reverse_native_map'][i],pack['reverse_kin_map'][i]))
        changed=a['event']!=b['event'] or (a['overlap'] is None)!=(b['overlap'] is None) or (a['overlap'] is not None and abs(a['overlap']-b['overlap'])>2e-12) or abs(a['regret']-b['regret'])>2e-12
        sensitivity.append(dict(pair_index=i,status='ALIGNMENT_SENSITIVE' if changed else 'UNCHANGED_UNDER_REVERSE',primary=a,reverse=b))
    out=HERE/f'posthoc/seed{seed}';out.mkdir(parents=True)
    write_json(out/'PAIRS.json',records);write_json(out/'ALIGNMENT_SENSITIVITY.json',sensitivity)
    if arrays:save_npz(out/'PAIR_SCALARS.npz',**{k:np.stack([r[k] for r in arrays]) for k in arrays[0]},mode=np.asarray([r['mode'] for r in records]),pair_index=np.asarray([r['pair_index'] for r in records],np.int32))
    else:save_npz(out/'PAIR_SCALARS.npz',domain=np.empty((0,401),bool),q_positive=np.empty((0,401),bool),p_positive=np.empty((0,401),bool),qJ=np.empty((0,401)),pJ=np.empty((0,401)),mode=np.empty(0,'U7'),pair_index=np.empty(0,np.int32))
    write_json(out/'REVIEW.json',dict(status='PASS',posthoc_after_all3_GTfree_lock=True,pairs=len(records),distinct_quality_windows_checked=len(checked),all401_quality_and_B_reconstructed=True,independent_401_loop_all_pairs=True,seconds=time.perf_counter()-timing))
    resource(out/'RESOURCE.json',g);finish(out,dict(source=source(),selection_lock_SHA=sha256(HERE/'ALL_SELECTIONS_LOCK.json'),seed=seed))
    print('PASS posthoc independent',seed,len(records),len(checked),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
