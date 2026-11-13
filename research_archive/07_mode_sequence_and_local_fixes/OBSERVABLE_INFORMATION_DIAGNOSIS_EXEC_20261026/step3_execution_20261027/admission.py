"""Observed-only token and raw-neighbor provenance; no future/quality selection.

Historical metadata supplies sequence/target identity only. Full20-derived start
is NOT used to disambiguate observed8. Ambiguous observed histories are retained
in the registry and block neighbor arms, not removed from the population.
"""
from core import *
LABELS=Path('<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/labels/eth')
RAW=Path('<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse/eth_data_train.npy')
METADATA=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
def fingerprint(x):return np.round(np.asarray(x,np.float32),4).tobytes()
def neighbor_features(table,frames,target_id):
    # table cols: integer source-index, decimal pedestrian ID, world x/y.
    t0=int(frames[-1]); by={}
    # All accesses before selection explicitly restricted to observed times.
    past=table[np.isin(table[:,0],frames)]
    for f,p,x,y in past:by[(int(f),int(p))]=np.array([x,y],np.float64)
    target=np.stack([by[(int(f),int(target_id))] for f in frames])
    eligible=sorted({int(p) for f,p,_,_ in past if int(f)==t0 and int(p)!=target_id
                     and (int(frames[-2]),int(p)) in by})
    eligible.sort(key=lambda p:(float(np.sum((by[(t0,p)]-target[-1])**2)),p))
    selected=eligible[:8]; out=np.zeros((8,8,3),np.float64)
    heading=target[-1]-target[4];norm=np.linalg.norm(heading)
    u=heading/norm if norm>.03 else np.array([1.,0.])
    for j,p in enumerate(selected):
        for k,f in enumerate(frames):
            if (int(f),p) in by:
                r=by[(int(f),p)]-target[k]
                out[j,k,:2]=(r[0]*u[0]+r[1]*u[1],-r[0]*u[1]+r[1]*u[0]);out[j,k,2]=1
    # Target local past parity proves orientation/units against fixed cache.
    rel=target-target[0]
    local=np.stack((rel[:,0]*u[0]+rel[:,1]*u[1],-rel[:,0]*u[1]+rel[:,1]*u[0]),-1)
    return out,np.asarray(selected,np.int64),local
def main():
    require12();g=guard('CPU');out=HERE/'prepared';out.mkdir(exist_ok=True)
    if (HERE/'FEATURE_PROVENANCE.json').exists():raise FileExistsError('Preserve completed admission')
    with np.load(METADATA) as z:meta={k:z[k] for k in z.files}
    lookup={int(v):i for i,v in enumerate(meta['sample_id'])}
    observations={s:obs(s) for s in ('opt','hold','inner')}
    allids=np.concatenate([o['sample_id'] for o in observations.values()]);assert len(np.unique(allids))==len(allids)
    neededseq=sorted({str(meta['sequence'][lookup[int(v)]]) for v in allids})
    raw=np.load(RAW,mmap_mode='r');tables={};indices={};files=[METADATA,RAW,Path(__file__),HERE/'core.py']
    neededkeys={(str(meta['sequence'][lookup[int(v)]]),int(meta['pedestrian_id'][lookup[int(v)]])) for v in allids}
    for seq in neededseq:
        p=LABELS/f'{seq}_train.txt';files.append(p);t=np.loadtxt(p,dtype=str)
        table=t[:,[0,1,13,15]].astype(np.float64);tables[seq]=table
        for qseq,ped in sorted(neededkeys):
            if qseq!=seq:continue
            rows=table[table[:,1]==ped];rows=rows[np.argsort(rows[:,0],kind='stable')]
            for j in range(len(rows)-7):
                r=rows[j:j+8]
                if np.all(np.diff(r[:,0])==1):indices.setdefault((seq,ped,fingerprint(r[:,2:])),[]).append(r[:,0].astype(np.int64))
        g.check()
    tokenchecks=[];registry=[];features={};validcounts={};future_tests=0;maxpast=0.
    for split,o in observations.items():
        token=o['tokens'];assert token.dtype==np.float32 and token.shape[1:]==(8,96)
        assert np.array_equal(token[:,-1],o['bundle'][:,42:138])
        assert np.array_equal(token.mean(1),o['bundle'][:,138:234])
        if split=='opt':
            for seed in SEEDS:
                for lo in range(0,len(token),32):
                    x=old_features(seed,np.arange(lo,min(lo+32,len(token))))
                    assert np.array_equal(x[:,0,42:138],token[lo:lo+len(x),-1])
                    assert np.array_equal(x[:,0,138:234],token[lo:lo+len(x)].mean(1))
        tokenchecks.append(dict(split=split,n=len(token),last_exact=True,original_float32_mean_exact=True,token_SHA=arrsha(token)))
        f=np.zeros((len(token),8,8,3),np.float32);valid=0
        for i,sid in enumerate(o['sample_id']):
            m=lookup[int(sid)];seq=str(meta['sequence'][m]);ped=int(meta['pedestrian_id'][m])
            world=np.array(raw[int(sid),0,:8,-2:],np.float32,copy=True)
            candidates=indices.get((seq,ped,fingerprint(world)),[])
            r=dict(split=split,sample_id=int(sid),track_key=str(o['track_key'][i]),observed8_source_matches=len(candidates))
            if len(candidates)!=1:r.update(status='BLOCKED_AMBIGUOUS_OBSERVED_TIME' if candidates else 'BLOCKED_NO_OBSERVED_SOURCE',possible_source_starts=[int(x[0]) for x in candidates])
            else:
                frames=candidates[0];v,ids,local=neighbor_features(tables[seq],frames,ped)
                error=float(np.max(np.abs((local-local[-1])-(o['past'][i].astype(float)-o['past'][i,-1]))));maxpast=max(maxpast,error)
                if error>2e-5:r.update(status='BLOCKED_LOCAL_PARITY',maxerror=error)
                else:
                    if future_tests<128:
                        original=tables[seq];short=original[original[:,0]<=frames[-1]].copy()
                        changed=original.copy();changed[changed[:,0]>frames[-1],2:]=1234567.
                        for altered in (short,changed):
                            vv,ii,ll=neighbor_features(altered,frames,ped)
                            assert np.array_equal(v,vv) and np.array_equal(ids,ii) and np.array_equal(local,ll)
                        future_tests+=1
                    f[i]=v;valid+=1;r.update(status='PASS_OBSERVED_ONLY',source_start=int(frames[0]),neighbor_IDs=ids.tolist())
            registry.append(r)
        features[split]=f;validcounts[split]=dict(n=len(token),admitted=valid,blocked=len(token)-valid)
        save_npz(out/f'observed_blocks_{split}.npz',sample_id=o['sample_id'],token=token,neighbor=f)
        g.check()
    raw._mmap.close();neighborok=all(v['blocked']==0 for v in validcounts.values())
    files += [PREVIOUS/f'prepared/observations_{s}.npz' for s in observations]
    sources={str(p.resolve()):sha256(p) for p in files}
    write_json(HERE/'TOKEN_REDUNDANCY_CHECK.json',dict(status='PASS',records=tokenchecks,fixed_cache_not_new_information=True))
    write_json(HERE/'PAST_ONLY_AUDIT.json',dict(status='PASS' if neighborok else 'BLOCKED_INPUT_PROVENANCE',population=validcounts,registry=registry,
        memory_only_future_mutation_and_truncation_tests=future_tests,max_local_past_error=maxpast,no_disk_deletion=True,
        no_future_survival_sidecar=True,no_GT_or_full20_time_disambiguation=True))
    write_json(HERE/'FEATURE_PROVENANCE.json',dict(status='PASS_BASE_TOKEN_AND_NEIGHBOR' if neighborok else 'PASS_BASE_TOKEN_NEIGHBOR_BLOCKED',
        admissible_arms=list(ARMS) if neighborok else ['C_base','C_token'],blocked_arms=[] if neighborok else ['C_neighbor','C_both'],
        source_files=sources,population=validcounts,observed8_matching_only=True,old_fixed_cache_future_curation_risk_retained=True))
    resource_report(HERE/'ADMISSION_RESOURCE.json',g)
    print(json.dumps(dict(status='ADMITTED',validcounts=validcounts,neighborok=neighborok,maxpast=maxpast),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
