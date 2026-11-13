from base import *
def sample_actions(eligible,target,ids,seed):
    actions=np.full((len(ids),32),-1,np.int16);weights=np.zeros_like(actions,np.float64);labels=np.zeros_like(actions,bool);pi=np.zeros_like(weights)
    for i,sid in enumerate(ids):
        pos=np.flatnonzero(eligible[i]&target[i]);neg=np.flatnonzero(eligible[i]&~target[i]);assert len(pos)
        if not len(neg):continue
        c=0
        for truth,pool in ((True,pos),(False,neg)):
            other=[int(a) for a in pool if a!=0];other.sort(key=lambda a:(hashlib.sha256(f'info-probe-v1|{seed}|{int(sid)}|{a}'.encode()).digest(),a))
            hasstop=0 in pool;chosen=([0] if hasstop else [])+other[:(15 if hasstop else 16)]
            n=len(pool);start=c
            for a in chosen:
                prob=1. if a==0 or n<=16 else (15/(n-1) if hasstop else 16/n)
                actions[i,c]=a;weights[i,c]=1/(2*n*prob);labels[i,c]=truth;pi[i,c]=prob;c+=1
            assert abs(weights[i,start:c].sum()-.5)<1e-12
        assert c<=32
    return dict(actions=actions,weights=weights,labels=labels,inclusion_probability=pi,sample_id=ids)
def observations(split):
    # Only observations/identity, never hold/inner future or action labels here.
    cache=ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
    d=read_json(OLD/f'populations/seed42/{split}/done.json');f=OLD/f'populations/seed42/{split}';ids=[];tracks=[]
    for n in d['chunks']:
        with np.load(f/n) as z:ids.append(z['sample_id']);tracks.append(z['track_key'])
    ids=np.concatenate(ids);tracks=np.concatenate(tracks)
    with np.load(cache) as z:
        lookup={int(v):i for i,v in enumerate(z['sample_id'])};rows=[lookup[int(v)] for v in ids];past=z['past'][rows];tokens=z['tokens'][rows]
    h=np.concatenate((past.reshape(-1,16).astype(np.float64),np.diff(past.astype(np.float64),axis=1).reshape(-1,14)/.4,np.diff(np.diff(past.astype(np.float64),axis=1)/.4,axis=1).reshape(-1,12)/.4),1)
    bundle=np.concatenate((h,tokens[:,-1].astype(np.float64),tokens.mean(1).astype(np.float64)),1).astype(np.float32)
    return dict(sample_id=ids,track_key=tracks,past=past,tokens=tokens,bundle=bundle)
def permutation(obs,seed,split,cuts):
    p=obs['past'].astype(np.float64);speed=np.linalg.norm((p[:,-1]-p[:,-2])/.4,axis=-1);bins=np.searchsorted(cuts,speed,side='right');seq=np.array([s.rsplit('|',1)[0] for s in obs['track_key']]);donor=np.full(len(speed),-1,np.int32);layers=[]
    for q in sorted(set(seq)):
        for b in range(3):
            ix=np.flatnonzero((seq==q)&(bins==b));ix=np.asarray(sorted(ix,key=lambda i:(hashlib.sha256(f'info-observation-permute-v1|{split}|{seed}|{int(obs["sample_id"][i])}'.encode()).digest(),int(obs['sample_id'][i]))),np.int32)
            if not len(ix):layers.append(dict(sequence=q,bin=b,n=0,shift=None));continue
            shift=next((s for s in range(1,len(ix)) if np.all(obs['track_key'][ix]!=obs['track_key'][np.roll(ix,-s)])),None)
            layers.append(dict(sequence=q,bin=b,n=len(ix),tracks=len(set(obs['track_key'][ix])),shift=shift))
            if shift is not None:donor[ix]=np.roll(ix,-shift)
    return donor,dict(status='PASS' if np.all(donor>=0) else 'BLOCKED_NO_LEGAL_LAYER_SHIFT',seed=seed,split=split,layers=layers,speed_cuts=cuts.tolist(),future_or_labels_used=False)
def main():
    g=guard('source_prepare');out=HERE/'prepared';out.mkdir(exist_ok=True)
    if (out/'done.json').exists():return completed(out,dict(source_inputs=read_json(HERE/'SOURCE_AND_ID_AUDIT.json')['source']))
    if list(out.glob('*.npz')):raise FileExistsError('Preserve incomplete prepared; explicit recovery needed')
    lab=opt_labels(42);tinyix=np.array(sorted(range(4096),key=lambda i:(hashlib.sha256(f'info-tiny-v1|{int(lab["sample_id"][i])}'.encode()).digest(),int(lab['sample_id'][i])))[:128]);x=old_features(42,tinyix)
    chunks=list(aux_chunks(42,'opt'));full={k:np.concatenate([z[k] for z in chunks]) for k in ('sample_id','track_key','group','native','kin','gt','quality','eligible','B','target')};del chunks
    assert np.array_equal(lab['sample_id'],full['sample_id'])
    save_npz(out/'tiny.npz',features=x,opt_row=tinyix,**{k:v[tinyix] for k,v in full.items()})
    write_json(HERE/'TINY_IDS.json',dict(pool_seed=42,init_seeds=list(SEEDS),sample_id=lab['sample_id'][tinyix].tolist(),opt_row=tinyix.tolist(),hash='info-tiny-v1',GT_selected=False,stats='original full4096 pool42'))
    groups={};row_refs={};row_conflicts=0
    tx=(x[...,:323]-stats(42)[0])/stats(42)[1];tx=np.concatenate((tx,x[...,323:]),-1).astype(np.float32)
    for i in range(128):
        for a in np.flatnonzero(lab['eligible'][tinyix[i]]):
            key=hashlib.sha256(tx[i,a].tobytes()).hexdigest();groups.setdefault(key,[]).append((i,int(a),bool(lab['target'][tinyix[i],a])))
    contradictions=[v for v in groups.values() if len({a[2] for a in v})>1];impossible=[]
    for i in range(128):
        first={}
        for a in np.flatnonzero(lab['eligible'][tinyix[i]]):first.setdefault(hashlib.sha256(tx[i,a].tobytes()).hexdigest(),int(a))
        if not any(lab['target'][tinyix[i],a] for a in first.values()):impossible.append(int(full['sample_id'][tinyix[i]]))
    write_json(HERE/'INPUT_COLLISIONS.json',dict(status='COMPLETE_EXACT_FLOAT32_BYTE_GROUP_AUDIT',eligible_rows=sum(map(len,groups.values())),unique_input_classes=len(groups),contradictory_BCE_classes=len(contradictions),contradictory_row_references=contradictions,within_window_first_tie_top1_impossible_IDs=impossible,top1_scope='only within-window identical-input first-tie feasibility; no claim of global simultaneous function satisfiability'))
    del x,tx,groups,full;g.check()
    obs={s:observations(s) for s in ('opt','hold','inner')}
    for s,d in obs.items():save_npz(out/f'observations_{s}.npz',**d)
    p=obs['opt']['past'].astype(np.float64);cuts=np.quantile(np.linalg.norm((p[:,-1]-p[:,-2])/.4,axis=-1),[1/3,2/3]);perms=[]
    for seed in SEEDS:
        label=opt_labels(seed);assert np.array_equal(label['sample_id'],obs['opt']['sample_id']);table=sample_actions(label['eligible'],label['target'],label['sample_id'],seed);save_npz(out/f'actions_seed{seed}.npz',**table)
        for s,d in obs.items():
            donor,report=permutation(d,seed,s,cuts);save_npz(out/f'permutation_seed{seed}_{s}.npz',donor=donor,sample_id=d['sample_id'],track_key=d['track_key']);perms.append(report)
        g.check()
    write_json(out/'PERMUTATION_MAP.json',dict(records=perms,blocked_seeds=[s for s in SEEDS if any(r['seed']==s and r['status']!='PASS' for r in perms)]))
    resource_report(out,g);finish(out,dict(source_inputs=read_json(HERE/'SOURCE_AND_ID_AUDIT.json')['source']),no_held_future_access=True)
    print('PREPARED',len(tinyix),'tiny; tables3; permutation statuses',[(r['seed'],r['split'],r['status']) for r in perms],flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:fail(e);raise
