"""Independent read-only identity/exposure audit. No model or training imports.

SOURCE indices are the existing metadata contract, not claimed video frames.
Unknown intervals stay unknown. Coordinate hashes are screens, not identity proof.
"""
import os,json,csv,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('F16_ROOT',Path(__file__).resolve().parents[1]))
OUT=Path(__file__).resolve().parent
INPUTS={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def register(p):
    p=Path(p);key=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else 'external_raw_focal_dataset'
    INPUTS[key]=(p,sha(p));return p

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v

def intervals(indices,track,start,unknown):
    result={}
    for i in indices:
        if not unknown[i] and start[i]>=0:result.setdefault(track[i],[]).append((int(start[i]),int(start[i])+19,int(i)))
    return result

def exposure(train,ev,ids,track,start,unknown):
    by=intervals(train,track,start,unknown);train_tracks=set(track[train]);same=[];full=[];future=[];contained=[];missing=[]
    for i in ev:
        if track[i] in train_tracks:same.append(int(ids[i]))
        if unknown[i] or start[i]<0:missing.append(int(ids[i]));continue
        candidates=by.get(track[i],[]);t=int(start[i]);end=t+19
        if any(a<=end and b>=t for a,b,_ in candidates):full.append(int(ids[i]))
        if any(a<=end and b>=t+8 for a,b,_ in candidates):future.append(int(ids[i]))
        if any(a<=t+8 and b>=end for a,b,_ in candidates):contained.append(int(ids[i]))
    return dict(n_eval=len(ev),same_track_windows=len(same),same_track_IDs=same,full_interval_overlap=len(full),full_interval_IDs=full,future_intersection=len(future),future_intersection_IDs=future,entire_future12_contained=len(contained),entire_future12_contained_IDs=contained,unknown_eval_intervals=missing,unknown_training_intervals=int(np.count_nonzero(unknown[train])))

def main():
    cache=register(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz')
    meta=register(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz')
    split=register(ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz')
    with np.load(cache,allow_pickle=False) as z:ids=z['sample_id'];groups=z['change_group']
    with np.load(meta,allow_pickle=False) as z:
        allmd={k:z[k].copy() for k in z.files};lookup={int(s):i for i,s in enumerate(z['sample_id'])};loc=np.array([lookup[int(s)] for s in ids]);md={k:z[k][loc].copy() for k in z.files}
    if len(ids)!=28791 or len(set(ids))!=28791:raise ValueError('Registry identity failed')
    track=np.array([str(s)+'::'+str(int(p)) for s,p in zip(md['sequence'],md['pedestrian_id'])]);start=md['start_frame'];unknown=md['ambiguous_track_time']|(start<0)
    with np.load(split,allow_pickle=False) as z:
        if not np.array_equal(z['sample_id'],ids[:25912]):raise ValueError('Split registry mismatch')
        splits={name:z[key].copy() for name,key in [('FIT','grouped_train_index'),('inner','inner_cal_index'),('outer','outer_eval_index')]}
    splits['original_CAL']=np.arange(25912,28791);report={'scope':'S1_CPU_READ_ONLY_NOT_AUTHOR_CLOSED','C1':{},'C2':{},'C3':{},'C4':{},'C5':{},'C6':{}};rows=[]
    for a,ai in splits.items():
        for b,bi in splits.items():
            if list(splits).index(b)<=list(splits).index(a):continue
            key=a+'__'+b;common=np.intersect1d(ids[ai],ids[bi]);report['C1'][key]=dict(count=len(common),sample_IDs=common)
            shared=np.intersect1d(track[ai],track[bi]);pair={}
            for g in [None,2]:
                aa=ai if g is None else ai[groups[ai]==g];bb=bi if g is None else bi[groups[bi]==g];st=np.intersect1d(track[aa],track[bb])
                pair['overall' if g is None else 'change']=dict(shared_tracks=len(st),track_keys=st,windows_a=int(np.isin(track[aa],st).sum()),windows_b=int(np.isin(track[bb],st).sum()),sample_IDs_a=ids[aa[np.isin(track[aa],st)]],sample_IDs_b=ids[bb[np.isin(track[bb],st)]])
            report['C2'][key]=pair
            for t in shared:
                for name,ix in [(a,ai),(b,bi)]:
                    for i in ix[track[ix]==t]:rows.append(dict(pair=key,track=t,split=name,sample_id=int(ids[i]),start=None if unknown[i] else int(start[i]),end=None if unknown[i] else int(start[i])+19,change=bool(groups[i]==2),unknown=bool(unknown[i])))
    fit=splits['FIT']
    for name in ['inner','outer','original_CAL']:
        report['C3'][name]={g:exposure(fit,ix,ids,track,start,unknown) for g,ix in [('overall',splits[name]),('change',splits[name][groups[splits[name]]==2])]}
    oi=splits['outer'][groups[splits['outer']]==2]
    report['C2']['outer_change_in_FIT']=dict(n_windows=len(oi),n_tracks=len(np.unique(track[oi])),seen_tracks=np.intersect1d(track[oi],track[fit]),seen_windows=int(np.isin(track[oi],track[fit]).sum()))
    # Identity-reuse screens: time gap flags alone do NOT prove reused IDs.
    gaps=[]
    for t in np.unique(track):
        ix=np.flatnonzero((track==t)&~unknown);times=np.unique(start[ix]);large=np.diff(times)
        if np.any(large>20):gaps.append(dict(track=t,source_index_gaps=large[large>20].tolist(),sample_IDs=ids[ix]))
    report['C4'].update(unknown_interval_IDs=ids[unknown],gaps_over20_screen=gaps,identity_reuse_proven=False,cal_sequence_names=sorted(set(md['sequence'][splits['original_CAL']])),noncal_sequence_names=sorted(set(md['sequence'][np.concatenate([fit,splits['outer'],splits['inner']])])),coordinate_hash_resolution_m=.001,raw_available=False)
    astra=os.environ.get('F16_ASTRA_ROOT')
    if astra:
        rawp=Path(astra)/'datasets/eth_ucy/processed_data_diverse/eth_data_train.npy'
        if rawp.exists():
            raw=np.load(register(rawp),mmap_mode='r');world=np.asarray(raw[ids,0,:,-2:],np.float32);hashes=[hashlib.sha256(np.rint(w.astype(float)*1000).astype('<i8').tobytes()).hexdigest() for w in world]
            pairs={};examples=[]
            for a,ai in splits.items():
                for b,bi in splits.items():
                    if list(splits).index(b)<=list(splits).index(a):continue
                    hm={}
                    for i in ai:hm.setdefault(hashes[i],[]).append(int(i))
                    matches=[]
                    for j in bi:
                        for i in hm.get(hashes[j],[]):matches.append([int(ids[i]),int(ids[j]),str(track[i]),str(track[j])])
                    pairs[a+'__'+b]=dict(n_pairs=len(matches),pairs=matches)
            # Inconsistent world position for same track and source index.
            positions={};conflicts=[]
            for i in np.flatnonzero(~unknown):
                for offset in range(20):
                    key=(track[i],int(start[i])+offset);value=world[i,offset].astype(float);old=positions.get(key)
                    if old is None:positions[key]=(value,int(ids[i]))
                    elif np.linalg.norm(old[0]-value)>2e-5:conflicts.append([str(key[0]),key[1],old[1],int(ids[i]),float(np.linalg.norm(old[0]-value))])
            report['C4'].update(raw_available=True,rounded_world_duplicates=pairs,same_track_time_position_conflicts=conflicts,n_conflicts=len(conflicts))
    for name,ix in splits.items():
        report['C5'][name]={str(s):dict(n=int(np.sum(md['scene_family'][ix]==s)),change=int(np.sum((md['scene_family'][ix]==s)&(groups[ix]==2)))) for s in np.unique(md['scene_family'][ix])}
    report['C5']['outer_seen_scene_families']=sorted(set(md['scene_family'][splits['outer']])&set(md['scene_family'][fit]))
    # Native producer train indices use full raw dataset indices. Rejoin full metadata.
    allids=allmd['sample_id'];alltrack=np.array([str(s)+'::'+str(int(p)) for s,p in zip(allmd['sequence'],allmd['pedestrian_id'])]);allstart=allmd['start_frame'];allu=allmd['ambiguous_track_time']|(allstart<0);rawlookup={int(s):i for i,s in enumerate(allids)}
    for stage in ['h3','v2']:
        p=ROOT/f'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/results/eth/{stage}/f16_temporal_social_cross/seed_1/val_scene_-1/split_indices.npz'
        if not p.exists():report['C6']['native_'+stage]={'status':'UNKNOWN_TRAIN_ID_ARTIFACT_MISSING'};continue
        with np.load(register(p),allow_pickle=False) as z:
            keys=z.files;key=next((k for k in ['train','train_indices','train_idx'] if k in keys),None)
            if key is None:report['C6']['native_'+stage]={'status':'UNKNOWN_SCHEMA','keys':keys};continue
            trainids=z[key].astype(int);ti=np.array([rawlookup[int(s)] for s in trainids])
        report['C6']['native_'+stage]={'status':'TRAIN_SPLIT_ARTIFACT_READ_NOT_RUNTIME_PROVENANCE','training_windows':len(ti),'split_key':key,'split_source':'actual fold decoder/encoder split_indices.npz'}
        for name in ['outer','original_CAL']:
            ei=np.array([rawlookup[int(s)] for s in ids[splits[name]]]);report['C6']['native_'+stage][name]=exposure(ti,ei,allids,alltrack,allstart,allu)
    # Actual HGB action row sample IDs; do not load model pickle.
    idlookup={int(s):i for i,s in enumerate(ids)}
    for seed in [42,43,44]:
        p=ROOT/f'CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k1_exec/seed{seed}/train_data/rows_seed{seed}.npz'
        if not p.exists():report['C6'][f'HGB_seed{seed}']={'status':'UNKNOWN_ROWS_MISSING'};continue
        with np.load(register(p),allow_pickle=False) as z:trainids=np.unique(z['sample_id'])
        ti=np.array([idlookup[int(s)] for s in trainids]);report['C6'][f'HGB_seed{seed}']=dict(status='ACTUAL_SAVED_ACTION_ROW_IDS',n_train=len(ti),**{name:exposure(ti,splits[name],ids,track,start,unknown) for name in ['outer','original_CAL']})
    kinp=register(ROOT/'KINEMATIC_DECODER_SPEED_CLAMP_20261018/kdec/train.py');kindata=register(ROOT/'KINEMATIC_DECODER_SPEED_CLAMP_20261018/kdec/data.py')
    report['C6']['KIN']=dict(status='SOURCE_RECIPE_FIT_NOT_PER_CHECKPOINT_TRAIN_ID_PROVEN',recipe_files=[kinp.relative_to(ROOT).as_posix(),kindata.relative_to(ROOT).as_posix()],n_recipe_train=len(fit),**{name:exposure(fit,splits[name],ids,track,start,unknown) for name in ['outer','original_CAL']})
    qp=ROOT/'CANDIDATE_ACTION_IDENTIFIABILITY_EXEC_20261019/COHORT_IDS.npz'
    if qp.exists():
        with np.load(register(qp),allow_pickle=False) as z:
            cohort={}
            for k in z.files:
                if not k.endswith('_ids'):continue
                ti=np.array([idlookup[int(s)] for s in z[k]])
                cohort[k]=dict(n=len(ti),evaluation_cohort_not_new_training=True,**{name:exposure(ti,splits[name],ids,track,start,unknown) for name in ['outer','original_CAL']})
        report['C6']['Q0_cohorts']=cohort
    report['C6']['limitations']=['Source recipe is not per-checkpoint training provenance.','Cached encoder/native reuse includes upstream training exposure; not unseen whole-model validation.','No claim all C6 inputs known; HGB model pickles not executed.']
    for row in rows:
        i=idlookup[row['sample_id']];row['interval_intersects_FIT']=None if row['split']=='FIT' else row['sample_id'] in report['C3'].get(row['split'],{}).get('overall',{}).get('full_interval_IDs',[])
        a,b=row['pair'].split('__');other=b if row['split']==a else a;eligible=splits[other][(track[splits[other]]==track[i])&~unknown[splits[other]]]
        row['interval_intersects_other_split']=None if unknown[i] else any(start[j]<=start[i]+19 and start[j]+19>=start[i] for j in eligible)
    with (OUT/'shared_tracks.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['track']);w.writeheader();w.writerows(rows)
    before={k:v[1] for k,v in INPUTS.items()};after={k:sha(p) for k,(p,_) in INPUTS.items()}
    if before!=after:raise RuntimeError('Input changed during overlap audit')
    report.update(input_sha256=before,inputs_unchanged=before==after,script_sha256=sha(__file__),author_review='OPEN',formal_freeze_allowed=False)
    (OUT/'OVERLAP_REPORT.json').write_text(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'C3':{k:{g:v['full_interval_overlap'] for g,v in groups.items()} for k,groups in report['C3'].items()},'outer_change':clean(report['C2']['outer_change_in_FIT']),'inputs_unchanged':True}))

if __name__=='__main__':main()
