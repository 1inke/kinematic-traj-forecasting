"""H0: frozen inputs; separate target-only long labels; never regenerate neighbors."""
import csv, itertools, traceback
from collections import Counter,defaultdict
import common as c
import numpy as np

def local(world, reference):
    # Exactly the old float32 world->local convention, including slow-motion axis.
    ref=np.asarray(reference,dtype=np.float32);world=np.asarray(world,dtype=np.float32)
    direction=ref[7]-ref[4];norm=np.linalg.norm(direction);unit=direction/max(float(norm),1e-8)
    if norm<=.03:unit=np.array([1.,0.],np.float32)
    rel=world-ref[7]
    return np.stack((rel[...,0]*unit[0]+rel[...,1]*unit[1],-rel[...,0]*unit[1]+rel[...,1]*unit[0]),-1)

def label_tables(sequences):
    tables={};sources={};schemas={};duplicate_rows=0
    root=c.RAW_ROOT/'datasets/eth_ucy/labels'
    for seq in sequences:
        # Already existing full files OR original eth train/val shards only;
        # full copies for other cross-validation folds are labels, not new data.
        full=sorted(root.glob('*/'+seq+'.txt'))
        paths=full if full else [root/'eth'/f'{seq}_train.txt',root/'eth'/f'{seq}_val.txt']
        table={};conflicts=set();per_file=[]
        for p in paths:
            assert p.is_file(),('MISSING_LABEL_SOURCE',str(p))
            text=np.loadtxt(p,dtype=str);assert text.ndim==2 and text.shape[1]>=16
            a=text[:,[0,1,13,15]].astype(float);assert np.isfinite(a).all()
            assert np.array_equal(a[:,:2],np.rint(a[:,:2])),('NONINTEGER_INDEX_ID',str(p))
            sources[str(p)]=c.sha(p)
            for f,ped,x,y in a:
                key=(int(f),int(ped));xy=np.array([x,y],float)
                if key in table:
                    duplicate_rows+=1
                    if not np.array_equal(table[key],xy):conflicts.add(key)
                else:table[key]=xy
            per_file.append(dict(path=str(p),rows=len(a),columns=text.shape[1],frame_column=0,pedestrian_column=1,world_xy_columns=[13,15]))
        tables[seq]=(table,conflicts)
        schemas[seq]=dict(files=per_file,merged_keys=len(table),conflicting_keys=[list(x) for x in sorted(conflicts)],unit='old world coordinates metres; old cache reproduction required',source_index_step=1,dt_seconds=.4,not_asserted_video_frame=True)
    return tables,sources,schemas,duplicate_rows

def overlap(start,end,lo,hi):return start<=hi and lo<=end

def main():
    begin=c.time.perf_counter();c.guard('startup');c.stage('H0_LOAD_FROZEN_INPUTS');d=c.load();md=d['meta'];ids=d['ids'];n=len(ids)
    sources={str(p):c.sha(p) for p in c.INPUTS.values()}
    sources.update({str(p):c.sha(p) for p in c.S.glob('*.md')})
    sources.update({str(p):c.sha(p) for p in c.H.glob('*.py')})
    tables,label_sources,schemas,duplicate_rows=label_tables(np.unique(md['sequence']));sources.update(label_sources)
    raw=np.load(c.INPUTS['raw'],mmap_mode='r');assert raw.shape[0]>int(ids.max()) and raw.shape[2:]==(20,6)
    owner=np.full(n,'unused_legacy_FIT',dtype='<U20')
    for name,ix in d['split'].items():owner[ix]=name
    assert len(np.unique(np.concatenate(list(d['split'].values()))))==24679
    # ALL 28791 old IDs are retained, including unused FIT registry entries.
    m20=np.zeros(n,bool);presence=np.zeros((n,8),bool);continuous=np.zeros(n,bool)
    y20=np.full((n,2),np.nan);old_error=np.full(n,np.nan);past_error=np.full(n,np.nan);raw_error=np.full(n,np.nan)
    reason=np.full(n,'AMBIGUOUS_TIME',dtype='<U64');known=(md['start_frame']>=0)&~md['ambiguous_track_time']
    t0=md['start_frame'].astype(np.int64)+7;t0[~known]=-1
    interval_by_track=defaultdict(list)
    for i in range(n):
        if known[i]:interval_by_track[d['track'][i]].append((int(t0[i]-7),int(t0[i]+12),str(owner[i]),i))
    c.stage('H0_CONNECT_TARGET_LABELS',known_time=int(known.sum()))
    mismatch_sources=[];raw_time_missing=[]
    for i in range(n):
        if i%256==0:c.guard()
        world=np.asarray(raw[int(ids[i]),0,:20,-2:],dtype=np.float32)
        # Reproduce old cached rotation even when absolute start is ambiguous.
        pe=float(np.max(np.abs(local(world[:8],world)-d['past'][i])));ge=float(np.max(np.abs(local(world[8:],world)-d['gt'][i])))
        assert max(pe,ge)<=2e-5,('CACHE_COORDINATE_MISMATCH',int(ids[i]),pe,ge)
        past_error[i]=pe;old_error[i]=ge
        if not known[i]:continue
        seq=str(md['sequence'][i]);ped=int(md['pedestrian_id'][i]);start=int(md['start_frame'][i]);table,conflicts=tables[seq]
        oldkeys=[(start+j,ped) for j in range(20)]
        if any(k not in table or k in conflicts for k in oldkeys):
            reason[i]='OLD_LABEL_SOURCE_CONFLICT_OR_MISSING';mismatch_sources.append(dict(id=int(ids[i]),sequence=seq,reason=reason[i]));continue
        raw_time=raw[int(ids[i]),0,:,1]
        if np.all(raw_time==-1):
            # Some original zara03 processed rows have a missing timestamp column.
            # Unique old metadata + all20 source coordinates verify identity/time;
            # do not pretend the absent column independently proves the join.
            raw_time_missing.append(int(ids[i]))
        else:assert np.array_equal(raw_time,np.arange(start,start+20)),('SOURCE_TIME_COLUMN_CONFLICT',int(ids[i]))
        orig=np.array([table[k] for k in oldkeys],float);re=float(np.max(np.abs(orig-world)));raw_error[i]=re
        if re>2e-5:
            reason[i]='OLD_WORLD_COORDINATE_MISMATCH';mismatch_sources.append(dict(id=int(ids[i]),sequence=seq,max_error=re));continue
        presence[i]=[(int(t0[i]+j),ped) in table and (int(t0[i]+j),ped) not in conflicts for j in range(13,21)]
        keys=[(int(t0[i]+j),ped) for j in range(21)]
        continuous[i]=all(k in table and k not in conflicts for k in keys)
        if not continuous[i]:reason[i]='MISSING_OR_CONFLICTING_FUTURE_13_20';continue
        y20[i]=local(table[(int(t0[i]+20),ped)],world)
        m20[i]=True;reason[i]='VALID_CONTINUOUS_20'
    c.write('SOURCE_SCHEMA.json',dict(schemas=schemas,consistent_duplicate_rows=duplicate_rows,old_source_mismatches=mismatch_sources,raw_timestamp_missing_ids=raw_time_missing,missing_raw_time_join='unique existing source metadata AND complete20 coordinate match; no nearest-coordinate guess'))
    assert not mismatch_sources,('STOP_LABEL_SOURCE_REPRODUCTION',len(mismatch_sources))
    # Label ownership is target-specific. Unknown held-out intervals -> conservative
    # mask of otherwise supported FIT windows sharing that same target track.
    held={'outer','inner','original_CAL'}
    unknown_held_tracks=set(d['track'][np.isin(owner,list(held))&~known])
    new_cross=np.zeros(n,bool);unknown_owner=np.zeros(n,bool);events=[];old_cross=Counter()
    for track,values in interval_by_track.items():
        for a,b in itertools.combinations(values,2):
            if a[2]==b[2] or 'unused_legacy_FIT' in (a[2],b[2]):continue
            if overlap(a[0],a[1],b[0],b[1]):old_cross['|'.join(sorted((a[2],b[2])))]+=1
    for i in d['split']['FIT']:
        if not known[i]:continue
        if d['track'][i] in unknown_held_tracks:unknown_owner[i]=True;m20[i]=False;reason[i]='UNKNOWN_HELD_INTERVAL_SAME_TRACK';continue
        for lo,hi,pop,j in interval_by_track[d['track'][i]]:
            if pop in held and overlap(int(t0[i]+13),int(t0[i]+20),lo,hi):
                new_cross[i]=True;m20[i]=False;reason[i]='NEW_LONG_LABEL_OVERLAPS_HELD_TARGET_INTERVAL';events.append(dict(FIT_id=int(ids[i]),held_id=int(ids[j]),held_split=pop,new_interval=[int(t0[i]+13),int(t0[i]+20)],held_interval=[lo,hi]))
    assert not any('original_CAL' not in k and v>0 for k,v in old_cross.items()),'STOP_TRACK_DISJOINT'
    # Exact full focal-content duplicates crossing target-disjoint populations
    # are not silently counted as independent evaluation windows.
    content={name:{} for name in d['split']}
    for name,ix in d['split'].items():
        for i in ix:
            content[name].setdefault(c.array_sha(np.asarray(raw[int(ids[i]),0,:20,-2:],np.float32)),[]).append(int(ids[i]))
    duplicates={}
    for a,b in itertools.combinations(d['split'],2):
        common=set(content[a])&set(content[b]);duplicates[a+'|'+b]=[dict(a_ids=content[a][k],b_ids=content[b][k]) for k in sorted(common)]
        if 'original_CAL' not in (a,b):assert not common,('STOP_EXACT_CROSS_SPLIT_FOCAL_DUPLICATE',a,b,len(common))
    summary={};vel=np.linalg.norm(d['v0'],axis=-1);disp=np.linalg.norm(d['gt'][:,-1],axis=-1)
    ae=np.linalg.norm(d['native'].astype(float)-d['gt'][:,None],axis=-1);ng=((ae.mean(-1)<=.2)&(ae[:,:,-1]<=.4)).any(1)
    for name,ix in d['split'].items():
        rows={}
        for family,labels in [('group',d['groups'].astype(str)),('sequence',md['sequence']),('scene',d['scene'])]:
            for label in np.unique(labels[ix]):
                j=ix[labels[ix]==label];rows[family+':'+str(label)]=dict(n=len(j),m20_true=int(m20[j].sum()),valid_tracks=len(np.unique(d['track'][j[m20[j]]])))
        selection={}
        for flag in (False,True):
            j=ix[m20[ix]==flag];selection[str(flag)]=dict(n=len(j),tracks=len(np.unique(d['track'][j])),speed_mean=float(vel[j].mean()) if len(j) else None,displacement_mean=float(disp[j].mean()) if len(j) else None,native_good=int(ng[j].sum()),native_coverage=float(ng[j].mean()) if len(j) else None)
        summary[name]=dict(n=len(ix),m20_true=int(m20[ix].sum()),valid_tracks=len(np.unique(d['track'][ix[m20[ix]]])),reasons=dict(Counter(reason[ix])),selection=selection,strata=rows)
    fit=d['split']['FIT'];change=fit[d['groups'][fit]==2];valid=fit[m20[fit]];vc=change[m20[change]]
    gates=dict(FIT_valid_windows_ge1000=len(valid)>=1000,FIT_valid_tracks_ge50=len(np.unique(d['track'][valid]))>=50,FIT_change_valid_windows_ge200=len(vc)>=200,FIT_change_valid_tracks_ge20=len(np.unique(d['track'][vc]))>=20)
    c.save('H0_labels.npz',sample_id=ids,y20=y20,m20=m20,future13_20_present=presence,continuous20=continuous,owner=owner,t0=t0,start_frame=md['start_frame'],reason=reason,old_local_error=old_error,past_local_error=past_error,old_world_error=raw_error,new_cross_held=new_cross,unknown_held_owner=unknown_owner)
    c.write('ID_REGISTRY.json',dict(n=n,used=24679,unused_original_FIT=n-24679,ids_by_split={k:ids[v] for k,v in d['split'].items()},unused_ids=ids[owner=='unused_legacy_FIT'],unused_reason='frozen grouped split, not eligible for either new training or evaluation; retained in registry'))
    with (c.H/'MASK_BY_ID.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['sample_id','sequence','pedestrian','start_index','t0','split','group','m20','reason'])
        for i in range(n):w.writerow([int(ids[i]),md['sequence'][i],int(md['pedestrian_id'][i]),int(md['start_frame'][i]),int(t0[i]),owner[i],int(d['groups'][i]),int(m20[i]),reason[i]])
    c.write('SPLIT_OVERLAP_AUDIT.json',dict(status='PASS',old_target_interval_overlap_pair_counts=dict(old_cross),new_FIT_long_interval_cross_events=events,new_FIT_windows_masked=int(new_cross[fit].sum()),unknown_held_owner_masked=int(unknown_owner[fit].sum()),exact_focal_duplicates=duplicates,target_disjoint_not_context_disjoint=True,original_CAL_not_independent=True))
    c.write('LONG_LABEL_AUDIT.json',dict(status='PASS' if all(gates.values()) else 'INFEASIBLE_LONG_LABEL_SUPPORT',summary=summary,gates=gates,all_28791_retained=True,all_evaluation_denominators_unchanged=True,ALL_primary=True,common_aux_mask_both_arms=True,raw_step=1,dt=.4,video_frame_unit_not_assumed=True,coordinate_tolerance=2e-5,max_old_local_error=float(old_error.max()),max_past_error=float(past_error.max()),max_world_error=float(np.nanmax(raw_error)),frozen_P2_and_neighbors_not_regenerated=True,legacy_original12_future_selection_remains=True))
    c.verify(sources);outputs={str(p):c.sha(p) for p in [c.H/k for k in ('H0_labels.npz','ID_REGISTRY.json','MASK_BY_ID.csv','SPLIT_OVERLAP_AUDIT.json','LONG_LABEL_AUDIT.json','SOURCE_SCHEMA.json')]}
    result=dict(status='PASS_H0_ALLOW_H1_PREFLIGHT' if all(gates.values()) else 'INFEASIBLE_LONG_LABEL_SUPPORT',at=c.now(),elapsed_seconds=c.time.perf_counter()-begin,summary=summary,gates=gates,source_files=sources,output_sha=outputs,source_initial_final_unchanged=True,resource=c.RESOURCE,no_training=True,no_model_forward=True)
    c.write('H0_RESULT.json',result);print(c.json.dumps(c.safe(dict(status=result['status'],summary=summary,gates=gates,seconds=result['elapsed_seconds'])),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        c.write('H0_FAILURE.json',dict(at=c.now(),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
