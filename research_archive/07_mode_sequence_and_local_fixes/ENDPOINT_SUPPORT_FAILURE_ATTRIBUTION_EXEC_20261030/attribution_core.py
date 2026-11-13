"""Bounded saved-array diagnosis. No Torch, model, checkpoint deserialization, or training."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
PARENT=ROOT.parent/'ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029'
P0=ROOT.parent/'ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029'
sys.path.insert(0,str(PARENT/'code'))
from io_contract import sha,load_npz,json_read,json_new,npz_new
from features import transform
from action_geometry import space,apply_action,decode
from motion_guard import inspect,pooled,junction,MOTION_NAMES
from evaluate import sample_metrics

TOL=2e-12
PORTS=('N','C','Q','S','O')
STRATA=('unique_support','multiple_support','failed_slot_in_success_window','original_failure_window')

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def write_json(p,v):json_new(p,clean(v))

def csv_new(p,rows):
    if not rows:raise ValueError('empty table; use explicit NA row')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(clean(rows))

def guard():
    p=ROOT.parent/'K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py'
    spec=importlib.util.spec_from_file_location('attribution_resource_v3',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.Guard(2*1024**3)

def verify_files(files,check=lambda:None):
    for i,(p,s) in enumerate(files.items()):
        if sha(p)!=s:raise AssertionError('SOURCE_SHA_CHANGED '+str(p))
        if i%50==0:check()

def source_lock(check):
    t=time.perf_counter();old=json_read(PARENT/'source_manifest.json')['files']
    assert len(old)==5140
    files=dict(old)
    def add(p,expected=None):
        p=Path(p).resolve();v=sha(p)
        if expected is not None and expected!=v:raise AssertionError('completed parent output SHA '+str(p))
        if str(p) in files and files[str(p)]!=v:raise AssertionError('overlapping inherited SHA '+str(p))
        files[str(p)]=v
    verify_files(old,check)
    final=json_read(PARENT/'FINAL_PRESERVATION.json')
    for rel,s in final['files'].items():add(PARENT/rel,s)
    add(PARENT/'FINAL_PRESERVATION.json')
    add(PARENT/'source_manifest.json')
    add(PROJECT/'docs/ENDPOINT_SUPPORT_FAILURE_ATTRIBUTION_DESIGN_20261030 - 副本.md')
    add(ROOT.parent/'K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
    for p in sorted(ROOT.glob('*.py'))+sorted(ROOT.glob('*.md'))+sorted(ROOT.glob('protocol.json')):add(p)
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write_json(ROOT/'SOURCE_EXTENSION.json',dict(status='LOCKED_OFFLINE_ONLY',inherited_count=5140,
        parent_immutable_outputs=len(final['files']),files=files,source_digest=digest,initial_seconds=time.perf_counter()-t,
        checkpoint_policy='SHA bytes only; never load model/pickle'))
    return files,time.perf_counter()-t

def observed():
    ids=load_npz(PARENT/'tiny_v2/indices.npz')
    xa=load_npz(P0/'prepared/X_train.npz');idx=ids['indices']
    x={k:np.array(v[idx],copy=True) for k,v in xa.items()}
    assert np.array_equal(x['sample_id'],ids['sample_id']) and len(idx)==128
    assert len(np.unique(x['sample_id']))==128 and len(np.unique(x['track_key']))==97
    stats=load_npz(P0/'prepared/statistics.npz')
    z=transform(x['past'],x['native'],stats,'H')
    assert np.array_equal(transform(x['past'],x['native'],stats,'G')[...,16:],z[...,16:])
    return x,z,idx

def nearest_maps(z,ids,tracks,sequence,check=lambda:None):
    """No GT/label/action/J inputs. Squared distances are computed from float32 model inputs."""
    a=np.asarray(z,dtype=np.float32).reshape(-1,40)
    if not np.isfinite(a).all():raise ValueError('nonfinite observed input')
    N=len(a);wid=np.repeat(np.asarray(ids),20);slot=np.tile(np.arange(20),len(ids))
    tr=np.repeat(tracks,20);seq=np.repeat(sequence,20);unique=sorted(set(tr.tolist()))
    members={t:np.flatnonzero(tr==t) for t in unique}
    maps={k:np.full((N,5),-1,np.int32) for k in ('G_main','H_main','G_all','H_all','hash_main','hash_all')}
    distances={k:np.full((N,5),np.nan) for k in ('G_main','H_main','G_all','H_all')}
    for q in range(N):
        if q%64==0:check()
        dp=((a[:,16:].astype(float)-a[q,16:])**2).sum(1)/24
        dh=((a[:,:16].astype(float)-a[q,:16])**2).sum(1)/16
        for scope,allowed in [('main',(tr!=tr[q])&(seq==seq[q])),('all',tr!=tr[q])]:
            ts=[t for t in unique if allowed[members[t][0]]]
            for arm,d in [('G',dp),('H',.5*(dp+dh))]:
                winners=[]
                for t in ts:
                    ix=members[t];o=np.lexsort((slot[ix],wid[ix],d[ix]));winners.append(int(ix[o[0]]))
                winners=sorted(winners,key=lambda j:(d[j],int(wid[j]),int(slot[j])))[:5]
                key=arm+'_'+scope;maps[key][q,:len(winners)]=winners;distances[key][q,:len(winners)]=d[winners]
            ts.sort(key=lambda t:hashlib.sha256(f'endpoint-support-attribution-v1|{wid[q]}|{slot[q]}|{t}'.encode()).hexdigest())
            refs=[]
            for t in ts[:5]:
                refs.append(min(members[t],key=lambda j:hashlib.sha256(
                    f'endpoint-support-attribution-candidate-v1|{wid[q]}|{slot[q]}|{wid[j]}|{slot[j]}'.encode()).hexdigest()))
            maps['hash_'+scope][q,:len(refs)]=refs
    return maps,distances

def seal_maps(x,z,check):
    t=time.perf_counter();maps,distances=nearest_maps(z,x['sample_id'],x['track_key'],x['sequence'],check)
    npz_new(ROOT/'neighbor_map.npz',**maps,**{k+'_distance_squared':v for k,v in distances.items()})
    write_json(ROOT/'NEIGHBOR_MAP_SEAL.json',dict(status='SEALED_BEFORE_LOADING_GT_OR_TEACHER',
        sha=sha(ROOT/'neighbor_map.npz'),at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),seconds=time.perf_counter()-t,
        selector_inputs=['actual float32 normalized observations','sample ID','track','sequence'],
        scientific_GT_or_teacher_loaded=False,source_file_hashing_is_not_label_association=True))
    return maps,distances,time.perf_counter()-t

def selected_labels(x,idx):
    ya=load_npz(P0/'prepared/GT_train.npz')
    gt=ya['gt'][idx];assert np.array_equal(ya['sample_id'][idx],x['sample_id'])
    labels={};masks={}
    # Container reads are declared; only selected rows enter scientific arithmetic.
    read_files=[]
    for kind,out in [('labels',labels),('masks',masks)]:
        for p in sorted((P0/f'P0_END_train_{kind}').glob('[0-9]*.npz')):
            rows=load_npz(p);lookup={int(s):i for i,s in enumerate(rows['sample_id'])}
            selected=[(j,lookup[int(s)]) for j,s in enumerate(x['sample_id']) if int(s) in lookup]
            if not selected:continue
            read_files.append(str(p.relative_to(P0)))
            for k,v in rows.items():
                if k not in out:out[k]=np.empty((128,)+v.shape[1:],dtype=v.dtype)
                for j,r in selected:out[k][j]=v[r]
    assert np.array_equal(labels['sample_id'],x['sample_id']) and np.array_equal(masks['sample_id'],x['sample_id'])
    write_json(ROOT/'INPUT_MANIFEST.json',dict(population_windows=128,tracks=97,candidates=2560,
        observed_container_rows=3103,GT_container_rows=3103,scientific_GT_rows=128,
        selected_label_mask_containers=read_files,selection=sha(PARENT/'tiny_v2/indices.npz'),
        all49_new_geometry_enumerations=0,old_J_matrix_safe_masks_reused=True,
        checkpoint_deserialization=False,held_GT_access=False))
    return gt,labels,masks

def quality(p,gt):
    d=np.linalg.norm(np.asarray(p,dtype=float)-gt[:,None],axis=-1)
    a,f=d.mean(-1),d[...,-1];j=np.maximum(a/.2,f/.4)
    return a,f,j,(a<=.2)&(f<=.4)

def build_ports(x,gt,labels,masks,check):
    initial=load_npz(PARENT/'tiny_v2/initial.npz');final=load_npz(PARENT/'tiny_v2/final.npz')
    for d in (initial,final):assert np.array_equal(d['sample_id'],x['sample_id'])
    n=x['native'];u=final['continuous_param'].astype(float);a=space('END')
    continuous=n+(np.arange(1,13)/12)[None,None,:,None]*(.4*u[:,:,None,:])
    oracle=np.empty_like(n)
    for i in range(128):
        check()
        for k in range(20):
            oracle[i,k]=apply_action(n[i,k],a,int(labels['action_id'][i,k]))
    paths=np.stack((n,continuous,final['raw_paths'],final['guarded_paths'],oracle))
    aid=np.stack((np.full((128,20),a.identity),np.full((128,20),-1),final['raw_action_id'],
                  final['executed_action_id'],labels['action_id']))
    failed=np.zeros((5,128,20,7),bool)
    for i in range(128):
        check()
        for k in range(20):
            for s in range(5):failed[s,i,k]=inspect(x['past'][i],n[i,k],paths[s,i,k])[1]
            assert decode(n[i,k],u[i,k],a)==final['raw_action_id'][i,k]
    assert np.array_equal(failed[2],final['rejection_reason'])
    assert np.array_equal(np.where(failed[2].any(-1)[...,None,None],n,paths[2]),paths[3])
    assert not failed[3].any() and not failed[4].any()
    values=[quality(p,gt) for p in paths];q={key:np.stack([v[j] for v in values]) for j,key in enumerate(('ADE','FDE','J','good'))}
    active=np.any(labels['target']!=0,axis=-1)
    assert int(active.sum())==2057
    assert [int(q['good'][j].any(1).sum()) for j in (0,2,3)]==[69,67,70]
    b=q['good'][0].any(1)
    for j,rescue,lost in [(2,8,10),(3,8,7)]:
        c=q['good'][j].any(1);assert int((~b&c).sum())==rescue and int((b&~c).sum())==lost
    # The teacher targets were float32 in the actual tiny training arithmetic.
    target=labels['target'].astype(np.float32).astype(float)
    e0=((initial['continuous_param'].astype(float)-target)**2).mean(-1)
    e1=((u-target)**2).mean(-1)
    npz_new(ROOT/'derived_ports.npz',sample_id=x['sample_id'],past=x['past'],native=n,gt=gt,
        track_key=x['track_key'],sequence=x['sequence'],group=x['group'],paths=paths,action_id=aid,failed=failed,
        continuous_param=u,target=target,teacher_action=labels['action_id'],safe=masks['safe'],J_matrix=labels['J_matrix'],
        initial_error=e0,final_error=e1,**q)
    return paths,aid,failed,q,e0,e1,active

def window_physics(p,past):
    v=np.diff(np.concatenate((np.zeros(p.shape[:-2]+(1,2)),p),-2),axis=-2)/.4
    aa=np.diff(v,axis=-2)/.4;jj=np.diff(v,n=2,axis=-2)/(.4**2)
    vn,an,jn=[np.linalg.norm(a,axis=-1) for a in (v,aa,jj)]
    ax=tuple(range(1,vn.ndim));axa=tuple(range(1,an.ndim));axj=tuple(range(1,jn.ndim))
    jun=np.linalg.norm(p[...,0,:]/.4-((past[:,-1]-past[:,-2])/.4)[:,None],axis=-1)/.4
    return np.stack((np.quantile(vn,.95,axis=ax),an.mean(axa),np.quantile(an,.95,axis=axa),
        jn.mean(axj),np.quantile(jn,.95,axis=axj),jun.mean(1),np.quantile(jun,.95,axis=1)),1)

def group_metrics(x,gt,paths,q,check):
    rows=[];base=q['good'][0].any(1);n=x['native']
    samples=[]
    for s,p in enumerate(paths):
        check();sm=sample_metrics(p,gt);samples.append(sm)
        for g,m in [('overall',np.ones(128,bool))]+[(name,x['group']==j) for j,name in enumerate(('stable','transition','change'))]:
            b,c=base[m],sm['covered'][m];pm,nm=pooled(p[m]),pooled(n[m])
            ju=np.concatenate([junction(x['past'][i],p[i]) for i in np.flatnonzero(m)])
            jn=np.concatenate([junction(x['past'][i],n[i]) for i in np.flatnonzero(m)])
            row=dict(port=PORTS[s],group=g,N=int(m.sum()),tracks=len(np.unique(x['track_key'][m])),
                native_good=int(b.sum()),good=int(c.sum()),rescue=int((~b&c).sum()),lost=int((b&~c).sum()),retained=int((b&c).sum()),
                lost_fraction=(b&~c).sum()/b.sum() if b.any() else None)
            row.update({key:float(sm[key][m].mean()) for key in ('minADE','minFDE','minJ','distinct','effective_good','early_ADE','middle_ADE','late_ADE','coverage_sensitive_03_05')})
            row.update({key:float(v) for key,v in zip(MOTION_NAMES,pm)})
            row.update({'native_'+key:float(v) for key,v in zip(MOTION_NAMES,nm)})
            row.update(motion_literal_pass=bool(np.all(pm<=nm*1.01)),junction_mean=ju.mean(),junction_p95=np.quantile(ju,.95),
                native_junction_mean=jn.mean(),native_junction_p95=np.quantile(jn,.95),
                junction_literal_pass=bool(ju.mean()<=1.01*jn.mean() and np.quantile(ju,.95)<=1.01*np.quantile(jn,.95)))
            rows.append(row)
    csv_new(ROOT/'all_groups_quality_motion.csv',rows);csv_new(ROOT/'ports_summary.csv',[r for r in rows if r['group']=='overall'])
    npz_new(ROOT/'port_window_metrics.npz',**{PORTS[s]+'_'+k:v for s,sm in enumerate(samples) for k,v in sm.items()})
    return rows

def event(b,c):return 'retained' if b and c else 'lost' if b else 'rescue' if c else 'unresolved'

def stage_tables(x,paths,aid,failed,q,e0,e1,active):
    good=q['good'];cov=good.any(2);b=good[0];m=b.sum(1)
    strata=np.where(b,np.where(m[:,None]==1,0,1),np.where(m[:,None]>0,2,3))
    d=e0-e1;dj=q['J'][0]-q['J'][3]
    rows=[];support=[];trans=[]
    for i in range(128):
        code=''.join(str(int(cov[s,i])) for s in range(4))
        trans.append(dict(sample_id=int(x['sample_id'][i]),track=str(x['track_key'][i]),group=int(x['group'][i]),
            N=int(cov[0,i]),C=int(cov[1,i]),Q=int(cov[2,i]),S=int(cov[3,i]),O=int(cov[4,i]),
            stage_code=code,support_count=int(m[i]),native_minJ_slot=int(q['J'][0,i].argmin()),event=event(cov[0,i],cov[3,i])))
        for k in range(20):
            row=dict(sample_id=int(x['sample_id'][i]),window_index=i,slot=k,track=str(x['track_key'][i]),group=int(x['group'][i]),
                stratum=STRATA[strata[i,k]],teacher_type='active' if active[i,k] else 'identity',teacher_action=int(aid[4,i,k]),
                initial_error=e0[i,k],final_error=e1[i,k],delta_MSE=d[i,k],delta_J=dj[i,k],
                support_count=int(m[i]),native_margin=1-q['J'][0,i,k],native_support=bool(b[i,k]),
                continuous_offset_norm=float(np.linalg.norm(paths[1,i,k,-1]-paths[0,i,k,-1])),
                C_outside_radius=bool(np.linalg.norm(paths[1,i,k,-1]-paths[0,i,k,-1])>.4),
                C_position_bound_violated=bool(np.any(np.linalg.norm(paths[1,i,k]-paths[0,i,k],axis=-1)>.4*np.arange(1,13)/12)),
                continuous_gap=q['J'][1,i,k]-q['J'][4,i,k],quantization_gap=q['J'][2,i,k]-q['J'][1,i,k],
                guard_gap=q['J'][3,i,k]-q['J'][2,i,k],S_oracle_gap=q['J'][3,i,k]-q['J'][4,i,k],
                executed_teacher_hit=bool(aid[3,i,k]==aid[4,i,k]),safe_utility_within_01=bool(q['J'][3,i,k]<=q['J'][4,i,k]+.01),
                discrete_parameter_distance=float(np.linalg.norm(space('END').normalized[aid[3,i,k]]-space('END').normalized[aid[4,i,k]])),
                guard_restored_native=bool(failed[2,i,k].any()),final_event=trans[-1]['event'])
            for s in range(5):
                for key in ('ADE','FDE','J','good'):row[PORTS[s]+'_'+key]=q[key][s,i,k]
                row[PORTS[s]+'_action']=int(aid[s,i,k]);row[PORTS[s]+'_failed_bits']=''.join(str(int(t)) for t in failed[s,i,k])
            assert abs(row['continuous_gap']+row['quantization_gap']+row['guard_gap']-row['S_oracle_gap'])<=TOL
            rows.append(row)
            if b[i,k]:support.append(row)
    csv_new(ROOT/'candidate_stage_audit.csv',rows);csv_new(ROOT/'window_stage_transitions.csv',trans)
    write_json(ROOT/'native_support_sets.json',dict(windows=trans,support_slots=support))
    gains=[]
    for g,gm in [('overall',np.ones(128,bool))]+[(name,x['group']==j) for j,name in enumerate(('stable','transition','change'))]:
        for s,name in enumerate(STRATA):
            for label,am in [('all',np.ones_like(active)),('active',active),('identity',~active)]:
                mask=(strata==s)&gm[:,None]&am;ws=mask.any(1)
                row=dict(group=g,stratum=name,teacher_type=label,candidates=int(mask.sum()),windows=int(ws.sum()),tracks=len(np.unique(x['track_key'][ws])))
                if not mask.any():row['status']='NA_EMPTY'
                else:
                    row['status']='DESCRIPTIVE_NATIVE_FROZEN_STRATUM'
                    for nm,value,total in [('delta_MSE',d,d.sum()),('delta_J',dj,dj.sum())]:
                        val=value[mask];sums=np.where(mask,value,0).sum(1)
                        row.update({nm+'_mean':val.mean(),nm+'_median':np.median(val),nm+'_p05':np.quantile(val,.05),nm+'_p95':np.quantile(val,.95),
                            nm+'_positive_fraction':(val>0).mean(),nm+'_negative_fraction':(val<0).mean(),nm+'_signed_sum':val.sum(),
                            nm+'_contribution_ratio':val.sum()/total if abs(total)>1e-12 else None,
                            nm+'_window_mean':(sums[ws]/mask.sum(1)[ws]).mean()})
                    row.update(final_lost_windows=int((ws&cov[0]&~cov[3]).sum()),final_rescue_windows=int((ws&~cov[0]&cov[3]).sum()),
                        native_support_became_bad=int((mask&b&~good[3]).sum()),exact_teacher_hit=(aid[3][mask]==aid[4][mask]).mean(),
                        safe_J_within_01=(q['J'][3][mask]<=q['J'][4][mask]+.01).mean())
                gains.append(row)
    csv_new(ROOT/'gain_allocation.csv',gains)
    return strata,trans,gains

def counterfactual(x,gt,paths,q,check,only_first=None):
    n,s=paths[0],paths[3];N=len(n) if only_first is None else only_first
    allrows=[];saved={}
    for mode in ('restore','edit'):
        arr={key:np.empty((N,20),dtype=bool if key=='covered' else float) for key in
            ('covered','minJ','distinct','effective_good','speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95','junction_mean','junction_p95')}
        for i in range(N):
            check();base=s[i] if mode=='restore' else n[i];insert=n[i] if mode=='restore' else s[i]
            batch=np.repeat(base[None],20,axis=0);batch[np.arange(20),np.arange(20)]=insert
            sm=sample_metrics(batch,np.repeat(gt[i:i+1],20,axis=0))
            ph=window_physics(batch,np.repeat(x['past'][i:i+1],20,axis=0))
            for key in ('covered','minJ','distinct','effective_good'):arr[key][i]=sm[key]
            for j,key in enumerate(('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95','junction_mean','junction_p95')):arr[key][i]=ph[:,j]
            for k in range(20):
                nb=bool(q['good'][0,i].any());sb=bool(q['good'][3,i].any());c=bool(sm['covered'][k])
                row=dict(window_index=i,sample_id=int(x['sample_id'][i]),slot=k,track=str(x['track_key'][i]),group=int(x['group'][i]),
                    native_covered=nb,S_covered=sb,event_vs_native=event(nb,c),event_vs_S=event(sb,c),
                    delta_minJ_vs_background=sm['minJ'][k]-q['J'][3 if mode=='restore' else 0,i].min(),K=20)
                row.update({key:val[i,k] for key,val in arr.items()});allrows.append((mode,row))
        saved.update({mode+'_'+k:v for k,v in arr.items()})
    if only_first is None:
        for mode in ('restore','edit'):csv_new(ROOT/f'single_slot_{mode}.csv',[r for mm,r in allrows if mm==mode])
        npz_new(ROOT/'counterfactual_metrics.npz',**saved)
    return saved

def utility(x,z,maps,labels,masks,q):
    a=z.reshape(-1,40).astype(float);target=labels['target'].reshape(-1,2)
    action=labels['action_id'].reshape(-1);safe=masks['safe'].reshape(-1,49);jm=labels['J_matrix'].reshape(-1,49)
    oracle=jm[np.arange(2560),action];base=q['J'][0].ravel();identity=space('END').identity
    rows=[];agg={}
    for key,mp in maps.items():
        metrics={nm:np.full((2560,5),np.nan) for nm in ('distance_squared','target_distance','same_action','same_identity','proposal_safe','proposal_gap','executed_gap','retained_native_good','rescue_native_bad')}
        for r in range(2560):
            for rank,j in enumerate(mp[r]):
                if j<0:continue
                act=int(action[j]);ok=bool(safe[r,act]);exe=act if ok else identity
                dp=((a[r,16:]-a[j,16:])**2).sum()/24;dh=((a[r,:16]-a[j,:16])**2).sum()/16
                d=dp if key.startswith('G') else .5*(dp+dh)
                vals=[d,np.linalg.norm(target[r]-target[j]),float(action[r]==act),float((action[r]==identity)==(act==identity)),
                    float(ok),jm[r,act]-oracle[r],jm[r,exe]-oracle[r],float(jm[r,exe]<=1) if base[r]<=1 else np.nan,
                    float(jm[r,exe]<=1) if base[r]>1 else np.nan]
                for nm,v in zip(metrics,vals):metrics[nm][r,rank]=v
                rows.append(dict(map=key,query_id=int(x['sample_id'][r//20]),query_slot=r%20,query_track=str(x['track_key'][r//20]),
                    rank=rank+1,reference_id=int(x['sample_id'][j//20]),reference_slot=int(j%20),reference_track=str(x['track_key'][j//20]),
                    query_teacher=int(action[r]),reference_teacher=act,executed_action=exe,reference_count=int((mp[r]>=0).sum()),
                    reference_status='SUFFICIENT_5' if (mp[r]>=0).sum()==5 else 'INSUFFICIENT_REFERENCE_TRACKS',
                    **{nm:v for nm,v in zip(metrics,vals)}))
        agg.update({key+'_'+nm:v for nm,v in metrics.items()})
    csv_new(ROOT/'neighbor_utility.csv',rows);npz_new(ROOT/'neighbor_utility_metrics.npz',**agg)
    summary=[]
    for key in maps:
        for top in (1,5):
            row=dict(map=key,top=top,queries=2560,windows=128,tracks=97,
                zero_reference_queries=int(((maps[key]>=0).sum(1)==0).sum()),under5_queries=int(((maps[key]>=0).sum(1)<5).sum()))
            for nm in metrics:
                v=agg[key+'_'+nm][:,:top];finite=np.isfinite(v);counts=finite.sum(1)
                cv=np.divide(np.nansum(v,1),counts,out=np.full(2560,np.nan),where=counts>0)
                wm=np.array([np.mean(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan for c in cv.reshape(128,20)])
                row[nm+'_window_mean']=float(np.mean(wm[np.isfinite(wm)])) if np.isfinite(wm).any() else None
                row[nm+'_valid_windows']=int(np.isfinite(wm).sum())
            # Hash reference has both metric distances, not a guaranteed farther control.
            js=maps[key].ravel();rs=np.repeat(np.arange(2560),5);valid=js>=0;rs,js=rs[valid],js[valid]
            dp=((a[rs,16:]-a[js,16:])**2).sum(1)/24;dh=((a[rs,:16]-a[js,:16])**2).sum(1)/16
            row.update(G_distance_mean=dp.mean(),H_distance_mean=(.5*(dp+dh)).mean())
            summary.append(row)
    csv_new(ROOT/'neighbor_summary.csv',summary)
    return agg,summary

def duplicate_bound(z,target):
    result={}
    for arm,xx in [('G',z[...,16:]),('H',z)]:
        xx=np.array(xx.reshape(2560,-1),copy=True);xx[xx==0]=0
        if not np.isfinite(xx).all():raise ValueError('nonfinite exact inputs')
        _,inv,cnt=np.unique(xx,axis=0,return_inverse=True,return_counts=True)
        groups=[];sse=0
        for g,n in enumerate(cnt):
            ix=np.flatnonzero(inv==g);yy=target.reshape(-1,2)[ix];mu=yy.mean(0);err=((yy-mu)**2).sum();sse+=err
            groups.append(dict(group=g,n=int(n),candidate_indices=ix.tolist(),target_mean=mu.tolist(),residual_squared_sum=err))
        result[arm]=dict(groups=len(cnt),duplicate_groups=int((cnt>1).sum()),duplicate_candidates=int(cnt[cnt>1].sum()),
            empirical_MSE_lower_bound=sse/2/2560,scope='exact float32 inputs, +0/-0 canonical, fullcandidate not activeonly, not Bayes',records=groups)
    return result

def bootstrap(x,q,strata,utilities):
    tracks=np.unique(x['track_key']);rng=np.random.default_rng(20261030)
    draws=rng.integers(0,len(tracks),(5000,len(tracks)));tidx=np.searchsorted(tracks,x['track_key'])
    counts=np.array([np.bincount(d,minlength=len(tracks)) for d in draws]);w=counts[:,tidx]
    cov=q['good'].any(2);b,s=cov[0],cov[3];dj=q['J'][0]-q['J'][3]
    unique=(strata==0).any(1);bad=~b
    h=utilities['H_main_executed_gap'];r=utilities['hash_main_executed_gap']
    def rowmean(v):
        n=np.isfinite(v).sum(1);return np.divide(np.nansum(v,1),n,out=np.full(len(v),np.nan),where=n>0)
    gap=rowmean(h).reshape(128,20).mean(1)-rowmean(r).reshape(128,20).mean(1)
    vals=[(s.astype(float)-b, np.ones(128,bool)),((b&~s).astype(float),b),
        (q['J'][3].min(1)-q['J'][0].min(1),np.ones(128,bool)),
        ((dj*(strata==0)).sum(1),unique),(dj.mean(1),bad),(gap,np.isfinite(gap))]
    names=['coverage_S_minus_N','lost_fraction_nativegood','minJ_S_minus_N','unique_support_window_deltaJ','native_failure_window_deltaJ','H_main_minus_hash_execJgap']
    bs=[];report=[]
    for nm,(v,mask) in zip(names,vals):
        den=w[:,mask].sum(1);num=w[:,mask]@v[mask];value=np.divide(num,den,out=np.full(5000,np.nan),where=den>0);bs.append(value)
        valid=np.isfinite(value);report.append(dict(metric=nm,estimate=v[mask].mean() if mask.any() else None,windows=int(mask.sum()),valid=int(valid.sum()),
            CI95=np.quantile(value[valid],[.025,.975]).tolist() if valid.sum()>=4750 else None,status='EXPLORATORY_TRAIN_SUBSET' if valid.sum()>=4750 else 'NOT_ESTIMABLE'))
    npz_new(ROOT/'BOOTSTRAP_DRAWS.npz',tracks=tracks,track_draws=draws,statistics=np.stack(bs),window_track_index=tidx)
    write_json(ROOT/'DESCRIPTIVE_CI.json',dict(draws=5000,seed=20261030,results=report,no_p_values=True,no_confirmatory_claim=True))
    return report
