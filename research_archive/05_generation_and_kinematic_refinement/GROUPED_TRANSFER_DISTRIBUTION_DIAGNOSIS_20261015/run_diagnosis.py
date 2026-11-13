"""Frozen cache-only diagnosis. Never import a legacy module or run a neural model."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='1'
import sys,json,hashlib,time,csv,ctypes,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import psutil
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SERIES=ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006'
NAMES=['experiment1_target_track','experiment2_recording_time','experiment3_scene_family']
TAG='20261015-grouped-shift-v1'
GIB=2**30

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(name,obj):
    def cast(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=cast,allow_nan=False),encoding='utf-8')
    os.replace(temp,p)

class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]

class Guard:
    def __init__(self):self.private=0;self.rss=0;self.phys=2**64;self.commit=2**64;self.calls=0
    def check(self):
        v=psutil.Process().memory_info();m=Mem();m.length=ctypes.sizeof(m)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        self.private=max(self.private,v.peak_pagefile);self.rss=max(self.rss,v.peak_wset)
        self.phys=min(self.phys,m.avail_phys);self.commit=min(self.commit,m.avail_page);self.calls+=1
        if max(v.rss,v.private)>6*GIB or min(m.avail_phys,m.avail_page)<6*GIB:raise MemoryError('STOP_RESOURCE process6GiB/reserve6GiB')
    def values(self):return dict(peak_private_GiB=self.private/GIB,peak_RSS_GiB=self.rss/GIB,min_physical_GiB=self.phys/GIB,min_commit_GiB=self.commit/GIB,checks=self.calls)

guard=Guard()
def stage(s,**kw):
    guard.check();print(s,json.dumps(kw),flush=True)
    write('runtime_state.json',dict(status='RUNNING',stage=s,at=datetime.now().astimezone().isoformat(),**kw))

def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%80==0:guard.check()
        assert digest(p)==h,('SOURCE_OR_OUTPUT_SHA_MISMATCH',p)

def add_files(dst,files):
    for p,h in files.items():
        if p in dst:assert dst[p]==h,('CONFLICTING_ORIGINAL_DIGEST',p)
        dst[p]=h

def load_np(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def kinetics(past,base):
    """All positions are prepared relative coordinates; future prediction starts at 0."""
    past=np.asarray(past,np.float64);base=np.asarray(base,np.float64)
    v=np.diff(past,axis=1)/.4;a=np.diff(v,axis=1)/.4
    speed=np.linalg.norm(v,axis=-1);acc=np.linalg.norm(a,axis=-1)
    last=v[:,-1];first=v[:,0]
    angle=np.arctan2(first[:,0]*last[:,1]-first[:,1]*last[:,0],np.sum(first*last,axis=-1))
    angle[(speed[:,0]<.03)|(speed[:,-1]<.03)]=0
    hist=np.c_[speed[:,-1],speed.mean(1),speed.std(1),acc.mean(1),np.quantile(acc,.95,axis=1),speed.sum(1)*.4,np.linalg.norm(past[:,-1]-past[:,0],axis=-1),np.abs(angle),np.sin(angle),np.cos(angle)]
    vv=np.diff(base,axis=2,prepend=np.zeros_like(base[:,:,:1]))/.4
    aa=np.diff(vv,axis=2)/.4;jj=np.diff(aa,axis=2)/.4
    cv=last[:,None,None,:]*(np.arange(1,13)*.4)[None,None,:,None]
    dist=np.linalg.norm(base-cv,axis=-1)
    ep=base[:,:,-1];epcenter=ep.mean(1,keepdims=True)
    native=np.c_[dist.mean((1,2)),dist[:,:,-1].mean(1),np.quantile(dist[:,:,-1],.95,axis=1),np.linalg.norm(vv-last[:,None,None],axis=-1).mean((1,2)),np.linalg.norm(vv,axis=-1).mean((1,2)),np.linalg.norm(vv[:,:,-1],axis=-1).std(1),np.linalg.norm(ep-epcenter,axis=-1).mean(1),np.linalg.norm(base-base.mean(1,keepdims=True),axis=-1).mean((1,2)),np.linalg.norm(aa,axis=-1).mean((1,2)),np.linalg.norm(jj,axis=-1).mean((1,2))]
    return hist,native

def nn_excluding_track(tree,ref_tracks,query,query_tracks):
    out=np.empty(len(query));k=min(8,len(ref_tracks))
    pending=np.arange(len(query))
    while len(pending):
        d,i=tree.query(query[pending],k=k,workers=1)
        if k==1:d=d[:,None];i=i[:,None]
        ok=ref_tracks[i]!=query_tracks[pending,None]
        found=ok.any(1);chosen=np.argmax(ok,axis=1)
        out[pending[found]]=d[np.where(found)[0],chosen[found]]
        pending=pending[~found]
        if len(pending):
            assert k<len(ref_tracks),'NO_DIFFERENT_TRACK_NEIGHBOR'
            k=min(k*4,len(ref_tracks));guard.check()
    return out

def fixtures():
    t=np.arange(-7,1)*.4;v=np.array([.7,-.2])
    past=np.tile(t[None,:,None]*v,(4,1,1));base=np.tile((np.arange(1,13)*.4)[None,None,:,None]*v,(4,20,1,1))
    p,b=kinetics(past,base)
    assert np.max(np.abs(b[:,[0,1,2,3,8,9]]))<1e-12
    tree=cKDTree([[0.],[1.],[3.]])
    assert np.array_equal(nn_excluding_track(tree,np.array(['a','b','c']),np.array([[0.],[3.]]),np.array(['a','c'])),[1.,2.])
    ade=np.array([[.1,.3]]);fde=np.array([[.5,.2]])
    assert not ((ade<=.2)&(fde<=.4)).any()
    assert np.maximum(ade/.2,fde/.4).min()>1
    write('FIXTURES.json',dict(status='PASS',constant_motion_error=float(np.max(np.abs(b[:,[0,1,2,3,8,9]]))),samepath_disjoint_rejected=True,same_track_nearest_excluded=True))

def metric_arrays(path,gt):
    e=np.linalg.norm(path.astype(np.float64)-gt[:,None].astype(np.float64),axis=-1)
    a=e.mean(-1);f=e[:,:,-1]
    return dict(minADE=a.min(1),minFDE=f.min(1),Q=(a+.5*f).min(1),J=np.maximum(a/.2,f/.4).min(1),coverage=((a<=.2)&(f<=.4)).any(1))

def audit_studies(ids,gt,groups):
    reports=[];quality={};files={};allrecords=[];allcis=[];maxerr=0.
    for name in NAMES:
        folder=SERIES/name;result=read(folder/'results/RESULT.json')
        assert result['status'].startswith('PASS') and result['jobs']==57
        assert len(result['records'])==216 and len(result['outputs_SHA'])==114 and len(result['models_SHA'])==18
        for p in ['results/exit_status.json','worker_state.json']:
            assert read(folder/p)['status']=='SUCCESS'
        for p in ['results/ARITHMETIC_AUDIT.json','results/FROZEN_AUDIT.json','results/PRESERVATION_AUDIT.json']:
            assert read(folder/p)['status']=='PASS'
        add_files(files,result['outputs_SHA']);add_files(files,result['models_SHA'])
        add_files(files,read(folder/'preflight.json')['source_hashes'])
        splitpath=folder/'identity_audit/splits.npz'
        if not splitpath.exists():splitpath=SERIES/'identity_audit/splits.npz'
        split=load_np(splitpath);assert np.array_equal(split['sample_id'],ids[:25912])
        for f in [folder/'results/RESULT.json',folder/'results/ARITHMETIC_AUDIT.json',folder/'results/FROZEN_AUDIT.json',folder/'results/PRESERVATION_AUDIT.json',folder/'worker_state.json',folder/'results/exit_status.json',folder/'preflight.json',splitpath]:files[str(f)]=digest(f)
        science=[]
        for p in result['outputs_SHA']:
            p=Path(p)
            if p.stem.endswith('_legacy'):continue
            guard.check();z=load_np(p);assert len(np.unique(z['sample_id']))==len(z['sample_id'])
            order=np.searchsorted(ids,z['sample_id']) if np.all(np.diff(ids)>0) else None
            if order is None:
                sort=np.argsort(ids);order=sort[np.searchsorted(ids[sort],z['sample_id'])]
            assert np.array_equal(ids[order],z['sample_id'])
            q=metric_arrays(z['trajectory'],gt[order])
            for k,val in q.items():
                if val.dtype==bool:assert np.array_equal(val,z[k]),(name,p.name,k)
                else:
                    err=float(np.max(np.abs(val-z[k])));maxerr=max(maxerr,err);assert err<=1e-12,(p.name,k,err)
            quality[name,p.stem]={k:z[k].copy() for k in ('sample_id','coverage','J','minADE','minFDE','effective_good','raw_good','all_distinct')}
            assert np.all((z['effective_good']<=z['raw_good']) & (z['effective_good']<=20))
            assert np.all((z['all_distinct']>=1)&(z['all_distinct']<=20))
            science.append(p.stem)
        assert len(science)==57
        for arm in ('random','grouped'):
            for seed in (1,2,3):
                path=folder/f'results/models/{arm}/seed_{seed}/balanced_protected/history.csv'
                with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
                assert len(rows)==80
                files[str(path)]=digest(path)
        for row in result['records']:
            key=f"{row['arm']}_seed{row['seed']}_{row['policy']}_{row['split']}"
            q=quality[name,key];native=quality[name,'native_'+row['split']]
            sort=np.argsort(ids);ix=sort[np.searchsorted(ids[sort],q['sample_id'])]
            mask=np.ones(len(ix),bool) if row['group']=='overall' else groups[ix]=={'stable':0,'transition':1,'change':2}[row['group']]
            assert row['n']==int(mask.sum()) and row['good']==int(q['coverage'][mask].sum())
            assert row['rescued']==int((q['coverage'][mask]&~native['coverage'][mask]).sum())
            assert row['lost']==int((~q['coverage'][mask]&native['coverage'][mask]).sum())
            for k in ('minADE','minFDE','Q','J'):
                if k=='Q':continue
                assert abs(float(q[k][mask].mean())-row[k])<1e-12
        rows=[dict(study=name,**r) for r in result['records']]
        cis=[dict(study=name,**r) for r in result['bootstrap']+result.get('target_track_bootstrap',[])]
        allrecords.extend(rows);allcis.extend(cis)
        reports.append(dict(study=name,records=len(rows),bootstrap_rows=len(cis),models=18,outputs=114,scientific_jobs=57,histories='6x80',all_audits='PASS',motion_failing_rows=sum(not r['motion_allfive_1pct'] for r in rows),split=split))
    assert len(allrecords)==648 and len(allcis)==720
    return reports,quality,files,allrecords,allcis,maxerr

def support(x,source,target,tracks):
    mu=x[source].mean(0);sd=np.maximum(x[source].std(0),.001)
    scaled=(x-mu)/sd
    unique=np.unique(tracks[source])
    refkeys={k for k in unique if int(hashlib.sha256((TAG+'|'+k).encode()).hexdigest(),16)%100<80}
    ref=source[np.array([k in refkeys for k in tracks[source]])]
    cal=source[np.array([k not in refkeys for k in tracks[source]])]
    assert len(ref)>100 and len(cal)>50 and len(set(tracks[cal]))>=10
    tree=cKDTree(scaled[ref]);den=np.sqrt(x.shape[1])
    dcal=nn_excluding_track(tree,tracks[ref],scaled[cal],tracks[cal])/den
    threshold=float(np.quantile(dcal,.95));assert np.isfinite(threshold)
    distance=nn_excluding_track(tree,tracks[ref],scaled[target],tracks[target])/den
    good=distance<=threshold
    return good,dict(source_n=len(source),target_n=len(target),reference_n=len(ref),reference_tracks=len(set(tracks[ref])),calibration_n=len(cal),calibration_tracks=len(set(tracks[cal])),dimensions=x.shape[1],threshold=threshold,source_calibration_out_rate=float(np.mean(dcal>threshold)),target_out_rate=float(np.mean(~good)),target_distance_median=float(np.median(distance)),source_distance_median=float(np.median(dcal)),target_tracks=len(set(tracks[target])),same_track_neighbors_excluded=True),distance

def shift_rows(x,names,source,target,contrast):
    rows=[]
    for j,name in enumerate(names):
        a=x[source,j];b=x[target,j];s=max(float(a.std()),1e-8);p=max(float(np.sqrt((a.var()+b.var())/2)),1e-8)
        rows.append(dict(contrast=contrast,feature=name,source_n=len(a),target_n=len(b),source_mean=float(a.mean()),target_mean=float(b.mean()),source_median=float(np.median(a)),target_median=float(np.median(b)),source_std=float(a.std()),target_std=float(b.std()),SMD_source=float((b.mean()-a.mean())/s),SMD_pooled=float((b.mean()-a.mean())/p),W1_source_std=float(wasserstein_distance(a,b)/s),source_p10=float(np.quantile(a,.1)),source_p90=float(np.quantile(a,.9)),target_p10=float(np.quantile(b,.1)),target_p90=float(np.quantile(b,.9))))
    return rows

def gaps(quality,name,split,target,tracks,groups,masks,family):
    rows=[];native=quality[name,'native_'+split]['coverage']
    for condition,mask in masks.items():
        for group in ('overall','change'):
            m=mask if group=='overall' else mask&(groups[target]==2)
            n=int(m.sum());nt=len(set(tracks[target][m]))
            row=dict(study=name,split=split,family=family,condition=condition,group=group,n=n,tracks=nt,thin=n<100 or nt<20)
            if n:
                r=np.stack([quality[name,f'random_seed{s}_final_{split}']['coverage'] for s in (1,2,3)])
                g=np.stack([quality[name,f'grouped_seed{s}_final_{split}']['coverage'] for s in (1,2,3)])
                rj=np.stack([quality[name,f'random_seed{s}_final_{split}']['J'] for s in (1,2,3)])
                gj=np.stack([quality[name,f'grouped_seed{s}_final_{split}']['J'] for s in (1,2,3)])
                row.update(native_coverage=float(native[m].mean()),random_coverage=float(r[:,m].mean()),grouped_coverage=float(g[:,m].mean()),grouped_minus_random_pp=float((g[:,m].mean()-r[:,m].mean())*100),grouped_minus_random_J=float((gj[:,m]-rj[:,m]).mean()),random_rescued=[int((v[m]&~native[m]).sum()) for v in r],random_lost=[int((~v[m]&native[m]).sum()) for v in r],grouped_rescued=[int((v[m]&~native[m]).sum()) for v in g],grouped_lost=[int((~v[m]&native[m]).sum()) for v in g])
            rows.append(row)
    return rows

def main():
    t=time.perf_counter();stage('FIXTURES_AND_SOURCE_LOCK');fixtures()
    guard.check();cachepath=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
    encpath=ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
    labelpath=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
    metapath=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
    old=read(SERIES/NAMES[2]/'old_results_manifest.json')['files'];files={}
    for p in (cachepath,encpath,labelpath,metapath):
        assert str(p) in old,('INPUT_NOT_PRIOR_LOCKED',str(p));files[str(p)]=old[str(p)]
    for p in (HERE/'DESIGN.md',HERE/'protocol.json',HERE/'CLAIM_INTENT.json',Path(__file__)):
        files[str(p)]=digest(p)
    verify(files)
    with np.load(encpath,allow_pickle=False) as z:ids=z['sample_id'];past=z['past'];tokens=z['tokens']
    assert len(ids)==28791 and len(np.unique(ids))==len(ids)
    with np.load(cachepath,allow_pickle=False) as z:
        assert np.array_equal(ids,z['sample_id']);groups=z['change_group'];base=z['candidate'][...,:24].reshape(-1,20,12,2)
    with np.load(labelpath,allow_pickle=False) as z:assert np.array_equal(ids,z['sample_id']);gt=z['future']
    with np.load(metapath,allow_pickle=False) as z:
        sort=np.argsort(z['sample_id']);take=sort[np.searchsorted(z['sample_id'][sort],ids)]
        assert np.array_equal(z['sample_id'][take],ids)
        sequence=z['sequence'][take];scene=z['scene_family'][take];ped=z['pedestrian_id'][take]
    tracks=np.array([f'{s}|{int(p)}' for s,p in zip(sequence,ped)])
    stage('FULL_THREE_SAVED_METRIC_REVIEW')
    studies,quality,others,records,cis,error=audit_studies(ids,gt,groups);add_files(files,others);verify(files)
    stage('FIXED_OBSERVABLE_FEATURES')
    p,c=kinetics(past,base);tok=tokens.astype(np.float64).mean(1)
    assert p.shape==(28791,10) and c.shape==(28791,10) and tok.shape==(28791,96)
    features={'past':p,'native_candidate':c,'frozen_tokens':tok,'all':np.c_[p,c,tok]}
    assert all(np.isfinite(v).all() for v in features.values())
    scalars=np.c_[p,c,np.linalg.norm(tok,axis=1),np.sqrt(np.mean(tokens.astype(np.float64)**2,axis=(1,2)))]
    names=['last_speed','mean_speed','std_speed','observed_accel_mean','observed_accel_p95','observed_length','observed_displacement','observed_abs_turn','observed_sin_turn','observed_cos_turn','candidate_CV_path_gap','candidate_CV_endpoint_gap','candidate_CV_endpoint_gap_p95','candidate_velocity_vobs_gap','candidate_speed_mean','candidate_endpoint_speed_std','candidate_endpoint_spread','candidate_path_spread','candidate_accel_mean','candidate_jerk_mean','token_mean_norm','token_RMS']
    np.savez_compressed(HERE/'observed_features.npz',sample_id=ids,scalar=scalars,scalar_names=np.array(names),past=p,native_candidate=c,frozen_tokens=tok,track=tracks,scene=scene)
    shifts=[];supports=[];paired=[]
    fit=np.arange(25912);cal=np.arange(25912,28791)
    for tag,src,tar in [('originalFIT_to_CAL',fit,cal),('universityFIT_to_CAL',fit[scene[fit]=='university'],cal)]:
        shifts+=shift_rows(scalars,names,src,tar,tag)
        for family,x in features.items():
            inside,row,d=support(x,src,tar,tracks);supports.append(dict(contrast=tag,family=family,**row))
    for study in studies:
        name=study['study'];split=study.pop('split');support_cache={}
        for targetname,target in [('outer',split['outer_eval_index']),('original_CAL',cal)]:
            for arm in ('random','grouped'):
                src=split[arm+'_train_index'];contrast=name+'|'+arm+'_to_'+targetname
                stage('DISTRIBUTION_SUPPORT',study=name,arm=arm,target=targetname)
                shifts+=shift_rows(scalars,names,src,target,contrast)
                for family,x in features.items():
                    inside,row,d=support(x,src,target,tracks);supports.append(dict(contrast=contrast,family=family,**row))
                    support_cache[arm,targetname,family]=inside
            for family in features:
                ri=support_cache['random',targetname,family];gi=support_cache['grouped',targetname,family]
                masks={'all':np.ones(len(target),bool),'both_supported':ri&gi,'grouped_unsupported':~gi,'random_supported_grouped_not':ri&~gi,'both_unsupported':~ri&~gi}
                paired+=gaps(quality,name,targetname,target,tracks,groups,masks,family)
    write('FULL_THREE_RECORDS.json',records);write('FULL_THREE_EXPLORATORY_CIS.json',cis)
    write('SUPPORT.json',supports);write('FEATURE_SHIFT.json',shifts);write('PAIRED_SUPPORT_GAPS.json',paired)
    with (HERE/'FEATURE_SHIFT.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(shifts[0]));writer.writeheader();writer.writerows(shifts)
    write('SOURCE_LOCK.json',dict(files=files,scope='all 3 original current sources,54 models,342 outputs,read inputs,docs/history; inherited full34k not all repeated'))
    stage('FINAL_ALL_READ_SOURCE_OUTPUT_SHA');verify(files)
    result=dict(status='PASS_FROZEN_DISTRIBUTION_DIAGNOSIS',studies=studies,records=len(records),original_exploratory_CIs=len(cis),saved_science_jobs=171,saved_quality_max_error=error,feature_rows=len(ids),feature_families={k:v.shape[1] for k,v in features.items()},shift_rows=len(shifts),support_rows=len(supports),paired_stratum_rows=len(paired),source_output_files=len(files),new_formal_count=0,new_model_forward=False,new_optimizer=False,new_full_clustering_replay=False,new_bootstrap=False,resource=guard.values(),elapsed_seconds=time.perf_counter()-t)
    write('RESULT.json',result);write('RESOURCE_USAGE.json',dict(status='PASS',**guard.values()))
    write('runtime_state.json',dict(status='COMPLETED_SAVED_ANALYSIS_PENDING',finished_at=datetime.now().astimezone().isoformat(),new_formal_count=0,no_worker_or_training=True))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        folder=HERE/'failures'/datetime.now().strftime('%Y%m%d_%H%M%S');folder.mkdir(parents=True,exist_ok=True)
        (folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        (folder/'source.py').write_bytes(Path(__file__).read_bytes())
        write('runtime_state.json',dict(status='STOP_FAILURE',failure=str(folder),resource=guard.values()))
        raise
