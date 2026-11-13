"""Analysis-only saved summary audit. No torch/model/forward/optimizer or old writes."""
import os,sys,json,hashlib,time,gc,ctypes
from pathlib import Path
import numpy as np,psutil
H=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(name,x):
    p=O/name;p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()),encoding='utf-8')
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%64==0:guard()
        assert sha(p)==h,('SHA',p)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
resource=dict(checks=0,peak_private=0,peak_RSS=0)
E=5700000000
def guard():
    p=psutil.Process().memory_info();v=Mem();v.length=ctypes.sizeof(v);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(v));resource['checks']+=1
    resource['peak_private']=max(resource['peak_private'],p.peak_pagefile);resource['peak_RSS']=max(resource['peak_RSS'],p.peak_wset)
    assert v.avail_phys>=1.3*max(E-p.rss,0) and v.avail_page>=1.3*max(E-p.private,0) and max(p.peak_pagefile,p.peak_wset)<1.3*E,'RESOURCE'
def close(a,b):assert np.max(np.abs(np.asarray(a,float)-np.asarray(b,float)))<=2e-12
def main():
    start=time.perf_counter()
    for n in ('worker_state.json','exit_status.json','ARITHMETIC_AUDIT.json','PRESERVATION.json','RESOURCE.json'):
        assert read(H/n)['status'] in ('SUCCESS','PASS')
    result=read(H/'RESULT.json');assert result['status']=='PASS_P1_MODE_PRESERVING_CONDITIONAL_REFINEMENT'
    source=read(H/'source_manifest.json')['files'];out=result['outputs_SHA'];verify(source);verify(out)
    jobs=read(H/'JOBS.json');runs=read(H/'RUNS.json');assert len(jobs)==52 and len(runs)==12
    rows=[];motionfail=[];protectfail=[];change=[];history=[]
    fields=['n','native_good','good','rescue','lost','coverage','lost_rate_native_good','ADE','FDE','J','native_ADE','native_FDE','native_J','distinct','effective_good','correction_mean','correction_max','motion','native_motion','motion_pass','owner_disagreement','owner_counts','owner_ties','native_owner_ties','nonowner_native_good','nonowner_good_lost','owner_repair_other_good_destroyed','native_owner_final_Q','support_bidirectional_mean']
    for key,row in jobs.items():
        guard();met=read(row['metrics']);cell=key.split('/')[0];split=key.split('/')[-1];seed=None if cell=='native' else int(key.split('/')[1][4:])
        with np.load(row['aux'],allow_pickle=False) as a:
            groups=a['groups'];good=a['good'].any(1);nv=a['native_vectors4'];ng=nv[:,0].astype(bool);ade=a['ADE'].min(1);fde=a['FDE'].min(1);jj=a['J'].min(1);ids=a['sample_id']
            for group in ('overall','stable','transition','change'):
                sel=np.ones(len(groups),bool) if group=='overall' else groups=={'stable':0,'transition':1,'change':2}[group];m=met[group];n=int(sel.sum())
                assert m['n']==n and m['good']==int(good[sel].sum()) and m['native_good']==int(ng[sel].sum())
                assert m['rescue']==int((sel&good&~ng).sum()) and m['lost']==int((sel&~good&ng).sum())
                for k,x in [('ADE',ade),('FDE',fde),('J',jj)]:close(m[k],x[sel].mean())
                assert np.array_equal(np.sort(m['rescue_IDs']),np.sort(ids[sel&good&~ng])) and np.array_equal(np.sort(m['lost_IDs']),np.sort(ids[sel&~good&ng]))
                ratios={k: m['motion'][k]/v if v else None for k,v in m['native_motion'].items()}
                fail=[k for k in m['motion'] if m['motion'][k]>1.01*m['native_motion'][k]]
                assert m['motion_pass']==(not fail)
                rec=dict(job=key,cell=cell,seed=seed,split=split,group=group,**{k:m[k] for k in fields},motion_ratios=ratios,motion_fail=fail,rescued_IDs=m['rescue_IDs'],lost_IDs=m['lost_IDs'])
                rows.append(rec)
                if group=='change':change.append(rec)
                if cell!='native' and fail:motionfail.append(dict(cell=cell,seed=seed,split=split,group=group,ratios=ratios,fail=fail))
                if cell!='native' and group in ('overall','stable','transition'):
                    bad=[k for k in ('ADE','FDE','J') if m[k]>1.01*m['native_'+k]]
                    if bad:protectfail.append(dict(cell=cell,seed=seed,split=split,group=group,fail=bad))
        del groups,good,nv,ng,ade,fde,jj,ids;gc.collect()
    paired={};orders={}
    for key,row in runs.items():
        folder=H/'runs'/key;man=read(folder/'run_manifest.json');hist=read(folder/'history.json');assert len(hist)==25 and sum(x['optimization_steps'] for x in hist)==1850
        cell,seed=key.split('/');orders.setdefault(seed,[x['batchorder_SHA'] for x in hist]);paired.setdefault(seed,man['paired_init_SHA'])
        assert orders[seed]==[x['batchorder_SHA'] for x in hist] and paired[seed]==man['paired_init_SHA']
        for r in hist:close(r['total'],r['task']+.05*r['diversity']+(r['anchor'] if cell in ('C','D') else 0))
        snapshots=read(folder/'snapshot_switching.json')
        history.append(dict(cell=cell,seed=seed,first_total=hist[0]['total'],last_total=hist[-1]['total'],last_task=hist[-1]['task'],last_anchor=hist[-1]['anchor'],last_disagreement_rate=hist[-1]['disagree']/hist[-1]['n'],snapshot_switching=snapshots,fixed_gradient_diagnostic=hist[-1]['fixed_FIT_coordinate_gradient']))
    stats=read(H/'STATISTICS.json')
    oc={cell:[next(r for r in change if r['cell']==cell and r['seed']==s and r['split']=='outer') for s in (42,43,44)] for cell in ('A','B','C','D')}
    measures=lambda r:np.array([r['good']/r['n'],r['lost']/r['native_good'],r['rescue']/(r['n']-r['native_good']),r['J']])
    coef={'B-A':[-1,1,0,0],'C-A':[-1,0,1,0],'D-A':[-1,0,0,1],'D-C':[0,0,-1,1],'D-B':[0,-1,0,1],'D-C-B+A':[1,-1,-1,1]}
    per=np.array([[measures(r) for r in oc[c]] for c in ('A','B','C','D')])
    for contrast,w in coef.items():
        x=np.einsum('csm,c->sm',per,w);close(x,stats['primary'][contrast]['per_seed']);close(x.mean(0),[r['estimate'] for r in stats['primary'][contrast]['rows']])
    d=stats['primary']['D-A'];r=d['rows'];per=np.asarray(d['per_seed']);douter=[x for x in rows if x['cell']=='D' and x['split']=='outer'];dmotion=[x for x in motionfail if x['cell']=='D']
    checks=dict(coverage_mean_ge2pp=r[0]['estimate']>=.02,coverage_each_seed_nondecreasing=bool((per[:,0]>=0).all()),coverage_family24_lower_positive=r[0]['CI_family'][0]>0,J_each_seed_nonworse=bool((per[:,3]<=0).all()),J_family24_upper_negative=r[3]['CI_family'][1]<0,rescue_mean_increases=r[2]['estimate']>0,lost_mean_not_increases=r[1]['estimate']<=0,D_native_coverage_each_seed_not_worse=all(x['good']>=x['native_good'] for x in oc['D']),D_native_coverage_family3_lower_positive=stats['absolute_native']['D']['coverage'][0]['CI_family'][0]>0,D_outer_each_group_native_lost_le1pct=all(x['lost_rate_native_good']<=.01 for x in douter),D_outer_protected_errors_pass=not any(x['cell']=='D' and x['split']=='outer' for x in protectfail),D_all4split_3seed_4group_five_motion_pass=not dmotion,complete_execution_audit=True)
    p2=all(checks.values())
    verify(source);verify(out)
    payload=dict(status='PASS',source_count=len(source),outputs_count=len(out),jobs=52,science_jobs=48,models=12,history_rows=300,steps=22200,rows=rows,change_rows=change,motion_failures=motionfail,protected_error_failures=protectfail,history=history,primary=stats,gate_checks=checks,P2_TRIGGERED=p2,not_all_model_gradient_or_cluster_CI_replay_this_supplement=True,worker_full_cluster_CI_independent_audit=read(H/'ARITHMETIC_AUDIT.json')['status'],all_saved_aux_counts_rescuelost_IDs_errors_and_CI_points_recomputed=True,source_output_SHA_initial_final_unchanged=True,resource=resource,elapsed_seconds=time.perf_counter()-start,helper_SHA=sha(__file__))
    write('REVIEW.json',payload);print(json.dumps(dict(status='PASS',P2_TRIGGERED=p2,checks=checks,motionfail_bycell={a:sum(x['cell']==a for x in motionfail) for a in ('A','B','C','D')},outputs=len(out),seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()

