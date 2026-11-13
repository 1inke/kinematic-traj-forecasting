"""CPU/NumPy saved-data completion summary; no torch or model replay."""
import sys,time,gc,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
import common as c
import numpy as np

def main():
    start=time.perf_counter();c.guard('startup')
    lock=c.read(c.OUT/'source_manifest.json');c.verify(lock['files'])
    required={name:c.read(c.OUT/(name+'.json')) for name in ('RESULT','RECOVERY_RESULT','ARITHMETIC_AUDIT','PRESERVATION','RESOURCE','worker_state','exit_status')}
    assert required['RESULT']['status']=='PASS_U0_UNPROJECTED_TEMPORAL_REFINEMENT'
    assert required['RECOVERY_RESULT']['status']=='SUCCESS'
    assert all(required[k]['status']=='PASS' for k in ('ARITHMETIC_AUDIT','PRESERVATION','RESOURCE'))
    assert all(required[k]['status']=='SUCCESS' for k in ('worker_state','exit_status'))
    jobs=required['RESULT']['jobs'];assert len(jobs)==28
    base={s:c.npz(c.PARENT/f'native/{s}_aux.npz') for s in c.SIZES}
    rows=[];subsets=[];native=[];motionfail=[];maxerror=0.;all_ids={}
    for key,row in jobs.items():
        c.guard();c.valid_done(row,required['RESULT']['parent_source_digest'])
        split=key.split('/')[-1];a=c.npz(row['aux']);metrics=c.read(row['metrics']);b=base[split]
        assert np.array_equal(a['sample_id'],b['sample_id']) and np.array_equal(a['groups'],b['groups'])
        if row['arm']=='native20':all_ids[split]=a['sample_id'];native.append(dict(split=split,groups={g:{k:r[k] for k in ('n','native_good','native_ADE','native_FDE','native_J')} for g,r in metrics.items()}));continue
        arm,seed=key.split('/')[:2];seed=int(seed[4:]);good=a['good'].any(1);bgood=b['good'].any(1)
        for group,mask in c.masks(a['groups']):
            r=metrics[group];rescue=mask&good&~bgood;lost=mask&~good&bgood
            counts=dict(n=int(mask.sum()),good=int((mask&good).sum()),native_good=int((mask&bgood).sum()),rescue=int(rescue.sum()),lost=int(lost.sum()))
            assert all(r[k]==v for k,v in counts.items());assert np.array_equal(a['sample_id'][rescue],r['rescue_IDs']) and np.array_equal(a['sample_id'][lost],r['lost_IDs'])
            for column,name in enumerate(('coverage','ADE','FDE','J')):
                value=float(a['vectors4'][mask,column].mean());maxerror=max(maxerror,c.close(value,r[name]))
            harm={k:100*(r['motion'][k]/r['native_motion'][k]-1) if r['native_motion'][k] else (0 if r['motion'][k]==0 else None) for k in r['motion']}
            failures=[k for k in r['motion'] if r['motion'][k]>1.01*r['native_motion'][k]]
            cell=dict(job=key,arm=arm,seed=seed,split=split,group=group,**counts,coverage_pp_change=100*(r['good']-r['native_good'])/r['n'],ADE=r['ADE'],FDE=r['FDE'],J=r['J'],ADE_gain_percent=100*(1-r['ADE']/r['native_ADE']),FDE_gain_percent=100*(1-r['FDE']/r['native_FDE']),lost_percent_native_good=100*r['lost_rate_native_good'],motion_harm_percent=harm,motion_failure_components=failures,motion_pass=r['motion_pass'],effective_good=r['effective_good'],distinct=r['distinct'],correction_mean=r['correction_mean'],correction_p95=r['correction_p95'],correction_max=r['correction_max'],Q_segments=r['Q_segments'],J_segments=r['J_segments'],native_success_mean_vectors4=r['native_success_mean_vectors4'],native_failure_mean_vectors4=r['native_failure_mean_vectors4'])
            rows.append(cell)
            if failures:motionfail.append(dict(job=key,group=group,components=failures,harm_percent=harm))
            for name,sel in (('rescue',rescue),('lost',lost),('retained',mask&good&bgood),('still_bad',mask&~good&~bgood)):
                if not sel.any():continue
                diff=a['window_motion5'][sel]-b['window_motion5'][sel]
                subsets.append(dict(job=key,group=group,subset=name,n=int(sel.sum()),sample_IDs=a['sample_id'][sel] if name in ('rescue','lost') else None,window_motion5_mean=a['window_motion5'][sel].mean(0),native_window_motion5_mean=b['window_motion5'][sel].mean(0),delta_window_motion5_mean=diff.mean(0),fraction_windows_any_motion_above_native1p01=float(np.any(a['window_motion5'][sel]>1.01*b['window_motion5'][sel],axis=1).mean()),not_pooled_motion_gate=True))
        del a,metrics;gc.collect()
    ids=np.concatenate(list(all_ids.values()));assert len(ids)==24679 and len(np.unique(ids))==24679
    histories=[]
    for arm in ('free_position','integrated_acceleration'):
        for seed in (42,43,44):
            h=c.read(c.PARENT/f'runs/{arm}/seed{seed}/history.json');assert len(h)==25 and sum(r['optimization_steps'] for r in h)==1850
            histories.append(dict(arm=arm,seed=seed,epochs=25,steps=1850,first_total=h[0]['total'],last_total=h[-1]['total'],total_decrease_percent=100*(1-h[-1]['total']/h[0]['total']),first_quality=h[0]['quality'],last_quality=h[-1]['quality'],first_diversity=h[0]['diversity'],last_diversity=h[-1]['diversity'],batch_preupdate_not_population_oracle=True))
    c.verify(lock['files'])
    c.write(OUT/'SUMMARY.json',dict(status='PASS_SAVED_SUMMARY',native=native,rows=rows,subset_rows=subsets,history=histories,bootstrap=c.read(c.PARENT/'BOOTSTRAP.json'),all_motion_failures=motionfail,quality_scalar_max_error=maxerror,source_count=len(lock['files']),source_initial_final_unchanged=True,unique_IDs=24679,no_torch_forward_backward_optimizer=True,not_all_geometry_replay=True,resource=dict(c.RESOURCE),elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(dict(status='PASS',scalar_max_error=maxerror,change=[r for r in rows if r['group']=='change'],histories=histories,motion_failed_cells=len(motionfail),motion_total_cells=len(rows)),ensure_ascii=False))

if __name__=='__main__':main()
