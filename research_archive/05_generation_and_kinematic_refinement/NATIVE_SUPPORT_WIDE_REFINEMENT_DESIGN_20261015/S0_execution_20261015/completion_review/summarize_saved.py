"""Independent CPU saved-slot/count/certificate supplement; no model or new solver."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as c
import numpy as np,time,gc

def main():
    t=time.perf_counter();files=c.read(c.H/'source_manifest.json')['files'];c.verify(files)
    result=c.read(c.H/'RESULT.json');audit=c.read(c.H/'ARITHMETIC_AUDIT.json');pres=c.read(c.H/'PRESERVATION.json');res=c.read(c.H/'RESOURCE.json');w=c.read(c.H/'worker_state.json');exit=c.read(c.H/'exit_status.json')
    assert w['status']==exit['status']=='SUCCESS' and audit['status']==pres['status']==res['status']=='PASS'
    assert result['status']=='PASS_DIAGNOSTIC_ONLY_NATIVE_SUPPORT_SAFE_OPPORTUNITY' and result['scientific_jobs']==4
    c.verify(result['outputs_SHA']);rows=[];allids=[];maximum=0.
    for name,row in result['records'].items():
        c.valid_done(c.read(c.H/'jobs'/f'{name}.done.json'),result['source_digest']);z=c.npz(row['output']);allids.extend(z['sample_id'].tolist());N=len(z['native']);ng=z['native20_slot_good'].any(1)
        # Explicit 2-component arithmetic, not the production np.linalg.norm.
        for port in c.PORTS:
            path=z['native' if port=='native20' else port];delta=path-z['gt'][:,None];e=np.sqrt(delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1]);a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);g=(a<=.2)&(f<=.4)
            for suf,val in [('ADE',a),('FDE',f),('J',j)]:maximum=max(maximum,c.close(val,z[port+'_slot_'+suf]))
            assert np.array_equal(g,z[port+'_slot_good'])
            for group,mask in c.masks(z['groups']):
                report=row['metrics'][port][group];gg=g.any(1);assert report['n']==int(mask.sum()) and report['good']==int((gg&mask).sum())
                assert report['rescue']==int((gg&~ng&mask).sum()) and report['lost']==int((~gg&ng&mask).sum())==0
                for key,val in [('ADE',a[mask].min(1).mean()),('FDE',f[mask].min(1).mean()),('J',j[mask].min(1).mean())]:maximum=max(maximum,c.close(val,report[key]))
        split=dict(split=name,n=N,groups={})
        for group,mask in c.masks(z['groups']):
            cert=row['certificate'][group];alpha=z['alpha'][mask];roots=z['roots'][mask];minimum=roots.min(-1);bind=roots==minimum[...,None]
            assert cert['candidates']==int(alpha.size) and cert['zero_alpha']==int((alpha==0).sum()) and cert['fallback']==int(z['fallback'][mask].sum())
            assert np.array_equal(bind.sum((0,1)),cert['binding_counts_33'])
            assert np.array_equal(z['zero_budget'][mask].sum((0,1)),cert['zero_budget_counts_33']) and np.array_equal(z['fallback_reasons'][mask].sum((0,1)),cert['fallback_reason_counts_33'])
            maximum=max(maximum,c.close(np.quantile(alpha,[0,.01,.1,.25,.5,.75,.9,.99,1]),cert['alpha_quantiles']))
            nonzero=np.any(z['safe20'][mask]!=z['native'][mask],axis=(-1,-2));assert int(nonzero.sum())==cert['actual_nonzero_correction']
            delta=z['safe20'][mask]-z['native'][mask];corr=np.sqrt((delta*delta).sum(-1)).max(-1)
            summary=dict(n=int(mask.sum()),candidates=int(alpha.size),native_good=row['metrics']['native20'][group]['good'],safe4_good=row['metrics']['safe4'][group]['good'],safe20_good=row['metrics']['safe20'][group]['good'],relaxed_ball_good=cert['relaxed_ball_good_windows'],zero_alpha=cert['zero_alpha'],nonzero_correction=cert['actual_nonzero_correction'],fallback=cert['fallback'],alpha_quantiles=cert['alpha_quantiles'],correction_max_m=float(corr.max()),correction_mean_candidate_max_m=float(corr.mean()),binding_by_order=[int(bind[...,:12].sum()),int(bind[...,12:23].sum()),int(bind[...,23:].sum())],zero_budget_candidates_by_order=[int(z['zero_budget'][mask][...,:12].any(-1).sum()),int(z['zero_budget'][mask][...,12:23].any(-1).sum()),int(z['zero_budget'][mask][...,23:].any(-1).sum())],roots_min_quantiles=np.quantile(minimum,[0,.5,.9,.99,1]).tolist(),fallback_alpha_positive=int((z['fallback'][mask]&(z['pre_fallback_alpha'][mask]>0)).sum()),native_to_safe20_ADE_gain_percent=100*(1-row['metrics']['safe20'][group]['ADE']/row['metrics']['native20'][group]['ADE']),native_to_safe20_FDE_gain_percent=100*(1-row['metrics']['safe20'][group]['FDE']/row['metrics']['native20'][group]['FDE']),all_motion_literal=all(row['metrics'][p][group]['motion_literal_1pct'] for p in c.PORTS),per_slot_rescue_windows=cert['per_slot_rescue_windows'],binding_counts_33=cert['binding_counts_33'],zero_budget_counts_33=cert['zero_budget_counts_33'])
            split['groups'][group]=summary
        rows.append(split);del z;gc.collect()
    assert len(allids)==24679 and len(set(allids))==24679
    d=result['decision'];g=rows[0]['groups']['change'];assert g['n']==4725 and g['native_good']==1937 and 100*(g['safe20_good']-1937)==d['integer_lhs'] and d['integer_rhs']==23625
    assert d['status']=='STOP_SAFE_OPPORTUNITY_UNDER_REGISTERED_MAP' and not d['S1_preflight_allowed']
    c.verify(files);c.verify(result['outputs_SHA'])
    c.write('completion_review/SUMMARY.json',dict(status='PASS',records=rows,decision=d,source_count=len(files),maxerror=maximum,worker_started_at=w['started_at'],worker_finished_at=w['finished_at'],production_resource=res,core_elapsed_seconds=result['elapsed_seconds']))
    c.write('completion_review/REVIEW.json',dict(status='PASS',jobs=4,slot_arrays_checked=48,uniqueIDs=24679,primary_outputs=len(result['outputs_SHA']),done_contracts=4,saved_slot_metrics_and_all_group_counts_certificate_recomputed=True,maxerror=maximum,production_full_cluster_motion_ray_audit='ARITHMETIC_AUDIT.json PASS max0, not repeated by this supplement',source1863_and_outputs_initial_final_SHA_PASS=True,no_model_forward_optimizer_newdata=True,not_external_review=True,source_digest=c.digest(files),review_script_SHA=c.sha(__file__),resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-t))
    print('COMPLETION_SUPPLEMENT PASS',maximum,len(files),flush=True)
    for row in rows:
        for g,r in row['groups'].items():print(row['split'],g,r,flush=True)

if __name__=='__main__':main()
