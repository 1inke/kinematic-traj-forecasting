"""Post-completion CPU saved arithmetic; no torch, model, optimizer or solver replay."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as c,evaluate as e
import numpy as np,time
def main():
    start=time.perf_counter();result=c.read(c.H/'RESULT.json');assert result['attempts']==640
    for name in ('worker_state.json','exit_status.json','ARITHMETIC_AUDIT.json','PRESERVATION.json','RESOURCE.json'):
        z=c.read(c.H/name);assert z['status'] in ('SUCCESS','PASS'),(name,z['status'])
    files=c.read(c.H/'source_manifest.json')['files'];c.verify(files);c.verify(result['outputs_SHA']);jobs=c.read(c.H/'JOBS.json');assert len(jobs)==640
    counters=dict(reconstructable=0,strict=0,robust=0,primal_internal_pass=0,component_unreconstructable=0);maxerr=0.;witnesses={}
    for key,row in jobs.items():
        c.verify(row['files']);z=c.npz(next(p for p in row['files'] if p.endswith('.npz')))
        if bool(z['primal_present']):
            u=z['primal'];finite=np.isfinite(u).all();defined=finite and (np.abs(u)<1).all()
            if defined:
                counters['reconstructable']+=1;assert 'raw' in z
                assert np.array_equal(z['raw'],np.arctanh(u).astype(np.float32))
                maxerr=max(maxerr,c.close(z['paths'],e.independent_paths(z['native'],z['controls'])))
                r=e.primal_residual(z['native'],z['gt'],int(z['slot']),u,row.get('gamma',0));c.compare(r,row['primal_residual'])
                primalpass=all(r[k]<=0 for k in ('gamma_low_excess','gamma_high_excess','max_internal_radius_excess','max_position_excess','margin_ADE_excess','margin_FDE_excess','accel_mean_excess','jerk_mean_excess'))
                assert primalpass==row['primal_internal_pass'];counters['primal_internal_pass']+=primalpass
                cert=e.certificate(z['native'],z['gt'],int(z['slot']),z,True);c.compare(cert,row['certificate']);strict=cert['pass_literal'] and cert['nonzero'];margin=min(1-cert['ADE']/.2,1-cert['FDE']/.4)
                assert strict==row['valid_witness'] and strict==row['STRICT_ACTUAL_WITNESS'];assert (strict and margin>=1e-4)==row['ROBUST_MARGIN_WITNESS'];c.close(margin,row['actual_margin'])
                counters['strict']+=strict;counters['robust']+=strict and margin>=1e-4
                if strict:witnesses.setdefault(row['index'],[]).append((margin,row['slot'],row['sample_id']))
            else:counters['component_unreconstructable']+=1
    fit=c.load_fit();p=c.npz(c.H/'portfolio.npz');chosen=c.read(c.H/'FULL_FIT_AUDIT.json')['chosen'];assert len(chosen)==len(witnesses)==result['rescued_windows']
    selected=[]
    for index,ww in witnesses.items():
        best=sorted(ww,key=lambda x:(-x[0],x[1]))[0];assert chosen[str(index)]==best[1]
        selected.append(dict(sample_id=int(best[2]),index=int(index),slot=int(best[1]),actual_margin=float(best[0]),strict=True,robust=best[0]>=1e-4))
    rr,aa=e.portfolio(p,p['paths'],True);nr,na=e.portfolio(p,p['native'],True);c.compare(rr,result['records']);aux=c.npz(c.H/'portfolio_aux.npz')
    for k in aa:maxerr=max(maxerr,c.close(aa[k],aux[k]))
    assert all(x['all5motion_pass'] for x in rr.values())==result['PORTFOLIO_GLOBAL_SAFE']
    c.verify(files);c.verify(result['outputs_SHA'])
    review=dict(status='PASS',source_count=len(files),source_digest=c.digest(files),outputs_verified=len(result['outputs_SHA']),attempts=640,counters=counters,max_saved_arithmetic_error=maxerr,all640_raw_no_postshrink_and_gamma_residual_label_exact=True,fullFIT_quality_original_complete_link_five_motion_recomputed=True,maxmargin_then_physical_slot_portfolio_selection_exact=True,selected=selected,native_records=nr,portfolio_records=rr,elapsed_seconds=time.perf_counter()-start,helper_SHA=c.sha(__file__),same_assistant_CPU_not_external=True,no_torch_model_gradient_solver_replay=True)
    c.write('completion_review/REVIEW.json',review);print('W0_COMPLETION_REVIEW_PASS',counters,maxerr,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_W0_COMPLETION_REVIEW');raise


