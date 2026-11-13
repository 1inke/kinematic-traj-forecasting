"""Post-completion CPU saved-output review: no solver, Torch, model or tuning."""
import sys,json,hashlib,collections,time,ast
from pathlib import Path
N=Path(__file__).resolve().parents[1];R=Path(__file__).resolve().parent
sys.path.insert(0,str(N))
import common as c
import evaluate as e
import numpy as np
c.H=R
def main():
    started=time.perf_counter()
    source=c.read(N/'source_manifest.json')['files'];result=c.read(N/'RESULT.json');jobs=c.read(N/'JOBS.json');registry=c.read(N/'sample_registry.json')
    assert result['status']=='PASS_DIAGNOSTIC_ONLY_F0_SAFE_REACHABILITY'
    assert len(source)==5732 and c.digest(source)==result['source_digest'];c.verify(source);c.verify(result['outputs_SHA'])
    assert c.read(N/'worker_state.json')['status']==c.read(N/'exit_status.json')['status']=='SUCCESS'
    audit=c.read(N/'ARITHMETIC_AUDIT.json');assert audit['status']=='PASS' and audit['attempts']==len(audit['all_attempt_saved_arithmetic'])==640
    assert all(a['status']=='PASS' for a in audit['all_attempt_saved_arithmetic'])
    assert len(jobs)==640 and len(registry['selection'])==48
    status=collections.Counter();reasons=collections.Counter();rejections=collections.Counter();exception_ids=[];optimal=[];windows=[]
    saved_fields={};wall=[];strict_primal=[];global_error=0.;case_rows=[]
    for wi,r in enumerate(registry['selection'][:32]):
        cases=[];passing=[]
        for slot in range(20):
            row=jobs[f'{wi}:{slot}'];assert row['source_digest']==result['source_digest']
            assert row['sample_id']==r['sample_id'] and row['index']==r['index'] and row['slot']==slot
            c.verify(row['files'])
            zp=next(p for p in row['files'] if p.endswith('.npz'));jp=next(p for p in row['files'] if p.endswith('.json'))
            z=c.npz(zp);disk=c.read(jp)
            for k in disk:global_error=max(global_error,c.compare(disk[k],row[k]))
            assert int(z['sample_id'])==r['sample_id'] and int(z['slot'])==slot
            status[row['status']]+=1;wall.append(row['wall_seconds']);cases.append(row['status'])
            rec={k:row.get(k) for k in ('sample_id','slot','index','status','wall_seconds','valid_witness','primal_rejected')}
            if z['primal_present']:
                u=z['primal'];assert u.shape==(20,12,2) and np.isfinite(u).all()
                residual=e.primal_residual(z['native'],z['gt'],slot,u);global_error=max(global_error,c.compare(residual,row['primal_residual']))
                rec['primal_residual']=residual
                if all(v<=0 for k,v in residual.items() if k!='finite'):strict_primal.append(dict(sample_id=r['sample_id'],slot=slot))
            if 'primal_rejected' in row:rejections[row['primal_rejected']]+=1
            if 'controls' in z:
                expected_raw=np.arctanh(z['primal']*(1-2**-20)).astype(np.float32)
                assert np.array_equal(expected_raw,z['raw'])
                control=np.tanh(z['raw']).astype(float);control/=np.maximum(1,np.sqrt((control*control).sum(-1,keepdims=True)))
                c.close(control,z['controls'],2e-7)
                global_error=max(global_error,c.close(z['paths'],e.independent_paths(z['native'],z['controls'])))
                global_error=max(global_error,c.close(z['correction'],z['paths']-z['native']))
                cert=e.certificate(z['native'],z['gt'],slot,z,True);global_error=max(global_error,c.compare(cert,row['certificate']))
                assert row['valid_witness']==(cert['pass_literal'] and cert['nonzero'])
                reasons.update(cert['reasons']);rec['certificate']=cert
                assert not any(k in cert['reasons'] for k in cert['motion']),'Expected all4 reconstructed cases to pass five-motion'
                rawvel=np.diff(z['paths'],axis=1,prepend=np.zeros_like(z['paths'][:,:1]))/.4
                rawacc=np.diff(rawvel,axis=1)/.4
                nativevel=np.diff(z['native'],axis=1,prepend=np.zeros_like(z['native'][:,:1]))/.4
                nativeacc=np.diff(nativevel,axis=1)/.4
                delta=np.sqrt((rawacc*rawacc).sum(-1)).mean(1)-np.sqrt((nativeacc*nativeacc).sum(-1)).mean(1)
                global_error=max(global_error,c.close(delta,cert['all20_candidate_delta_accel_mean']))
                rec['other_slots_delta_accel_sum']=float(np.delete(delta,slot).sum())
                rec['target_delta_accel']=float(delta[slot])
            if row['valid_witness']:passing.append(slot)
            if row['status']=='SOLVER_EXCEPTION':exception_ids.append(dict(request_id=row['request_id'],sample_id=row['sample_id'],slot=slot,traceback=row['traceback']))
            if row['status']=='optimal':optimal.append(rec)
            case_rows.append(rec)
        windows.append(dict(**r,attempts=20,status_counts=dict(collections.Counter(cases)),valid_slots=passing,chosen_slot=min(passing) if passing else None))
        if wi%8==0:print('CPU_REVIEW_ATTEMPTS',20*(wi+1),flush=True)
    assert dict(status)==result['solver_status_counts']
    # Independently read the complete saved FIT population and original saved per-slot fields.
    z=c.npz(N/'portfolio.npz');assert len(z['sample_id'])==18909 and len(set(z['sample_id']))==18909
    assert np.array_equal(z['paths'],z['native']),'zero accepted cases must retain full native portfolio bitwise'
    recorded=c.read(N/'FULL_FIT_AUDIT.json');assert recorded['chosen']=={}
    native_aux=c.npz(N/'native_aux.npz');aux=c.npz(N/'portfolio_aux.npz')
    a,f,j,g,segments=c.errors(z['paths'],z['gt'],True);cg=c.clusters(z['paths'],g,True)
    for key,value in dict(ADE=a,FDE=f,J=j,good=g,clusters=cg,segments=segments,sample_id=z['sample_id']).items():
        if value.dtype.kind in 'biu':assert np.array_equal(value,aux[key]) and np.array_equal(value,native_aux[key])
        else:global_error=max(global_error,c.close(value,aux[key]),c.close(value,native_aux[key]))
    population={}
    for group,mask in c.masks(z['groups']):
        motion=c.motion(z['paths'][mask],True);ng=int(g[mask].any(1).sum())
        check=dict(n=int(mask.sum()),native_good=ng,good=ng,rescue=0,lost=0,coverage=float(g[mask].any(1).mean()),minADE=float(a[mask].min(1).mean()),minFDE=float(f[mask].min(1).mean()),minJ=float(j[mask].min(1).mean()),effective_good=float(cg[mask,1].mean()),distinct=float(cg[mask,0].mean()),singleJslot_segments=segments[mask].mean(0),motion=motion,native_motion=motion,all5motion_pass=True,rescued_ids=[],lost_ids=[])
        global_error=max(global_error,c.compare(check,result['records'][group]),c.compare(check,recorded['portfolio'][group]),c.compare(check,recorded['native'][group]))
        population[group]=check
    # All48 registry rows must refer to exact saved identity/track/scene/sequence, all success16 untouched.
    for r in registry['selection']:
        ii=r['index'];assert int(z['sample_id'][ii])==r['sample_id']
        for key in ('track','scene','sequence'):assert str(z[key][ii])==r[key]
        assert bool(g[ii].any())==r['native_good']
    assert len({r['track'] for r in registry['selection']})==48
    preserved=c.read(N/'PRESERVATION.json');assert preserved['status']=='PASS' and preserved['initial_final_SHA_unchanged']
    resource=c.read(N/'RESOURCE.json');assert resource['status']=='PASS' and resource['child_max_peak_private']>10*2**20
    assert not result['SAFE_WITNESS_PRESENT'] and result['rescued_windows']==0
    c.verify(source);c.verify(result['outputs_SHA'])
    assert not any(k in sys.modules for k in ('torch','cvxpy','clarabel'))
    output=dict(status='PASS',source_count=len(source),outputs_verified=len(result['outputs_SHA']),input_source_and_completed_output_SHA_before_after=True,
      attempts=640,registry=48,independent_saved_JSON_and_NPZ_fields=True,raw_from_fixed_primal_shrink_float32_EXACT=True,
      independent_controls_recurrence_all_actual_raw_cases=True,completeFIT_original_SciPy_complete_link_and_per_slot_quality_and_fourgroups_motion=True,
      max_arithmetic_error=global_error,solver_status_counts=dict(status),primal_rejections=dict(rejections),actual_reconstruction_rejections=dict(reasons),
      strict_continuous_primal_cases=strict_primal,optimal_cases=optimal,exception_cases=exception_ids,windows=windows,
      population=population,elapsed_seconds=time.perf_counter()-started,review_resource=c.RESOURCE,no_Torch_model_forward_optimizer_solver_or_retraining=True,same_assistant_saved_arithmetic_not_external_peer_review=True)
    c.write(R/'REVIEW.json',output);c.write(R/'ALL_CASES.json',case_rows)
    print('CPU_REVIEW_PASS',global_error,dict(status),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_CPU_COMPLETION_REVIEW');raise
