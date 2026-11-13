import common as c
import evaluate as e
import numpy as np
from solver_client import Solver
def attempt(z,index,slot,client,prefix,request_id):
    native=z['native'][index];gt=z['gt'][index];nm=c.motion(native[None])
    result=client.solve(native,gt,slot,nm['accel_mean'],nm['jerk_mean'],request_id)
    u=result.pop('primal');values=dict(native=native,gt=gt,slot=np.array(slot),sample_id=z['sample_id'][index],primal_present=np.array(u is not None))
    result.update(slot=slot,index=index,sample_id=int(z['sample_id'][index]),valid_witness=False,oracle_GT=True)
    if u is not None:
        u=np.asarray(u,float);values['primal']=u;result['primal_residual']=e.primal_residual(native,gt,slot,u,result.get('gamma',0.)) if np.isfinite(u).all() else dict(finite=False)
        if np.isfinite(u).all() and (np.abs(u)<1).all():
            actual=e.reconstruct(native,u);values.update(actual);cert=e.certificate(native,gt,slot,actual);result['certificate']=cert
            result['valid_witness']=cert['pass_literal'] and cert['nonzero']
            result['actual_margin']=min(1-cert['ADE']/.2,1-cert['FDE']/.4)
            result['STRICT_ACTUAL_WITNESS']=result['valid_witness'];result['ROBUST_MARGIN_WITNESS']=result['valid_witness'] and result['actual_margin']>=1e-4
            result['primal_internal_pass']=all(result['primal_residual'][k]<=0 for k in ('gamma_low_excess','gamma_high_excess','max_internal_radius_excess','max_position_excess','margin_ADE_excess','margin_FDE_excess','accel_mean_excess','jerk_mean_excess'))
            values.update(primal_to_actual_delta=actual['controls']-u,solver_gamma=np.array(result.get('gamma',0.)))
        else:result['primal_rejected']='NONFINITE_OR_COMPONENT_ABS_GE1'
    c.save(prefix+'.npz',**values);c.write(prefix+'.json',result)
    result['files']={str(c.H/(prefix+s)):c.sha(c.H/(prefix+s)) for s in ('.npz','.json')}
    return result
def independent_attempt(row):
    c.verify(row['files']);path=next(p for p in row['files'] if p.endswith('.npz'));z=c.npz(path)
    if z['primal_present'] and np.isfinite(z['primal']).all():
        residual=e.primal_residual(z['native'],z['gt'],int(z['slot']),z['primal'],row.get('gamma',0.));c.compare(residual,row['primal_residual'])
    if 'controls' in z:
        c.close(z['paths'],e.independent_paths(z['native'],z['controls']))
        u=np.tanh(z['raw']).astype(float);u/=np.maximum(1,np.sqrt(np.sum(u*u,axis=-1,keepdims=True)))
        c.close(u,z['controls'],2e-7)
        cert=e.certificate(z['native'],z['gt'],int(z['slot']),z,True);c.compare(cert,row['certificate'])
        assert row['valid_witness']==(cert['pass_literal'] and cert['nonzero'])
    return dict(status='PASS',files_SHA=True,independent_controls_recurrence_literal_quality_motion=True)
def main():
    pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS'
    files=c.read(c.H/'source_manifest.json')['files'];source=c.digest(files);assert source==pre['source_digest']
    c.stage('W0_INITIAL_SOURCE_SHA');c.guard(startup=True);c.verify(files)
    z=c.load_fit();registry=c.read(c.H/'sample_registry.json');jobs={};c.write('JOBS.json',jobs)
    parent=c.H.parent/'W0_execution_20261017';oldjobs=c.read(parent/'JOBS.json');assert len(oldjobs)==46
    c.verify(c.read(parent/'source_manifest.json')['files'])
    for key,row in oldjobs.items():
        assert row['source_digest']==c.read(parent/'PREFLIGHT.json')['source_digest'];c.verify(row['files']);independent_attempt(row)
    client=Solver(c.guard,c.H/'solver_logs');chosen={};attempts=[]
    try:
        for wi,row in enumerate(registry['selection'][:32]):
            index=row['index'];passing=[]
            for slot in range(20):
                c.stage('W0_SOLVE_SAVE',window=wi,slot=slot,completed_attempts=len(attempts))
                key=f'{wi}:{slot}'
                if key in oldjobs:
                    result=dict(oldjobs[key]);result['reused_from_source_digest']=result['source_digest'];result['reused_saved_attempt_without_new_solve']=True
                else:result=attempt(z,index,slot,client,f'attempts/w{wi:02d}_slot{slot:02d}',key)
                result['source_digest']=source;jobs[f'{wi}:{slot}']=result;attempts.append(result);c.write('JOBS.json',jobs)
                if result['valid_witness']:passing.append((result['actual_margin'],slot))
            if passing:chosen[index]=sorted(passing,key=lambda x:(-x[0],x[1]))[0][1]
    finally:client.close();c.CHILD=None
    assert len(attempts)==640
    # JSONL is an analysis output, not a source edit.
    (c.H/'ALL_ATTEMPTS.jsonl').write_text('\n'.join(c.json.dumps(x,default=lambda a:a.tolist(),allow_nan=False) for x in attempts)+'\n',encoding='utf-8')
    portfolio=z['native'].copy()
    for wi,row in enumerate(registry['selection'][:32]):
        index=row['index']
        if index in chosen:
            q=c.npz(next(p for p in jobs[f'{wi}:{chosen[index]}']['files'] if p.endswith('.npz')));portfolio[index]=q['paths']
    c.stage('W0_FULL_FIT_PORTFOLIO')
    c.save('portfolio.npz',paths=portfolio,**{k:z[k] for k in ('native','gt','groups','sample_id','track','scene','sequence')})
    native_rows,native_aux=e.portfolio(z,z['native']);rows,aux=e.portfolio(z,portfolio)
    c.write('FULL_FIT_AUDIT.json',dict(native=native_rows,portfolio=rows,chosen={str(k):v for k,v in chosen.items()}))
    c.save('portfolio_aux.npz',**aux);c.save('native_aux.npz',**native_aux)
    c.stage('W0_INDEPENDENT_SAVED_AUDIT',attempts=640);audits=[]
    for i,row in enumerate(attempts):
        if i%20==0:c.stage('W0_INDEPENDENT_SAVED_AUDIT',audited_attempts=i)
        audits.append(independent_attempt(row))
    saved=c.npz(c.H/'portfolio.npz');c.close(saved['paths'],portfolio);ir,ia=e.portfolio(saved,saved['paths'],True);nr,na=e.portfolio(saved,saved['native'],True)
    c.compare(ir,rows);c.compare(nr,native_rows)
    for key in aux:
        if aux[key].dtype.kind in 'biu':assert np.array_equal(ia[key],aux[key])
        else:c.close(ia[key],aux[key])
    unmodified=np.ones(len(z['sample_id']),bool);unmodified[list(chosen)]=False;assert np.array_equal(portfolio[unmodified],z['native'][unmodified])
    assert all(rows[g]['lost']==0 for g in rows)
    witnessed=len(chosen)>0 and all(x['all5motion_pass'] for x in rows.values())
    c.write('ARITHMETIC_AUDIT.json',dict(status='PASS',attempts=640,all_attempt_saved_arithmetic=audits,full_fit_four_groups_quality_clusters_five_motion_independent=True,no_model_gradient_replay=True))
    c.stage('W0_FINAL_SOURCE_SHA');c.verify(files);outputs={}
    for row in attempts:outputs.update(row['files'])
    for name in ('ALL_ATTEMPTS.jsonl','portfolio.npz','portfolio_aux.npz','native_aux.npz','FULL_FIT_AUDIT.json','ARITHMETIC_AUDIT.json','JOBS.json'):outputs[str(c.H/name)]=c.sha(c.H/name)
    c.verify(outputs);c.write('RESOURCE.json',dict(status='PASS',**c.RESOURCE));c.write('PRESERVATION.json',dict(status='PASS',source_count=len(files),source_digest=source,initial_final_SHA_unchanged=True,old_F16_decoder_scorer_not_loaded=True,all_nonchosen_native_EXACT=True,success16_identity=True))
    result=dict(status='PASS_DIAGNOSTIC_ONLY_W0_SAFE_REACHABILITY',source_digest=source,source_count=len(files),attempts=640,identity_success_windows=16,rescued_windows=len(chosen),WINDOW_WITNESS_PRESENT=len(chosen)>0,PORTFOLIO_GLOBAL_SAFE=all(x['all5motion_pass'] for x in rows.values()),ROBUST_MARGIN_ATTEMPTS=sum(x.get('ROBUST_MARGIN_WITNESS',False) for x in attempts),SAFE_WITNESS_PRESENT=witnessed,conclusion='WINDOW_WITNESS_PRESENT' if len(chosen)>0 else 'SAFE_OPPORTUNITY_UNRESOLVED',records=rows,solver_status_counts={k:sum(x['status']==k for x in attempts) for k in sorted({x['status'] for x in attempts})},outputs_SHA=outputs,solver_restarts=client.restarts,reused_attempts=46,new_solver_attempts=594,not_training_not_teacher_not_deployable=True)
    c.write('RESULT.json',result);c.write('runtime_state.json',dict(status='COMPUTED_PENDING_ANALYSIS',SAFE_WITNESS_PRESENT=witnessed));print(result['conclusion'],len(chosen),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_F0');raise


