"""Explicit per-port parameter error/all49 existing-label utility, no new path enumeration."""
import time
import numpy as np
import attribution_core as c

def main():
    z=c.load_npz(c.ROOT/'derived_ports.npz');a=c.space('END')
    target=z['target'];exact=a.normalized[z['teacher_action']]
    params=np.stack((np.zeros_like(target),z['continuous_param'],a.normalized[z['action_id'][2]],a.normalized[z['action_id'][3]],exact))
    errors=((params-target[None])**2).mean(-1);exact_errors=((params-exact[None])**2).mean(-1)
    assert np.allclose(errors[0],z['initial_error'],atol=2e-12,rtol=0)
    assert np.array_equal(errors[1],z['final_error'])
    rows=[]
    for s,port in enumerate(c.PORTS):
        for i in range(128):
            for k in range(20):
                rows.append(dict(port=port,sample_id=int(z['sample_id'][i]),slot=k,parameter_x=params[s,i,k,0],parameter_y=params[s,i,k,1],
                    parameter_MSE_vs_actual_float32_teacher=errors[s,i,k],parameter_MSE_vs_exact_END49_teacher=exact_errors[s,i,k],
                    ADE=z['ADE'][s,i,k],FDE=z['FDE'][s,i,k],J=z['J'][s,i,k],good=z['good'][s,i,k],
                    action_id=int(z['action_id'][s,i,k]),seven_failed_bits=''.join(map(str,z['failed'][s,i,k].astype(int))),
                    C_only_diagnostic=port=='C',oracle_uses_GT=port=='O'))
    c.csv_new(c.ROOT/'five_port_candidate_parameters.csv',rows)
    # Full fixed action table is existing saved J/safe, not any new END/TIME geometry run.
    teacher_distance=np.linalg.norm(a.normalized[None,None]-exact[:,:,None],axis=-1)
    prediction_distance=np.linalg.norm(a.normalized[None,None]-z['continuous_param'][:,:,None],axis=-1)
    Jgap=z['J_matrix']-z['J'][4,:,:,None]
    c.npz_new(c.ROOT/'all49_saved_action_utility.npz',sample_id=z['sample_id'],action_normalized=a.normalized,
        safe=z['safe'],J=z['J_matrix'],safe_oracle_Jgap=Jgap,parameter_distance_to_teacher=teacher_distance,
        parameter_distance_to_continuous_prediction=prediction_distance)
    # Independent action/slot loop checks every125440 table entries from saved sources.
    mx=0.
    for i in range(128):
        for k in range(20):
            oracle_J=z['J'][4,i,k]
            for ac in range(49):
                dd=np.sqrt(sum(float(a.normalized[ac,d]-exact[i,k,d])**2 for d in range(2)))
                err=abs(dd-teacher_distance[i,k,ac]);mx=max(mx,err);assert err<=2e-12
                assert abs(Jgap[i,k,ac]-(z['J_matrix'][i,k,ac]-oracle_J))<=2e-12
    c.write_json(c.ROOT/'FIVE_PORT_PARAMETER_SUPPLEMENT.json',dict(status='PASS_ALL12800_PORT_ERRORS_AND125440_EXISTING_ACTION_UTILITIES',
        source_files={str(c.ROOT/'five_port_parameter_supplement.py'):c.sha(c.ROOT/'five_port_parameter_supplement.py')},
        candidate_ports=12800,existing_saved_action_cells=125440,new_path_enumerations=0,
        old_raw_param_target='exact END49 normalized float64',actual_training_target='old label cast float32 then exact float64 for saved arithmetic',
        independent_scalar_loop=True,max_error=mx,no_Torch=True))
    print('All five-port parameter errors and existing49-action utilities checked',flush=True)

if __name__=='__main__':main()
