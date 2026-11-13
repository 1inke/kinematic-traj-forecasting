"""Exactly 48 independent A0-state interventions, not accumulated training."""
from task_adapter import *
from optimizer_step import step

def decision(records):
    means={};valid_counts={};conflicts={};ratios=[];zero=[]
    for s in SEEDS:
        rows=[r for r in records if r['seed']==s];valid=[r for r in rows if r['valid_double_group']]
        valid_counts[str(s)]=len(valid);conflicts[str(s)]=sum(any(v>r['arms']['U']['solver_tolerance'] for v in r['arms']['U']['raw_Gd0']) for r in valid)/len(valid) if valid else None
        means[str(s)]={arm:float(np.mean([r['arms'][arm]['h'] for r in valid])) if valid else None for arm in ARMS}
        ratios.extend(r['arms']['P']['norm_ratio'] for r in valid if r['arms']['U']['d0_norm']!=0)
        zero.extend(r['arms']['P']['exact_zero_P'] for r in valid)
    avg={a:float(np.mean([means[str(s)][a] for s in SEEDS])) if all(means[str(s)][a] is not None for s in SEEDS) else None for a in ARMS}
    gates=dict(support=all(v>=12 for v in valid_counts.values()),conflict=sum(v is not None and v>=.2 for v in conflicts.values())>=2,actual_harm=avg['U'] is not None and avg['U']>0,mean_h_P_half_U=avg['U'] is not None and avg['U']>0 and avg['P']<=.5*avg['U'],mean_h_P_less_N=avg['P'] is not None and avg['P']<avg['N'],each_seed_P_le_U=all(means[str(s)]['P'] is not None and means[str(s)]['P']<=means[str(s)]['U'] for s in SEEDS),norm_median=len(ratios)>0 and float(np.median(ratios))>=.25,zero_fraction=len(zero)>0 and float(np.mean(zero))<=.1,numerical_certificates=True)
    return dict(status='G0_PASS_ALLOW_G1' if all(gates.values()) else 'STOP_G0_NO_FULL_TRAINING',gates=gates,valid_batches=valid_counts,conflict_fractions=conflicts,mean_h_by_seed=means,equal_seed_mean_h=avg,norm_ratio_median=float(np.median(ratios)) if ratios else None,zero_fraction=float(np.mean(zero)) if zero else None,formal_training=0,all48_reported=True)

def main():
    source=verify_manifest();guard=Guard(E);guard.check('startup');data=load_data();out=HERE/'G0';out.mkdir(exist_ok=True)
    with np.load(HERE/'identity_audit/G0_fixed.npz') as z:indices=z['index'];assert np.array_equal(z['sample_id'],data['FIT']['ids'][indices])
    records=[]
    for s in SEEDS:
        sets=load_sets(s);assert np.array_equal(sets['sample_id'],data['FIT']['ids'])
        for b in range(16):
            guard.check();stage('G0_INDEPENDENT_ONE_STEP',seed=s,batch=b,completed=len(records),total=48)
            idx=indices[b*256:(b+1)*256];labels=sets['membership'][idx];arms={};vectors={};saved={}
            for arm in ARMS:
                m,o,g=start(s);x,y=tensor_batch(data['FIT'],idx,'cuda')
                if arm=='U':saved['before']=predict(m,data['FIT']['x'][idx])
                rec,vec=step(m,o,x,y,labels,arm,measure_after=True,save_vectors=True);arms[arm]=rec
                for key,value in vec.items():vectors[arm+'_'+key]=value
                saved[arm]=predict(m,data['FIT']['x'][idx]);del m,o;torch.cuda.empty_cache()
            assert np.array_equal(vectors['U_d0'],vectors['N_d0']) and np.array_equal(vectors['U_d0'],vectors['P_d0'])
            assert np.array_equal(vectors['U_theta0'],vectors['P_theta0'])
            # G0 ideal N and P norms match before float32 writeback.
            assert abs(np.linalg.norm(vectors['N_dN'])-np.linalg.norm(vectors['P_dP']))<=1e-10
            valid=all(v>=8 for v in arms['U']['counts']) and all(v>0 for v in arms['U']['gradient_norms'])
            prefix=out/f'seed{s}_batch{b:02d}'
            np.savez_compressed(str(prefix)+'_vectors.npz',**vectors)
            np.savez_compressed(str(prefix)+'_paths.npz',**saved,gt=data['FIT']['gt'][idx],membership=labels,sample_id=data['FIT']['ids'][idx])
            records.append(dict(seed=s,batch=b,valid_double_group=valid,batch_ID_SHA=arraysha(data['FIT']['ids'][idx]),arms=arms,vectors=str(prefix.name)+'_vectors.npz',paths=str(prefix.name)+'_paths.npz'))
            write_json(out/'PACKETS.json',records)
    result=decision(records);write_json(out/'DECISION.json',result)
    assert verify_manifest()==source;guard.check();write_json(out/'RESOURCE.json',guard.observations)
    outputs={f.name:sha256(f) for f in out.iterdir() if f.is_file() and f.name!='RESULT.json'}
    write_json(out/'RESULT.json',dict(status='SUCCESS_G0_REVIEW_REQUIRED',decision=result,source_digest=source,packets=48,output_SHA=outputs,no_accumulated_training=True))
    return result
