"""Exactly one frozen END49 lookup witness, not a prediction/teacher."""
import numpy as np
from contracts import ROOT,write,npz_new
from checks import full_metrics,independent_paths,compare,verdict
from motion_guard import inspect

def run(d,bank):
    out=ROOT/'O0_witness';out.mkdir(exist_ok=False)
    p=d['x']['native'].copy();actions=np.full((128,20),bank.a.identity,dtype=np.int32);nc=(d['nj']<=1).any(1)
    choices=[]
    for i in range(128):
        if nc[i]:continue
        candidates=[(float(d['labels']['J_matrix'][i,k,a]),k,a) for k in range(20) for a in range(49) if d['safe'][i,k,a]]
        j,k,a=min(candidates)
        if j<=1:
            p[i,k]=bank.paths[i,k,a];actions[i,k]=a
            assert inspect(d['x']['past'][i],d['x']['native'][i,k],p[i,k])[0]
            assert np.array_equal(np.delete(p[i],k,0),np.delete(d['x']['native'][i],k,0))
            choices.append(dict(window=i,sample_id=int(d['x']['sample_id'][i]),slot=k,action=a,J=j))
    u=bank.a.normalized[actions]
    direct=independent_paths(u,d);compare(direct['S'],p)
    z,rows=full_metrics(p,d);assert int(z['covered'].sum())==90 and int((~nc&z['covered']).sum())==21 and not(nc&~z['covered']).any()
    result=verdict(p,p,d,bank.ref)
    status='FULL_FEASIBLE_WITNESS' if result['all_pass'] else 'COVERAGE_WITNESS_ONLY'
    npz_new(out/'witness.npz',paths=p,action_id=actions,sample_id=d['x']['sample_id'],**z)
    write(out/'RESULT.json',dict(status=status,choices=choices,groups=rows,verdict=result,GT_DIAGNOSTIC_ONLY=True,not_teacher=True))
    print('O0',status,'good90 rescue21 lost0',flush=True)
    return result
