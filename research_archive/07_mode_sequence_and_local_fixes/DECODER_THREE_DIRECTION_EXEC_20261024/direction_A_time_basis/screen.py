"""Opt-in CPU A-Screen. Does not launch A-Train even if the screen passes."""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import *
from basis import knot3,svd4,projector,project,cv,screen_gate
from f16.resource import Guard
sys.path.insert(0,str(ROOT))
from audit.verify import independent_metrics,compare

def run(out,estimate):
    out=owned_output(out,'direction_A_time_basis')
    if out.exists(): raise FileExistsError('New screen directory required; preserve all old evidence')
    out.mkdir(parents=True)
    guard=Guard(estimate);guard.check('startup')
    lock_sources(out/'source_manifest.json'); data=load_data()
    psi,singular=svd4(data['FIT']['gt'],data['FIT']['past'])
    bases={'knot3':knot3(),'svd4':psi}
    np.savez_compressed(out/'bases.npz',**bases,singular_values_FIT=singular)
    # Lock BOTH bases before looking at any inner projection results.
    write_json(out/'basis_manifest.json',dict(FIT_ID_SHA=digest_dict(data['FIT']['ids'].tolist()),
        fitting='FIT float64 noncentered SVD only',sha256=sha256(out/'bases.npz'),
        bases={k:dict(M=len(v),psi=v,projector=projector(v)) for k,v in bases.items()}))
    all_rows={}; decisions={}; source=verify_sources(out/'source_manifest.json')
    for name,psi in bases.items():
        per={}
        for split,d in data.items():
            guard.check();y=project(d['gt'],d['past'],psi)
            dist=np.linalg.norm(y-d['gt'],axis=-1);a=dist.mean(1);f=dist[:,-1]
            vel=np.diff(y,axis=1,prepend=np.zeros_like(y[:,:1]))/.4
            accel=np.linalg.norm(np.diff(vel,axis=1)/.4,axis=-1)
            join=np.linalg.norm(vel[:,0]-(d['past'][:,-1]-d['past'][:,-2])/.4,axis=-1)
            oracle={}
            for g,num in [('overall',None),('stable',0),('transition',1),('change',2)]:
                mask=np.ones(len(y),bool) if num is None else d['groups']==num
                oracle[g]=dict(n=int(mask.sum()),good=float(((a<=.2)&(f<=.4))[mask].mean()),
                    ADE_quantiles=np.quantile(a[mask],[0,.5,.9,.95,.99,1]),
                    FDE_quantiles=np.quantile(f[mask],[0,.5,.9,.95,.99,1]),
                    acceleration_mean=accel[mask].mean(),acceleration_by_time=accel[mask].mean(0),
                    velocity_join_mean=join[mask].mean())
            per[split]={'oracle':oracle,'seeds':{}}
            np.savez_compressed(out/f'{name}_oracle_{split}.npz',sample_id=d['ids'],paths=y,ADE=a,FDE=f)
            for seed in SEEDS:
                guard.check(); ids,old=old_paths(seed,split)
                assert np.array_equal(ids,d['ids'])
                projected=project(old,d['past'],psi)
                # Independent least-squares coordinate implementation, not P@r.
                residual=(old-cv(d['past'])[:,None]).transpose(2,0,1,3).reshape(12,-1)
                coeff=np.linalg.lstsq(psi.T,residual,rcond=None)[0]
                rebuilt=(psi.T@coeff).reshape(12,len(old),20,2).transpose(1,2,0,3)+cv(d['past'])[:,None]
                np.testing.assert_allclose(projected,rebuilt,atol=2e-12,rtol=0)
                path=out/f'{name}_seed{seed}_{split}.npz'
                np.savez_compressed(path,sample_id=ids,paths=projected,gt=d['gt'],native=d['native'],groups=d['groups'],track=d['track'])
                rows,e=report_paths(projected,d,out/f'{name}_seed{seed}_{split}_metrics.json')
                with np.load(path) as z:
                    assert np.array_equal(z['paths'],projected)
                    independent=independent_metrics(z)
                audit=compare({g:{k:rows[g][k] for k in independent[g]} for g in independent},independent)
                write_json(out/f'{name}_seed{seed}_{split}_audit.json',audit)
                if audit['status']!='PASS':raise ValueError(('SAVED_A_SCREEN_AUDIT_FAILED',audit))
                olde=errors(old,d['gt']);change=d['groups']==2
                per[split]['seeds'][str(seed)]=dict(metrics=rows,
                    old_change_good=int(olde['coverage'][change].sum()),
                    old_change_accel_mean=motion(old[change])['accel_mean'],
                    paths_SHA=sha256(path))
                del old,projected,rebuilt
        inner=per['inner'];s=[inner['seeds'][str(k)] for k in SEEDS]
        decisions[name]=screen_gate(inner['oracle']['change']['good'],
            [v['metrics']['change']['good'] for v in s],[v['old_change_good'] for v in s],166,
            np.mean([v['metrics']['change']['motion']['accel_mean'] for v in s])/
            np.mean([v['old_change_accel_mean'] for v in s]))
        all_rows[name]=per
    eligible=[k for k,v in decisions.items() if v['passed']]
    # If exact coverage ties, preregistered candidate order knot3 then svd4.
    selected=max(eligible,key=lambda k:sum(all_rows[k]['inner']['seeds'][str(s)]['metrics']['change']['good'] for s in SEEDS)) if eligible else None
    guard.check();assert verify_sources(out/'source_manifest.json')==source
    write_json(out/'RESOURCE.json',dict(estimate=estimate,factor=1.3,observations=guard.observations))
    write_json(out/'RESULT.json',dict(status='COMPLETED',decisions=decisions,selected_basis=selected,
        conclusion='ALLOW_A_TRAIN_PREFLIGHT_REQUIRES_HUMAN' if selected else 'SCREENING_DOES_NOT_SUPPORT_TRAINING_INVESTMENT',
        records=all_rows,source_digest=source,bases_SHA=sha256(out/'bases.npz'),
        limitation='L2 screen is neither feasibility proof nor learned/deployable upper bound'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--estimated-peak-gib',type=float,required=True,help='Explicit inventory estimate, not a fixed GiB limit')
    a=ap.parse_args();run(a.output,int(a.estimated_peak_gib*2**30))
