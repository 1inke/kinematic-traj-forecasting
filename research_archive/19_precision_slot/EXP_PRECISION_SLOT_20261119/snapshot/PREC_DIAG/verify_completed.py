"""Independently check saved geometry and candidate substitutions on CPU."""
import json
from pathlib import Path
import numpy as np
from run_prec_diag import ROOT, OLD, read, sha, write, arrays, run_folder, maneuver_labels


def errors(paths, truth):
    delta = paths - truth[:, None]
    distance = np.hypot(delta[..., 0], delta[..., 1])
    return distance.sum(-1) / 12, distance[..., -1]


def main():
    done = read(ROOT/'DONE.json')
    worker = read(ROOT/'standalone/WORKER_EXIT.json')
    assert done['status']=='SUCCESS' and worker['status']=='SUCCESS' and worker['exit_code']==0
    inputs = read(ROOT/'INPUT_MANIFEST.json')['files']
    for path, digest in (inputs | done['files']).items():
        assert sha(path)==digest,path
    result = read(ROOT/'RESULT.json')
    fit,dev = (arrays(OLD/f'C0/{split}.npz') for split in ('fit','dev'))
    labels = maneuver_labels(dev['past'],dev['future'])
    rows = np.arange(len(labels))
    past = dev['past'][:,0].astype(float)
    pos = past[:,-1]
    cv = pos[:,None] + (past[:,-1]-past[:,-2])[:,None]*np.arange(1,13)[None,:,None]
    still = np.broadcast_to(pos[:,None],cv.shape)
    target = dev['future'].astype(float)
    hypotheses = {'CV':cv[:,None],'stationary':still[:,None],'CV_stationary':np.stack([cv,still],axis=1)}
    masks = {label:labels==label for label in set(labels)}
    masks.update(overall=np.ones(len(rows),bool),stable=dev['group']==0,transition=dev['group']==1,change=dev['group']==2,
                 turn_any=np.isin(labels,['turn','turn_and_speed']),decel_stop=np.isin(labels,['deceleration','stop']))
    masks = {key:value for key,value in masks.items() if value.any()}
    cases = 0
    for arm,seeds in result['families'].items():
        for seed,record in seeds.items():
            standard = run_folder(arm,int(seed))
            f,p = (arrays(standard/f'{split}/predictions.npz') for split in ('fit','dev'))
            def align(saved, subset):
                lookup = {v:i for i,v in enumerate(saved['ids'])}
                return np.array([lookup[v] for v in subset['ids']])
            fx,dx = align(f,fit),align(p,dev)
            fa,ff = errors(f['paths'][fx],fit['future'])
            fg = (fa<=.2)&(ff<=.4)
            sole = ((fg.sum(1)==1)[:,None]&fg).sum(0)
            wins = np.bincount(fa.argmin(1),minlength=20) + np.bincount(ff.argmin(1),minlength=20)
            fixed = sorted(range(20),key=lambda j:(sole[j],wins[j],j))[:2]
            assert fixed==record['fixed_slots']
            paths,logits = p['paths'][dx],p['logits'][dx]
            top = logits.argmax(1)
            dynamic = np.array([sorted(range(20), key=lambda j:(z[j],-j))[:2] for z in logits])
            w = arrays(ROOT/'windows'/f'{arm}_s{seed}.npz')
            np.testing.assert_array_equal(w['ids'],dev['ids'])
            np.testing.assert_array_equal(w['maneuver'],labels)
            np.testing.assert_array_equal(w['CV_stationary_identical'],(cv==still).all((1,2)))
            def measure(trajectories, key, groups):
                a,fde = errors(trajectories,target)
                good = (a<=.2)&(fde<=.4)
                v = dict(minADE=a.min(1),minFDE=fde.min(1),top1_ADE=a[rows,top],top1_FDE=fde[rows,top],
                         covered=good.any(1),top1_good=good[rows,top])
                for name,value in v.items():
                    np.testing.assert_allclose(value,w[f'{key}__{name}'],atol=2e-12,rtol=2e-12)
                    for g,mask in masks.items():
                        np.testing.assert_allclose(value[mask].mean(),groups[g][name],atol=2e-12,rtol=2e-12)
                return v
            baseline = measure(paths,'base',record['baseline'])
            for scheme,new in hypotheses.items():
                key = 'union_'+scheme
                united = measure(np.concatenate([paths,new],axis=1),key,record['options'][key]['groups'])
                cases += 1
                for policy,slots in [('lowest_logit',dynamic),('fixed_fit',np.broadcast_to(fixed,dynamic.shape))]:
                    key = policy+'_'+scheme
                    slots = slots[:,:new.shape[1]]
                    np.testing.assert_array_equal(slots,w[key+'__retired_slots'])
                    updated = paths.copy()
                    updated[rows[:,None],slots] = new
                    v = measure(updated,key,record['options'][key]['groups'])
                    ae,fe = errors(paths,target)
                    original_good = (ae<=.2)&(fe<=.4)
                    original_good[rows[:,None],slots] = False
                    ae[rows[:,None],slots] = np.inf
                    fe[rows[:,None],slots] = np.inf
                    for metric,remaining in [('minADE',ae.min(1)),('minFDE',fe.min(1))]:
                        for name,expected in [('delete_harm_m',remaining-baseline[metric]),
                                              ('add_after_delete_gain_m',remaining-v[metric]),
                                              ('lost_vs_union_m',v[metric]-united[metric]),
                                              ('net_gain_m',baseline[metric]-v[metric])]:
                            np.testing.assert_allclose(w[f'{key}__{metric}_{name}'],expected,atol=2e-12,rtol=2e-12)
                    np.testing.assert_array_equal(w[key+'__deleted_original_coverage'],baseline['covered']&~original_good.any(1))
                    cases += 1
    assert cases==72
    report = dict(status='PASS',input_hashes=len(inputs),output_hashes=len(done['files']),
                  evaluated_model_seed_pairs=8,baseline_populations=8,variant_populations=cases,
                  window_records=(cases+8)*len(rows),fit_slot_ranking='EXACT',dev_errors_and_coverage='MATCH',
                  additive_deletion_decomposition='MATCH',new_training=False,new_inference=False)
    write(ROOT/'INDEPENDENT_VERIFICATION.json',report)
    print(json.dumps(report))


if __name__=='__main__':
    main()
