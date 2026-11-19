"""Recompute pilot decisions and isolate the learned slot using saved predictions."""
import json
import numpy as np
from run_pilot import ROOT, OLD, DIAG, CONTROLS, NEW, read, sha, verify, load, control_folder, replace_json
import sys
sys.path.insert(0,str(DIAG))
from run_prec_diag import maneuver_labels


def measure(paths, truth, logits):
    delta = paths-truth[:,None]
    d = np.hypot(delta[...,0],delta[...,1])
    a,f = d.sum(-1)/12,d[:,:,-1]
    top = logits.argmax(1)
    rows = np.arange(len(top))
    good = (a<=.2)&(f<=.4)
    m = dict(minADE=a.min(1),minFDE=f.min(1),top1_ADE=a[rows,top],top1_FDE=f[rows,top],
             covered=good.any(1),top1_good=good[rows,top])
    return m,a,f,good,top


def main():
    done = read(ROOT/'DONE.json')
    assert done['status']=='SUCCESS' and done['new_runs']==2
    worker = read(ROOT/'standalone/WORKER_EXIT.json')
    assert worker['status']=='SUCCESS' and worker['exit_code']==0
    sources = read(ROOT/'SOURCE_MANIFEST.json')['files']
    files = dict(sources)
    files.update(done['files'])
    order = read(ROOT/'CONTROLS.json')['controls']['B0']['order_hashes']
    epochs = 0
    for arm in NEW:
        directory = ROOT/'runs'/arm
        r = read(directory/'DONE.json')
        files.update(r['files'])
        expected = read(ROOT/'CONTROLS.json')['controls'][r['contract']['reused_control']]
        assert r['train']['parameters']==expected['parameters']
        receipts = [read(directory/f'ckpt_epoch{e:03d}.json') for e in range(1,33)]
        assert [x['order_hash'] for x in receipts]==order
        assert [x['epoch'] for x in receipts]==list(range(1,33))
        assert [x['updates'] for x in receipts]==[340*e for e in range(1,33)]
        for x in receipts[-2:]:
            assert sha(x['file'])==x['sha256']
        epochs += len(receipts)
    verify(files)
    report = read(ROOT/'RESULT.json')
    d = load(OLD/'C0/dev.npz')
    label = maneuver_labels(d['past'],d['future'])
    past,truth = d['past'][:,0].astype(float),d['future'].astype(float)
    cv = past[:,-1,None] + (past[:,-1]-past[:,-2])[:,None] * np.arange(1,13)[None,:,None]
    masks = dict(overall=np.ones(len(label),bool),stable=d['group']==0,transition=d['group']==1,change=d['group']==2,
                 turn_any=np.isin(label,['turn','turn_and_speed']),decel_stop=np.isin(label,['deceleration','stop']))
    masks.update({v:label==v for v in set(label)})
    by_model = {}
    diagnostic = {}
    for arm in CONTROLS+NEW:
        folder = control_folder(arm) if arm in CONTROLS else ROOT/'runs'/arm
        saved = load(folder/'dev_predictions.npz')
        np.testing.assert_array_equal(saved['ids'],d['ids'])
        np.testing.assert_array_equal(saved['group'],d['group'])
        paths, logits = saved['paths'].astype(float), saved['logits']
        row=np.arange(len(paths))
        cases=[(arm,paths,np.full(len(paths),19) if arm in NEW else None)]
        if arm in CONTROLS:
            slot=np.array([min(range(20),key=lambda j:(z[j],-j)) for z in logits])
            edited=paths.copy()
            edited[row,slot]=cv
            cases.append((arm+'_offline_CV',edited,slot))
        for name,trajectories,slot in cases:
            w,a,f,g,top=measure(trajectories,truth,logits)
            if slot is not None:
                w.update(prior_is_top1=top==slot,prior_is_minADE=a.argmin(1)==slot,
                         prior_is_minFDE=f.argmin(1)==slot,prior_good=g[row,slot],
                         prior_only_good=g[row,slot]&(g.sum(1)==1))
            saved_windows=load(ROOT/'windows'/f'{name}.npz')
            np.testing.assert_array_equal(saved_windows['maneuver'],label)
            by_model[name]=w
            for k,v in w.items():
                np.testing.assert_allclose(v,saved_windows[k],atol=2e-12,rtol=2e-12)
                for group,mask in masks.items():
                    np.testing.assert_allclose(v[mask].mean(),report['models'][name]['groups'][group][k],atol=2e-12,rtol=2e-12)
        if arm in NEW:
            replacement=paths.copy()
            replacement[:,19]=cv
            original,oa,of,_,top=measure(paths,truth,logits)
            replaced,ra,rf,_,_=measure(replacement,truth,logits)
            shift=np.linalg.norm(paths[:,19]-cv,axis=-1)
            info={}
            for g in ('overall','stationary','steady','change','turn_any','decel_stop'):
                m=masks[g]
                group_info=dict(n=int(m.sum()),deviation_from_cv_endpoint_mean_m=float(shift[m,-1].mean()),
                               prior_slot_learned_ADE_m=float(oa[m,19].mean()),
                               prior_slot_learned_FDE_m=float(of[m,19].mean()),
                               prior_slot_pure_CV_ADE_m=float(ra[m,19].mean()),
                               prior_slot_pure_CV_FDE_m=float(rf[m,19].mean()),
                               learned_slot_FDE_better_than_CV_fraction=float((of[m,19]<rf[m,19]).mean()),
                               learned_slot_higher_than_CV_FDE_fraction=float((of[m,19]>rf[m,19]).mean()))
                group_info['restoring_CV_relative_gains']={
                    k:float(1-replaced[k][m].mean()/original[k][m].mean()) for k in ('minADE','minFDE','top1_FDE')}
                info[g]=group_info
            diagnostic[arm]=info
    guarded = report['outcome']['compared_for_protection']
    for name, c in report['comparisons'].items():
        arm,ref=name.split('_vs_')
        current,base=by_model[arm],by_model[ref]
        for key, group in c['groups'].items():
            m=masks[key]
            for metric, gain in group['relative_gains'].items():
                np.testing.assert_allclose(1-current[metric][m].mean()/base[metric][m].mean(),gain,atol=2e-12)
            assert group['lost']==int((base['covered'][m]&~current['covered'][m]).sum())
            assert group['rescued']==int((current['covered'][m]&~base['covered'][m]).sum())
        thresholds=[('overall','top1_FDE',.02),('change','minFDE',.01),
                    ('turn_any','minFDE',.01),('decel_stop','minFDE',.01)]
        failures=[(g,k) for g,k,t in thresholds if current[k][masks[g]].mean()/base[k][masks[g]].mean()-1>t]
        assert failures==[(x['group'],x['metric']) for x in c['error_failures']]
    exact_pass=all((1-by_model['Prior_self'][k].mean()/by_model['E3_self'][k].mean())>=.015 for k in ('minADE','minFDE'))
    assert exact_pass==report['outcome']['screening_gain_pass']==False
    assert all(report['comparisons'][a]['protection_pass'] for a in guarded)==report['outcome']['protection_pass']==False
    replace_json(ROOT/'POSTHOC_CV_SLOT.json',dict(scope='Swap the learned CV slot back to exact CV, keeping all other saved paths and logits.',
                 comparison_is_descriptive=True,models=diagnostic))
    verification = dict(status='PASS',sha_files_checked=len(files),epochs_verified=epochs,
                        paired_batch_orders='MATCH',evaluation_populations=len(by_model),window_records=len(label)*len(by_model),
                        full_trajectory_metrics='MATCH',coverage_exchange='MATCH',error_protection='MATCH',
                        screening_gain_pass=exact_pass,scientific_verdict_more_seeds=False,
                        separate_analysis='POSTHOC_CV_SLOT.json',new_training=False,new_inference=False)
    replace_json(ROOT/'INDEPENDENT_VERIFICATION.json',verification)
    print(json.dumps(verification))


if __name__=='__main__':
    main()
