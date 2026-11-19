"""Re-evaluate fixed observed-history rules on sealed predictions."""
from pathlib import Path
import json
import math
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
PILOT = WORK/'qoder_work/EXP_PREC_PILOT_EXEC_20261119'
DIAG = WORK/'qoder_work/EXP_PREC_DIAG_EXEC_20261119'
FAST = WORK/'qoder_work/EXPLORE_FAST_EXEC_20261116'
sys.path.insert(0,str(DIAG))
from run_prec_diag import OLD,read,sha,write,arrays,groups,maneuver_labels,quality,window_metrics,summarize
from run_prec_diag import pooled,motion_arrays,motion_summary,motion_differences,fit_reference

RULES = {
    'original': ('none',False),
    'always_CV': ('all',False),
    'CV_if_stationary': ('stationary',False),
    'CV_if_steady': ('steady',False),
    'CV_if_stable_history': ('stable_history',False),
    'demote_if_other': ('none',True),
    'stable_CV_and_demote': ('stable_history',True),
}
METRICS = ('minADE','minFDE','top1_ADE','top1_FDE')


def observation_masks(past):
    delta = np.diff(past,axis=1)
    step_len = np.hypot(delta[...,0],delta[...,1])
    speed = step_len/.4
    avg = speed.mean(1)
    stationary = speed.max(1)<=.10
    net = np.sum(delta,axis=1)
    net_len = np.hypot(net[:,0],net[:,1])
    efficiency = net_len/np.maximum(step_len.sum(1),1e-12)
    agreement = (delta[:,-1]*net).sum(1)/np.maximum(step_len[:,-1]*net_len,1e-12)
    speed_deviation = np.max(np.abs(speed-avg[:,None]),axis=1)
    steady = ~stationary & (avg>=.10) & (speed_deviation<=.25*avg) & (efficiency>=.98) & (agreement>=math.cos(math.radians(15)))
    return dict(none=np.zeros(len(past),bool),all=np.ones(len(past),bool),stationary=stationary,
                steady=steady,stable_history=stationary|steady)


def summary(paths, top, truth, past, subsets, gt, fitref, tails):
    q = quality(paths,truth)
    v = window_metrics(q,top)
    row = np.arange(len(top))
    v.update(prior_top1=top==19,prior_joint_good=q['good'][:,19],
             prior_top1_bad_with_good_alternative=(top==19)&~q['good'][:,19]&q['good'][:,:19].any(1),
             selected_slot=top)
    stats = summarize({k:a for k,a in v.items() if k!='selected_slot'},subsets)
    return dict(groups=stats,motion=motion_summary(paths,past,q,top,subsets,fitref,tails,gt)),v


def compare(result, metrics, ref_result, ref_metrics, subsets):
    motion, undefined = motion_differences(ref_result['motion'],result['motion'])
    detail = {}
    for name,m in subsets.items():
        if not m.any():
            continue
        a,b = result['groups'][name],ref_result['groups'][name]
        item = dict(improvement={k:(b[k]-a[k])/b[k] if b[k]>0 else None for k in METRICS})
        for k in ('covered','top1_good'):
            prev, cur = ref_metrics[k][m],metrics[k][m]
            item[k] = dict(rescue=int((cur&~prev).sum()),lost=int((prev&~cur).sum()),
                           difference_pp=100*(float(cur.mean())-float(prev.mean())))
        item['score'] = {k:a[k]-b[k] for k in ('prior_top1','prior_top1_bad_with_good_alternative')}
        detail[name] = item
    error = []
    for g,k,tol in [('overall','top1_FDE',.02),('change','minFDE',.01),('turn_any','minFDE',.01),('decel_stop','minFDE',.01)]:
        value = detail[g]['improvement'][k]
        if value is None or -value>tol:
            error.append(dict(group=g,metric=k,relative_gain=value,maximum_harm=tol))
    return dict(groups=detail,error_flags=error,motion_flags=motion,undefined=undefined,
                protection_pass=not(error or motion or undefined))


def read_paths(path, data):
    p = arrays(path)
    np.testing.assert_array_equal(p['ids'],data['ids'])
    np.testing.assert_array_equal(p['group'],data['group'])
    assert p['paths'].shape==(len(data['ids']),20,12,2) and p['logits'].shape==(len(data['ids']),20)
    assert np.isfinite(p['paths']).all() and np.isfinite(p['logits']).all()
    return p['paths'].astype(float),p['logits']


def run():
    start = time.monotonic()
    if (ROOT/'DONE.json').exists():
        assert all(sha(p)==digest for p,digest in read(ROOT/'DONE.json')['files'].items())
        print('ALREADY_COMPLETE')
        return
    seal = read(PILOT/'DONE.json')['files']
    previous = read(PILOT/'SOURCE_MANIFEST.json')['files']
    for key in ('RESULT.json','SOURCE_MANIFEST.json'):
        assert sha(PILOT/key)==seal[str(PILOT/key)]
    for arm in ('Prior','Prior_self'):
        receipt = PILOT/'runs'/arm/'DONE.json'
        assert sha(receipt)==seal[str(receipt)]
        previous.update(read(receipt)['files'])
    pred_e3 = FAST/'runs/E3_self_E32_s42/dev_predictions.npz'
    dataset = OLD/'C0/dev.npz'
    inputs = [dataset,pred_e3,OLD/'C0/MOTION_REFERENCE.json',OLD/'C0/EVALUATION_CONTRACT.json',
              *[PILOT/'runs'/a/'dev_predictions.npz' for a in ('Prior','Prior_self')]]
    for p in inputs:
        assert sha(p)==previous[str(p)],p
    write(ROOT/'INPUTS.json',dict(files={str(p):sha(p) for p in [*inputs,ROOT/'PROTOCOL.md',Path(__file__).resolve()]}))
    data = arrays(dataset)
    assert len(data['ids'])==4753
    past,truth = data['past'][:,0].astype(float),data['future'].astype(float)
    mask = observation_masks(past)
    assert not (mask['stationary']&mask['steady']).any()
    subsets = groups(data,maneuver_labels(data['past'],data['future']))
    for name,h in [('obs_stationary',mask['stationary']),('obs_steady',mask['steady']),('obs_other',~mask['stable_history'])]:
        subsets[name] = h
    gt = {g:pooled(motion_arrays(truth[m,None],past[m])) for g,m in subsets.items() if m.any()}
    fitref = fit_reference(OLD/'C0/MOTION_REFERENCE.json')
    tails = read(OLD/'C0/EVALUATION_CONTRACT.json')['tail_thresholds']
    cv = past[:,-1,None]+(past[:,-1]-past[:,-2])[:,None]*np.arange(1,13)[None,:,None]
    p,s = read_paths(pred_e3,data)
    ref,ref_w = summary(p,s.argmax(1),truth,past,subsets,gt,fitref,tails)
    pilot = read(PILOT/'RESULT.json')
    for k in ('minADE','minFDE','top1_FDE'):
        np.testing.assert_allclose(ref_w[k].mean(),pilot['models']['E3_self']['groups']['overall'][k],atol=2e-12)
    report = {}
    for arm in ('Prior_self','Prior'):
        p,s = read_paths(PILOT/'runs'/arm/'dev_predictions.npz',data)
        top = s.argmax(1)
        other_top = s[:,:19].argmax(1)
        report[arm] = {}
        window_arrays = {name:np.asarray(value) for name,value in mask.items()}
        window_arrays.update(ids=data['ids'],group=data['group'])
        for name,(rule,demote) in RULES.items():
            use_cv = mask[rule]
            paths = p.copy()
            paths[use_cv,19] = cv[use_cv]
            new_top = np.where((top==19)&~mask['stable_history']&demote,other_top,top)
            result,w = summary(paths,new_top,truth,past,subsets,gt,fitref,tails)
            if name=='original':
                baseline,bw = result,w
                for k in ('minADE','minFDE','top1_FDE'):
                    np.testing.assert_allclose(bw[k].mean(),pilot['models'][arm]['groups']['overall'][k],atol=2e-12)
            if rule=='none':
                for k in ('minADE','minFDE','covered'):
                    np.testing.assert_array_equal(w[k],bw[k])
            if not demote:
                np.testing.assert_array_equal(new_top,top)
            for k,v in w.items():
                window_arrays[name+'__'+k] = v
            old = compare(result,w,baseline,bw,subsets)
            relative = compare(result,w,ref,ref_w,subsets)
            total = relative['groups']['overall']['improvement']
            result.update(cv_n=int(use_cv.sum()),cv_changed_n=int(np.any(paths[:,19]!=p[:,19],axis=(1,2)).sum()),
                          score_changed_n=int((new_top!=top).sum()),original_prior_top1_rate=float((top==19).mean()),
                          vs_original=old,vs_E3_self=relative,
                          versus_E3_self_dual_gain_pass=total['minADE']>=.015 and total['minFDE']>=.015)
            report[arm][name] = result
        np.savez_compressed(ROOT/f'{arm}_windows.npz',**window_arrays)
    output = dict(status='COMPLETE',population=4753,models=report,E3_self=ref,observed_history_counts={
        'stationary':int(mask['stationary'].sum()),'steady':int(mask['steady'].sum()),
        'other':int((~mask['stable_history']).sum())},elapsed_seconds=time.monotonic()-start,
        seed=42,epochs=32,no_new_training=True,no_new_inference=True)
    write(ROOT/'RESULT.json',output)
    names=('INPUTS.json','RESULT.json','Prior_windows.npz','Prior_self_windows.npz')
    write(ROOT/'DONE.json',dict(status='SUCCESS',files={str(ROOT/n):sha(ROOT/n) for n in names}))
    print(json.dumps(dict(status='SUCCESS',seconds=output['elapsed_seconds'],history=output['observed_history_counts'])))


if __name__=='__main__':
    run()
