"""Check changes from saved paths and rule masks with independent distance arithmetic."""
from pathlib import Path
import numpy as np
from run_history_diag import ROOT, OLD, PILOT, read, sha, write, arrays, RULES


def verify():
    inputs = read(ROOT/'INPUTS.json')['files']
    outputs = read(ROOT/'DONE.json')['files']
    assert all(sha(p)==digest for p,digest in (inputs|outputs).items())
    d = arrays(OLD/'C0/dev.npz')
    past = d['past'][:,0].astype(float)
    cv = past[:, -1, None] + np.arange(1,13)[None,:,None] * (past[:, -1]-past[:, -2])[:,None]
    truth = d['future'].astype(float)
    distance = lambda p: np.sqrt(np.sum((p-truth[:,None])**2,axis=-1))
    report = read(ROOT/'RESULT.json')
    n = len(past)
    cases = 0
    for arm in ('Prior','Prior_self'):
        prediction = arrays(PILOT/'runs'/arm/'dev_predictions.npz')
        w = arrays(ROOT/f'{arm}_windows.npz')
        np.testing.assert_array_equal(w['ids'],d['ids'])
        step = (past[:,1:]-past[:,:-1])/.4
        speed = np.sqrt((step**2).sum(-1))
        np.testing.assert_array_equal(w['stationary'],np.all(speed<=.10,axis=1))
        mean = speed.sum(1)/7
        chord = (past[:,-1]-past[:,0])/.4
        chord_size = np.sqrt((chord**2).sum(-1))
        steady = (~w['stationary'] & (mean>=.10) & (speed.max(1)<=mean*1.25) & (speed.min(1)>=mean*.75) &
                  (chord_size>=.98*speed.sum(1)) &
                  ((chord*step[:,-1]).sum(1)>=np.cos(np.pi/12)*chord_size*speed[:,-1]))
        np.testing.assert_array_equal(w['steady'],steady)
        np.testing.assert_array_equal(w['stable_history'],w['stationary']|steady)
        p = prediction['paths'].astype(float)
        orig_top = prediction['logits'].argmax(1)
        fallback = np.argsort(-prediction['logits'][:,:19],axis=1,kind='stable')[:,0]
        for name,(rule,demote) in RULES.items():
            error = distance(p)
            error[w[rule],19] = distance(cv[:,None])[w[rule],0]
            ade,fde = error.sum(-1)/12,error[:,:,-1]
            good = (ade<=.2)&(fde<=.4)
            top = orig_top.copy()
            changed = (orig_top==19)&~w['stable_history']&demote
            top[changed] = fallback[changed]
            np.testing.assert_array_equal(w[name+'__selected_slot'],top)
            assert report['models'][arm][name]['score_changed_n']==int(changed.sum())
            assert report['models'][arm][name]['cv_n']==int(w[rule].sum())
            for key,val in dict(minADE=ade.min(1), minFDE=fde.min(1),covered=good.any(1),
                                top1_ADE=ade[np.arange(n),top], top1_FDE=fde[np.arange(n),top],
                                top1_good=good[np.arange(n),top]).items():
                np.testing.assert_allclose(w[name+'__'+key],val,atol=2e-12,rtol=2e-12)
            cases += 1
    write(ROOT/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',files_sha256=len(inputs)+len(outputs),
          populations=cases,window_records=cases*n,history_decisions='MATCH',
          complete_geometry_and_top1='MATCH',new_training=False,new_inference=False))


if __name__=='__main__':
    verify()
