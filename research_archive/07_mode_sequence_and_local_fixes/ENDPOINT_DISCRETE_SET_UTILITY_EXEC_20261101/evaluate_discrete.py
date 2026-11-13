"""Full saved selected K20, scores, all-action labels and literal group checks."""
import numpy as np
import contracts_discrete as c
from decode_discrete import select,apply,selection_metrics
from targets_discrete import loss_numpy

def save(folder,score,d,arm,stage,extra=None):
    folder.mkdir(parents=True,exist_ok=False)
    choice=select(score,d['valid']);paths,slot,action=apply(d['x']['native'],d['x']['past'],choice,d['valid'])
    metrics,rows=c.reference.full_metrics(paths,d);native,base=c.reference.full_metrics(d['x']['native'],d)
    chosen=selection_metrics(score,choice,d);loss,_,perwin=loss_numpy(score,d['value'],d['valid'])
    c.npz_new(folder/'output.npz',score=score,valid=d['valid'],value=d['value'],q_edit=d['q_edit'],choice=choice,slot=slot,action=action,
        paths=paths,sample_id=d['x']['sample_id'],**chosen,**metrics)
    c.write(folder/'METRICS.json',dict(arm=arm,stage=stage,groups=rows,native_groups=base,balanced_MSE=loss,
        unweighted_legal_MSE=float(((score.astype(float)-d['value'])**2)[d['valid']].mean()) if d['valid'].any() else None,
        window_classes=perwin,noop_fraction=float((choice==0).mean()),legal_edit_fraction=float(d['valid'].mean()),selected_positive=int((chosen['selected_value']>0).sum()),
        selected_negative=int((chosen['selected_value']<0).sum()),selected_zero=int((chosen['selected_value']==0).sum()),regret=float(chosen['regret'].mean()),
        clip_both_above2_fraction=float(((d['nj'].min(1)[:,None,None]>2)&(d['q_edit']>2)&d['valid']).sum()/max(int(d['valid'].sum()),1)),
        not_probability=True,extra=extra or {}))
    c.write(folder/'done.json',dict(status='SAVED_FULL_K20_REVIEW_REQUIRED',contract=c.contract(d,arm,stage),files={p.name:c.sha(p) for p in folder.iterdir() if p.is_file()}))
    return rows

def read(folder,d,arm,stage):
    done=c.json_read(folder/'done.json')
    if done['contract']!=c.contract(d,arm,stage):raise ValueError('saved source/stat/split/actionorder mismatch')
    for rel,s in done['files'].items():
        if c.sha(folder/rel)!=s:raise ValueError('saved output SHA mismatch')
    z=c.load_npz(folder/'output.npz')
    if not np.array_equal(z['sample_id'],d['x']['sample_id']):raise ValueError('saved IDs mismatch')
    return z

def gates(rows,base):
    out=[]
    for r,n in zip(rows,base):
        if r['N']==0:out.append(dict(group=r['group'],status='NOT_ESTIMABLE'));continue
        limit=1 if r['group']=='change' else 1.01
        out.append(dict(group=r['group'],errors=all(r[k]<=n[k]*limit for k in ('minADE','minFDE')),
            motion=all(r[k]<=n[k]*1.01 for k in c.reference.NAMES),lost=r['lost'],rescue=r['rescue'],N=r['N']))
    return out
