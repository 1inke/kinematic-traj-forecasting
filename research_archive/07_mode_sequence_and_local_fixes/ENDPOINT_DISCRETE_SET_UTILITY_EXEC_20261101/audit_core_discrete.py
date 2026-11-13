"""Independent CPU source-bound arithmetic. Does not call model forward."""
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist
import contracts_discrete as c
from targets_discrete import independent_targets,loss_numpy
from motion_guard import inspect

def saved(folder,d,arm,stage):
    from evaluate_discrete import read
    z=read(folder,d,arm,stage);s=z['score'];valid=d['valid'];n=d['x']['native']
    selected=[]
    for i in range(len(s)):
        # Independent Python first-max order: NOOP, physical slot, original action index.
        choices=[(0.,0)]
        choices.extend((float(s[i,k,a]),1+k*48+a) for k in range(20) for a in range(48) if valid[i,k,a])
        best=max(v for v,j in choices);ch=next(j for v,j in choices if v==best);selected.append(ch)
    assert np.array_equal(selected,z['choice'])
    p=n.copy()
    for i,ch in enumerate(selected):
        if ch:
            k,a=divmod(ch-1,48);aa=int(c.NONIDENTITY[a]);p[i,k]=n[i,k]+(np.arange(1,13)/12)[:,None]*c.ACTIONS.params[aa]
            assert inspect(d['x']['past'][i],n[i,k],p[i,k])[0]
            assert np.array_equal(np.delete(z['paths'][i],k,0),np.delete(n[i],k,0))
        else:assert np.array_equal(z['paths'][i],n[i])
    assert np.array_equal(p,z['paths'])
    q,groups=c.reference.independent_groups(p,d);metrics=c.json_read(folder/'METRICS.json')
    c.reference.compare(dict(groups=metrics['groups']),dict(groups=groups))
    for k,v in q.items():c.reference.compare(v,z[k])
    c.reference.compare(q['good'].any(1),z['covered'])
    for name in ('ADE','FDE','J'):c.reference.compare(q[name].min(1),z['min'+name])
    c.reference.compare(((q['ADE']<=.3)&(q['FDE']<=.5)).any(1),z['coverage_sensitive_03_05'])
    for i in range(len(p)):
        ds=np.stack([pdist(p[i,:,t]) for t in range(12)],1)
        lab=fcluster(linkage(np.maximum(ds.mean(1)/.1,ds[:,-1]/.2),method='complete'),1,criterion='distance')
        assert len(np.unique(lab))==z['distinct'][i] and len(np.unique(lab[q['good'][i]]))==z['effective_good'][i]
        winner=int(q['J'][i].argmin());err=np.sqrt(((p[i,winner]-d['gt'][i])**2).sum(-1))
        assert winner==z['winner'][i]
        for k,sl in [('early_ADE',slice(0,4)),('middle_ADE',slice(4,8)),('late_ADE',slice(8,12))]:c.reference.compare(float(err[sl].mean()),z[k][i])
    value,qe=independent_targets(d['nj'],d['labels']['J_matrix'][:,:,c.NONIDENTITY])
    assert np.array_equal(value,z['value']) and np.array_equal(qe,z['q_edit'])
    loss,_,parts=loss_numpy(s,value,valid);assert abs(loss-metrics['balanced_MSE'])<=2e-12
    for i,ch in enumerate(selected):
        vv=value[i][valid[i]];best=max(0,float(vv.max())) if len(vv) else 0.
        val=0. if ch==0 else float(value[i].reshape(-1)[ch-1])
        assert val==z['selected_value'][i] and best-val==z['regret'][i]
        allv=np.r_[0.,value[i][valid[i]]];alls=np.r_[0.,s[i][valid[i]]]
        chosen_score=0. if ch==0 else float(s[i].reshape(-1)[ch-1])
        assert int((allv==best).sum())==z['oracle_exact_ties'][i]
        assert int(((value[i][valid[i]]>=best-1e-6)&(value[i][valid[i]]<best)).sum())+int(0<best<=1e-6)==z['oracle_near_ties_1e6'][i]
        assert int((alls==chosen_score).sum())==z['score_exact_ties'][i] and chosen_score==z['selected_score'][i]
        for label,mask in [('positive',value[i]>0),('negative',value[i]<0),('zero',value[i]==0)]:assert int((valid[i]&mask).sum())==z[label+'_count'][i]
        assert bool(z['oracle_hit'][i])==bool(best>0 and val==best)
        if ch:
            assert s[i].reshape(-1)[ch-1]>0
            c.reference.compare(q['J'][i].min(),qe[i].reshape(-1)[ch-1])
    return dict(status='PASS_SECOND_CPU_ALL_SCORES_SELECTION_FIXED19_LABEL_REGRET_QUALITY_CLUSTERS_POOLED',groups=groups,windows=len(p),actions=len(p)*960)
