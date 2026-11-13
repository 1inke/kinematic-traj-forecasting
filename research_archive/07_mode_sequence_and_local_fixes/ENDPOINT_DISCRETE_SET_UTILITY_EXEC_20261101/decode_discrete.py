"""Observation-only mask/select/apply APIs. At most one edit, no GT fallback."""
import numpy as np
import contracts_discrete as c
from motion_guard import inspect
from build_teacher import masks_for_window

def select(score,valid):
    s=np.asarray(score);v=np.asarray(valid,bool)
    if s.shape!=v.shape or s.shape[1:]!=(20,48) or not np.isfinite(s).all():raise ValueError('score finite/shape')
    flat=np.where(v,s,-np.inf).reshape(len(s),960)
    return np.concatenate((np.zeros((len(s),1),s.dtype),flat),1).argmax(1)

def apply(native,past,choice,valid):
    out=np.array(native,copy=True);slot=np.full(len(out),-1,np.int32);action=np.full(len(out),c.ACTIONS.identity,np.int32)
    mismatches=[]
    for i,ch in enumerate(choice):
        if ch==0:continue
        if not 1<=ch<=960:raise ValueError('choice range')
        k,a=divmod(int(ch)-1,48);a=int(c.NONIDENTITY[a])
        if not valid[i,k,np.flatnonzero(c.NONIDENTITY==a)[0]]:raise ValueError('masked selected action')
        proposal=native[i,k]+(np.arange(1,13)/12)[:,None]*c.ACTIONS.params[a]
        passed,failed=inspect(past[i],native[i,k],proposal)
        if not passed:
            mismatches.append(dict(window=i,slot=k,action=a,failed=failed.tolist()));continue
        out[i,k]=proposal;slot[i]=k;action[i]=a
        assert np.array_equal(np.delete(out[i],k,0),np.delete(native[i],k,0))
    if mismatches:raise RuntimeError('GUARD_IMPLEMENTATION_MISMATCH '+str(mismatches))
    return out,slot,action

def mask_observed(past,native):
    return np.stack([masks_for_window(h,n,c.ACTIONS)['safe'][:,c.NONIDENTITY] for h,n in zip(past,native)])

def selection_metrics(score,choice,d):
    value=d['value'];valid=d['valid'];best=np.maximum(0,np.where(valid,value,-np.inf).reshape(len(value),-1).max(1))
    vv=np.concatenate((np.zeros((len(value),1)),value.reshape(len(value),-1)),1)
    selected=vv[np.arange(len(value)),choice];validv=np.where(valid,value,-np.inf).reshape(len(value),-1)
    oracle_ties=(validv==best[:,None]).sum(1)+(best==0)
    chosen_score=np.concatenate((np.zeros((len(score),1)),score.reshape(len(score),-1)),1)[np.arange(len(score)),choice]
    validscore=np.concatenate((np.zeros((len(score),1)),np.where(valid,score,-np.inf).reshape(len(score),-1)),1)
    return dict(selected_value=selected,best_value=best,regret=best-selected,oracle_hit=(best>0)&(selected==best),
        oracle_exact_ties=oracle_ties,oracle_near_ties_1e6=((validv>=best[:,None]-1e-6)&(validv<best[:,None])).sum(1)+((best>0)&(best<=1e-6)),
        score_exact_ties=(validscore==chosen_score[:,None]).sum(1),selected_score=chosen_score,
        positive_count=((value>0)&valid).sum((1,2)),negative_count=((value<0)&valid).sum((1,2)),zero_count=((value==0)&valid).sum((1,2)))
