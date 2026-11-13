"""Saved-score diagnostics only, never changes a deployment choice."""
import numpy as np,math
KS=(1,5,10)
def random_hit(m,b,k):
    if not m:return np.nan
    t=min(k,m)
    return 1.-(math.comb(m-b,t)/math.comb(m,t) if m-b>=t else 0.)
def diagnose(scores,eligible,B,quality,actual):
    s=np.asarray(scores,np.float64);e=np.asarray(eligible,bool);b=np.asarray(B,bool);q=np.asarray(quality,np.float64)
    assert s.shape==e.shape==b.shape and s.shape[1]==401 and q.shape==s.shape+(3,)
    assert np.isfinite(s).all() and e[:,0].all() and not b[:,0].any() and not (b&~e).any()
    chosen=np.where(e,s,-np.inf).argmax(1);assert np.array_equal(chosen,actual)
    out={};rows=[]
    for i in range(len(s)):
        ii=np.flatnonzero(e[i]);ed=ii[ii!=0];order=ed[np.argsort(-s[i,ed],kind='stable')]
        # A separately expressed physical-action loop checks the entire ordering.
        independent=sorted((int(a) for a in ed),key=lambda a:(-float(s[i,a]),a));assert order.tolist()==independent
        ng=bool(q[i,0,2]<=1);rescue=b[i]&(q[i,:,2]<=1)&~ng;bn=bool(b[i].any());rn=bool(rescue.any());has=bool(len(ed))
        best=int(order[0]) if has else 0;good=bool(has and b[i,best]);rhit=bool(has and rescue[best]);margin=float(s[i,best]-s[i,0]) if has else np.nan
        rb=np.flatnonzero(b[i,order])+1;rr=np.flatnonzero(rescue[order])+1
        rankb=float(rb[0]) if len(rb) else np.nan;rankr=float(rr[0]) if len(rr) else np.nan
        if not has:event='NO_ELIGIBLE_EDIT'
        elif bn:event=('PASS_GOOD' if margin>0 else 'BLOCKED_GOOD') if good else 'BAD_INTERNAL'
        else:event='BAD_NO_B_RELEASE' if margin>0 else 'STOP_CORRECT_NOB'
        a=int(chosen[i]);real=q[i,a];base=q[i,0];harm=bool((real>base).any() or (ng and real[2]>1));exec_=a!=0
        row=dict(best_edit=best,margin=margin,has_edit=has,B_nonempty=bn,rescue_opportunity=rn,native_good=ng,actual_good=bool(real[2]<=1),actual_rescue=bool(not ng and real[2]<=1),actual_lost=bool(ng and real[2]>1),actual_action=a,
            actual_executed=exec_,actual_harmful=exec_ and harm,actual_ineffective=exec_ and not b[i,a] and not harm,
            best_B=good,best_rescue=rhit,blocked_good=good and margin<=0,blocked_rescue=rhit and margin<=0,internal_bad=bn and not good,
            event=event,rank_B=rankb,rank_rescue=rankr,MRR_B=1/rankb if len(rb) else np.nan,MRR_rescue=1/rankr if len(rr) else np.nan,
            eligible_edits=len(ed),B_count=int(b[i].sum()),rescue_count=int(rescue.sum()),best_edit_tie_count=int((s[i,ed]==s[i,best]).sum()) if has else 0,
            STOP_tie=has and margin==0,all_eligible_equal=bool(np.ptp(s[i,ii])==0))
        for k in KS:
            take=min(k,len(order));row[f'hitB{k}']=bool(b[i,order[:take]].any());row[f'hitRescue{k}']=bool(rescue[order[:take]].any())
            row[f'recallB{k}']=float(b[i,order[:take]].sum()/b[i].sum()) if bn else np.nan
            row[f'recallRescue{k}']=float(rescue[order[:take]].sum()/rescue.sum()) if rn else np.nan
            row[f'randomHitB{k}']=random_hit(len(ed),int(b[i].sum()),k);row[f'randomHitRescue{k}']=random_hit(len(ed),int(rescue.sum()),k)
            row[f'physicalHitB{k}']=bool(b[i,ed[:take]].any());row[f'physicalHitRescue{k}']=bool(rescue[ed[:take]].any())
        rows.append(row)
    for k in rows[0]:out[k]=np.asarray([r[k] for r in rows])
    return out
def quantiles(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return np.quantile(v,[0,.05,.25,.5,.75,.95,1]).tolist() if len(v) else None
def summarize(d,mask):
    n=int(mask.sum());b=mask&d['B_nonempty'];r=mask&d['rescue_opportunity'];fail=mask&~d['native_good'];edit=mask&d['has_edit']
    out=dict(n=n,B_nonempty=int(b.sum()),rescue_opportunities=int(r.sum()),original_failures=int(fail.sum()),eligible_edit_windows=int(edit.sum()),margin_quantiles=quantiles(d['margin'][mask]),events={str(k):int(v) for k,v in zip(*np.unique(d['event'][mask],return_counts=True))},margin_by_event={str(k):quantiles(d['margin'][mask&(d['event']==k)]) for k in np.unique(d['event'][mask])})
    for k in ('native_good','actual_good','actual_rescue','actual_lost','actual_executed','actual_harmful','actual_ineffective','best_B','best_rescue','blocked_good','blocked_rescue','internal_bad','STOP_tie','all_eligible_equal'):
        out[k]=int(d[k][mask].sum())
    out['best_B_conditional']=float(d['best_B'][b].mean()) if b.any() else None
    out['best_B_unconditional']=float(d['best_B'][mask].mean()) if n else None
    out['best_rescue_conditional_opportunity']=float(d['best_rescue'][r].mean()) if r.any() else None
    out['best_rescue_among_native_failure']=float(d['best_rescue'][fail].mean()) if fail.any() else None
    out['MRR_B']=float(d['MRR_B'][b].mean()) if b.any() else None;out['MRR_rescue']=float(d['MRR_rescue'][r].mean()) if r.any() else None
    out['rank_B_quantiles']=quantiles(d['rank_B'][mask]);out['rank_rescue_quantiles']=quantiles(d['rank_rescue'][mask]);out['best_edit_multiple_tie_windows']=int(((d['best_edit_tie_count']>1)&mask).sum())
    for k in KS:
        for label,den in (('B',b),('Rescue',r)):
            out[f'top{k}_{label}']=dict(hit_count=int(d[f'hit{label}{k}'][mask].sum()),conditional_hit=float(d[f'hit{label}{k}'][den].mean()) if den.any() else None,unconditional_hit=float(d[f'hit{label}{k}'][mask].mean()) if n else None,
                recall_fraction=float(d[f'recall{label}{k}'][den].mean()) if den.any() else None,random_hit_reference=float(d[f'randomHit{label}{k}'][den].mean()) if den.any() else None,physical_action_order_hit=float(d[f'physicalHit{label}{k}'][den].mean()) if den.any() else None)
    return out
