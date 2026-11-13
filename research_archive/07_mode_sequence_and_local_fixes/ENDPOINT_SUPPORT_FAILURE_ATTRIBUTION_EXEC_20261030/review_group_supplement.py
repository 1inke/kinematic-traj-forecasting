"""Explicit second-implementation group physics/support reconciliation extension."""
import csv
import numpy as np
import attribution_core as c

def main():
    c.write_json(c.ROOT/'GROUP_REVIEW_EXTENSION.json',dict(status='EXPLICIT_SAVED_ONLY_REVIEW_EXTENSION',
        files={str(c.ROOT/'review_group_supplement.py'):c.sha(c.ROOT/'review_group_supplement.py')},
        purpose='All20 group pooled physical/junction plus Native-support strata and signed sum reconciliation'))
    z=c.load_npz(c.ROOT/'derived_ports.npz');p=z['paths'];gt=z['gt'];j=z['J'];good=z['good'];m=good[0].sum(1)
    st=np.empty((128,20),int)
    for i in range(128):
        for k in range(20):st[i,k]=0 if good[0,i,k] and m[i]==1 else 1 if good[0,i,k] else 2 if m[i]>0 else 3
    rows=list(csv.DictReader((c.ROOT/'all_groups_quality_motion.csv').open(encoding='utf-8-sig')))
    maxerr=0.
    for row in rows:
        port=c.PORTS.index(row['port']);mask=np.ones(128,bool) if row['group']=='overall' else z['group']==['stable','transition','change'].index(row['group'])
        val=p[port,mask];native=p[0,mask]
        for pre,pp in [('',val),('native_',native)]:
            v=(pp-np.concatenate((np.zeros(pp.shape[:-2]+(1,2)),pp[...,:-1,:]),axis=-2))/.4
            a=(v[...,1:,:]-v[...,:-1,:])/.4
            dv=v[...,1:,:]-v[...,:-1,:];jj=(dv[...,1:,:]-dv[...,:-1,:])/(.4**2)
            vn,an,jn=[np.sqrt((xx*xx).sum(-1)).ravel() for xx in (v,a,jj)]
            stats=[np.quantile(vn,.95),an.mean(),np.quantile(an,.95),jn.mean(),np.quantile(jn,.95)]
            for nm,vv in zip(c.MOTION_NAMES,stats):
                err=abs(float(row[pre+nm])-vv);maxerr=max(maxerr,err);assert err<=2e-12
            past=z['past'][mask];vo=(past[:,-1]-past[:,-2])/.4
            junction=np.sqrt(((pp[:,:,0,:]/.4-vo[:,None])**2).sum(-1))/.4
            for nm,vv in [('junction_mean',junction.mean()),('junction_p95',np.quantile(junction,.95))]:
                err=abs(float(row[pre+nm])-vv);maxerr=max(maxerr,err);assert err<=2e-12
        cov=good[port,mask].any(1);b=good[0,mask].any(1)
        for nm,vv in [('N',mask.sum()),('good',cov.sum()),('rescue',(~b&cov).sum()),('lost',(b&~cov).sum()),('retained',(b&cov).sum())]:assert int(row[nm])==vv
    gains=list(csv.DictReader((c.ROOT/'gain_allocation.csv').open(encoding='utf-8-sig')))
    dm=z['initial_error']-z['final_error'];dj=j[0]-j[3];active=np.any(z['target']!=0,axis=-1)
    for row in gains:
        mask=(st==c.STRATA.index(row['stratum']))
        if row['group']!='overall':mask&=(z['group']==['stable','transition','change'].index(row['group']))[:,None]
        if row['teacher_type']!='all':mask&=active if row['teacher_type']=='active' else ~active
        assert int(row['candidates'])==mask.sum() and int(row['windows'])==mask.any(1).sum()
        if not mask.any():assert row['status']=='NA_EMPTY';continue
        for key,vv in [('delta_MSE',dm),('delta_J',dj)]:
            assert abs(float(row[key+'_signed_sum'])-vv[mask].sum())<=2e-12
            assert abs(float(row[key+'_mean'])-vv[mask].mean())<=2e-12
    table=list(csv.DictReader((c.ROOT/'candidate_stage_audit.csv').open(encoding='utf-8-sig')))
    for row in table:
        i,k=int(row['window_index']),int(row['slot']);assert row['stratum']==c.STRATA[st[i,k]]
        assert (row['native_support']=='True')==good[0,i,k]
        for nm,val in [('delta_MSE',dm[i,k]),('delta_J',dj[i,k])]:assert abs(float(row[nm])-val)<=2e-12
    c.write_json(c.ROOT/'completion_review/GROUP_SUPPLEMENT.json',dict(status='PASS_ALL20_GROUPS_AND_NATIVE_STRATA',
        groups=20,motion_values=100,junction_values=40,all2560candidate_strata=True,all192_gain_cells=True,
        gain_cells_actual=len(gains),max_arithmetic_error=maxerr,same_assistant_not_external=True))
    print('Group/support/signed gains independent supplement PASS',maxerr,flush=True)

if __name__=='__main__':main()
