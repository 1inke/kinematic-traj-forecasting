"""Second saved-array implementation and literal bracket/ID/summary verification."""
import numpy as np
import common as c
import direction_metrics as dm,retention_intervals as ri,attribution as at

def candidate_loop(p,q,y):
    result=[]
    for k in range(p.shape[1]):
        ep=p[:,k]-y;eq=q[:,k]-y;r=q[:,k]-p[:,k]
        en=np.sqrt(ep[...,0]*ep[...,0]+ep[...,1]*ep[...,1]);qn=np.sqrt(eq[...,0]*eq[...,0]+eq[...,1]*eq[...,1]);rn=np.sqrt(r[...,0]*r[...,0]+r[...,1]*r[...,1]);dp=ep[...,0]*r[...,0]+ep[...,1]*r[...,1]
        point=rn.copy();nonzero=en>0;point[nonzero]=dp[nonzero]/en[nonzero]
        a=en.mean(1);f=en[:,11];aa=qn.mean(1);ff=qn[:,11];ac=a/.2;fc=f/.4
        da=point.mean(1);df=point[:,11];dj=np.empty(len(a))
        for mode,mask in enumerate((ac>fc,ac<fc,ac==fc)):
            dj[mask]=(da/.2 if mode==0 else df/.4 if mode==1 else np.maximum(da/.2,df/.4))[mask]
        j=np.maximum(ac,fc);jj=np.maximum(aa/.2,ff/.4);delta=jj-j;near=abs(ac-fc)<=1e-10;uncertain=near|(abs(dj)<=1e-10);kind=np.zeros(len(a),np.int8)
        kind[(~uncertain)&(dj>1e-10)]=1;kind[(~uncertain)&(dj < -1e-10)&(delta>0)]=2;kind[(~uncertain)&(delta<0)]=3
        er=np.sqrt((ep*ep).sum((1,2)));rr=np.sqrt((r*r).sum((1,2)));valid=(er>0)&(rr>0);cos=np.full(len(a),np.nan);cos[valid]=dp.sum(1)[valid]/(er[valid]*rr[valid])
        d=dict(A0=a,F0=f,J0=j,good0=(a<=.2)&(f<=.4),A1=aa,F1=ff,J1=jj,good1=(aa<=.2)&(ff<=.4),dpoint=point,dA=da,dF=df,dJ=dj,near_tie=near,uncertain=uncertain,kind=kind,deltaJ=delta,deltaA=aa-a,deltaF=ff-f,mse_cross=2*dp.mean(1),mse_step=(r*r).sum(2).mean(1),deltaMSE=(eq*eq).sum(2).mean(1)-(ep*ep).sum(2).mean(1),endpoint_cross=2*dp[:,-1],endpoint_step=(r[:,-1]*r[:,-1]).sum(1),endpoint_deltaMSE=(eq[:,-1]*eq[:,-1]).sum(1)-(ep[:,-1]*ep[:,-1]).sum(1),path_cosine=cos,cosine_defined=valid,zero_error_points=(en==0).sum(1),tiny_nonzero_error_points=((en>0)&(en<=1e-10)).sum(1),correction_mean=rn.mean(1),correction_endpoint=rn[:,-1],correction_max=rn.max(1),segments0=en.reshape(-1,3,4).mean(2),segments1=qn.reshape(-1,3,4).mean(2))
        result.append(d)
    return {key:np.stack([r[key] for r in result],axis=1) for key in result[0]}

def process(z,independent=False):
    d=candidate_loop(z['native'],z['paths'],z['gt']) if independent else dm.candidate(z['native'],z['paths'],z['gt'])
    dm.check_identities(d);x=ri.exits(z['native'],z['paths'],z['gt'],d,independent);w=ri.windows(z,d,x)
    w['native_accel']=at.acceleration(z['native'],independent);w['refined_accel']=at.acceleration(z['paths'],independent);w['delta_accel']=w['refined_accel']-w['native_accel']
    return dict(d,**x),w,at.summarize(w,d)

def arrays_equal(a,b):
    assert a.keys()==b.keys();error=0.
    for k in a:
        x,y=a[k],b[k]
        if x.dtype.kind in 'bUSiu':assert np.array_equal(x,y),k
        else:error=max(error,c.close(x,y))
    return error

def review_job(z,production,window,summary):
    # Candidate calculations independently loop over slots on the entire job.
    d,w,s=process(z,True)
    errors=[arrays_equal(production,d),arrays_equal(window,w),c.compare(summary,s)]
    return dict(status='PASS',max_error=max(errors),candidate_slots=int(np.prod(d['J0'].shape)),windows=len(w['sample_id']),independent_point_and_slot_direction_formulas=True,bracket80_literal_checked=True,all_summary_counts_denominators_and_signed_accel_contributions_checked=True)
