import numpy as np

def exits(p,q,gt,d,independent=False):
    g=d['good0'];bad=g&~d['good1'];lo=np.full(g.shape,np.nan);hi=lo.copy();lo[g]=1.;hi[g]=1.
    rr,kk=np.where(bad)
    if len(rr):
        native=p[rr,kk];r=q[rr,kk]-native;y=gt[rr];left=np.zeros(len(rr));right=np.ones(len(rr))
        for _ in range(80):
            mid=(left+right)/2;path=native+mid[:,None,None]*r;e=path-y
            if independent:
                dist=np.sqrt(e[...,0]**2+e[...,1]**2);a=np.mean(dist,axis=1);f=dist[:,11]
            else:dist=np.sqrt((e*e).sum(-1));a=dist.mean(-1);f=dist[:,-1]
            good=(a<=.2)&(f<=.4);left=np.where(good,mid,left);right=np.where(good,right,mid)
        lo[rr,kk]=left;hi[rr,kk]=right
        for value,expected in ((left,True),(right,False)):
            e=native+value[:,None,None]*r-y;dist=np.sqrt((e*e).sum(-1));good=(dist.mean(-1)<=.2)&(dist[:,-1]<=.4)
            assert np.all(good==expected),'EXIT_BRACKET_LITERAL_FAILURE'
    return dict(exit_lo=lo,exit_hi=hi,exited=bad,native_good_no_exit=g&d['good1'])

def windows(z,d,x):
    j0=d['J0'].argmin(1);j1=d['J1'].argmin(1);ii=np.arange(len(j0));g0=d['good0'].sum(1);c0=g0>0;c1=d['good1'].any(1)
    event=np.select([c0&c1,c0&~c1,~c0&c1],[1,2,3],default=0).astype(np.int8)
    margin=np.where(c0,1-d['J0'].min(1),d['J0'].min(1)-1);binning=np.where(margin<=.02,0,np.where(margin<=.1,1,2)).astype(np.int8)
    gbin=np.minimum(g0,3).astype(np.int8);counts=np.stack([(d['kind']==k).sum(1) for k in range(4)],1)
    gc=np.stack([((d['kind']==k)&d['good0']).sum(1) for k in range(4)],1)
    gfrac=np.divide(gc,g0[:,None],out=np.full(gc.shape,np.nan,dtype=float),where=g0[:,None]>0)
    loss=event==2;lastlo=np.max(np.where(d['good0'],x['exit_lo'],-np.inf),axis=1);lasthi=np.max(np.where(d['good0'],x['exit_hi'],-np.inf),axis=1)
    lastlo=np.where(loss,lastlo,np.nan);lasthi=np.where(loss,lasthi,np.nan)
    return dict(sample_id=z['sample_id'],groups=z['groups'],track=z['track'],scene=z['scene'],sequence=z['sequence'],j0=j0,j1=j1,Q0=(d['A0']+.5*d['F0']).argmin(1),Q1=(d['A1']+.5*d['F1']).argmin(1),C0=c0,C1=c1,event=event,g0=g0,margin=margin,margin_bin=binning,g0_bin=gbin,J0_min=d['J0'].min(1),J1_min=d['J1'].min(1),native_ADE=d['A0'].min(1),native_FDE=d['F0'].min(1),dJ_fixed=d['dJ'][ii,j0],dA_fixed=d['dA'][ii,j0],dF_fixed=d['dF'][ii,j0],deltaJ_fixed=d['deltaJ'][ii,j0],deltaA_fixed=d['deltaA'][ii,j0],deltaF_fixed=d['deltaF'][ii,j0],uncertain_fixed=d['uncertain'][ii,j0],kind_fixed=d['kind'][ii,j0],all20_type_counts=counts,all20_type_fraction=counts/20,G0_type_counts=gc,G0_type_fraction=gfrac,G0_failed_slots=(d['good0']&~d['good1']).sum(1),no_definite_descent_all20=~((d['dJ'] < -1e-10)&~d['uncertain']).any(1),all20_certain=~d['uncertain'].any(1),last_G0_exit_lo=lastlo,last_G0_exit_hi=lasthi,last_G0_exit_le_half=loss&(lasthi<=.5),last_G0_exit_gt_half=loss&(lastlo>.5),last_G0_exit_half_bracket_uncertain=loss&~((lasthi<=.5)|(lastlo>.5)),correction_mean=d['correction_mean'].mean(1),correction_endpoint=d['correction_endpoint'].mean(1),Jslot_switched=j0!=j1,J_segments_native_fixed=d['segments0'][ii,j0],J_segments_refined_fixed=d['segments1'][ii,j0],cosine_fixed=d['path_cosine'][ii,j0],cosine_defined_fixed=d['cosine_defined'][ii,j0])
