import common as c
import numpy as np

def evaluate(z,ind=False):
    p=z['paths'];native=z['native'];gt=z['gt'];a,f,j,g,early=c.errors(p,gt,ind);na,nf,nj,ng,_=c.errors(native,gt,ind);cc=c.clusters(p,g,ind);gg=g.any(1);basegood=ng.any(1)
    error=np.sqrt(((p-gt[:,None])**2).sum(-1));qw=(a+.5*f).argmin(1);jw=j.argmin(1);nw=(na+.5*nf).argmin(1)
    aux=dict(ADE=a,FDE=f,J=j,good=g,clusters=cc,vectors4=np.stack((gg,a.min(1),f.min(1),j.min(1)),1),native_vectors4=np.stack((basegood,na.min(1),nf.min(1),nj.min(1)),1),Q_winner=qw,J_winner=jw,native_Q_winner=nw,Q_segments=error[np.arange(len(p)),qw].reshape(-1,3,4).mean(-1),J_segments=early)
    v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4;acc=np.diff(v,axis=2)/.4;jerk=np.diff(acc,axis=2)/.4
    mag=[np.sqrt((x*x).sum(-1)).reshape(len(p),-1) for x in (v,acc,jerk)]
    aux['window_motion5']=np.stack((np.quantile(mag[0],.95,axis=1),mag[1].mean(1),np.quantile(mag[1],.95,axis=1),mag[2].mean(1),np.quantile(mag[2],.95,axis=1)),1)
    rows={}
    for name,m in c.masks(z['groups']):
        if not m.any():rows[name]=dict(n=0);continue
        mot=c.motion(p[m],ind);base=c.motion(native[m],ind);den=int((basegood&m).sum());drift=np.linalg.norm(p[m]-native[m],axis=-1)
        rows[name]=dict(n=int(m.sum()),tracks=len(set(z['track'][m])),good=int((gg&m).sum()),native_good=den,coverage=float(gg[m].mean()),rescue=int((gg&~basegood&m).sum()),lost=int((~gg&basegood&m).sum()),retained=int((gg&basegood&m).sum()),lost_rate_native_good=float((~gg&basegood&m).sum()/den) if den else 0.,ADE=float(a[m].min(1).mean()),FDE=float(f[m].min(1).mean()),J=float(j[m].min(1).mean()),native_ADE=float(na[m].min(1).mean()),native_FDE=float(nf[m].min(1).mean()),native_J=float(nj[m].min(1).mean()),effective_good=float(cc[m,1].mean()),distinct=float(cc[m,0].mean()),Q_segments=aux['Q_segments'][m].mean(0),J_segments=early[m].mean(0),motion=mot,native_motion=base,motion_pass=all(mot[k]<=1.01*base[k] for k in mot),correction_mean=float(drift.mean()),correction_p95=float(np.quantile(drift,.95)),correction_max=float(drift.max()),Q_winner_switch_rate=float((qw[m]!=nw[m]).mean()),Q_winner_counts=np.bincount(qw[m],minlength=20),J_winner_counts=np.bincount(jw[m],minlength=20),rescue_IDs=z['sample_id'][gg&~basegood&m],lost_IDs=z['sample_id'][~gg&basegood&m],native_success_mean_vectors4=(aux['vectors4'][basegood&m].mean(0) if np.any(basegood&m) else None),native_failure_mean_vectors4=(aux['vectors4'][~basegood&m].mean(0) if np.any(~basegood&m) else None))
    return rows,aux

def validate(z,reported,aux_saved,arm):
    import model as m
    if arm!='native20':
        p,u,r=m.reconstruct(z['native'],z['raw'],arm,z['controls']);c.close(p,z['paths']);c.close(r,z['correction'])
        assert np.all(np.linalg.norm(z['correction'],axis=-1)<=.20*(np.arange(1,13)/12)**2+2e-12)
    else:assert np.array_equal(z['paths'],z['native'])
    rows,aux=evaluate(z,True);c.compare(rows,reported)
    for k,v in aux.items():
        if v.dtype==bool or np.issubdtype(v.dtype,np.integer):assert np.array_equal(v,aux_saved[k]),k
        else:c.close(v,aux_saved[k])
    return dict(status='PASS',full_saved_quality_original_complete_link_motion_recomputed=True,motion_PASS_not_required_for_execution=True,no_new_model_forward=True)
