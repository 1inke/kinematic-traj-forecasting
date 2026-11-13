import common as c
import numpy as np
c.extract(c.PARENT/'metrics.py',('evaluate','validate'),globals())
legacy_evaluate=evaluate;legacy_validate=validate

def directional(z,ind=False):
    namespace=dict(np=np,TOL=1e-10)
    c.extract(c.U1/'direction_metrics.py',('quality','candidate'),namespace)
    rows=[]
    for start in range(0,len(z['paths']),256):
        c.guard();p=z['native'][start:start+256];q=z['paths'][start:start+256];gt=z['gt'][start:start+256]
        d=namespace['candidate'](p,q,gt,ind);ii=np.arange(len(p));j0=d['J0'].argmin(1);g0=d['good0'];ng=g0.any(1);margin=np.where(ng,1-d['J0'].min(1),d['J0'].min(1)-1)
        rows.append(dict(fixed_dJ=d['dJ'][ii,j0],fixed_deltaJ=d['deltaJ'][ii,j0],fixed_uncertain=d['uncertain'][ii,j0],all20_dJ=d['dJ'].mean(1),all20_deltaJ=d['deltaJ'].mean(1),G0_dJ_sum=np.where(g0,d['dJ'],0).sum(1),G0_deltaJ_sum=np.where(g0,d['deltaJ'],0).sum(1),g0=g0.sum(1),native_good=ng,margin=margin,margin_bin=np.where(margin<=.02,0,np.where(margin<=.1,1,2)),g0_bin=np.minimum(g0.sum(1),3),original_j0=j0))
    return {key:np.concatenate([row[key] for row in rows]) for key in rows[0]}

def evaluate(z,ind=False):
    import objectives as o
    rows,aux=legacy_evaluate(z,ind);aux.update(directional(z,ind));parts=o.numpy_components(z['paths'],z['native'],z['gt'])
    aux.update(keep_values=parts['values'],keep_slot=parts['slot'],keep_active=parts['active'],native_reference_J=parts['reference'],native_good_mask=parts['good0'])
    for group,mask in c.masks(z['groups']):
        row=rows[group]
        if not mask.any():continue
        row.update(fixed_native_j0_dJ=float(aux['fixed_dJ'][mask].mean()),fixed_native_j0_deltaJ=float(aux['fixed_deltaJ'][mask].mean()),all20_dJ=float(aux['all20_dJ'][mask].mean()),all20_deltaJ=float(aux['all20_deltaJ'][mask].mean()),fixed_j0_uncertain=int(aux['fixed_uncertain'][mask].sum()))
        good=aux['native_good'];fail=~good;new=aux['good'].any(1)
        row['native_failure_fixed_dJ']=float(aux['fixed_dJ'][mask&fail].mean()) if (mask&fail).any() else None
        row['native_failure_rescue_rate']=float((new&mask&fail).sum()/(mask&fail).sum()) if (mask&fail).any() else None
        row['G0_dJ']=float(aux['G0_dJ_sum'][mask].sum()/aux['g0'][mask].sum()) if aux['g0'][mask].sum() else None
        row['G0_deltaJ']=float(aux['G0_deltaJ_sum'][mask].sum()/aux['g0'][mask].sum()) if aux['g0'][mask].sum() else None
        row['strata']=[]
        for success in (False,True):
            for b in range(3):
                for g in ((1,2,3) if success else (0,)):
                    ss=mask&(good==success)&(aux['margin_bin']==b)&(aux['g0_bin']==g);n=int(ss.sum());tracks=len(set(z['track'][ss]))
                    row['strata'].append(dict(native_success=success,margin_bin=b,g0_bin=g,n=n,tracks=tracks,thin=n<20 or tracks<5,lost=int((ss&good&~new).sum()),rescue=int((ss&fail&new).sum())))
    return rows,aux

def population(net,data,ix,cell):
    import torch,model as m,objectives as o
    records=[]
    with torch.inference_mode():
        for start in range(0,len(ix),128):
            ii=ix[start:start+128];base=m.inputs(data,ii)[1];out=net(*m.inputs(data,ii))[0];gt=torch.from_numpy(data['gt'][ii]).to(m.DEVICE);loss,d=o.loss(out,base,gt,cell)
            records.append(dict(n=len(ii),total=float(loss),base=float(d['base']),quality=float(d['quality']),diversity=float(d['diversity']),keep=float(d['keep'])))
    n=sum(r['n'] for r in records)
    return dict(n=n,**{k:sum(r[k]*r['n'] for r in records)/n for k in records[0] if k!='n'},population_not_training_history=True)

def validate(z,reported,aux_saved,arm):
    import model as m
    if arm!='native20':
        p,u,r=m.reconstruct(z['native'],z['raw'],'integrated_acceleration',z['controls']);c.close(p,z['paths']);c.close(r,z['correction'])
        assert np.all(np.linalg.norm(r,axis=-1)<=.2*(np.arange(1,13)/12)**2+2e-12)
    else:assert np.array_equal(z['paths'],z['native'])
    rows,aux=evaluate(z,True);c.compare(rows,reported)
    for k,v in aux.items():
        if v.dtype==bool or np.issubdtype(v.dtype,np.integer):assert np.array_equal(v,aux_saved[k]),k
        else:c.close(v,aux_saved[k])
    return dict(status='PASS',complete_quality_clusters_five_motion_direction_strata_keep=True,no_model_forward=True)
