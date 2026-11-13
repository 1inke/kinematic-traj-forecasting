"""Complete single-edit set utility. No unique teacher or oracle class target."""
import numpy as np
import contracts_discrete as c

def targets(j_native,j_action):
    nj=np.asarray(j_native,float);ja=np.asarray(j_action,float)
    other=np.broadcast_to(nj[:,None,:],(len(nj),20,20)).copy()
    other[:,np.arange(20),np.arange(20)]=np.inf
    q=np.minimum(other.min(-1)[...,None],ja);qn=nj.min(1)[:,None,None]
    v=(q<=1).astype(float)-(qn<=1).astype(float)+.5*(np.clip(qn,0,2)-np.clip(q,0,2))
    return v,q

def independent_targets(nj,ja):
    v=np.empty_like(ja);q=np.empty_like(ja)
    for i in range(len(nj)):
        original=min(nj[i]);oldgood=original<=1
        for k in range(20):
            for a in range(ja.shape[2]):
                allJ=nj[i].copy();allJ[k]=ja[i,k,a];new=min(allJ)
                q[i,k,a]=new;v[i,k,a]=int(new<=1)-int(oldgood)+.5*(min(max(original,0),2)-min(max(new,0),2))
    return v,q

def label_data(d):
    d['value'],d['q_edit']=targets(d['nj'],d['labels']['J_matrix'][:,:,c.NONIDENTITY]);return d

def loss_numpy(score,value,valid):
    score=np.asarray(score,float);value=np.asarray(value,float);valid=np.asarray(valid,bool)
    grad=np.zeros_like(score);losses=[];parts=[]
    for i in range(len(score)):
        groups=[valid[i]&(value[i]>0),valid[i]&(value[i]<0),valid[i]&(value[i]==0)];nonempty=sum(m.any() for m in groups)
        total=0.;row={}
        for name,m in zip(('positive','negative','zero'),groups):
            n=int(m.sum());row[name+'_count']=n;row[name+'_MSE']=float(((score[i][m]-value[i][m])**2).mean()) if n else None
            if n:
                total+=row[name+'_MSE']/nonempty
                grad[i][m]=2*(score[i][m]-value[i][m])/(n*nonempty*len(score))
        losses.append(total);parts.append(row)
    return float(np.mean(losses)),grad,parts
