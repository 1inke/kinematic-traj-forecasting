from common import *
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage,fcluster
FEATURES=('endpoint_x','endpoint_y','sin_turn','cos_turn','mean_speed','delta_speed','straightness','turn_defined')
def features(native,past,floor):
    # ONLY past and predictions: future GT/groups/IDs are not function inputs.
    p=np.asarray(native,float);past=np.asarray(past,float);q0=past[:,-1];prev=np.concatenate((np.broadcast_to(q0[:,None,None],(len(p),p.shape[1],1,2)),p[:,:,:-1]),axis=2)
    v=(p-prev)/.4;s=np.linalg.norm(v,axis=-1);end=p[:,:,-1]-q0[:,None];early=p[:,:,3]-q0[:,None];late=p[:,:,-1]-p[:,:,7]
    L=(s*.4).sum(-1);disp=np.linalg.norm(end,axis=-1);es=s[:,:,:4].mean(-1);ls=s[:,:,8:].mean(-1);mean=s.mean(-1);ds=ls-es;ratio=ds/np.maximum(es,floor)
    normE=np.linalg.norm(early,axis=-1);normL=np.linalg.norm(late,axis=-1)
    defined=(normE>floor*1.6)&(normL>floor*1.6);turn=wrapped(np.arctan2(late[...,1],late[...,0])-np.arctan2(early[...,1],early[...,0]));turn=np.where(defined,turn,np.nan)
    low=(mean<=floor)&(disp<=floor*4.8);h=past[:,-1]-past[:,-5];hd=np.linalg.norm(h,axis=-1)>floor*1.6;ha=np.arctan2(h[:,1],h[:,0])
    dx=end[...,0]*np.cos(ha[:,None])+end[...,1]*np.sin(ha[:,None]);dy=-end[...,0]*np.sin(ha[:,None])+end[...,1]*np.cos(ha[:,None])
    dx=np.where(hd[:,None],dx,disp);dy=np.where(hd[:,None],dy,0)
    straight=np.divide(disp,L,out=np.zeros_like(disp),where=L>0)
    phi=np.stack((dx,dy,np.where(defined,np.sin(turn),0),np.where(defined,np.cos(turn),0),mean,ds,straight,defined.astype(float)),axis=-1)
    velacc=np.diff(v,axis=2)/.4;jerk=np.diff(velacc,axis=2)/.4
    motion=np.stack((np.quantile(s,.95,axis=2),np.linalg.norm(velacc,axis=-1).mean(-1),np.quantile(np.linalg.norm(velacc,axis=-1),.95,axis=2),np.linalg.norm(jerk,axis=-1).mean(-1),np.quantile(np.linalg.norm(jerk,axis=-1),.95,axis=2)),axis=-1)
    return dict(phi=phi,turn=turn,turn_defined=defined,low=low,mean_speed=mean,early_speed=es,late_speed=ls,delta_speed=ds,speed_ratio=ratio,length=L,displacement=disp,straightness=straight,history_heading_defined=hd,relative_displacement_angle=np.where(hd[:,None]&(disp>floor*4.8),wrapped(np.arctan2(end[...,1],end[...,0])-ha[:,None]),np.nan),motion=motion)
def categories(f,theta,floor,joint=False):
    out=np.full(f['mean_speed'].shape,1,np.int16);ok=f['turn_defined']&~f['low'];out[ok]=2
    out[ok&(f['turn']>np.deg2rad(theta))]=3;out[ok&(f['turn']< -np.deg2rad(theta))]=4;out[f['low']]=0
    trend=np.ones_like(out);trend[(f['speed_ratio']>.2)&(f['delta_speed']>=floor)]=2;trend[(f['speed_ratio']<-.2)&(-f['delta_speed']>=floor)]=0
    if joint:out=np.where(out>=2,2+(out-2)*3+trend,out).astype(np.int16)
    return out,trend
def configuration(train):
    past=train['past'].astype(float);sv=np.linalg.norm(np.diff(past,axis=1),axis=-1)/.4
    resolution=float(np.quantile(abs(np.spacing(train['past'])),.95));emp=float(np.quantile(sv,.05));floor=max(emp,4*resolution/.4,np.finfo(float).eps)
    f=features(train['native'],train['past'],floor);mu=f['phi'].mean((0,1));sd=f['phi'].std((0,1));sd[sd<1e-12]=1
    scaled=(f['phi']-mu)/sd;nearest=[]
    for q in scaled:
        d=squareform(pdist(q));np.fill_diagonal(d,np.inf);nearest.extend(d.min(1))
    tau=float(np.quantile(nearest,.5));tau=max(tau,1e-12)
    return dict(version='fixed_native_mode_audit_v1',dt=.4,slow_floor=floor,empirical_history_speed_Q05=emp,numerical_speed_floor=4*resolution/.4,slow_floor_is_operational_proxy_not_sensor_noise=True,feature_names=FEATURES,phi_mean=mu.tolist(),phi_std=sd.tolist(),near_morphology_tau=tau,turn_degrees=20,sensitivity_turn=[15,30],slow_multipliers=[.5,2],clustering_multipliers=[.5,1,2],quality_ADE=.2,quality_FDE=.4,delete_numeric_tolerance=2e-12,bootstrap=20000,bootstrap_seed=20261027,no_GT_used_to_fit=True)
def variants():
    return [('coarse20',20,1,False,None),('joint20',20,1,True,None),('coarse15',15,1,False,None),('coarse30',30,1,False,None),('floor_half',20,.5,False,None),('floor_double',20,2,False,None),('cluster_half',20,1,False,.5),('cluster_main',20,1,False,1),('cluster_double',20,1,False,2)]
def assign(x,cfg,variant):
    name,angle,mult,joint,cluster=variant;floor=cfg['slow_floor']*mult;f=features(x['native'],x['past'],floor)
    phi=(f['phi']-np.asarray(cfg['phi_mean']))/np.asarray(cfg['phi_std'])
    mode,trend=categories(f,angle,floor,joint)
    if cluster is not None:
        mode=np.array([fcluster(linkage(pdist(q),method='complete'),cfg['near_morphology_tau']*cluster,criterion='distance')-1 for q in phi],np.int16)
    reps=np.zeros_like(mode,bool);rep_neartie=np.zeros_like(mode,bool);count=np.zeros_like(mode,np.int16);distrep=np.zeros_like(mode,float)
    M=[];Me=[];P=[];same=[];near=[]
    for i,(q,labels) in enumerate(zip(phi,mode)):
        D=squareform(pdist(q));ns=[]
        for label in np.unique(labels):
            ix=np.flatnonzero(labels==label);cost=D[np.ix_(ix,ix)].mean(1);winner=ix[np.argmin(cost)];reps[i,winner]=True;rep_neartie[i,ix[np.abs(cost-cost.min())<=2e-12]]=True;count[i,ix]=len(ix);distrep[i,ix]=D[ix,winner];ns.append(len(ix))
        ns=np.asarray(ns);p=ns/20.;M.append(len(ns));Me.append(np.exp(-(p*np.log(p)).sum()));P.append(p.max());same.append((ns*(ns-1)/2).sum()/190);near.append(float((pdist(q)<=cfg['near_morphology_tau']).mean()))
    structure=(count>=2)&~reps
    return dict(**f,mode=mode,trend=trend,representative=reps,representative_neartie=rep_neartie,mode_count=count,distance_to_medoid=distrep,structural=structure,M=np.asarray(M),Meff=np.asarray(Me),Pmax=np.asarray(P),same_pair=np.asarray(same),near_phi_pair=np.asarray(near))
