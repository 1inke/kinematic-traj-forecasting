from common import *
from legacy_morphology import features as old_features,categories
FLOOR=.006982410783134709
def variants():
    return [(f'a{a}_{h}_s{n}',a,h,n) for a in (20,15,30) for h in ('history3','robust4') for n in (2,3)]
def heading(past,kind='history3'):
    d=np.diff(past.astype(float),axis=1)
    if kind=='history3':
        u=past[:,-1].astype(float)-past[:,-4];parts=d[:,-3:];valid=np.linalg.norm(u,axis=-1)>FLOOR*1.2
    else:
        parts=d[:,-4:];ok=np.linalg.norm(parts,axis=-1)>.4*FLOOR
        u=np.zeros((len(past),2))
        for i in range(len(past)):
            if ok[i].any():u[i]=np.median(parts[i,ok[i]],axis=0)
        valid=ok.any(1)&(np.linalg.norm(u,axis=-1)>.4*FLOOR)
    length=np.linalg.norm(parts,axis=-1).sum(-1)
    coherence=np.divide(np.linalg.norm(parts.sum(1),axis=-1),length,out=np.zeros(len(past)),where=length>0)
    valid&=coherence>=.5
    return np.arctan2(u[:,1],u[:,0]),valid,coherence
def trend(ds,base):
    ratio=ds/np.maximum(base,FLOOR);v=np.ones(ds.shape,np.int8)
    v[(ratio>.2)&(ds>=FLOOR)]=2;v[(ratio<-.2)&(-ds>=FLOOR)]=0
    boundary=((np.abs(np.abs(ratio)-.2)<=.01)&(np.abs(ds)>=FLOOR))|((np.abs(np.abs(ds)-FLOOR)<=.05*FLOOR)&(np.abs(ratio)>.2))
    return v,ratio,boundary
def extract(p,past,angle=20,hkind='history3',sustain=2):
    p=np.asarray(p,float);past=np.asarray(past,float)
    old=old_features(p,past,FLOOR);l0,internaltrend=categories(old,angle,FLOOR)
    z=np.concatenate((np.broadcast_to(past[:,-1,None,None],(len(p),p.shape[1],1,2)),p),axis=2)
    d=np.diff(z,axis=2);speed=np.linalg.norm(d,axis=-1)/.4
    ha,hvalid,coh=heading(past,hkind)
    early=p[:,:,3]-past[:,-1,None];late=p[:,:,-1]-p[:,:,7];net=p[:,:,-1]-past[:,-1,None]
    ev=np.linalg.norm(early,axis=-1)>FLOOR*1.6
    nv=np.linalg.norm(net,axis=-1)>FLOOR*4.8
    entry=wrapped(np.arctan2(early[...,1],early[...,0])-ha[:,None])
    entryvalid=hvalid[:,None]&ev&~old['low']
    entry=np.where(entryvalid,entry,np.nan)
    netangle=np.where(hvalid[:,None]&nv,wrapped(np.arctan2(net[...,1],net[...,0])-ha[:,None]),np.nan)
    labels=np.full(entry.shape,3,np.int8);labels[entryvalid]=0
    threshold=np.deg2rad(angle)
    labels[entryvalid&(entry>threshold)]=1;labels[entryvalid&(entry< -threshold)]=2
    smooth=np.empty_like(d);smooth[:,:,0]=d[:,:,0];smooth[:,:,1:]=d[:,:,:-1]+d[:,:,1:]
    widths=np.array([1]+[2]*11)
    lv=np.linalg.norm(smooth,axis=-1)/(widths*.4)>FLOOR
    localangle=wrapped(np.arctan2(smooth[...,1],smooth[...,0])-ha[:,None,None]);lv&=hvalid[:,None,None]
    side=np.zeros(lv.shape,np.int8);side[lv&(localangle>threshold)]=1;side[lv&(localangle< -threshold)]=-1
    pos=np.zeros(entry.shape,bool);neg=pos.copy();onframe=np.zeros(entry.shape,np.int8)
    for t in range(12-sustain+1):
        a=np.all(side[:,:,t:t+sustain]==1,axis=2);b=np.all(side[:,:,t:t+sustain]==-1,axis=2)
        found=a|b;onframe[(onframe==0)&found]=t+1;pos|=a;neg|=b
    localturn=wrapped(np.diff(np.arctan2(smooth[...,1],smooth[...,0]),axis=2))
    validpair=lv[:,:,:-1]&lv[:,:,1:]
    abs_turn=(np.abs(localturn)*validpair).sum(-1)
    complex_=(pos&neg)|(abs_turn>=np.pi)
    onsetvalid=hvalid[:,None]&(lv.sum(-1)>=sustain)&~old['low']
    onset=np.zeros(entry.shape,np.int8)
    onset[(onframe>=1)&(onframe<=4)]=1;onset[(onframe>=5)&(onframe<=8)]=2;onset[onframe>=9]=3
    onset[complex_&onsetvalid]=4;onset[~onsetvalid]=5;onframe[~onsetvalid]=-1
    es=speed[:,:,:4].mean(-1);ms=speed[:,:,4:8].mean(-1);ls=speed[:,:,8:].mean(-1)
    hs=np.linalg.norm(np.diff(past[:,-4:],axis=1),axis=-1).mean(-1)/.4
    startds=es-hs[:,None];starttrend,startratio,sb=trend(startds,hs[:,None])
    intrend,inratio,ib=trend(ls-es,es)
    lowbits=(es<=FLOOR).astype(np.int8)+2*(ms<=FLOOR)+4*(ls<=FLOOR)
    stageboundary=np.any(np.abs(np.stack((es,ms,ls,old['mean_speed']),-1)-FLOOR)<=.05*FLOOR,axis=-1)
    angboundary=(entryvalid&(np.abs(np.abs(entry)-threshold)<=np.deg2rad(1)))|(old['turn_defined']&(np.abs(np.abs(old['turn'])-threshold)<=np.deg2rad(1)))|np.any(lv&(np.abs(np.abs(localangle)-threshold)<=np.deg2rad(1)),axis=-1)
    boundary=angboundary|stageboundary|sb|ib
    return dict(L0=l0,entry=labels,onset=onset,onset_frame=onframe,onset_valid=onsetvalid,onset_reliable=onsetvalid&~complex_,history_heading=ha,history_heading_valid=hvalid,history_coherence=coh,future_low=old['low'],entry_valid=entryvalid,entry_angle=entry,internal_angle=old['turn'],net_angle=netangle,early_speed=es,middle_speed=ms,late_speed=ls,history_speed=hs,start_delta_speed=startds,internal_delta_speed=ls-es,start_trend=starttrend,internal_trend=intrend,low_bits=lowbits,start_ratio=startratio,internal_ratio=inratio,abs_local_turn=abs_turn,complex_turn=complex_,boundary=boundary,angle_boundary=angboundary,speed_boundary=stageboundary|sb|ib,local_valid=lv,local_angle=np.where(lv,localangle,np.nan),length=old['length'],straightness=old['straightness'])
def associate(f,g):
    entry=(f['entry']==g['entry'])&(f['entry']!=3)&g['entry_valid']
    on=(f['onset']==g['onset'])&(f['onset']!=5)&g['onset_valid']
    speed=(f['start_trend']==g['start_trend'])&(f['internal_trend']==g['internal_trend'])&(f['low_bits']==g['low_bits'])
    eo=entry&on;full=eo&speed
    invalid=~g['entry_valid'][:,0];bd=g['boundary'][:,0]
    partial=(~invalid)&~g['onset_valid'][:,0]&~bd
    reason=np.full(len(entry),4,np.int8)
    reason[~full.any(1)]=3;reason[~eo.any(1)]=2;reason[~entry.any(1)]=1
    reason[partial]=5;reason[invalid|bd]=0
    return dict(entry_match=entry,onset_match=on,speed_match=speed,entry_onset_match=eo,full_match=full,reason=reason,GT_invalid=invalid,GT_boundary=bd,partial_valid=partial,common_valid=~(invalid|bd|partial))
