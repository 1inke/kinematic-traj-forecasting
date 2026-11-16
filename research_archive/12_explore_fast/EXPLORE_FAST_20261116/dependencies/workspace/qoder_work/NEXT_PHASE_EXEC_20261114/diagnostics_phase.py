"""Fixed-anchor respline and descriptive geometry, using saved arrays only."""
import argparse
import numpy as np
from common import *
import exchange_null as ex
from evaluate_saved import save_population
from scipy.spatial.distance import pdist


def zero(seed,split,reference):
    data=dict(np.load(HERE/'C0'/f'{split}.npz'))
    d=HERE/'E2_zero'/f's{seed}_{split}'
    with np.load(reference) as ref:
        base=ex.quality(ref['paths'],data['future'])
    result={}
    for variant in ('natural','clamped'):
        with np.load(d/f'respline_{variant}.npz') as p:
            summary,q=ex.summarize(p,data,ex.fit_reference(HERE/'C0/MOTION_REFERENCE.json'))
            save_population(d/f'audited_{variant}',p['paths'],p['logits'],data)
        result[variant]=dict(summary=summary,exchange=ex.pair_exchange(base['coverage'],q['coverage'],data['group']))
    write_json(d/'DIAGNOSTIC_RESULT.json',dict(results=result,interpretation='Fixed linear-trained anchors rejoined; diagnostic, not a mathematical lower bound on trained performance.'))


def geometry(name,split):
    data=dict(np.load(HERE/'C0'/f'{split}.npz'));folder=HERE/'runs'/name/split
    with np.load(folder/'predictions.npz') as p:
        paths=p['paths'].astype(np.float64);logits=p['logits'];q=ex.quality(paths,data['future'])
    nn=[];same=[];overshoot=[];anchors=paths[:,:,[3,7,11]];past=data['past'][:,0].astype(float)
    for i,pp in enumerate(paths):
        ds=np.linalg.norm(pp[:,None]-pp[None,:],axis=-1);dd=ds.mean(-1);np.fill_diagonal(dd,np.inf);nn.append(dd.min(1))
        mask=np.triu(np.ones((20,20),bool),1);same.append(float(((ds.mean(-1)<=.1)&(ds[:,:,-1]<=.2))[mask].mean()))
    # Per-time perpendicular distance from its anchor segment; straight-line paths have zero.
    starts=np.concatenate([np.broadcast_to(past[:,-1,None,None],(len(paths),20,1,2)),anchors[:,:,:-1]],2)
    ends=anchors;segment=np.repeat(np.arange(3),4)
    aa=starts[:,:,segment];bb=ends[:,:,segment];vv=bb-aa
    alpha=np.clip(((paths-aa)*vv).sum(-1)/np.maximum((vv*vv).sum(-1),1e-20),0,1)
    deviation=np.linalg.norm(paths-aa-alpha[...,None]*vv,axis=-1)
    speed=np.linalg.norm(past[:,-1]-past[:,-2],axis=-1)/.4
    v1=past[:,-1]-past[:,-2];v2=past[:,-2]-past[:,-3]
    turning=np.abs(np.arctan2(v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0],(v1*v2).sum(-1)))>np.deg2rad(10)
    nn=np.asarray(nn);same=np.asarray(same);rows={}
    for g,mask in ex.group_masks(data['group']).items():
        rows[g]=dict(n=int(mask.sum()),nearest_neighbor_meanpath_p10_p50_p90=np.quantile(nn[mask],[.1,.5,.9]).tolist(),
            within_same_distance_cluster_pair_fraction=float(same[mask].mean()),
            anchor_polyline_deviation_p50_p95_max=[float(np.quantile(deviation[mask],.5)),float(np.quantile(deviation[mask],.95)),float(deviation[mask].max())])
    ports={}
    motion=ex.motion_arrays(paths,past)
    for label,mask in [('stationary',speed<.05),('slow',(.05<=speed)&(speed<.5)),('normal',(.5<=speed)&(speed<1.5)),('fast',speed>=1.5),('observed_turn',turning),('no_observed_turn',~turning)]:
        if mask.any():ports[label]=dict(n=int(mask.sum()),coverage=float(q['coverage'][mask].mean()),junction_accel_mean=float(motion['junction_acceleration'][mask].mean()))
    write_json(folder/'GEOMETRY_DIAGNOSTICS.json',dict(groups=rows,observation_strata=ports,
        thresholds='m/s stationary<.05, slow<.5, normal<1.5, fast>=1.5; observed turn>10deg; descriptive predeclared engineering bins'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode');ap.add_argument('name');ap.add_argument('split');ap.add_argument('--reference');a=ap.parse_args()
    if a.mode=='zero':zero(int(a.name),a.split,a.reference)
    elif a.mode=='geometry':geometry(a.name,a.split)
    else:raise ValueError('UNKNOWN_DIAGNOSTIC')
