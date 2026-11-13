"""Endpoint-only extension of f16.scene_prior.HistogramPrior's hierarchy.

The old module is NOT edited. 257 mutually exhaustive bins, target purge,
unknown-time rejection, and nonsequence cold fallback replace its old 256-bin
normalization. Query API has no current future/GT argument.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from f16.scene_prior import HistogramPrior

ARMS=('M0_motion_iso','M1_heading_global','PRIOR_static','SHUF_static',
      'XSCENE_static','PRIOR_causal','SHUF_causal','XSCENE_causal')

@dataclass(frozen=True)
class Record:
    sample_id:int
    track:str
    sequence:str
    scene:str
    t0:int
    end_index:int
    cell:tuple
    heading:int
    endpoint:tuple
    verified_time:bool

def bucket(xy):
    xy=np.asarray(xy,float)
    if xy.shape!=(2,) or not np.isfinite(xy).all(): raise ValueError('Finite XY required')
    if np.any(np.abs(xy)>6): return 256
    ix=np.minimum(np.floor((xy+6)/.75).astype(int),15)
    return int(ix[0]*16+ix[1])

def histogram(records):
    return np.bincount([bucket(r.endpoint) for r in records],minlength=257).astype(np.int64)

def probability(counts):
    c=np.asarray(counts)
    if c.shape!=(257,) or np.any(c<0): raise ValueError('257 nonnegative counts required')
    return (c+.10)/(c.sum()+25.7)

def modes(counts):
    ix=np.argsort(-np.asarray(counts)[:256],kind='stable')[:4]
    return np.stack((-6+(ix//16+.5)*.75,-6+(ix%16+.5)*.75),1)

def heading(past_world):
    p=np.asarray(past_world,float);d=p[-1]-p[-2]
    if np.linalg.norm(d)<.02: d=p[-1]-p[0]
    if np.linalg.norm(d)<.02: return -1
    angle=(np.arctan2(d[1],d[0])+np.pi)%(2*np.pi)-np.pi
    return min(7,int(np.floor((angle+np.pi)/(2*np.pi/8))))

class EndpointPrior(HistogramPrior):
    def modes(self,answer,m=4):
        if m!=4: raise ValueError('Only registered top4 modes')
        return modes(answer['counts'])

    def __init__(self,records,fit_ids):
        super().__init__(records,fit_ids=fit_ids,kind='static',bins=16,extent=6.,alpha=.10)
        self.records=tuple(records)
        if len({r.sample_id for r in records})!=len(records): raise ValueError('Duplicate source ID')
        if {r.sample_id for r in records}!=set(map(int,fit_ids)): raise ValueError('FIT-only library ID mismatch')
        # Fixed permutation is constructed ONCE using FIT support, never GT query.
        self.static_maps={};rng=np.random.default_rng(20261023)
        for scene,h in sorted({(r.scene,r.heading) for r in records}):
            local=[r for r in records if (r.scene,r.heading)==(scene,h)]
            cells=sorted({r.cell for r in local})
            supported=[c for c in cells if sum(r.cell==c for r in local)>5]
            self.static_maps[(scene,h)]=dict(zip(supported,[supported[i] for i in rng.permutation(len(supported))]))

    def query(self,*,arm,sample_id,track,sequence,scene,t0,cell,heading,verified_time):
        if arm not in ARMS: raise ValueError(arm)
        eligible=[r for r in self.records if r.track!=track]
        def answer(chosen,level,cold=False):
            c=histogram(chosen)
            return dict(counts=c,probability=probability(c),support=len(chosen),level=level,cold_start=cold,
                source_ids=np.array([r.sample_id for r in chosen],np.int64),
                source_tracks=[r.track for r in chosen],source_end_indices=[r.end_index for r in chosen])
        if arm=='M0_motion_iso': return answer(eligible,'M0')
        if arm.startswith('XSCENE'):
            rs=[r for r in eligible if r.scene!=scene and (heading<0 or r.heading==heading)]
            return answer(rs,'cross_scene_heading' if heading>=0 else 'cross_scene_no_heading')
        if arm=='M1_heading_global':
            rs=[r for r in eligible if r.scene==scene and r.heading==heading] if heading>=0 else []
            return answer(rs,'scene_heading') if len(rs)>5 else answer(eligible,'M0')
        causal=arm.endswith('causal')
        if causal and not verified_time:
            return dict(status='time_unverifiable',counts=None,probability=None,level='time_unverifiable',cold_start=False)
        rs=[r for r in eligible if r.scene==scene]
        if causal: rs=[r for r in rs if r.verified_time and r.sequence==sequence and r.end_index<=t0]
        target_cell=tuple(cell)
        if arm.startswith('SHUF') and heading>=0:
            if causal:
                supported=sorted({r.cell for r in rs if r.heading==heading})
                payload=json.dumps([int(sample_id),20261023],separators=(',',':')).encode()
                sd=int.from_bytes(hashlib.sha256(payload).digest()[:8],'little')
                perm=np.random.default_rng(sd).permutation(len(supported))
                mapping=dict(zip(supported,[supported[i] for i in perm]))
            else: mapping=self.static_maps.get((scene,heading),{})
            target_cell=mapping.get(target_cell,target_cell)
        levels=[]
        if heading>=0: levels.append(([r for r in rs if r.cell==target_cell and r.heading==heading],'cell_heading'))
        levels.append(([r for r in rs if r.cell==target_cell],'cell'))
        if heading>=0: levels.append(([r for r in rs if r.heading==heading],'scene_heading'))
        for chosen,label in levels:
            if len(chosen)>5: return answer(chosen,label)
        if causal: return answer([r for r in eligible if r.sequence!=sequence],'M0_nonsequence',True)
        return answer(eligible,'M0')
