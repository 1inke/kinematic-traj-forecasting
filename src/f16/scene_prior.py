"""A-line auditable histogram bank. No current-window GT in query API.

This is an S1 component, not evidence that A0 information gates passed. Source
records and hyperparameters must be preregistered before actual A0 evaluation.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Fragment:
    sample_id:int
    track:str
    sequence:str
    scene:str
    end_index:int
    cell:tuple[int,int]
    heading:int
    endpoint:tuple[float,float]
    midpoint:tuple[float,float]

class HistogramPrior:
    def __init__(self,records,*,fit_ids,kind,bins,extent,alpha):
        if kind not in ('static','causal') or bins<2 or extent<=0 or alpha<=0:raise ValueError('Explicit prior contract required')
        self.records=tuple(records);self.fit_ids=frozenset(fit_ids);self.kind=kind;self.bins=bins;self.extent=extent;self.alpha=alpha
        if kind=='static' and any(r.sample_id not in self.fit_ids for r in self.records):raise ValueError('Static prior includes non-FIT source')
    def query(self,*,track,sequence,scene,t0,cell,heading):
        # Exclude EVERY frame/fragment from the current target, even in causal mode.
        eligible=[r for r in self.records if r.track!=track and (self.kind!='causal' or (r.sequence==sequence and r.end_index<=t0))]
        local=[r for r in eligible if r.scene==scene]
        levels=[([r for r in local if r.cell==cell and r.heading==heading],'cell_heading'),([r for r in local if r.cell==cell],'cell'),([r for r in local if r.heading==heading],'scene_heading'),(eligible,'global')]
        chosen,label=next(((rs,name) for rs,name in levels if len(rs)>5),levels[-1])
        histogram=np.full((self.bins,self.bins),self.alpha,dtype=float)
        for r in chosen:
            x,y=r.endpoint
            # Out-of-range mass is tracked rather than clipped to boundary modes.
            if -self.extent<=x<=self.extent and -self.extent<=y<=self.extent:
                a=min(self.bins-1,int((x+self.extent)/(2*self.extent)*self.bins));b=min(self.bins-1,int((y+self.extent)/(2*self.extent)*self.bins));histogram[a,b]+=1
        density=histogram/histogram.sum()
        return dict(density=density,support=len(chosen),level=label,source_ids=[r.sample_id for r in chosen],source_tracks=[r.track for r in chosen],source_end_indices=[r.end_index for r in chosen],cold_start=not bool(eligible),out_of_range=sum(not(-self.extent<=r.endpoint[0]<=self.extent and -self.extent<=r.endpoint[1]<=self.extent) for r in chosen),causal_same_recording_only=self.kind=='causal')
    def modes(self,answer,m=4):
        p=answer['density'];order=np.argsort(-p.ravel(),kind='stable')[:m];a,b=np.unravel_index(order,p.shape);step=2*self.extent/self.bins
        return np.stack((-self.extent+(a+.5)*step,-self.extent+(b+.5)*step),1)
