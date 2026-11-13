import common as c
import numpy as np
TAG='20261016-safe-reachability-v1'
def rank(kind,value):
    # Explicit UTF8 canonical JSON encoding removes concatenation ambiguities.
    return c.digest([TAG,kind,str(value)])
def select(z):
    good=c.errors(z['native'],z['gt'])[3].any(1);mask=z['groups']==2;used=set();selected=[]
    tracks=sorted(set(z['track'][mask]),key=lambda t:(rank('track',t),t))
    for desired,count,label in ((False,32,'native_failure'),(True,16,'native_success_identity')):
        chosen=0
        for track in tracks:
            if track in used:continue
            ii=np.flatnonzero(mask&(z['track']==track)&(good==desired))
            if not len(ii):continue
            index=min(ii,key=lambda i:(rank('sample_id',int(z['sample_id'][i])),int(z['sample_id'][i])))
            selected.append(dict(index=int(index),sample_id=int(z['sample_id'][index]),track=str(track),sequence=str(z['sequence'][index]),scene=str(z['scene'][index]),stratum=label,native_good=bool(desired)))
            used.add(track);chosen+=1
            if chosen==count:break
        assert chosen==count,('INSUFFICIENT_TRACKS',label,chosen)
    assert len(selected)==48 and len(used)==48
    return dict(tag=TAG,serialization='SHA256 canonical JSON [tag,kind,str(canonical sequence::pedestrian or sample_id)] UTF8 ensure_ascii default; sorted hash then lexical tie',selection=selected,solver_attempts=640,FIT_n=18909,FIT_change_n=4725,ambiguous_times_retained_not_independent_claim=True,GT_strata_oracle_only=True)
