"""Second implementation: expand sampled tracks to row indices each draw."""
import numpy as np

def independent_draws(treatment,control,tracks,draws=20000,seed=20261020):
    t=np.asarray(treatment);c=np.asarray(control);tr=np.asarray(tracks);values=np.unique(tr);rows=[np.flatnonzero(tr==x) for x in values];rng=np.random.Generator(np.random.PCG64(seed));answer=np.empty(draws)
    for i in range(draws):
        sampled_tracks=rng.integers(0,len(values),len(values));sampled_seeds=rng.integers(0,len(t),len(t));ix=np.concatenate([rows[j] for j in sampled_tracks]);per=[]
        for s in sampled_seeds:per.append(float(np.mean(t[s,ix])-np.mean(c[s,ix])))
        answer[i]=sum(per)/len(per)
    return answer
