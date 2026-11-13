"""Separate vectorized Hungarian primal/dual arithmetic for matching audit."""
import numpy as np
def hungarian(cost):
    a=np.asarray(cost,np.float64);n=len(a);u=np.zeros(n+1);v=np.zeros(n+1);p=np.zeros(n+1,np.int32);way=np.zeros(n+1,np.int32)
    for i in range(1,n+1):
        p[0]=i;j0=0;minimum=np.full(n+1,np.inf);used=np.zeros(n+1,bool)
        while True:
            used[j0]=True;i0=p[j0];cur=a[i0-1]-u[i0]-v[1:];available=~used[1:]
            improved=available&(cur<minimum[1:]);minimum[1:][improved]=cur[improved];way[1:][improved]=j0
            candidates=np.flatnonzero(available)+1;j1=int(candidates[np.argmin(minimum[candidates])]);delta=minimum[j1]
            u[p[used]]+=delta;v[used]-=delta;minimum[~used]-=delta;j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    mapping=np.empty(n,np.int32)
    for j in range(1,n+1):mapping[p[j]-1]=j-1
    total=float(a[np.arange(n),mapping].sum());scale=max(1,abs(total))
    assert np.min(a-u[1:,None]-v[None,1:])>=-2e-12*scale
    assert abs(total-u[1:].sum()-v[1:].sum())<=2e-12*scale
    return mapping,total
