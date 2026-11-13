from common import *
def remove_metrics(a,f,j,good):
    n,k=a.shape;delta=[]
    for v in (j,a,f):
        s=np.sort(v,axis=1);small=s[:,0];unique=(v==small[:,None]).sum(1)==1
        delta.append(np.where((v==small[:,None])&unique[:,None],(s[:,1]-small)[:,None],0))
    lost=good&(good.sum(1)==1)[:,None]
    return dict(deltaJ=delta[0],deltaA=delta[1],deltaF=delta[2],lost_if_deleted=lost,strict_gt_removable=~lost&(delta[0]==0)&(delta[1]==0)&(delta[2]==0),tol_gt_removable=~lost&(delta[0]<=2e-12)&(delta[1]<=2e-12)&(delta[2]<=2e-12),winJ=j==j.min(1)[:,None],winA=a==a.min(1)[:,None],winF=f==f.min(1)[:,None],nearwinJ=abs(j-j.min(1)[:,None])<=2e-12,nearwinA=abs(a-a.min(1)[:,None])<=2e-12,nearwinF=abs(f-f.min(1)[:,None])<=2e-12)
def scalar_check(a,f,j,good,record):
    for row in range(len(a)):
        for k in range(20):
            keep=[l for l in range(20) if l!=k]
            for name,v in [('J',j),('A',a),('F',f)]:
                value=min(float(v[row,l]) for l in keep)-min(map(float,v[row]));assert abs(record['delta'+name][row,k]-value)<=2e-12
            lost=any(good[row]) and not any(good[row,l] for l in keep);assert record['lost_if_deleted'][row,k]==lost
