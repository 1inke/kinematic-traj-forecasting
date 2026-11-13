import common as c
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import sklearn

def calculate(d):
    ix=d['split']['FIT'];out=dict(FIT_ids=d['ids'][ix])
    for prefix,x in [('p2',d['p2'][ix].astype(float)),('v',d['v0'][ix].astype(float)),('z',d['gt'][ix,-1].astype(float))]:
        out[prefix+'_mean']=x.mean(0);sd=x.std(0);out[prefix+'_std']=np.where(sd<1e-6,1.,sd)
    residual=d['gt'][ix,-1].astype(float)-4.8*d['v0'][ix]
    with threadpool_limits(1):km=KMeans(n_clusters=20,init='k-means++',n_init=1,random_state=20261019,max_iter=100,tol=1e-4,algorithm='lloyd').fit(residual)
    centers=km.cluster_centers_;order=np.lexsort((np.arange(20),centers[:,1],centers[:,0]));inverse=np.argsort(order)
    out.update(centers12=centers[order],centers_original_order=order,kmeans_label=inverse[km.labels_],kmeans_iterations=np.array(km.n_iter_),sklearn_version=np.array(sklearn.__version__))
    return out

def main():
    d=c.load();assert c.read('H0_RESULT.json')['status']=='PASS_H0_ALLOW_H1_PREFLIGHT'
    stats=calculate(d);c.save('shared_statistics.npz',**stats)
    c.write('STATISTICS_CONTRACT.json',dict(FIT_ids_SHA=c.array_sha(stats['FIT_ids']),FIT_n=len(stats['FIT_ids']),file_SHA=c.sha(c.H/'shared_statistics.npz'),KMeans=dict(n_clusters=20,init='k-means++',n_init=1,random_state=20261019,max_iter=100,tol=1e-4,algorithm='lloyd',sklearn_version=str(stats['sklearn_version'])),normalization='float64 FIT-only mean/std; std<1e-6 ->1; model float32 buffers',z_stats='old 12-step displacement, shared by both arms',slot_sort='center_x,center_y,original_cluster_index'))
if __name__=='__main__':main()
