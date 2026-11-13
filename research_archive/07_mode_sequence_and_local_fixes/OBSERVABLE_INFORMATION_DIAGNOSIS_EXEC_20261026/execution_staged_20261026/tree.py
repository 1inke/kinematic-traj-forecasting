"""CPU-only fixed HGB, shared IPW rows. No Torch or midway-resume claim."""
from base import *
G=guard('TREE')
import sklearn,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

def parameters(seed):return dict(loss='log_loss',learning_rate=.05,max_iter=64,max_leaf_nodes=8,max_depth=3,min_samples_leaf=20,l2_regularization=.1,max_bins=64,early_stopping=False,categorical_features=None,random_state=seed)

def training_rows(seed,limit=None):
    with np.load(HERE/f'prepared/actions_seed{seed}.npz') as z:a=z['actions'];w=z['weights'];y=z['labels']
    if limit:a,w,y=a[:limit],w[:limit],y[:limit]
    valid=w>0;ix,slot=np.nonzero(valid);mm=np.load(OLD/f'prepared_v1/features_seed{seed}.npy',mmap_mode='r');x=np.array(mm[ix,a[ix,slot]],copy=True);mm._mmap.close();x=normalize(x,seed)
    assert len(x)<=131072;return x,y[ix,slot],w[ix,slot],ix,a[ix,slot]

def fit(arm,seed):
    if arm=='T_perm':raise RuntimeError('Registered permutation admission is BLOCKED; no fit')
    out=HERE/'step2'/f'{arm}_seed{seed}';out.mkdir(parents=True,exist_ok=True);c=dict(source=source(),arm=arm,seed=seed,table_sha=sha256(HERE/f'prepared/actions_seed{seed}.npz'),stats_sha=sha256(OLD/f'prepared_v1/stats_seed{seed}.npz'),parameters=parameters(seed),sklearn=sklearn.__version__,midfit_resume=False)
    if (out/'done.json').exists():completed(out,c);return
    if list(out.iterdir()):raise RuntimeError('Preserve partial HGB; no exact midfit resume or blind rerun')
    x,y,w,ix,a=training_rows(seed)
    if arm=='T_geom':x=x[:,234:324].copy()
    write_json(out/'manifest.json',dict(**c,rows=len(x),columns=x.shape[1],row_ID_sha=arrsha(np.stack((ix,a),1)),weight_sha=arrsha(w),original_weights_unscaled=True))
    mark('HGB_FIT_FIXED64',arm=arm,seed=seed,rows=len(x));G.check();t=time.perf_counter()
    with threadpool_limits(limits=1):m=HistGradientBoostingClassifier(**parameters(seed)).fit(x,y,sample_weight=w)
    assert m.n_iter_==64;joblib.dump(m,out/'final.joblib');write_json(out/'TRAIN.json',dict(iterations=m.n_iter_,seconds=time.perf_counter()-t));resource_report(out,G);finish(out,c)

def predict(arm,seed,split):
    from evaluate_saved import save_population
    out=HERE/'step2'/f'{arm}_seed{seed}';d=read_json(out/'done.json');completed(out,d['contract']);lock=read_json(HERE/'ALL_PROBE_FINALS_LOCK.json');assert lock['models'][f'{arm}_seed{seed}']==sha256(out/'final.joblib');m=joblib.load(out/'final.joblib')
    def score(raw):
        x=normalize(raw,seed)
        if arm=='T_geom':x=x[...,234:324]
        with threadpool_limits(limits=1):return m.decision_function(x.reshape(-1,x.shape[-1])).reshape(raw.shape[:2])
    save_population(arm,seed,split,score,G)

if __name__=='__main__':
    try:
        if sys.argv[1]=='fit':fit(sys.argv[2],int(sys.argv[3]))
        else:predict(sys.argv[2],int(sys.argv[3]),sys.argv[4])
    except BaseException as e:fail(e);raise
