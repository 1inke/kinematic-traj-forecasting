"""Saved paths only; independent scalar/complete-link/motion reconstruction, no neural forward."""
import core as c
import numpy as np,time

def compare(a,b):
    if isinstance(a,dict):assert a.keys()==b.keys();return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(list,np.ndarray)):return float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
    if isinstance(a,(str,bool)):assert a==b;return 0.
    return abs(float(a)-float(b))

def main():
    t=time.perf_counter();jobs=c.read(c.H/'JOBS.json');files=c.read(c.H/'source_manifest.json')['files'];c.verify(files);data,sp=c.load_data();lookup={int(x):i for i,x in enumerate(data['ids'])};maximum=0.;rows=[]
    for name,r in jobs.items():
        c.guard();assert c.sha(r['file'])==r['SHA'];z=c.npz(r['file']);ix=np.array([lookup[int(x)] for x in z['sample_id']]);assert len(np.unique(ix))==len(ix)
        assert np.array_equal(z['gt'],data['gt'][ix]) and np.array_equal(z['groups'],data['groups'][ix]) and np.array_equal(z['track'],data['track'][ix])
        p,res,lam=c.decode_numpy(data['anchor'][ix],z['bounded_controls'],'position3');assert np.max(abs(p-z['refined4']))<=2e-12
        assert np.max(abs(res-z['residual']))<=2e-12
        expected=data['native'][ix].astype(np.float64).copy();expected[np.arange(len(ix))[:,None],z['slots']]=p;assert np.max(abs(expected-z['paths']))<=2e-12
        keep=np.ones((len(ix),20),bool);keep[np.arange(len(ix))[:,None],z['slots']]=False;assert np.array_equal(z['paths'][keep],data['native'][ix][keep])
        meta=r['metadata'];baseline=f"s{meta['seed']}_step0_probe" if meta['split']=='probe' else f"s{meta['seed']}_p{meta['packet']}_step0_source"
        base=c.npz(jobs[baseline]['file'])['paths'];met=c.metrics(z,base,True);error=compare(met,r['metrics']);maximum=max(maximum,error);assert error<=2e-12,(name,error)
        rows.append(dict(name=name,n=len(ix),independent_metric_error=error,fixed16_EXACT=True,position_bound=bool(np.all(np.linalg.norm(z['residual'],axis=-1)<=.20*(np.arange(1,13)/12.)**2+2e-15))))
    c.verify({r['file']:r['SHA'] for r in jobs.values()});c.verify(files)
    c.write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=len(rows),rows=rows,max_error=maximum,all_complete_quality_clusters_pooled_five_motion=True,NoNeuralForward=True,NoOptimizer=True,source_output_SHA_initial_final=True,elapsed_seconds=time.perf_counter()-t))
    print('SAVED_AUDIT_PASS',len(rows),maximum,flush=True)

if __name__=='__main__':main()
