import common as c,metrics as v,model as m,objectives as o
import numpy as np,torch,gc,time
def audit(data,files,source,jobs,runs):
    start=time.perf_counter();c.verify(files);records=[];paired={};orders={}
    for key,row in jobs.items():
        c.stage('INDEPENDENT_SAVED_AUDIT',job=key,done=len(records),total=52);c.valid_done(row,source)
        z=c.npz(row['output']);aux=c.npz(row['aux']);split=key.split('/')[-1];ix=data['split'][split]
        for k in ('native','sample_id','gt','groups','track','scene','sequence'):assert np.array_equal(z[k],data[k][ix]),k
        v.validate(z,c.read(row['metrics']),aux,row['arm']);records.append(dict(job=key,n=len(ix),status='PASS'));del z,aux;gc.collect()
    for key,row in runs.items():
        c.valid_done(row,source);folder=c.H/'runs'/key;state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);hist=c.read(folder/'history.json');m.exact(state['history'],hist);assert len(hist)==25 and sum(r['optimization_steps'] for r in hist)==1850
        cell=key.split('/')[0];seed=key.split('/')[1];manifest=state['manifest']
        if seed in paired:assert paired[seed]==manifest['paired_init_SHA'];assert orders[seed]==[r['batchorder_SHA'] for r in hist]
        else:paired[seed]=manifest['paired_init_SHA'];orders[seed]=[r['batchorder_SHA'] for r in hist]
        assert manifest['owner_labels_SHA']==c.sha(c.H/'responsibility_labels.npz')
        for r in hist:
            assert r['n']==18909 and r['optimization_steps']==74 and sum(r['owner_counts'])==18909
            c.close(r['total'],r['task']+.05*r['diversity']+(r['anchor'] if cell in ('C','D') else 0))
            assert sum(x['n'] for x in r['steps'])==18909
        snapshots=[c.npz(folder/f'diagnostic_epoch{e:02d}.npz') for e in (0,1,13,25)]
        for q in snapshots:
            ii=data['split']['FIT'][:256];assert np.array_equal(q['sample_id'],data['ids'][ii])
            d=o.numpy_components(q['paths'],data['native'][ii],data['gt'][ii],cell)
            for k in ('Q','Q0','owner','dynamic','fixed'):
                c.close(q[k],d[k])
        c.write(folder/'snapshot_switching.json',dict(epochs=[0,1,13,25],adjacent_dynamic_switches=[int((a['dynamic']!=b['dynamic']).sum()) for a,b in zip(snapshots,snapshots[1:])],denominator=256,not_all_optimizer_step_switch_rate=True))
        assert all(torch.isfinite(x).all() for x in state['model'].values());assert state['model']['native_identity_flag'].dtype==torch.bool
    import bootstrap
    c.compare(c.read(c.H/'STATISTICS.json'),bootstrap.compute(True));c.verify(files)
    c.write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=52,science_jobs=48,native_jobs=4,models=12,history_rows=300,steps=22200,records=records,all_saved_independent_quality_original_clusters_motion_responsibility_drift_controls_bootstrap=True,not_model_gradient_replay=True,elapsed_seconds=time.perf_counter()-start))



