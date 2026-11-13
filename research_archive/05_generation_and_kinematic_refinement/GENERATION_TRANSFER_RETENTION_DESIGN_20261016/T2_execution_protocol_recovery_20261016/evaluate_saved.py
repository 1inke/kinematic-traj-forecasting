import common as c,model as m,metrics as v,objectives as o
import numpy as np,torch,gc,time

def audit(data,files,source,jobs,runs):
    started=time.perf_counter();c.verify(files);records=[]
    for key,row in jobs.items():
        c.stage('INDEPENDENT_SAVED_AUDIT',job=key,done=len(records),total=28);c.valid_done(row,source if key.startswith('C/') else c.read(c.T1/'PREFLIGHT.json')['source_digest']);z=c.npz(row['output']);aux=c.npz(row['aux']);met=c.read(row['metrics']);split=key.split('/')[-1];ix=data['split'][split]
        for field in ('native','sample_id','gt','groups','track','scene','sequence'):assert np.array_equal(z[field],data[field][ix]),field
        v.validate(z,met,aux,row['arm'])
        if row['population'] is not None:
            independent=o.numpy_components(z['paths'],z['native'],z['gt']);pop=row['population'];c.close(pop['base'],independent['base']);c.close(pop['keep'],independent['keep']);c.close(pop['total'],pop['base']+(pop['keep'] if row['cell']=='B' else 0));assert pop['n']==len(ix)
        records.append(dict(job=key,n=len(ix),complete_quality_original_clusters_motion_direction_keep_recomputed=True));del z,aux;gc.collect()
    for key,row in runs.items():
        c.valid_done(row,source);folder=c.H/'runs'/key;state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);history=c.read(folder/'history.json');m.exact(state['history'],history);assert len(history)==25 and sum(r['optimization_steps'] for r in history)==1850
        for r in history:
            assert r['n']==18909 and r['optimization_steps']==74 and sum(r['winner_counts'])==18909 and sum(r['g0_counts'])==18909
            assert sum(r['keep_slot_counts'])==r['success'] and np.sum(r['winner_keep_cross'])==r['success'] and r['keep_active']<=r['success']
            c.close(r['base'],r['quality']+.05*r['diversity']);c.close(r['total'],r['base']+(r['keep'] if key.startswith('B/') else 0))
        st=c.npz(folder/'statistics.npz');assert np.array_equal(st['training_ids'],data['ids'][data['split']['FIT']]);assert all(torch.isfinite(x).all() for x in state['model'].values());assert state['model']['native_identity_flag'].dtype==torch.bool
    import bootstrap
    c.compare(c.read(c.H/'STATISTICS.json'),bootstrap.compute(True));c.verify(files);c.write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=28,science_jobs=24,new_science_jobs=12,reused_A_science_jobs=12,native_jobs=4,models=3,history_rows=75,steps=5550,records=records,independent_saved_math_and_bootstrap=True,not_all_model_or_optimizer_replay=True,elapsed_seconds=time.perf_counter()-started))
