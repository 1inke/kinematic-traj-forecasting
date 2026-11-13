import common as c,model as m,metrics as v
import numpy as np,torch,gc,time

def main():
    start=time.perf_counter();files=c.read(c.H/'source_manifest.json')['files'];c.verify(files);data=c.load();jobs=c.read(c.H/'JOBS.json');runs=c.read(c.H/'RUNS.json');assert len(jobs)==28 and len(runs)==6;rows=[]
    for key,row in jobs.items():
        c.stage('U0_INDEPENDENT_SAVED_AUDIT',job=key,done=len(rows),total=28);c.valid_done(row,c.digest(files));z=c.npz(row['output']);aux=c.npz(row['aux']);met=c.read(row['metrics']);split=key.split('/')[-1];ix=data['split'][split]
        for k in ('sample_id','gt','groups','track','scene','sequence','native'):assert np.array_equal(z[k],data[k][ix]),k
        v.validate(z,met,aux,row['arm']);rows.append(dict(job=key,n=len(ix),full_quality_original_clusters_motion_independent=True,all_motion_criteria_reported_not_suppressed=True));del z,aux;gc.collect()
    epochs=0
    for key,row in runs.items():
        c.valid_done(row,c.digest(files));path=c.H/'runs'/key/'checkpoint.pt';state=torch.load(path,map_location='cpu',weights_only=False);history=state['history'];assert len(history)==25 and all(r['n']==18909 and r['optimization_steps']==74 for r in history);m.exact(history,c.read(c.H/'runs'/key/'history.json'));assert sum(r['optimization_steps'] for r in history)==1850;epochs+=len(history)
        assert all(torch.isfinite(t).all() for t in state['model'].values()) and state['model']['native_identity_flag'].dtype==torch.bool
    from statistics_ci import compute
    old=c.read(c.H/'BOOTSTRAP.json');new=compute(True);c.compare(old,new);assert epochs==150
    c.verify(files);c.write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=28,scientific_jobs=24,native_jobs=4,models=6,histories=epochs,steps=11100,records=rows,independent_primary_CI_saved_draws=True,source_initial_final_unchanged=True,saved_math_not_allmodel_gradient_replay=True,elapsed_seconds=time.perf_counter()-start))

if __name__=='__main__':main()
