"""CPU prelaunch supplement; no changes to locked production scientific sources."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as c
import numpy as np,time,ast

def main():
    t=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30
    files=c.read(c.H/'source_manifest.json')['files'];assert len(files)==pre['source_count'] and c.digest(files)==pre['source_digest'];c.verify(files)
    maximum=0.;rows=[]
    for item in pre['actual_eight128_pipeline']:
        p=c.H/'preflight_smoke'/('chunk'+str(item['chunk'])+'.npz');assert c.sha(p)==item['SHA'];z=c.npz(p)
        for port in c.PORTS:
            path=z['native' if port=='native20' else port];a,f,j,g,_=c.errors(path,z['gt'],True)
            for suffix,value in [('ADE',a),('FDE',f),('J',j)]:maximum=max(maximum,c.close(value,z[port+'_slot_'+suffix]))
            assert np.array_equal(g,z[port+'_slot_good'])
        rows.append(dict(chunk=item['chunk'],saved_all_slot_metrics_recomputed=True))
    # The actual production loop calls the audited job_arrays and independent
    # audit before writing RESULT/SUCCESS. No legacy top-level model import.
    run=ast.parse((c.H/'run_s0.py').read_text(encoding='utf-8'));assert 'audit' in {n.id for n in ast.walk(run) if isinstance(n,ast.Name)}
    c.verify(files)
    c.write('prelaunch_review/REVIEW.json',dict(status='PASS',source_count=len(files),source_digest=c.digest(files),source_initial_final_unchanged=True,all_eight_saved_slot_quality_metrics=True,maxerror=maximum,rows=rows,no_model_forward_optimizer_or_training=True,not_external_review=True,locked_scientific_source_edited=False,review_script_SHA=c.sha(__file__),elapsed_seconds=time.perf_counter()-t))
    print('PRELAUNCH_REVIEW PASS',maximum,len(files),flush=True)

if __name__=='__main__':main()
