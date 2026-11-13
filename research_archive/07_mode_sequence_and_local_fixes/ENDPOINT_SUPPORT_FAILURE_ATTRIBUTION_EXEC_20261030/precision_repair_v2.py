"""Repair only new counterfactual junction precision, preserving all v1 evidence."""
import csv
import time
import numpy as np
import attribution_core as c

def main():
    g=c.guard();g.check(mode='startup',force=True)
    c.write_json(c.ROOT/'FAILURE_REVIEW_JUNCTION_PRECISION_V1.json',dict(
        status='PRESERVED_IMPLEMENTATION_PRECISION_MISMATCH_NOT_SOURCE_DATA_FAILURE',
        actual_origin='review_group_adapter.py -> review_group_supplement.py line30 junction assertion',
        source_files={str(c.ROOT/n):c.sha(c.ROOT/n) for n in ('attribution_core.py','review_group_supplement.py','review_saved.py')},
        trace='AssertionError at abs(saved_junction - review_float32_junction) <=2e-12',
        diagnosis='Registered motion_guard.junction casts observed past to float64 BEFORE subtracting/dividing. New counterfactual helper and its second reviewer used float32 past subtraction; five-port group production used original float64 helper and was correct.',
        observed_max_group_junction_difference=1.334953609877232e-7,
        no_tolerance_change=True,no_original_experiment_change=True,
        scope='v1 counterfactual junction mean/p95 only; original coverage/paths/quality/clusters/five future-motion unchanged',
        v1_review_pass_scope='v1 full reviewer matched v1 counterfactual arithmetic, not corrected float64 junction; superseded by full v2 reviewer'))
    z=c.load_npz(c.ROOT/'derived_ports.npz');old=c.load_npz(c.ROOT/'counterfactual_metrics.npz');new={k:v.copy() for k,v in old.items()}
    maxdiff=0.
    for mode in ('restore','edit'):
        for i in range(128):
            g.check();past=z['past'][i].astype(np.float64);vobs=(past[-1]-past[-2])/.4
            for k in range(20):
                p=z['paths'][3 if mode=='restore' else 0,i].copy()
                p[k]=z['paths'][0 if mode=='restore' else 3,i,k]
                jj=np.sqrt(np.sum((p[:,0]/.4-vobs)**2,axis=1))/.4
                for nm,val in [('junction_mean',jj.mean()),('junction_p95',np.quantile(jj,.95))]:
                    maxdiff=max(maxdiff,abs(old[mode+'_'+nm][i,k]-val));new[mode+'_'+nm][i,k]=val
        rows=list(csv.DictReader((c.ROOT/f'single_slot_{mode}.csv').open(encoding='utf-8-sig')))
        for row in rows:
            i,k=int(row['window_index']),int(row['slot'])
            for nm in ('junction_mean','junction_p95'):row[nm]=new[mode+'_'+nm][i,k]
        c.csv_new(c.ROOT/f'single_slot_{mode}_v2.csv',rows)
    for k in old:
        if 'junction' not in k:assert np.array_equal(old[k],new[k])
    c.npz_new(c.ROOT/'counterfactual_metrics_v2.npz',**new)
    c.write_json(c.ROOT/'PRECISION_REPAIR_V2.json',dict(status='CORRECTED_NEW_COUNTERFACTUAL_JUNCTION_ONLY_REVIEW_PENDING',
        max_junction_difference=maxdiff,sets=5120,nonjunction_fields_bitwise_unchanged=True,
        paths_quality_coverage_clusters_future5motion_unchanged=True,arithmetic_tol=2e-12,scientific_epsilon=0,
        old_outputs_preserved=True,files={str(c.ROOT/'precision_repair_v2.py'):c.sha(c.ROOT/'precision_repair_v2.py')},
        observations=g.observations))
    print('Repaired CF junction float64 only; maximum difference',maxdiff,flush=True)

if __name__=='__main__':main()
