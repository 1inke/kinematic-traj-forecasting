from core import *
import core,sys
sys.modules['base']=core
stat=module('_step3_exact_registered_bootstrap',PREVIOUS.parent/'execution_staged_20261026/analysis_statistics.py')
def main():
    g=guard('CPU');arms=read_json(HERE/'FEATURE_PROVENANCE.json')['admissible_arms'];arrays={};samples={}
    for arm in arms:
        rows=[]
        for seed in SEEDS:
            with np.load(HERE/f'populations/{arm}_seed{seed}/hold/audit/sample_metrics.npz') as z:
                rows.append(z['good'].astype(float));samples[(arm,seed)]={k:z[k] for k in z.files}
        arrays[arm]=np.stack(rows)
    base=samples[('C_base',42)];groups=base['group'];tracks=base['track_key'];native=np.stack([samples[('C_base',s)]['native_good'] for s in SEEDS]).astype(float)
    for sample in samples.values():
        for k in ('sample_id','group','track_key'):assert np.array_equal(base[k],sample[k])
    comparisons={}
    for name,a,b in [('Ctoken_minus_Cbase','C_token','C_base'),('Cneighbor_minus_Cbase','C_neighbor','C_base'),('Cboth_minus_Ctoken','C_both','C_token')]:
        if a in arrays and b in arrays:comparisons[name]=arrays[a]-arrays[b]
    for arm,v in arrays.items():comparisons[arm+'_minus_native']=v-native
    records,draw=stat.bootstrap(comparisons,groups,tracks)
    with np.load(PREVIOUS/'statistics/BOOTSTRAP_DRAWS.npz') as z:
        for k in ('track_draws','seed_draws','tracks'):assert np.array_equal(z[k],draw[k]),k
    for k,r in records.items():r['positive_gate']=bool(r['status']=='ESTIMABLE' and min(r['point_per_seed'])>=0 and r['mean']>=.02 and r['CI_family6'][0]>0 and r['fixed_seed_CI_family6'][0]>0)
    out=HERE/'statistics';out.mkdir();save_npz(out/'BOOTSTRAP_DRAWS.npz',**draw)
    write_json(out/'PRIMARY_COMPARISONS.json',dict(family_fixed=6,records=records,same_step2_track_seed_order_EXACT=True,
        absent_comparisons='BLOCKED_INPUT_PROVENANCE_NOT_NEGATIVE',step2_primary_SHA=sha256(PREVIOUS/'statistics/PRIMARY_COMPARISONS.json'),all_20000_independent=True,
        token_information_signal=bool(records.get('Ctoken_minus_Cbase',{}).get('positive_gate',False) and records.get('C_token_minus_native',{}).get('positive_gate',False))))
    resource_report(out,g);finish(out,dict(source=source(),family=6,draws=20000))
if __name__=='__main__':main()
