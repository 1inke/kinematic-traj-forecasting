"""Posthoc descriptive strata only; fixed outputs, no inference or new gate."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
from collections import Counter
import numpy as np
out=Path(__file__).resolve().parent
assert c.read_json(out/'REVIEW.json')['status']=='PASS_SAVED_COMPLETION_SUPPLEMENT'
results=[]
for seed in c.SEEDS:
    pairs=c.read_json(c.HERE/f'posthoc/seed{seed}/PAIRS.json')
    for mode in ('primary','reverse'):
        rows=[r for r in pairs if r['mode']==mode];conf=[r for r in rows if r['event']=='CONFLICT']
        both=[r for r in rows if r['both_positive']]
        for group in ('overall',0,1,2):
            xx=[r for r in rows if group=='overall' or r['group']==group];cc=[r for r in xx if r['event']=='CONFLICT']
            results.append(dict(seed=seed,mode=mode,group=group,pairs=len(xx),conflicts=len(cc),both_positive_pairs=sum(r['both_positive'] for r in xx),
                conflict_both_rescue_opportunities=sum(r['q_rescue_opportunity'] and r['p_rescue_opportunity'] for r in cc),
                allpairs_both_rescue_opportunities=sum(r['q_rescue_opportunity'] and r['p_rescue_opportunity'] for r in xx),
                conflict_either_nativegood=sum(r['q_native_good'] or r['p_native_good'] for r in cc),
                conflict_zero_J_regret_literal=sum(r['regret']==0 for r in cc),
                no_conflict_regret_positive_pairs=sum(r['event']!='CONFLICT' and r['regret']>0 for r in xx),
                opportunity_nativegood_composition=dict(Counter(f"qgood{int(r['q_native_good'])}_pgood{int(r['p_native_good'])}" for r in cc)),
                zero_regret_is_literal_no_epsilon=True))
c.write_json(out/'CONFLICT_STRATA.json',dict(status='DESCRIPTIVE_ONLY_UNDERPOWERED',source=c.source(),rows=results,source_helper_SHA=c.sha256(Path(__file__)),no_B_quality_or_threshold_changed=True))
print('PASS descriptive strata',len(results),flush=True)
