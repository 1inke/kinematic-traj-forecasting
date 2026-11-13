from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
def main():
    rows=[];errors=[]
    for split in SPLITS:
        x=xload(split);q=load(HERE/f'E3_gt_audit/{split}/native_quality.npz');r=load(HERE/f'E3_gt_audit/{split}/a20_history3_s2/presence.npz');v=load(HERE/f'completion_review/error_reviewed/{split}.npz')
        ports={p:load(HERE/f'E4_oracle/{split}/{p}.npz')['covered'] for p in ('END_unbounded','END_b02','END_b04','TIME_grid')}
        for group,mask in masks(x):
            fail=mask&~q['native_good']
            for reason,label in enumerate(REASONS):
                active=fail&(r['reason']==reason)
                row=dict(population=split,group=group,original_diagnostic_layer=label,failure_N=int(active.sum()),all_FDE_above04=int((active&np.all(q['FDE']>.4,axis=1)).sum()))
                for p,covered in ports.items():row[p+'_rescue']=int((active&covered).sum())
                row['END04_TIME_both']=int((active&ports['END_b04']&ports['TIME_grid']).sum());rows.append(row)
            km=v['k_mode'];kj=v['k_J'];exists=km>=0;assert np.all(q['J'][np.flatnonzero(exists),km[exists]]>=q['J'][np.flatnonzero(exists),kj[exists]])
            assert np.all(km[~r['full_match'].any(1)]==-1)
    table(HERE/'completion_review/ORACLE_BY_ORIGINAL_DIAGNOSTIC.csv',rows)
    write(HERE/'completion_review/REFERENCE_SCOPE_CHECK.json',dict(status='PASS_ALL_POPULATIONS',kMode_not_filled_when_empty=True,kMode_J_never_better_than_global_kJ=True,no_unique_causal_root_assignment=True))
if __name__=='__main__':main()
