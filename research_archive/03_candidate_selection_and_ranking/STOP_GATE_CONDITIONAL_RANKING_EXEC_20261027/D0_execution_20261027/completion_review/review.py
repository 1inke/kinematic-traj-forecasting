from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from shared import *
HERE=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def main():
    g=Guard(2147483648);g.check('startup');h=verify_manifest(HERE);p=read_json(HERE/'PRESERVATION.json')
    for f,hh in p['outputs'].items():assert sha256(HERE/f)==hh,f
    groups=read_json(HERE/'ALL_GROUPS.json');matrix=read_json(HERE/'SOURCE_REUSE_MATRIX.json');assert len(matrix)==63 and len(groups)==252
    n=0
    for row in matrix:
        folder=HERE/f"populations/{row['model']}_seed{row['seed']}/{row['split']}";d=read_json(folder/'done.json');completed(folder,d['contract']);z=load_npz(folder/'PER_WINDOW_RANK_MARGIN.npz');n+=len(z['sample_id'])
        assert np.array_equal(z['blocked_good'],z['best_B']&(z['margin']<=0));assert np.array_equal(z['blocked_rescue'],z['best_rescue']&(z['margin']<=0))
        assert np.array_equal(z['actual_action'],np.where(z['has_edit']&(z['margin']>0),z['best_edit'],0))
        assert np.array_equal(z['internal_bad'],z['B_nonempty']&~z['best_B'])
        for gi,name in enumerate(GROUPS):
            mask=np.ones(len(z['group']),bool) if gi==0 else z['group']==gi-1;r=next(x for x in groups if x['model']==row['model'] and x['seed']==row['seed'] and x['split']==row['split'] and x['group']==name)
            assert r['n']==int(mask.sum())
            for key in ('best_B','best_rescue','blocked_good','blocked_rescue','internal_bad','actual_good','actual_rescue','actual_lost','native_good','actual_executed'):
                assert r[key]==int(z[key][mask].sum())
        g.check()
    # New explicit zero-score BCE derivative supplement, no Torch/model.
    zero=[]
    for negatives in (1,5,400):
        eps=1e-5
        for truth in (0,1):
            f=lambda x:.5*np.logaddexp(0,-x if truth else x)/(1 if truth else negatives)
            v=(f(eps)-f(-eps))/(2*eps);expected=-.25 if truth else .25/negatives
            assert abs(v-expected)<=1e-10;zero.append(dict(negatives=negatives,STOP_positive=truth,expected=expected,numeric=v))
    assert n==205317;verify_manifest(HERE)
    for f,hh in p['outputs'].items():assert sha256(HERE/f)==hh,f
    rootnames=('RESULT.json','RESOURCE.json','PRESERVATION.json','STOP_WEIGHT_AUDIT.json','SOURCE_REUSE_MATRIX.json','ALL_GROUPS.json','ALL_GROUPS.csv','BOOTSTRAP.json','BOOTSTRAP_DRAWS.npz')
    write_json(OUT/'REVIEW.json',dict(status='PASS_SAVED_D0_COMPLETION',source=h,source_count=len(read_json(HERE/'source_manifest.json')['files']),all_source_and_outputs_initial_final_SHA=True,populations=63,groups=252,repeated_model_windows=n,all_saved_gate_rank_events_recomputed=True,full_original_scores_paths_source_checked_by_main_audit=True,all20000_independent_bootstrap_inherited_MAIN_SHA=sha256(HERE/'BOOTSTRAP.json'),zero_score_STOP_derivative_explicit=zero,root_SHA={x:sha256(HERE/x) for x in rootnames},new_helper_SHA=sha256(Path(__file__)),no_Torch_forward_training=True,same_assistant_not_external=True))
    write_json(OUT/'RESOURCE.json',dict(last=g.check(),observations=g.observations));print('PASS D0 COMPLETION',n,flush=True)
if __name__=='__main__':main()
