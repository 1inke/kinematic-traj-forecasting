"""CPU-only saved-path reviewer, no model class or Torch import."""
from core import *
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist
from src.f16.eval import errors,evaluate
legacy={'np':np,'linkage':linkage,'fcluster':fcluster,'pdist':pdist,'NAMES':NAMES}
pure(OLD.parent/'E0_execution_20261025/review_saved.py',('direct','compare','independent_population'),legacy)
rank={'np':np,'math':__import__('math'),'KS':(1,5,10)}
pure(D0/'ranking.py',('random_hit','diagnose','quantiles','summarize'),rank)
def audit(seed,split,folder=None):
    g=guard('CPU');folder=Path(folder) if folder else HERE/f'populations/seed{seed}/{split}';done=read_json(folder/'done.json');completed(folder,done['contract'])
    out=folder/'audit';out.mkdir(exist_ok=True);expected=dict(source=source0(),population_SHA=sha256(folder/'done.json'))
    if (out/'done.json').exists():completed(out,expected);return read_json(out/'REVIEW.json')
    parts={a:[] for a in ARMS};diag={a:[] for a in ARMS};ref=[];quality_error=0.
    for j,(name,z) in enumerate(aux(seed,split)):
        if j not in done['chunks']:break
        ref.append(z)
        for a in ARMS:
            q=load_npz(folder/f'{a}_chunk{j:04d}.npz')
            for k in ('sample_id','eligible','B','target'):assert np.array_equal(q[k],z[k])
            ch=np.where(q['eligible'],q['scores'],-np.inf).argmax(1);assert np.array_equal(ch,q['chosen'])
            for i,act in enumerate(ch):
                p=z['native'][i].copy()
                if act:r,l=divmod(int(act)-1,20);p[r]=z['kin'][i,l]
                assert np.array_equal(p,q['paths'][i])
                dp=np.linalg.norm(p-z['gt'][i],axis=-1);A=dp.mean(-1);F=dp[:,-1];J=np.maximum(A/.2,F/.4)
                v=np.array([A.min(),F.min(),J.min()]);quality_error=max(quality_error,float(abs(v-z['quality'][i,act]).max()));assert quality_error<=2e-12
            parts[a].append(q);diag[a].append(rank['diagnose'](q['scores'],z['eligible'],z['B'],z['quality'],ch))
        g.check()
    full={k:np.concatenate([z[k] for z in ref]) for k in ('sample_id','track_key','group','native','gt','quality','eligible','B','target')};summaries={}
    for a in ARMS:
        paths=np.concatenate([z['paths'] for z in parts[a]]);scores=np.concatenate([z['scores'] for z in parts[a]]);chosen=np.concatenate([z['chosen'] for z in parts[a]])
        rows,cc=legacy['independent_population'](paths,full['gt'],full['group'],full['native'],full['sample_id'],full['track_key']);ee=errors(paths,full['gt']);ne=errors(full['native'],full['gt']);ng=ne['coverage'];dd={k:np.concatenate([x[k] for x in diag[a]]) for k in diag[a][0]}
        allrank={}
        for gi,group in enumerate(GROUPS):
            mask=np.ones(len(chosen),bool) if not gi else full['group']==gi-1;allrank[group]=rank['summarize'](dd,mask)
            if mask.any():
                rows[group].update(native_ADE=float(ne['ADE'][mask].min(1).mean()),native_FDE=float(ne['FDE'][mask].min(1).mean()))
                hp=dd['has_edit']&mask;truth=dd['B_nonempty'];pred=dd['margin']>0
                allrank[group]['gate_confusion']=dict(TP=int((hp&truth&pred).sum()),FP=int((hp&~truth&pred).sum()),TN=int((hp&~truth&~pred).sum()),FN=int((hp&truth&~pred).sum()),no_eligible_edit=int((mask&~dd['has_edit']).sum()))
        save_npz(out/f'{a}_sample_metrics.npz',sample_id=full['sample_id'],track_key=full['track_key'],group=full['group'],good=ee['coverage'],native_good=ng,chosen=chosen,ADE=ee['ADE'],FDE=ee['FDE'],J=ee['J'],Q_slot=ee['Q_winner'],J_slot=ee['J_winner'],Q_segments=ee['Q_segments'],J_segments=ee['J_segments'])
        save_npz(out/f'{a}_rank_margin.npz',sample_id=full['sample_id'],track_key=full['track_key'],group=full['group'],**dd)
        write_json(out/f'{a}_METRICS.json',rows);write_json(out/f'{a}_RANK_GATE.json',allrank);summaries[a]=rows;g.check()
    report=dict(status='PASS_ALL3_SAVED_POPULATIONS',n=len(full['sample_id']),samecandidate_quality_max_error=quality_error,all401_argmax_fixed19_STOP_EXACT=True,full_original_complete_link_and_independent_pdist=True,all_group_pooledfive_motion=True,no_model_gradient_replay=True,no_Torch=True)
    assert 'torch' not in sys.modules;write_json(out/'REVIEW.json',report);resource(out/'RESOURCE_CPU.json',g);finish(out,expected);mark('D1_INDEPENDENT_SAVED_AUDIT',seed=seed,split=split,n=len(full['sample_id']));return report
if __name__=='__main__':audit(int(sys.argv[1]),sys.argv[2])
