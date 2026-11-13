from core import *
from independent import reconstruct
from src.f16.eval import errors,evaluate,motion
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,fcluster
def run(seed,split,folder=None,existing_guard=None):
    g=existing_guard if existing_guard is not None else guard(True);g.check();f=Path(folder) if folder else HERE/f'populations/seed{seed}/{split}';done=read_json(f/'done.json');completed(f,done['contract']);ct=dict(source=source(),population_done=sha256(f/'done.json'));out=f/'audit';out.mkdir(exist_ok=True)
    if (out/'done.json').exists():completed(out,ct);return read_json(out/'REVIEW.json')
    chunks=[load(f/n) for n in done['chunks']];z={k:np.concatenate([q[k] for q in chunks]) for k in chunks[0]};st=load(HERE/'prepared/statistics.npz');maximum=0.;groups={}
    for a in ('B0',)+ARMS:
        p=z['native'] if a=='B0' else z[a+'_paths']
        if a!='B0':
            rebuilt=reconstruct(z['native'],z[a+'_anchors'],z[a+'_gate'],st['basis'],st['basis_l1_bound']);maximum=max(maximum,float(abs(rebuilt-p).max()));assert maximum<=2e-12 and np.linalg.norm(p-z['native'],axis=-1).max()<=.2+2e-12
        rows,e=evaluate(p,z['gt'],z['group'],z['native'],z['sample_id'],z['track_key']);independent=[]
        for pp,good in zip(p,e['good']):
            d=np.stack([pdist(pp[:,t]) for t in range(12)]);labels=fcluster(linkage(np.maximum(d.mean(0)/.1,d[-1]/.2),method='complete'),1,criterion='distance');independent.append([len(set(labels)),len(set(labels[good]))])
        independent=np.array(independent);ne=errors(z['native'],z['gt']);dp=np.linalg.norm(p-z['gt'][:,None],axis=-1);aa=dp.mean(-1);ff=dp[:,:,-1];good=((aa<=.2)&(ff<=.4)).any(1);assert np.array_equal(good,e['coverage'])
        for gi,name in enumerate(GROUPS):
            mask=np.ones(len(p),bool) if gi==0 else z['group']==gi-1
            if not mask.any():continue
            r=rows[name];assert r['distinct']==independent[mask,0].mean() and r['effective_good']==independent[mask,1].mean()
            assert r['good']==int(good[mask].sum());v=(np.diff(np.concatenate((np.zeros_like(p[mask,:,:1]),p[mask]),axis=2),axis=2))/.4;acc=(v[:,:,1:]-v[:,:,:-1])/.4;j=(acc[:,:,1:]-acc[:,:,:-1])/.4;s,ac,jj=[np.sqrt((w*w).sum(-1)) for w in (v,acc,j)];ind=[np.quantile(s,.95),ac.mean(),np.quantile(ac,.95),jj.mean(),np.quantile(jj,.95)];assert max(abs(x-y) for x,y in zip(ind,r['motion'].values()))<=2e-12
            r.update(native_ADE=float(ne['ADE'][mask].min(1).mean()),native_FDE=float(ne['FDE'][mask].min(1).mean()))
            if a!='B0':r['gate_quantiles']=np.quantile(z[a+'_gate'][mask],[0,.25,.5,.75,1]).tolist()
        subsets={};ng=ne['coverage']
        for name,mask in dict(retained=good&ng,lost=~good&ng,rescue=good&~ng,unresolved=~good&~ng).items():subsets[name]=dict(n=int(mask.sum()),IDs=z['sample_id'][mask].tolist(),motion=motion(p[mask]) if mask.any() else None,native_motion=motion(z['native'][mask]) if mask.any() else None)
        save(out/(a+'_metrics.npz'),sample_id=z['sample_id'],track_key=z['track_key'],group=z['group'],coverage=e['coverage'],native_good=ne['coverage'],ADE=e['ADE'],FDE=e['FDE'],J=e['J'],Q_slot=e['Q_winner'],J_slot=e['J_winner'],Q_segments=e['Q_segments'],J_segments=e['J_segments'],distinct=independent[:,0],effective_good=independent[:,1]);write_json(out/(a+'_GROUPS.json'),rows);write_json(out/(a+'_EVENTS.json'),subsets);groups[a]=rows;g.check()
    report=dict(status='PASS_SAVED_ALL3_PORTS',n=len(z['native']),reconstruction_max_error=maximum,all_samecandidate_quality=True,full_original_complete_link_independent_pdist=True,all_group_five_motion_independent=True,no_Torch_or_model_replay=True);assert 'torch' not in sys.modules;write_json(out/'REVIEW.json',report);resource(out/'RESOURCE.json',g);finish(out,ct);return report
if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2])
