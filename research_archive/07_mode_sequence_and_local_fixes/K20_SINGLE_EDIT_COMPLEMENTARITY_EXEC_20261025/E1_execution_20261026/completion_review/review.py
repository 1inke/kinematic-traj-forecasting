"""Post-completion saved-data/source review; no model forward or optimizer updates."""
from pathlib import Path
import sys,json,csv,hashlib
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from env import *
from source_control import verify
from review_saved import independent_population
from src.f16.eval import motion
import torch
OUT=Path(__file__).resolve().parent
def table(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    guard=Guard(ESTIMATE);guard.check('startup');source=verify()
    extension=read_json(HERE/'source_extension_prelaunch.json')
    assert extension['base_source_digest']==source
    for path,h in extension['extension_files'].items():assert sha256(ROOT/path)==h
    preservation=read_json(HERE/'PRESERVATION.json');assert preservation['source_digest']==source
    for fn,h in preservation['outputs'].items():assert sha256(HERE/fn)==h
    extras=['PRIMARY_COMPARISON.json','BOOTSTRAP_DRAWS.npz','COST_CACHED.json','ARITHMETIC_AUDIT.json','ALL6_FINALS_LOCK.json','RESOURCE.json','RESULT.json','PRESERVATION.json','source_extension_prelaunch.json','PRELAUNCH_REVIEW.json','worker_state.json','exit_status.json']
    extra_SHA={fn:sha256(HERE/fn) for fn in extras}
    assert read_json(HERE/'worker_state.json')['status']==read_json(HERE/'exit_status.json')['status']=='SUCCESS'
    runs=[]
    for seed in SEEDS:
        for arm in ARMS:
            folder=HERE/f'runs/{arm}_seed{seed}';d=read_json(folder/'done.json')
            for fn,h in d['outputs'].items():assert sha256(folder/fn)==h
            ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
            h=read_json(folder/'history.json');assert len(h)==8 and h[-1]['steps']==1024 and ck['history']==h and ck['contract']==d['contract']
            assert all(torch.isfinite(t).all() for t in ck['model'].values())
            assert {'optimizer','rng','loader_rng'}.intersection(ck.keys())
            runs.append(dict(seed=seed,arm=arm,epochs=8,steps=1024,checkpoint_SHA=sha256(folder/'checkpoint.pt'),checkpoint_keys=list(ck)))
            del ck;guard.check()
    metrics_rows=[];motion_rows=[];decision_rows=[];events=[];inner={a:[] for a in ARMS};bases=[];ids0=groups0=tracks0=None
    for seed in SEEDS:
        for split in ('opt','hold','inner'):
            folder=HERE/f'populations/seed{seed}/{split}';d=read_json(folder/'done.json')
            for fn,h in d['outputs'].items():assert sha256(folder/fn)==h
            native=[];gt=[];ids=[];groups=[];tracks=[];qq=[];bb=[];valid=[];paths={a:[] for a in ARMS};actions={a:[] for a in ARMS}
            for fn in d['chunks']:
                with np.load(folder/fn) as z:
                    native.append(z['native']);gt.append(z['gt']);ids.append(z['sample_id']);groups.append(z['group']);tracks.append(z['track_key']);qq.append(z['quality']);bb.append(z['B']);valid.append(z['eligible'])
                    for arm in ARMS:
                        with np.load(folder/(arm+'_'+fn[len('aux_'):])) as o:
                            act=np.where(z['eligible'],o['scores'],-np.inf).argmax(1);assert np.array_equal(act,o['action'])
                            for i,a in enumerate(act):
                                p=z['native'][i].copy()
                                if a:r,l=divmod(int(a)-1,20);p[r]=z['kin'][i,l]
                                assert np.array_equal(p,o['paths'][i])
                            paths[arm].append(o['paths']);actions[arm].append(act)
                guard.check()
            native,gt,ids,groups,tracks,qq,bb,valid=[np.concatenate(x) for x in (native,gt,ids,groups,tracks,qq,bb,valid)]
            nd=np.linalg.norm(native-gt[:,None],axis=-1);nativeq=np.stack((nd.mean(-1).min(1),nd[:,:,-1].min(1),np.maximum(nd.mean(-1)/.2,nd[:,:,-1]/.4).min(1)),1)
            assert np.max(abs(nativeq-qq[:,0]))<=2e-12
            for arm in ARMS:
                pp=np.concatenate(paths[arm]);act=np.concatenate(actions[arm]);rows,_=independent_population(pp,gt,groups,native,ids,tracks)
                assert e0.digest(rows)==e0.digest(read_json(folder/f'{arm}_metrics.json'))
                assigned=read_json(folder/f'{arm}_assignment.json');sel=qq[np.arange(len(act)),act];ng=nativeq[:,2]<=1;good=sel[:,2]<=1;isB=bb[np.arange(len(act)),act];harm=(sel>nativeq).any(1)|(ng&~good)
                for gi,g in enumerate(('overall','stable','transition','change')):
                    m=np.ones(len(ids),bool) if gi==0 else groups==gi-1;rr=rows[g];aa=assigned[g];assert aa['n']==int(m.sum())
                    checks=dict(B_nonempty=int((bb.any(1)&m).sum()),chosen_in_B=int((isB&m).sum()),STOP_omitted_opportunity=int(((act==0)&bb.any(1)&m).sum()),correct_STOP_no_B=int(((act==0)&~bb.any(1)&m).sum()),executed=int(((act!=0)&m).sum()),harmful=int(((act!=0)&harm&m).sum()),ineffective=int(((act!=0)&~isB&~harm&m).sum()),eligible_actions=int(valid[m].sum()))
                    assert all(aa[k]==v for k,v in checks.items())
                    baseline=nativeq[m].mean(0)
                    scalar={k:v for k,v in rr.items() if not isinstance(v,(dict,list))}
                    scalar.update(seed=seed,split=split,arm=arm,group=g,native_ADE=float(baseline[0]),native_FDE=float(baseline[1]),native_J=float(baseline[2]),protected_error_pass=bool(rr['ADE']<=1.01*baseline[0] and rr['FDE']<=1.01*baseline[1]),change_5pct_each_error=bool(rr['ADE']<=.95*baseline[0] and rr['FDE']<=.95*baseline[1]),lost_1pct_pass=bool(rr['lost_rate_native_good'] is None or rr['lost_rate_native_good']<=.01))
                    metrics_rows.append(scalar);decision_rows.append(dict(seed=seed,split=split,arm=arm,group=g,n=aa['n'],**checks))
                    for k,v in rr['motion'].items():motion_rows.append(dict(seed=seed,split=split,arm=arm,group=g,metric=k,value=v,native=rr['native_motion'][k],ratio=v/rr['native_motion'][k] if rr['native_motion'][k] else None,PASS=v<=1.01*rr['native_motion'][k]))
                for i in range(len(ids)):
                    event='retained' if ng[i] and good[i] else 'lost' if ng[i] else 'rescue' if good[i] else 'unresolved'
                    if act[i]!=0 or event in ('lost','rescue'):
                        mm=motion(pp[i:i+1]);nm=motion(native[i:i+1])
                        events.append(dict(seed=seed,split=split,arm=arm,sample_id=int(ids[i]),track=str(tracks[i]),group=int(groups[i]),action=int(act[i]),event=event,chosen_B=bool(isB[i]),harmful=bool(harm[i]),native_J=float(nativeq[i,2]),selected_J=float(sel[i,2]),**{k:mm[k] for k in mm},window_motion_pass=all(mm[k]<=1.01*nm[k] for k in mm)))
                if split=='inner':inner[arm].append(sel)
            if split=='inner':
                bases.append(nativeq)
                if ids0 is None:ids0,groups0,tracks0=ids,groups,tracks
                else:assert np.array_equal(ids,ids0) and np.array_equal(groups,groups0) and np.array_equal(tracks,tracks0)
            guard.check()
    with np.load(HERE/'BOOTSTRAP_DRAWS.npz') as z:
        unique,ix=np.unique(tracks0,return_inverse=True);assert np.array_equal(unique,z['track_keys']);n=len(unique);m=groups0==2;counts=np.bincount(ix,weights=m,minlength=n)
        s={a:np.asarray(v) for a,v in inner.items()};base=np.asarray(bases);summed={a:np.stack([np.bincount(ix,weights=(s[a][j,:,2]<=1)*m,minlength=n) for j in range(3)]) for a in ARMS}
        asum=np.asarray([[np.bincount(ix,weights=(s['S'][j,:,k]-base[j,:,k])*m,minlength=n) for k in range(2)] for j in range(3)])
        rng=np.random.default_rng(20261025);maximum=0
        for b in range(20000):
            td=rng.integers(n,size=n);sd=rng.integers(3,size=3);assert np.array_equal(td,z['tracks'][b]) and np.array_equal(sd,z['seeds'][b]);den=counts[td].sum()
            if den==0:assert not z['valid'][b];continue
            diff=(summed['S'][:,td].sum(1)-summed['I'][:,td].sum(1))/den;ab=asum[:,:,td].sum(2)[sd].mean(0)/den
            maximum=max(maximum,float(abs(diff[sd].mean()-z['primary'][b])),float(abs(diff.mean()-z['fixed_seed'][b])),float(abs(ab-z['absolute_S_native_ADE_FDE'][b]).max()))
        assert maximum<=1e-12
    assert verify()==source
    for fn,h in extra_SHA.items():assert sha256(HERE/fn)==h
    for fn,h in preservation['outputs'].items():assert sha256(HERE/fn)==h
    for p,h in extension['extension_files'].items():assert sha256(ROOT/p)==h
    table('ALL72_GROUPS.csv',metrics_rows);table('ALL360_MOTION.csv',motion_rows);table('ALL72_DECISIONS.csv',decision_rows);table('EDIT_AND_COVERAGE_EVENTS.csv',events)
    report=dict(status='PASS_COMPLETION_SOURCE_OUTPUT_AND_SAVED_DATA',source=source,source_count=700,prelaunch_extension_count=1,primary_output_SHA=extra_SHA,run_contracts=runs,preservation_output_count=len(preservation['outputs']),full_group_rows=len(metrics_rows),full_motion_rows=len(motion_rows),full_decision_rows=len(decision_rows),event_rows=len(events),bootstrap_20000_maxerror=maximum,motion_failures=[r for r in motion_rows if not r['PASS']],protected_error_failures=[r for r in metrics_rows if r['group']!='change' and not r['protected_error_pass']],lost_failures=[r for r in metrics_rows if not r['lost_1pct_pass']],saved_only_no_model_forward=True,no_optimizer_update=True,no_delete=True,helper_SHA=sha256(Path(__file__)),resource=guard.check())
    write_json(OUT/'REVIEW.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('run_contracts','primary_output_SHA','resource')},ensure_ascii=False))
if __name__=='__main__':main()
