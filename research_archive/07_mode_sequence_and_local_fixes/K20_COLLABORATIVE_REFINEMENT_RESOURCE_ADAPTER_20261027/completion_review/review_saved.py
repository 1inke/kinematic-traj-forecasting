"""Finite CPU saved-data completion supplement. No Torch/model/optimizer replay."""
from pathlib import Path
import sys, json, csv
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from core import *
from independent import reconstruct
from src.f16.eval import errors, motion
from scipy.spatial.distance import pdist

OUT=Path(__file__).resolve().parent
def close(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.dtype.kind in 'bUSO' or b.dtype.kind in 'bUSO':assert np.array_equal(a,b)
    else:assert a.shape==b.shape and np.max(np.abs(a-b),initial=0)<=2e-12

def preserved():
    digest0=verify();p=read_json(BASE/'PRESERVATION.json')
    assert p['status']=='PASS' and p['source']==digest0
    for n,h in p['outputs'].items():assert sha256(BASE/n)==h,n
    return digest0,len(p['outputs'])

def main():
    g=guard(True);h,output_count=preserved();print('INITIAL SOURCE AND OUTPUT SHA PASS',flush=True)
    root_names=['RESULT.json','RESOURCE.json','ARITHMETIC_AUDIT.json','PRESERVATION.json','ALL6_FINALS_LOCK.json','statistics/PRIMARY.json','statistics/BOOTSTRAP_DRAWS.npz','cost/COST_MODULE.json','PRODUCTION_SOURCE_LOCK.json','PRELAUNCH_REVIEW.json','PREFLIGHT_REAL_STAGE_ADMISSION.json','RESOURCE_POLICY_TESTS.json','startup_check.json','halfway_check.json','worker_state.json','exit_status.json']
    root_hash={n:sha256(BASE/n) for n in root_names}
    result=read_json(BASE/'RESULT.json');assert result['status']=='SUCCESS_COLLABORATIVE_REFINEMENT_REVIEW_REQUIRED'
    assert read_json(BASE/'worker_state.json')['status']==read_json(BASE/'exit_status.json')['status']=='SUCCESS'
    locked=read_json(BASE/'ALL6_FINALS_LOCK.json');assert locked['source']==h and locked['epochs']==48 and locked['updates']==4656 and len(locked['models'])==6
    histories=[]
    for seed in SEEDS:
        order=[]
        for arm in ARMS:
            folder=BASE/f'runs/{arm}_seed{seed}';ct=contract(seed,arm);d=completed(folder,ct);hist=read_json(folder/'history.json')
            assert len(hist)==8 and [r['epoch'] for r in hist]==list(range(1,9)) and hist[-1]['updates']==776 and d['updates']==776
            order.append([r['order_SHA'] for r in hist]);histories.append(dict(seed=seed,arm=arm,epochs=len(hist),updates=hist[-1]['updates'],last_preupdate_epoch_loss=hist[-1],done_SHA=sha256(folder/'done.json')))
            assert sha256(folder/'checkpoint.pt')==locked['models'][str((folder/'checkpoint.pt').relative_to(BASE))]
        assert order[0]==order[1]
    stats=load(BASE/'prepared/statistics.npz');split_ids=load(BASE/'prepared/split.npz');rows=[];extra=[];geom=[];held={};max_error=0.;total_windows=0;reviews=[];port_keys=['B0','B1','B2']
    for seed in SEEDS:
        for split in SPLITS:
            f=BASE/f'populations/seed{seed}/{split}';done=read_json(f/'done.json');completed(f,done['contract']);aud=read_json(f/'audit/done.json');completed(f/'audit',aud['contract']);review=read_json(f/'audit/REVIEW.json');assert review['status']=='PASS_SAVED_ALL3_PORTS';reviews.append(dict(seed=seed,split=split,review_SHA=sha256(f/'audit/REVIEW.json'),full_original_cluster_and_motion_audit=review))
            chunks=[load(f/n) for n in done['chunks']];z={k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]};assert len(z['sample_id'])==len(np.unique(z['sample_id']));expected=load(BASE/f'prepared/{split}.npz');assert np.array_equal(z['sample_id'],expected['sample_id']);ne=errors(z['native'],z['gt'])
            for arm in port_keys:
                p=z['native'] if arm=='B0' else z[arm+'_paths'];saved=load(f/f'audit/{arm}_metrics.npz');e=errors(p,z['gt']);total_windows+=len(p)
                for field,key in [('ADE','ADE'),('FDE','FDE'),('J','J'),('coverage','coverage'),('Q_slot','Q_winner'),('J_slot','J_winner'),('Q_segments','Q_segments'),('J_segments','J_segments')]:close(saved[field],e[key])
                close(saved['native_good'],ne['coverage']);assert np.array_equal(saved['sample_id'],z['sample_id'])
                if arm!='B0':
                    rb=reconstruct(z['native'],z[arm+'_anchors'],z[arm+'_gate'],stats['basis'],stats['basis_l1_bound']);err=float(abs(rb-p).max());max_error=max(max_error,err);assert err<=2e-12 and np.linalg.norm(p-z['native'],axis=-1).max()<=.2+2e-12
                groups=read_json(f/f'audit/{arm}_GROUPS.json');events=read_json(f/f'audit/{arm}_EVENTS.json');good=e['coverage'];ng=ne['coverage']
                for i,name in enumerate(GROUPS):
                    mask=np.ones(len(p),bool) if i==0 else z['group']==i-1;r=groups[name];rescue=good&~ng&mask;lost=~good&ng&mask
                    assert r['n']==int(mask.sum()) and r['good']==int(good[mask].sum()) and r['native_good']==int(ng[mask].sum()) and r['rescue']==int(rescue.sum()) and r['lost']==int(lost.sum())
                    assert r['good']-r['native_good']==r['rescue']-r['lost'];assert r['rescue_IDs']==z['sample_id'][rescue].tolist() and r['lost_IDs']==z['sample_id'][lost].tolist()
                    close(r['ADE'],e['ADE'][mask].min(1).mean());close(r['FDE'],e['FDE'][mask].min(1).mean());close(r['J'],e['J'][mask].min(1).mean())
                    rows.append(dict(seed=seed,split=split,arm=arm,group=name,**r))
                for name,mask in dict(retained=good&ng,lost=~good&ng,rescue=good&~ng,unresolved=~good&~ng).items():assert events[name]['n']==int(mask.sum()) and events[name]['IDs']==z['sample_id'][mask].tolist()
                # Original worker independently recalculated full SciPy/pdist clusters
                # and global motion; supplement binds that receipt, not a fake replay.
                tri=np.triu_indices(20,1);md=[];ed=[];dups=[]
                for start in range(0,len(p),128):
                    q=p[start:start+128];dist=np.linalg.norm(q[:,tri[0]]-q[:,tri[1]],axis=-1);md.append(dist.mean(-1));ed.append(dist[:,:,-1]);dups.append((dist.mean(-1)<=.1)&(dist[:,:,-1]<=.2))
                means=np.concatenate(md);ends=np.concatenate(ed);dup=np.concatenate(dups)
                geom.append(dict(seed=seed,split=split,arm=arm,all190_meanpath_quantiles=np.quantile(means,[0,.25,.5,.75,.95,1]).tolist(),all190_endpoint_quantiles=np.quantile(ends,[0,.25,.5,.75,.95,1]).tolist(),near_duplicate_pair_fraction=float(dup.mean()),correction_mean=float(np.linalg.norm(p-z['native'],axis=-1).mean()),correction_p95=float(np.quantile(np.linalg.norm(p-z['native'],axis=-1),.95))))
                if split=='hold':held[seed,arm]=dict(coverage=good.copy(),track=z['track_key'],group=z['group'],IDs=z['sample_id'])
                if split=='opt':
                    for part in ('train','val'):
                        ix=split_ids[part];maskpart=np.zeros(len(p),bool);maskpart[ix]=True
                        for gi,name in enumerate(GROUPS):
                            mask=maskpart&(np.ones(len(p),bool) if gi==0 else z['group']==gi-1);mot=motion(p[mask]);nm=motion(z['native'][mask]);rescue=mask&good&~ng;lost=mask&~good&ng
                            extra.append(dict(seed=seed,split=part,arm=arm,group=name,n=int(mask.sum()),tracks=len(np.unique(z['track_key'][mask])),good=int(good[mask].sum()),native_good=int(ng[mask].sum()),rescue=int(rescue.sum()),lost=int(lost.sum()),ADE=float(e['ADE'][mask].min(1).mean()),FDE=float(e['FDE'][mask].min(1).mean()),J=float(e['J'][mask].min(1).mean()),motion=mot,native_motion=nm,motion_pass=all(mot[k]<=1.01*nm[k] for k in mot)))
            g.check();print('SAVED REVIEW',seed,split,flush=True)
    bs=load(BASE/'statistics/BOOTSTRAP_DRAWS.npz');primary=read_json(BASE/'statistics/PRIMARY.json');base=held[42,'B0'];tracks=np.unique(base['track']);assert np.array_equal(bs['track_keys'],tracks) and np.array_equal(bs['sample_id'],base['IDs']);ix=np.searchsorted(tracks,base['track']);change=base['group']==2;rng=np.random.default_rng(20261027);draw=rng.integers(len(tracks),size=(20000,len(tracks)),dtype=np.int32);seed_draw=rng.integers(3,size=(20000,3),dtype=np.int32);assert np.array_equal(draw,bs['track_draws']) and np.array_equal(seed_draw,bs['seed_draws']);delta_errors=[]
    for ref in ('B1','B0'):
        delta=np.stack([held[s,'B2']['coverage'].astype(int)-held[s,ref]['coverage'].astype(int) for s in SEEDS]);values=[]
        per_track_den=[int(np.sum(change&(ix==t))) for t in range(len(tracks))]
        per_track_num=[delta[:,change&(ix==t)].sum(1) for t in range(len(tracks))]
        for d in draw:
            # Independent explicit whole-track expansion via per-track sufficient sums.
            den=sum(per_track_den[t] for t in d);num=np.add.reduce([per_track_num[t] for t in d])
            values.append(num/den if den else np.full(3,np.nan))
        values=np.asarray(values);old=bs['B2-'+ref+'_values'];close(values,old);delta_errors.append(float(np.nanmax(abs(values-old))))
        paired=values[np.arange(20000)[:,None],seed_draw].mean(1);fixed=values.mean(1);rec=primary['comparisons']['B2-'+ref]
        for a,b in ((paired,rec['paired']),(fixed,rec['fixed_seed'])):
            ok=np.isfinite(a);assert int(ok.sum())==b['valid'] and int((~ok).sum())==b['invalid'];close(np.quantile(a[ok],[.0125,.9875]),b['CI97_5']);close(np.quantile(a[ok],[.025,.975]),b['CI95'])
        print('ALL20000 INDEPENDENT',ref,flush=True);g.check()
    write_json(OUT/'ALL108_GROUPS.json',rows);write_json(OUT/'TRAIN_VAL72_GROUPS.json',extra);write_json(OUT/'CANDIDATE_GEOMETRY.json',geom);write_json(OUT/'HISTORIES.json',histories)
    keys=['seed','split','arm','group','n','tracks','good','native_good','rescue','lost','ADE','FDE','J','lost_rate_native_good','motion_pass','distinct','effective_good','correction_mean','correction_p95']
    with (OUT/'ALL108_GROUPS.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    assert preserved()==(h,output_count);assert all(sha256(BASE/n)==hh for n,hh in root_hash.items());resource(OUT/'RESOURCE.json',g)
    write_json(OUT/'REVIEW.json',dict(status='PASS_CPU_SAVED_COMPLETION_REVIEW',base_source=h,source_count=len(read_json(BASE/'source_manifest.json')['files']),source_and_all_worker_outputs_initial_final_SHA_EXACT=True,worker_output_file_count=output_count,explicit_root_SHA=root_hash,new_helper_SHA=sha256(Path(__file__)),histories48_updates4656_done_models6_PASS=True,reviewed_populations=9,science_ports=18,native_ports=9,repeated_model_window_rows=total_windows,all108group_counts_events_slot_ADE_FDE_J_Q_EXACT=True,train_val72groups_separate=True,all_saved_reconstruct_max_error=max_error,bootstrap_actual20000_bothcomparisons_max_error=max(delta_errors),original_complete_link_pdist_and_five_motion_receipts=reviews,no_Torch_model_or_gradient_replay=True,not_external_review=True,no_delete=True))
    assert 'torch' not in sys.modules;print('PASS COMPLETION CPU SAVED',total_windows,flush=True)
if __name__=='__main__':main()
