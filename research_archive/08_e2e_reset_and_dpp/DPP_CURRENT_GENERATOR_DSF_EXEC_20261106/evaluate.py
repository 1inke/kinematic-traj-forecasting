"""New output namespace, canonical quality/cluster/motion + independent score/EC."""
from run_common import *
from independent import decoder_np

def labels(split):
    if split=='dev' and not (RUN/'ALL6_FINALS_LOCK.json').exists():raise ValueError('DEV_GT_LOCK_NOT_RELEASED')
    with np.load(BASE/'C0'/f'{split}.npz',allow_pickle=False) as f:return {k:f[k] for k in ('ids','past','future','group')}

def evaluate_population(folder,paths,logits,control,cache,split,guard=None,preflight=False):
    from evaluate_saved import evaluate as canonical_evaluate
    sys.path.insert(0,str(STAGE_A));from scoring import kinematic,dpp,turning
    data=labels(split)
    if preflight:
        lookup={str(x):i for i,x in enumerate(data['ids'])};ix=np.array([lookup[str(x)] for x in cache['ids']]);data={k:v[ix] for k,v in data.items()}
    if not np.array_equal(data['ids'],cache['ids']):raise ValueError('EVAL_ID_ORDER')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    metrics,e,audit=canonical_evaluate(paths,logits,data,guard)
    for row in metrics.values():row.pop('LEGACY_SHADOW',None)
    c=read(STAGE_A/'CALIBRATION.json');kin=kinematic(data['past'][:,0].astype(np.float64),paths,c)
    quality=np.ones_like(kin['qkin']) if control is None else None
    arm='native' if control is None else folder.parent.name.split('_seed')[0]
    quality=np.ones_like(kin['qkin']) if arm in ('native','uniform') else kin['qkin']
    ds=dpp(paths,quality,c['k'],c['epsilon_q']);validgood=(kin['kin_valid']&e['good']).any(1)
    groups={};native_errors=np.linalg.norm(cache['native'].astype(np.float64)-data['future'][:,None].astype(np.float64),axis=-1)
    nativegood=((native_errors.mean(-1)<=.2)&(native_errors[:,:,-1]<=.4)).any(1)
    gtturn=turning(data['past'][:,0].astype(np.float64),data['future'][:,None].astype(np.float64))[:,0]
    for g,name in enumerate(('overall','stable','transition','change')):
        mask=np.ones(len(paths),bool) if g==0 else data['group']==g-1;n=int(mask.sum());good=e['coverage'];rescue=mask&good&~nativegood;lost=mask&~good&nativegood
        groups[name]=dict(N=n,good=int(good[mask].sum()),native_good=int(nativegood[mask].sum()),rescue=int(rescue.sum()),lost=int(lost.sum()),lost_rate=float(lost.sum()/nativegood[mask].sum()) if nativegood[mask].any() else None,rescue_IDs=data['ids'][rescue].tolist(),lost_IDs=data['ids'][lost].tolist(),kin_valid_coverage=float(validgood[mask].mean()) if n else None,kin_violation_rate=float((~kin['kin_valid'][mask]).mean()) if n else None,qkin_mean=float(kin['qkin'][mask].mean()) if n else None,qkin_underflow=int((kin['qkin'][mask]==0).sum()),EC=float(ds['ec'][mask].mean()) if n else None)
    np.savez_compressed(folder/'predictions.npz',paths=paths,logits=logits,ids=data['ids'],past=data['past'][:,0],future=data['future'],groups=data['group'])
    np.savez_compressed(folder/'window_metrics.npz',**e)
    from f16.eval import clusters
    cc=clusters(paths,e['good'])
    aux=dict(ids=data['ids'],groups=data['group'],qkin=kin['qkin'],logqkin=kin['logqkin'],penalty_v=kin['penalty_v'],penalty_a=kin['penalty_a'],penalty_join=kin['penalty_join'],kin_valid=kin['kin_valid'],EC=ds['ec'],marginal=ds['marginal'],kin_valid_coverage=validgood,kin_violation_rate=(~kin['kin_valid']).mean(1),effective_good=cc[:,1],distinct=cc[:,0],gt_turn=gtturn)
    if control is not None:aux['control']=control
    np.savez_compressed(folder/'auxiliary.npz',**aux)
    write(folder/'metrics.json',metrics);write(folder/'support_and_quality.json',groups);write(folder/'audit.json',audit)
    write(folder/'done.json',dict(status='SUCCESS_SAVED_CANONICAL_INDEPENDENT_CLUSTER_MOTION',source=sha(RUN/'PRODUCTION_SOURCE_LOCK.json') if (RUN/'PRODUCTION_SOURCE_LOCK.json').exists() else sha(RUN/'PRE_SOURCE.json'),files=seal(folder.iterdir())))
    return metrics,e,groups

def independent_saved_review(adapter,cache,folder,arm,guard):
    sys.path.insert(0,str(STAGE_A))
    import importlib.util
    sp=importlib.util.spec_from_file_location('stageA_independent',STAGE_A/'independent.py');im=importlib.util.module_from_spec(sp);sp.loader.exec_module(im)
    done=read(folder/'done.json');check_files(done['files'])
    with np.load(folder/'predictions.npz',allow_pickle=False) as f:paths=f['paths'];ids=f['ids']
    with np.load(folder/'auxiliary.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    if not np.array_equal(ids,cache['ids']):raise ValueError('SAVED_INDEPENDENT_IDS')
    weights={k:v.detach().cpu().numpy().astype(np.float64) for k,v in adapter.model.decoder.state_dict().items()}
    c=read(STAGE_A/'CALIBRATION.json');errorq=errorec=errormarg=errorpath=rawscaled=0.
    for start in range(0,len(ids),64):
        sl=slice(start,start+64);y=paths[sl];past=cache['raw_past'][sl,0].astype(np.float64)
        q,valid,raw=im.score_loop(past,y,c,return_details=True)
        if not np.array_equal(valid,a['kin_valid'][sl]):raise ValueError('INDEPENDENT_VALID')
        errorq=max(errorq,float(np.max(abs(q-a['qkin'][sl]))))
        for key,value in raw.items():rawscaled=max(rawscaled,float(np.max(abs(value-a[key][sl])/(2e-12+2e-12*abs(value)))))
        quality=np.ones_like(q) if arm in ('uniform','native') else q
        ec,marg=im.spectral_ec_loop(y,quality,c['k'],c['epsilon_q'],return_marginal=True)
        errorec=max(errorec,float(np.max(abs(ec-a['EC'][sl]))));errormarg=max(errormarg,float(np.max(abs(marg-a['marginal'][sl]))))
        if arm!='native':
            reconstructed=decoder_np(a['control'][sl],cache,sl,weights);errorpath=max(errorpath,float(np.max(abs(reconstructed-y))))
        guard.check()
    if max(errorq,errorec,errormarg)>2e-12 or errorpath>2e-5 or rawscaled>1:raise ValueError('INDEPENDENT_SAVED_SCORE_EC_OR_DECODER_FAILURE')
    check_files(done['files'])
    report=dict(status='PASS_ALL_SAVED_QKIN_EC_MARGINAL_AND_CONTROL_RECONSTRUCTION',N=len(ids),score_error=errorq,EC_error=errorec,marginal_error=errormarg,raw_penalty_logq_scaled_error=rawscaled,decoder64_vs_production32_max_path_error=errorpath,arithmetic_tolerance_not_scientific_threshold=True,not_whole_model_or_gradient_replay=True)
    write(folder/'independent_review.json',report);return report
