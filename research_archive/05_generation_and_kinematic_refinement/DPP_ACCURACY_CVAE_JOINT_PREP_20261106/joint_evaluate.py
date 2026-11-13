"""Canonical complete K20 quality/cluster/motion; no fictional CVAE scorer."""
from pathlib import Path
import sys, numpy as np
from joint_common import BASE,STAGE,labels,module,read,write,seal,legacy,verify
from joint_losses import quality,expected_cardinality

def canonical_api():
    # This evaluator also works as the FIRST stage of a fresh CVAE-only process.
    # Never rely on an earlier existing-h3 load to initialize the `common` namespace.
    sys.path.insert(0,str(BASE))
    if 'common' in sys.modules and Path(sys.modules['common'].__file__).resolve()!=BASE/'common.py':
        raise RuntimeError('CANONICAL_COMMON_NAMESPACE_COLLISION')
    module('common',BASE/'common.py')
    return module('joint_canonical_saved',BASE/'evaluate_saved.py')

def evaluate(folder,paths,control,data,split,execution=None,guard=None,fixture=False,reconstruct=None):
    target,group=labels(split,data['ids'],execution=execution,preflight=fixture)
    # Original imported canonical evaluator has no output side effects on import.
    canonical=canonical_api()
    p=Path(folder);p.mkdir(parents=True,exist_ok=False)
    d=dict(ids=data['ids'],past=data['raw_past'],future=target,group=group)
    # Scores are not trained for either family. Remove ALL fake top1 reports.
    logits=np.zeros((len(paths),20),np.float32)
    metrics,e,audit=canonical.evaluate(paths,logits,d,guard)
    for row in metrics.values():
        for key in ('LEGACY_SHADOW','score_top1_ADE','score_top1_FDE'):row.pop(key,None)
        if 'tail_risk' in row:row['tail_risk'].pop('score_top1_not_probability',None)
        row['scorer_status']='NOT_TRAINED_NOT_EVALUATED_NO_SLOT0_PROXY'
    scoring=module('joint_stageA_score',STAGE/'scoring.py');cal=read(STAGE/'CALIBRATION.json')
    kin=scoring.kinematic(data['raw_past'][:,0].astype(np.float64),paths,cal)
    dpp=scoring.dpp(paths,kin['qkin'],cal['k'],cal['epsilon_q'])
    dd=np.linalg.norm(data['native'].astype(np.float64)-target[:,None].astype(np.float64),axis=-1)
    ng=((dd.mean(-1)<=.2)&(dd[:,:,-1]<=.4)).any(1)
    support={}; valid_good=(kin['kin_valid']&e['good']).any(1)
    cc=canonical.clusters(paths,e['good'])
    for i,g in enumerate(('overall','stable','transition','change')):
        m=np.ones(len(paths),bool) if i==0 else group==i-1
        lost=m&ng&~e['coverage'];resc=m&~ng&e['coverage'];den=int((m&ng).sum())
        support[g]=dict(N=int(m.sum()),native_good=den,good=int(e['coverage'][m].sum()),
            rescue=int(resc.sum()),lost=int(lost.sum()),lost_rate=float(lost.sum()/den) if den else None,
            rescue_IDs=data['ids'][resc].tolist(),lost_IDs=data['ids'][lost].tolist(),
            kin_valid_coverage=float(valid_good[m].mean()),kin_violation_rate=float((~kin['kin_valid'][m]).mean()))
    np.savez_compressed(p/'predictions.npz',paths=paths,control=control,ids=data['ids'],future=target,past=data['raw_past'][:,0],groups=group)
    np.savez_compressed(p/'window_metrics.npz',**e)
    np.savez_compressed(p/'auxiliary.npz',ids=data['ids'],groups=group,qkin=kin['qkin'],kin_valid=kin['kin_valid'],
        EC=dpp['ec'],marginal=dpp['marginal'],effective_good=cc[:,1],kin_valid_coverage=valid_good,
        kin_violation_rate=(~kin['kin_valid']).mean(1))
    write(p/'metrics.json',metrics);write(p/'support.json',support);write(p/'audit.json',audit)
    independent=module('joint_stageA_independent',STAGE/'independent.py')
    maxq=maxec=maxm=maxpath=0.
    for start in range(0,len(paths),64):
        sl=slice(start,start+64)
        q,v=independent.score_loop(data['raw_past'][sl,0].astype(np.float64),paths[sl],cal)
        ec,marg=independent.spectral_ec_loop(paths[sl],q,cal['k'],cal['epsilon_q'],return_marginal=True)
        if not np.array_equal(v,kin['kin_valid'][sl]):raise ValueError('INDEPENDENT_KIN_VALID')
        maxq=max(maxq,float(np.max(abs(q-kin['qkin'][sl]))));maxec=max(maxec,float(np.max(abs(ec-dpp['ec'][sl]))))
        maxm=max(maxm,float(np.max(abs(marg-dpp['marginal'][sl]))))
        if reconstruct is not None:
            maxpath=max(maxpath,float(np.max(abs(reconstruct(control[sl],data,sl)-paths[sl]))))
        if guard:guard.check()
    if max(maxq,maxec,maxm)>2e-12:raise ValueError('SAVED_DPP_INDEPENDENT')
    if reconstruct is not None and maxpath>2e-5:raise ValueError('INDEPENDENT_NUMPY64_DECODER_RECONSTRUCTION')
    with np.load(p/'predictions.npz',allow_pickle=False) as f:
        if not np.array_equal(f['paths'],paths) or not np.array_equal(f['ids'],data['ids']):raise ValueError('SAVED_READBACK')
    write(p/'independent_review.json',dict(status='PASS_SAME_SLOT_FULLK_CLUSTER_PDIST_MOTION_AND_INDEPENDENT_Q_EC',
        maxq=maxq,maxEC=maxec,maxmarginal=maxm,max_decoder64_vs_prod32_error=maxpath,
        control_decoder_reconstructed=reconstruct is not None,no_model_or_gradient_replay=True,arithmetic_not_scientific_tolerance=True))
    write(p/'done.json',dict(status='PASS_NEW_POPULATION',files=seal(p.iterdir())))
    return metrics,support
