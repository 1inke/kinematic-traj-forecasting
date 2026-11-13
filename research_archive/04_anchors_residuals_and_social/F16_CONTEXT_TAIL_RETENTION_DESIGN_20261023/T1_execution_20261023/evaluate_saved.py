from core import *

def data_with_metadata(d,paths):
    result=dict(d,paths=paths,sample_id=d['ids'])
    with np.load(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz') as z:
        lookup={int(v):i for i,v in enumerate(z['sample_id'])};ix=[lookup[int(v)] for v in d['ids']]
        result.update(sequence=z['sequence'][ix].astype(str),scene=z['scene_family'][ix].astype(str))
        assert np.array_equal(result['track'],np.array([str(a)+'::'+str(int(b)) for a,b in zip(z['sequence'][ix],z['pedestrian_id'][ix])]))
    return result

def save_evaluate(run,split,paths,d,guard):
    before={p.relative_to(ROOT).as_posix():sha256(p) for p in (OLD/'statistics.npz',OLD/'split_ids.npz')}
    path=run/f'predictions_{split}.npz'
    np.savez_compressed(path,paths=paths,native=d['native'],gt=d['gt'],sample_id=d['ids'],groups=d['groups'],track=d['track'])
    with np.load(path) as z:assert np.array_equal(z['paths'],paths) and np.array_equal(z['sample_id'],d['ids'])
    metrics,er=evaluate(paths,d['gt'],d['groups'],d['native'],d['ids'],d['track'])
    write_json(run/f'metrics_{split}.json',metrics);np.savez_compressed(run/f'sample_metrics_{split}.npz',**er)
    guard.check()
    audit=old.independent.audit_file(path,run/f'metrics_{split}.json',frozen_cache=old.prior.PATHS['cache'],frozen_labels=old.prior.PATHS['labels'],frozen_metadata=old.prior.PATHS['metadata'],frozen_manifest=ROOT/'config/data_manifest.v1.json',root=ROOT)
    assert audit['status']=='PASS',audit
    write_json(run/f'audit_{split}.json',audit)
    enriched=data_with_metadata(d,paths);out=diag.compute(enriched);summary=diag.summary(out)
    review=diag_review.review_job(enriched,out,summary);assert review['status']=='PASS',review
    np.savez_compressed(run/f'diagnostics_{split}.npz',**out);write_json(run/f'diagnostics_{split}.json',summary);write_json(run/f'diagnostic_audit_{split}.json',review)
    native=errors(d['native'],d['gt']);excess=np.maximum(er['J'].min(1)-native['J'].min(1),0)
    population={}
    for group,num in [('overall',None),('stable',0),('transition',1),('change',2)]:
        mask=np.ones(len(paths),bool) if num is None else d['groups']==num
        population[group]=dict(n=int(mask.sum()),R=float(excess[mask].mean()),active_fraction=float((excess[mask]>0).mean()))
    # Base population is N-weighted. Diversity is over 190 pairs, never window p95 averaging.
    total=0.;rel=0.
    for i in range(0,len(paths),256):
        p=paths[i:i+256].astype(float);y=d['gt'][i:i+256].astype(float);distance=np.linalg.norm(p-y[:,None],axis=-1)
        endpoint=np.linalg.norm(p[:,PAIR[0],-1]-p[:,PAIR[1],-1],axis=-1)
        total+=(distance.mean(-1)+.5*distance[:,:,-1]).min(1).sum()+.05*np.maximum(.3-endpoint,0).sum()/190
        rel+=.2*excess[i:i+256].sum()
    write_json(run/f'population_loss_{split}.json',dict(groups=population,base=total/len(paths),relative_penalty=rel/len(paths),dtype='float64 evaluation, distinct from float32 history preupdate'))
    if run.name.startswith('P_relative_seed'):
        seed=int(run.name.split('seed')[-1]);fixed={}
        with np.load(DIAG/f'D0_D1/seed{seed}_{split}/per_window.npz') as z:
            assert np.array_equal(z['sample_id'],d['ids']);labels=z['partition']
            for tag,label in [('retained',0),('rescue',1),('lost',2),('neither',3)]:
                mask=labels==label;fixed[tag]=dict(n=int(mask.sum()),good=int(er['coverage'][mask].sum()),mean_J=float(er['J'][mask].min(1).mean()) if mask.any() else None,mean_delta_J_vs_oldA=float((er['J'][mask].min(1)-z['student_Jmin'][mask]).mean()) if mask.any() else None)
        write_json(run/f'FIXED_OLD_A_SUBSETS_{split}.json',fixed)
    assert before=={k:sha256(ROOT/k) for k in before}
    guard.check();return metrics
