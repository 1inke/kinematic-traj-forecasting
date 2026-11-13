"""Final evaluation only; held GT cannot be accessed until all six final models lock."""
import numpy as np
import torch
from env import *
from data_pipeline import CACHE,OptData,contract
from train_paired import read_completed,lock_all_finals,initialize
from editor import choose
from feature_builder import build
from review_saved import independent_population,direct,compare
def check_final_unlock(source):
    locks=read_json(HERE/'ALL6_FINALS_LOCK.json')
    if locks['status']!='ALL6_FIXED_FINAL8_LOCKED' or locks['source']!=source:raise ValueError('Held GT locked until all6finals')
    for name,v in locks['locks'].items():
        folder=HERE/'runs'/name
        for key,file in (('checkpoint','checkpoint.pt'),('done','done.json'),('manifest','run_manifest.json'),('history','history.json')):
            if sha256(folder/file)!=v[key]:raise ValueError('Held-GT unlock final SHA')
def population(seed,split,source,limit=None):
    from data_contract import CACHE as RAWCACHE,ENC,SPLIT,LABEL,kinpath,OLD
    registry=np.load(E0/'SPLIT_REGISTRY.npz')
    with np.load(ROOT/SPLIT) as z:fit=z['grouped_train_index'];inner=z['inner_cal_index'];seq=z['sequence'];ped=z['pedestrian_id'];g0=z['loss_change_group']
    if split=='opt':
        ix=registry['opt_row'];raw=fit[ix]
        with np.load(E0/f'seed{seed}/input_pools.npz') as z:native=z['native'];kin=z['kin'];gt=z['gt'];ids=z['sample_id'];groups=z['group'];tracks=z['track_key']
    elif split in ('hold','inner'):
        check_final_unlock(source)
        if split=='hold':
            ix=registry['hold_row'];raw=fit[ix];kin=np.asarray(np.load(ROOT/kinpath(seed),mmap_mode='r')[ix],np.float64)
        else:
            raw=inner;p=ROOT/f'COMPLEMENTARY_POOL_VALIDATION_20261018/artifacts/v1/kin_inner_seed{seed}.npy';old=read_json(ROOT/OLD)
            if sha256(p)!=old['files'][f'kin_inner_seed{seed}.npy']['sha256']:raise ValueError('Historical inner KIN SHA')
            kin=np.asarray(np.load(p,allow_pickle=False),np.float64);assert kin.shape==(1856,20,12,2)
        with np.load(ROOT/RAWCACHE) as z:native=z['candidate'][raw,:,:24].reshape(-1,20,12,2).astype(np.float64);ids=z['sample_id'][raw]
        # Only reached after SHA-bound all-six lock.
        with np.load(ROOT/LABEL) as z:assert np.array_equal(z['sample_id'][raw],ids);gt=z['future'][raw].astype(np.float64)
        groups=g0[raw];tracks=np.array([f'{s}|{int(t)}' for s,t in zip(seq[raw],ped[raw])])
    else:raise ValueError('Only opt/hold/inner')
    with np.load(ROOT/ENC) as z:assert np.array_equal(z['sample_id'][raw],ids);past=z['past'][raw];tokens=z['tokens'][raw]
    if limit is not None:
        native,kin,gt,ids,groups,tracks,past,tokens=(x[:limit] for x in (native,kin,gt,ids,groups,tracks,past,tokens))
    return dict(native=native,kin=kin,gt=gt,ids=ids,groups=groups,tracks=tracks,past=past,tokens=tokens)
def _action_chunk(seed,split,data,start,end):
    if split=='opt' and end-start==128 and start%128==0:
        with np.load(E0/f'seed{seed}/chunk_{start:05d}_{end:05d}.npz') as z:return {k:z[k] for k in ('motion','quality','eligible','rejects','B','target','margin')}
    records=[e0space.enumerate_window(data['native'][i],data['kin'][i],data['gt'][i]) for i in range(start,end)]
    return {k:np.asarray([r[k] for r in records]) for k in ('motion','quality','eligible','rejects','B','target','margin')}
def write_population(seed,split,data,models,source,folder,guard):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists():
        d=read_json(folder/'done.json')
        if d['source']!=source:raise ValueError('Saved population source')
        for fn,ss in d['outputs'].items():
            if sha256(folder/fn)!=ss:raise ValueError('Saved population output SHA')
        return d
    if list(folder.glob('aux_*.npz')):raise FileExistsError('Preserve partial eval; do not blindly overwrite')
    before={arm:{k:v.detach().cpu().clone() for k,v in model.state_dict().items()} for arm,model in models.items()}
    paths={a:[] for a in models};chosen={a:[] for a in models};quality=[];B_all=[];valid_all=[];outputs={};chunks=[]
    for start in range(0,len(data['ids']),128):
        end=min(start+128,len(data['ids']));aux=_action_chunk(seed,split,data,start,end)
        if split=='opt':raw=np.asarray(np.load(CACHE/f'features_seed{seed}.npy',mmap_mode='r')[start:end]).copy()
        else:raw=np.asarray([build(data['past'][i],data['tokens'][i],data['native'][i],data['kin'][i],aux['motion'][i-start]) for i in range(start,end)],np.float32)
        af=f'aux_{start:05d}_{end:05d}.npz'
        np.savez_compressed(folder/af,sample_id=data['ids'][start:end],track_key=data['tracks'][start:end],group=data['groups'][start:end],native=data['native'][start:end],kin=data['kin'][start:end],gt=data['gt'][start:end],**aux)
        chunks.append(af);outputs[af]=sha256(folder/af)
        for arm,model in models.items():
            model.eval()
            with torch.no_grad():scores=model(torch.from_numpy(raw).to(next(model.parameters()).device)).cpu().numpy()
            if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite saved action scores')
            action=np.where(aux['eligible'],scores,-np.inf).argmax(1)
            pp=np.asarray([e0space.materialize(data['native'][i],data['kin'][i],action[i-start]) for i in range(start,end)])
            fn=f'{arm}_{start:05d}_{end:05d}.npz';np.savez_compressed(folder/fn,sample_id=data['ids'][start:end],scores=scores,action=action,paths=pp)
            outputs[fn]=sha256(folder/fn);paths[arm].append(pp);chosen[arm].append(action)
        quality.append(aux['quality']);B_all.append(aux['B']);valid_all.append(aux['eligible']);guard.check()
    q=np.concatenate(quality);bb=np.concatenate(B_all);vv=np.concatenate(valid_all)
    for arm,model in models.items():
        assert all(torch.equal(before[arm][k],v.cpu()) for k,v in model.state_dict().items())
        pp=np.concatenate(paths[arm]);acts=np.concatenate(chosen[arm]);metrics,cc=independent_population(pp,data['gt'],data['groups'],data['native'],data['ids'],data['tracks'])
        row=np.arange(len(acts));isB=bb[row,acts];selectedq=q[row,acts];base=q[:,0];harm=(selectedq>base).any(1)|((base[:,2]<=1)&(selectedq[:,2]>1));nonstop=acts!=0
        assignment={}
        for gi,name in enumerate(('overall','stable','transition','change')):
            mask=np.ones(len(acts),bool) if gi==0 else data['groups']==gi-1
            assignment[name]=dict(n=int(mask.sum()),B_nonempty=int((bb.any(1)&mask).sum()),chosen_in_B=int((isB&mask).sum()),STOP_omitted_opportunity=int((~nonstop&bb.any(1)&mask).sum()),correct_STOP_no_B=int((~nonstop&~bb.any(1)&mask).sum()),executed=int((nonstop&mask).sum()),harmful=int((nonstop&harm&mask).sum()),ineffective=int((nonstop&~isB&~harm&mask).sum()),eligible_actions=int(vv[mask].sum()),B_count_quantiles=np.quantile(bb[mask].sum(1),[0,.5,.95,1]).tolist() if mask.any() else [])
        for name,value in ((f'{arm}_metrics.json',metrics),(f'{arm}_assignment.json',assignment)):
            write_json(folder/name,value);outputs[name]=sha256(folder/name)
    done=dict(status='SAVED_POPULATION_PENDING_AUDIT',source=source,seed=seed,split=split,n=len(data['ids']),chunks=chunks,arms=list(models),eval_weights_buffers_exact=True,outputs=outputs)
    write_json(folder/'done.json',done);return done
def audit_population(folder,source,guard):
    d=read_json(folder/'done.json');assert d['source']==source
    paths={a:[] for a in d['arms']};ids=[];groups=[];tracks=[];gt=[];native=[];maxerr=0.;direct_actions=0
    for fn in d['chunks']:
        with np.load(folder/fn) as zz:z={k:zz[k] for k in zz.files}
        count=len(z['sample_id']);ids.append(z['sample_id']);groups.append(z['group']);tracks.append(z['track_key']);gt.append(z['gt']);native.append(z['native'])
        for i in range(count):
            if d['split']!='opt':
                ref,_=direct(z['native'][i],z['kin'][i],z['gt'][i]);saved={name:z[name][i] for name in ('motion','quality','margin','rejects','eligible','B','target')};saved['action']=ref['action'];maxerr=max(maxerr,compare(saved,ref));direct_actions+=400
        suffix=fn[len('aux_'):]
        for arm in d['arms']:
            with np.load(folder/(arm+'_'+suffix)) as out:
                assert np.array_equal(out['sample_id'],z['sample_id']);act=np.where(z['eligible'],out['scores'],-np.inf).argmax(1)
                assert np.array_equal(act,out['action'])
                for i,a in enumerate(act):
                    expected=z['native'][i].copy()
                    if a:r,l=divmod(int(a)-1,20);expected[r]=z['kin'][i,l]
                    if not np.array_equal(expected,out['paths'][i]):raise ValueError('Chosen action/fixed19/STOP path')
                paths[arm].append(out['paths'])
        guard.check()
    arrays=[np.concatenate(v) for v in (ids,groups,tracks,gt,native)];ids,groups,tracks,gt,native=arrays
    for arm in d['arms']:
        rows,_=independent_population(np.concatenate(paths[arm]),gt,groups,native,ids,tracks)
        if e0.digest(rows)!=e0.digest(read_json(folder/f'{arm}_metrics.json')):raise ValueError('Saved population full quality/cluster/motion')
    for fn,ss in d['outputs'].items():
        if sha256(folder/fn)!=ss:raise ValueError('Final population SHA')
    report=dict(status='PASS_SAVED_POPULATION',n=d['n'],arms=d['arms'],allK20_clusters_physics=True,direct_new_action_audits=direct_actions,opt_actions_reuse_complete_E0_by_source_SHA=d['split']=='opt',max_abs_error=maxerr,no_model_replay=True)
    write_json(folder/'ARITHMETIC_AUDIT.json',report);return report
