"""P0/P1: preserve originals, evidence inventory, E0 corrected shared-track CI."""
from common import *

def exposure(train_ids,eval_ids,md):
    lookup={int(s):i for i,s in enumerate(md['sample_id'])};ti=np.array([lookup[int(s)] for s in train_ids]);ei=np.array([lookup[int(s)] for s in eval_ids]);tracks=np.array([str(s)+'::'+str(int(p)) for s,p in zip(md['sequence'],md['pedestrian_id'])]);starts=md['start_frame'];unknown=md['ambiguous_track_time']|(starts<0);by={}
    for i in ti:
        if not unknown[i]:by.setdefault(tracks[i],[]).append((int(starts[i]),int(starts[i])+19))
    same=np.isin(tracks[ei],tracks[ti]);over=[];future=[]
    for i in ei:
        if unknown[i]:continue
        t=int(starts[i]);matches=by.get(tracks[i],[])
        if any(a<=t+19 and b>=t for a,b in matches):over.append(int(md['sample_id'][i]))
        if any(a<=t+19 and b>=t+8 for a,b in matches):future.append(int(md['sample_id'][i]))
    return dict(eval_N=len(ei),same_sample_N=int(np.isin(eval_ids,train_ids).sum()),same_track_N=int(same.sum()),interval_N=len(over),future_intersection_N=len(future),unknown_eval_N=int(unknown[ei].sum()),unknown_train_N=int(unknown[ti].sum()),interval_IDs=over)

def main():
    if (HERE/'source_manifest_before.json').exists():raise RuntimeError('P0_EXISTS_DO_NOT_OVERWRITE_EVIDENCE')
    paths=current_sources();write_json(HERE/'source_manifest_before.json',dict(evidence_kind='POSTHOC_RECONSTRUCTION',historical_lock=False,files=capture(paths)))
    p=read_json(HERE/'protocol.json');guard=Guard(p['resource']['estimate_bytes']);guard.check('startup');data=data_load();inventory=[]
    for seed,arm in RUNS:
        ck=checkpoint(seed,arm)
        if not ck.exists():raise FileNotFoundError(ck)
        r=old_result(seed)['results'][arm]
        inventory.append(dict(seed=seed,arm=arm,checkpoint=str(ck),size=ck.stat().st_size,sha256_now=sha256(ck),source_binding='UNKNOWN_TRAINING_TIME',params_reported=r['params'],epochs_saved=len(r['history']),optimizer_RNG='NOT_PRESENT_AS_REPORTED_NOT_RESTORED',inner_good=r['eval']['change']['good_count'],motion_gate=r['eval']['change']['motion_gate']))
    if len(inventory)!=11:raise ValueError('INVENTORY_NOT_11')
    write_csv(HERE/'RUN_INVENTORY.csv',inventory)
    lineage=[]
    for name,path in PATHS.items():
        with np.load(path,allow_pickle=False) as z:arrays={key:dict(shape=list(z[key].shape),dtype=str(z[key].dtype)) for key in z.files}
        lineage.append(dict(role=name,artifact_path=str(path),file_sha256_now=sha256(path),arrays=arrays,producer_checkpoint_sha256=None,producer_training_ids_sha256=None,historical_source_binding='UNKNOWN',evidence_kind='POSTHOC_RECONSTRUCTION',replay_verified=False,coordinate='focal-local last observed point zero' if name in ['cache','labels','encoder'] else 'metadata / index',time_unit='0.4s sample interval; source indices not verified video frames'))
    kinrows=[]
    for seed in SEEDS:
        for split in ['FIT','inner']:
            path=kin_path(split,seed);ix=data['split'][split];a=np.load(path,mmap_mode='r')
            if a.shape!=(len(ix),20,12,2):raise ValueError('KIN_CACHE_SHAPE')
            lineage.append(dict(role=f'E1_KIN_{split}_seed{seed}',artifact_path=str(path),file_sha256_now=sha256(path),array_sha256_now=array_sha(a),shape=list(a.shape),dtype=str(a.dtype),ID_row_mapping='producer loops data.split indices in current source; full numeric checkpoint-row replay required in preflight',mapping_ids_sha256=array_sha(data['ids'][ix]),producer_checkpoint=str(ROOT/f'KINEMATIC_DECODER_SPEED_CLAMP_20261018/results/short/KIN_seed{seed}.pt'),producer_checkpoint_sha256=sha256(ROOT/f'KINEMATIC_DECODER_SPEED_CLAMP_20261018/results/short/KIN_seed{seed}.pt'),producer_training_ids_sha256=None,historical_source_binding='UNKNOWN',evidence_kind='POSTHOC_RECONSTRUCTION',replay_verified=False,upstream_exposure_status='SOURCE_RECIPE_FIT_NOT_PER_CHECKPOINT_MANIFEST'))
        e0=read_json(E0/'RESULT.json');actual=Path(e0['results'][str(seed)]['kin_src']);a=np.load(actual,mmap_mode='r');oldsha=e0['results'][str(seed)]['kin_sha']
        if array_sha(a)!=oldsha:raise ValueError('E0_ACTUAL_ARRAY_SHA_CHANGED')
        locked=[k for k in e0['inputs_sha_before'] if f'kin_original_CAL_seed{seed}' in k]
        lineage.append(dict(role=f'E0_actual_KIN_CAL_seed{seed}',artifact_path=str(actual),file_sha256_now=sha256(actual),array_sha256_now=array_sha(a),array_sha_matches_original_record=True,original_file_manifest_paths=locked,actual_file_in_original_manifest=any(Path(k).name==actual.name and (ROOT/k).resolve()==actual.resolve() for k in locked),evidence_kind='POSTHOC_RECONSTRUCTION',historical_source_binding='ACTUAL_ARRAY_HASH_PRESENT_FILE_PATH_LOCK_MISMATCH',producer_checkpoint_sha256=sha256(ROOT/f'KINEMATIC_DECODER_SPEED_CLAMP_20261018/results/short/KIN_seed{seed}.pt'),producer_training_ids_sha256=None,ID_row_mapping='original_CAL registry indices25912..28790, numeric replay verification pending',upstream_exposure_status='UNKNOWN_RUNTIME_BINDING'))
    write_json(HERE/'CACHE_LINEAGE.json',dict(entries=lineage,note='Current hash does not backdate a training-time lock; no old source changed.'))
    with np.load(PATHS['metadata'],allow_pickle=False) as z:md={k:z[k].copy() for k in z.files}
    cal_index=np.load(ROOT/'qoder_work/v1_1/cal_v11_index.npy');evalsets={'FIT':data['ids'][data['split']['FIT']],'inner':data['ids'][data['split']['inner']],'cal_v11':data['ids'][cal_index]};exrows=[];traces={}
    for model_name,train_ids,level in [('student_current_recipe',evalsets['FIT'],'CURRENT_SOURCE_RECIPE_NOT_BOUND_OLD_CHECKPOINT'),('KIN_current_recipe',evalsets['FIT'],'CURRENT_SOURCE_RECIPE_NOT_BOUND_OLD_CHECKPOINT')]:
        for name,eval_ids in evalsets.items():
            v=exposure(train_ids,eval_ids,md);traces[model_name+'__'+name]=v;exrows.append(dict(model=model_name,evaluation=name,evidence=level,**{k:v[k] for k in ['eval_N','same_sample_N','same_track_N','interval_N','future_intersection_N','unknown_eval_N','unknown_train_N']}))
    for stage in ['h3','v2']:
        base=ROOT/f'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/results/eth/{stage}/f16_temporal_social_cross/seed_1/val_scene_-1';sp=base/'split_indices.npz'
        with np.load(sp,allow_pickle=False) as z:
            key=next(k for k in ['train','train_indices','train_idx'] if k in z.files);train_ids=z[key].copy()
        for name,eval_ids in evalsets.items():
            v=exposure(train_ids,eval_ids,md);traces['native_'+stage+'__'+name]=v;exrows.append(dict(model='native_'+stage,evaluation=name,evidence='ACTUAL_SPLIT_ARTIFACT_ASSOCIATED_BY_CURRENT_CACHE_PRODUCER_SOURCE_NOT_RUNTIME_LOCK',**{k:v[k] for k in ['eval_N','same_sample_N','same_track_N','interval_N','future_intersection_N','unknown_eval_N','unknown_train_N']}))
    write_csv(HERE/'EXPOSURE_AUDIT.csv',exrows);write_json(HERE/'EXPOSURE_ID_TRACE.json',traces)
    # Preserve exact historical RNG convention from prior corrected sensitivity.
    clean=cal_index;ch=data['groups'][clean]==2
    with (ROOT/'qoder_work/v1_1/cal_v11_windows.csv').open(encoding='utf-8-sig') as f:csvrows=list(csv.DictReader(f))
    tracks=np.array([r['track'] for r in csvrows]);deltas=[];e0rows=[]
    for seed in SEEDS:
        kin=np.load(Path(read_json(E0/'RESULT.json')['results'][str(seed)]['kin_src']))
        for name,ix in [('clean',clean),('dropped',np.setdiff1d(np.arange(25912,28791),clean))]:
            idx=np.load(E0/f'k0c_indices_{name}_seed{seed}.npy');pool=np.concatenate([data['native'][ix],kin[ix-25912]],1);pred=pool[np.arange(len(ix))[:,None],idx]
            def good(a):
                e=np.linalg.norm(a.astype(float)-data['gt'][ix,None],axis=-1);return ((e.mean(-1)<=.2)&(e[:,:,-1]<=.4)).any(1)
            ng=good(data['native'][ix]);pg=good(pred);mask=data['groups'][ix]==2;delta=pg[mask].astype(float)-ng[mask].astype(float)
            if not np.array_equal(delta,np.load(E0/f'delta_change_{name}_seed{seed}.npy')):raise ValueError('E0_SAVED_DELTA_MISMATCH')
            e0rows.append(dict(seed=seed,pop=name,change_N=int(mask.sum()),native=int(ng[mask].sum()),oracle=int(pg[mask].sum()),rescue=int((~ng[mask]&pg[mask]).sum()),lost=int((ng[mask]&~pg[mask]).sum())))
            if name=='clean':deltas.append(delta)
    d=np.stack(deltas);tr=tracks[ch];uniq=np.unique(tr);counts=np.array([(tr==u).sum() for u in uniq]);sums=np.stack([d[:,tr==u].sum(1) for u in uniq],axis=1);rng=np.random.default_rng(20261020);draw=rng.integers(len(uniq),size=(20000,len(uniq)));means=sums[:,draw].sum(-1)/counts[draw].sum(-1)[None,:];fixed=means.mean(0);sd=rng.integers(3,size=(20000,3));hier=means[sd,np.arange(20000)[:,None]].mean(1)
    write_json(HERE/'E0_STATISTICS_CORRECTED.json',dict(evidence_kind='POSTHOC_SENSITIVITY',rows=e0rows,point_pp=float(d.mean()*100),tracks=len(uniq),draws=20000,rng='PCG64(20261020), all track draws then paired seed draws as preserved prior review',shared_track_fixed_seed_CI_pp=(np.quantile(fixed,[.025,.975])*100).tolist(),shared_track_paired_seed_CI_pp=(np.quantile(hier,[.025,.975])*100).tolist(),not_full_W_or_motion_audit=True))
    guard.check();verify_sources();write_json(HERE/'P0_P1_RECORD.json',dict(status='PRESERVED_AND_E0_COVERAGE_STATISTICS_CORRECTED',models=11,planned_jobs=22,resource=guard.observations,old_inputs_unchanged=True));progress('P0_P1_DONE',done_jobs=0)
    print('P0 complete:11 checkpoints, current source/file lineage, exposure table and fixed shared-track CI.',flush=True)

if __name__=='__main__':main()
