"""Explicit original-ID joins and frozen historical cache provenance, E0 FIT labels only."""
import math,hashlib
import numpy as np
from common import *
CACHE='REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
LABEL='F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
ENC='GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
SPLIT='GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'
OLD='CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/freeze/input_manifest.json'
def kinpath(seed):return f'UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_FIT_seed{seed}.npy'
def keyhash(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def load(save=False):
    old=read_json(ROOT/OLD)
    required={'eth_train.npz':CACHE,'labels_train.npz':LABEL,'train_encoder.npz':ENC,'splits.npz':SPLIT}
    for seed in SEEDS:
        required[f'kin_FIT_seed{seed}.npy']=kinpath(seed)
        required[f'KIN_seed{seed}.pt']=f'KINEMATIC_DECODER_SPEED_CLAMP_20261018/results/short/KIN_seed{seed}.pt'
    for name,p in required.items():
        if sha256(ROOT/p)!=old['files'][name]['sha256']:raise ValueError('Historical frozen input differs: '+name)
    with np.load(ROOT/SPLIT,allow_pickle=False) as z:sp={k:z[k].copy() for k in ('sample_id','sequence','pedestrian_id','scene_family','loss_change_group','grouped_train_index','grouped_train_ids','inner_cal_index','inner_cal_ids')}
    fit=sp['grouped_train_index'];inner=sp['inner_cal_index']
    assert len(fit)==18909 and len(inner)==1856 and np.array_equal(sp['sample_id'][fit],sp['grouped_train_ids']) and np.array_equal(sp['sample_id'][inner],sp['inner_cal_ids'])
    # Canonical identity delimiter explicitly frozen; no reinterpretation after seeing outcomes.
    tracks=np.array([f'{s}|{int(t)}' for s,t in zip(sp['sequence'],sp['pedestrian_id'])])
    ft=tracks[fit];it=tracks[inner];assert not set(ft)&set(it)
    holdtracks=set()
    for seq in sorted(set(sp['sequence'][fit])):
        ts=sorted(set(ft[sp['sequence'][fit]==seq]),key=lambda t:(keyhash('single-edit-hold-v1|'+t),t))
        holdtracks.update(ts[:math.ceil(.2*len(ts))])
    h=np.flatnonzero(np.isin(ft,list(holdtracks)));pool=np.flatnonzero(~np.isin(ft,list(holdtracks)))
    assert len(pool)>=4096 and len(h)>=1000
    opt=np.array(sorted(pool,key=lambda i:(keyhash('single-edit-train-v1|'+str(int(sp['sample_id'][fit[i]]))),int(sp['sample_id'][fit[i]])))[:4096],np.int64)
    assert len(set(ft[h][sp['loss_change_group'][fit[h]]==2]))>=20
    assert (sp['loss_change_group'][inner]==2).sum()==497 and len(set(it[sp['loss_change_group'][inner]==2]))>=20
    with np.load(ROOT/CACHE,allow_pickle=False) as z:
        ids=z['sample_id'];assert np.array_equal(ids[:25912],sp['sample_id'])
        native=z['candidate'][fit,:,:24].reshape(-1,20,12,2).astype(np.float64)
        groups=z['change_group'][fit];assert np.array_equal(groups,sp['loss_change_group'][fit])
    with np.load(ROOT/ENC,allow_pickle=False) as z:
        assert np.array_equal(z['sample_id'],ids)
        past=z['past'][fit];tokens=z['tokens'][fit]
    assert past.shape==(18909,8,2) and tokens.shape==(18909,8,96) and np.all(past[:,-1]==0)
    # Only FIT_opt GT is retained/used; hold and inner labels are not built or evaluated here.
    with np.load(ROOT/LABEL,allow_pickle=False) as z:
        assert np.array_equal(z['sample_id'],ids);gt=z['future'][fit[opt]].astype(np.float64)
    assert len(np.unique(ids))==len(ids) and np.isfinite(native).all() and np.isfinite(past).all() and np.isfinite(tokens).all() and np.isfinite(gt).all()
    for seed in SEEDS:
        kin=np.load(ROOT/kinpath(seed),mmap_mode='r',allow_pickle=False)
        assert kin.shape==(18909,20,12,2) and np.isfinite(kin).all()
        # Original producer loads grouped_train_index explicitly. Match prior frozen full-array digest too.
        old_array=kin.astype(np.float64)
        hh=hashlib.sha256(np.ascontiguousarray(old_array).tobytes());hh.update(str(old_array.shape).encode());hh.update(str(old_array.dtype).encode())
        assert hh.hexdigest()==old['binding'][f'kin_FIT_seed{seed}']['sha']
    registry=dict(sample_id=ids[fit],track_key=ft,sequence=sp['sequence'][fit],scene=sp['scene_family'][fit],group=groups,opt_row=opt,hold_row=h,pool_row=pool,unused_row=np.setdiff1d(pool,opt),inner_ids=ids[inner],inner_tracks=it)
    audit=dict(status='PASS_EXPLICIT_ID_AND_HISTORICAL_SHA',n_FIT=18909,FIT_opt=4096,FIT_hold=len(h),FIT_opt_pool=len(pool),unused_pool=len(pool)-4096,hold_change_tracks=len(set(ft[h][groups[h]==2])),inner_n=1856,inner_change=497,inner_change_tracks=len(set(it[sp['loss_change_group'][inner]==2])),track_key='sequence|decimal_pedestrian_id',fit_id_sha=array_sha(ids[fit]),opt_id_sha=array_sha(ids[fit[opt]]),opt_tracks=len(set(ft[opt])),opt_change=int((groups[opt]==2).sum()),opt_change_tracks=len(set(ft[opt][groups[opt]==2])),source_row_space='KIN cache row i = original grouped_train_index[i], original loader + current IDs + historical full array SHA; not shape-only',context='train_encoder tokens decoder-prehook target8x96; recent last token and mean8tokens; original native_alignment0 recorded; no new forward',historical_training_lock_claim=False,no_outer_CAL_evaluation=True,inputs={p:{'sha256':sha256(ROOT/p),'size':(ROOT/p).stat().st_size} for p in required.values()})
    if save:
        np.savez_compressed(HERE/'SPLIT_REGISTRY.npz',**registry)
        write_json(HERE/'IDENTITY_AUDIT.json',audit)
        write_json(HERE/'INPUT_MANIFEST.json',dict(**audit,shape_native=list(native.shape),dtype_native=str(native.dtype),context_shape=list(tokens.shape),coordinates='cached focal heading metres, last observed origin, no new transform',historical_manifest=OLD,historical_manifest_SHA=sha256(ROOT/OLD)))
    return dict(native=native[opt],gt=gt,groups=groups[opt],ids=ids[fit[opt]],tracks=ft[opt],opt_row=opt,identity=audit)
def kin_opt(seed,rows):return np.asarray(np.load(ROOT/kinpath(seed),mmap_mode='r',allow_pickle=False)[rows],np.float64)
