"""Resume only unsealed CPU labels, preserve every original/partial artifact.

Existing terminal stages are SHA-verified. Partial complete NPZ shards are
copied byte-for-byte into a new namespace, then independently reviewed by the
full completion review. No completed mask or teacher action is recomputed here.
"""
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from io_contract import json_new,json_read,npz_new,load_npz,sha,seal,checked_seal,verify_manifest
from action_geometry import space
from build_teacher import teacher_for_window,write_masks,write_labels
from p0_driver import guard,terminal,resource,emit,now


def resume_labels(observed_path,gt_path,maskroot,partial,out,g):
    receipt=checked_seal(maskroot);x=load_npz(observed_path);y=load_npz(gt_path)
    if not np.array_equal(x['sample_id'],y['sample_id']):raise ValueError('GT ID mismatch')
    gt=y['gt'];out.mkdir(exist_ok=False);files=[];best=[];base=[];raw=[];offset=0;copied=[]
    s=space(receipt['branch'])
    for mf in sorted(maskroot.glob('[0-9]*.npz')):
        g.check();m=load_npz(mf);count=len(m['sample_id'])
        if not np.array_equal(m['sample_id'],x['sample_id'][offset:offset+count]):raise ValueError('mask ID')
        old=partial/mf.name;dest=out/mf.name
        if old.exists():
            z=load_npz(old)
            if not np.array_equal(z['sample_id'],m['sample_id']) or z['target'].shape!=(count,20,2) or z['J_matrix'].shape!=(count,20,len(s.params)):raise ValueError('partial shard contract')
            digest=sha(old)
            with dest.open('xb') as f:f.write(old.read_bytes())
            if sha(dest)!=digest or sha(old)!=digest:raise ValueError('partial copy SHA')
            copied.append(dict(path=str(old),sha=digest))
        else:
            rows=[]
            for local in range(count):
                g.check();i=offset+local
                rows.append(teacher_for_window(x['native'][i],gt[i],s,{k:m[k][local] for k in ('legal','safe')}))
            z={k:np.stack([r[k] for r in rows]) for k in rows[0]}
            z['sample_id']=m['sample_id'];npz_new(dest,**z)
        files.append(dest);best.extend(z['best_J'].min(1));base.extend(z['identity_J'].min(1));raw.extend(z['unfiltered_J'].min(1));offset+=count
        if offset%256==0:emit('P0_LABEL_RESUME',branch=s.branch,windows=offset,reused_shards=len(copied))
    if offset!=len(gt):raise ValueError('incomplete labels')
    b,c,r=(np.asarray(v)<=1 for v in (base,best,raw));change=x['group']==2;rescue=change&~b&c
    if (b&~c).any():raise AssertionError('oracle lost')
    tracks=len(np.unique(x['track_key'][rescue]));threshold=int(np.ceil(.02*change.sum()))
    json_new(out/'opportunity.json',dict(change_N=int(change.sum()),change_native_good=int((change&b).sum()),change_safe_good=int((change&c).sum()),
        change_rescue=int(rescue.sum()),change_unfiltered_good=int((change&r).sum()),rescue_tracks=tracks,required_rescue=threshold,
        opportunity_pass=bool(change.any() and rescue.sum()>=threshold and tracks>=5),note='Only train authorizes training; oracle zero lost is structural.'))
    json_new(out/'PARTIAL_REUSE.json',dict(files=copied,all_shards_require_full_independent_review=True,no_old_output_overwrite=True))
    files += [out/'opportunity.json',out/'PARTIAL_REUSE.json']
    seal(out,files,stage='TEACHERS_SEALED',branch=s.branch,observed_sha=sha(observed_path),gt_sha=sha(gt_path),windows=len(gt),mask_done_sha=sha(maskroot/'done.json'))
    terminal(out,g)


def main():
    begin=time.perf_counter();g=guard();verify_manifest(ROOT/'source_manifest.json')
    ext=ROOT/'p0_resume_source_extension.json'
    verify_manifest(ext)
    registry={b:{p:dict(masks=f'P0_{b}_{p}_masks',labels=f'P0_{b}_{p}_labels') for p in ('train','val')} for b in ('END','TIME')}
    for pop in ('train','val'):
        for kind in ('masks','labels'):checked_seal(ROOT/registry['END'][pop][kind])
    checked_seal(ROOT/'P0_TIME_train_masks')
    dest=ROOT/'P0_TIME_train_labels_resume_20261029'
    resume_labels(ROOT/'prepared/X_train.npz',ROOT/'prepared/GT_train.npz',ROOT/'P0_TIME_train_masks',ROOT/'P0_TIME_train_labels',dest,g)
    registry['TIME']['train']['labels']=dest.name
    x,y=ROOT/'prepared/X_val.npz',ROOT/'prepared/GT_val.npz'
    masks=ROOT/'P0_TIME_val_masks';labels=ROOT/'P0_TIME_val_labels'
    emit('P0_TIME_VAL_MASKS_START');write_masks(x,masks,'TIME',g.check);terminal(masks,g)
    emit('P0_TIME_VAL_LABELS_START');write_labels(x,y,masks,labels,g.check);terminal(labels,g)
    json_new(ROOT/'P0_STAGE_REGISTRY.json',registry)
    result={b:{p:json_read(ROOT/registry[b][p]['labels']/'opportunity.json') for p in ('train','val')} for b in ('END','TIME')}
    json_new(ROOT/'P0_RESULT.json',dict(status='P0_SAVED_OUTPUTS_REQUIRE_COMPLETION_REVIEW',branches=result,held_GT_read=False,no_training=True,resource=resource(g)))
    emit('P0_COMPLETE_SAVED_OUTPUTS',branches=result)
    # Separate CPU process releases all label-generation arrays first.
    code=subprocess.call([sys.executable,'-B',str(ROOT/'p0_review.py')])
    if code:raise RuntimeError('completion review failed, exit '+str(code))
    verify_manifest(ROOT/'source_manifest.json');verify_manifest(ext)
    json_new(ROOT/'P0_WORKER_SUCCESS.json',dict(status='SUCCESS',at=now(),elapsed_seconds=time.perf_counter()-begin,source_unchanged=True,review_sha=sha(ROOT/'p0_completion_review/REVIEW.json')))
    emit('P0_WORKER_SUCCESS',elapsed_seconds=time.perf_counter()-begin)


if __name__=='__main__':
    try:main()
    except Exception:
        json_new(ROOT/'P0_RESUME_FAILURE.json',dict(status='FAILURE',at=now(),traceback=traceback.format_exc()))
        traceback.print_exc();raise
