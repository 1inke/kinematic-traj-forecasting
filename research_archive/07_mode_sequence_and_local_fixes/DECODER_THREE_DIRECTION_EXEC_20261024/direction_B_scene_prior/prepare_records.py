"""Explicit metadata adapter. Requires registered raw WORLD observations.

No raw future values or non-FIT/inner samples are accessed. Past-only rigid
alignment is checked against cached local past. Unknown times stay unknown.
"""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from prior import heading
from f16.resource import Guard

def prepare(raw_path,out,raw_expected_sha,estimate):
    raw_path=Path(raw_path);out=owned_output(out,'direction_B_scene_prior')
    if out.exists(): raise FileExistsError('Never overwrite a record registry')
    guard=Guard(estimate);guard.check('startup')
    if sha256(raw_path)!=raw_expected_sha: raise ValueError('RAW_SHA_MISMATCH')
    data=load_data();raw=np.load(raw_path,mmap_mode='r',allow_pickle=False)
    mdpath=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
    with np.load(mdpath) as z:
        ix={int(v):i for i,v in enumerate(z['sample_id'])}
        metadata={k:z[k].copy() for k in z.files if k!='sample_id'}
    records=[];maxerror=0.
    for split,d in data.items():
        for i,sid in enumerate(d['ids']):
            if i%128==0:guard.check()
            # IDs are original raw dataset row IDs, not current subset offsets.
            world=np.asarray(raw[int(sid),0,:8,-2:],np.float64)
            if world.shape!=(8,2) or not np.isfinite(world).all(): raise ValueError('RAW_PAST_SHAPE')
            centered=world-world[-1];local=d['past'][i].astype(float)
            u,_,vh=np.linalg.svd(centered.T@local)
            # Proper-rotation Kabsch: rank-one straight paths otherwise admit
            # an arbitrary reflected SVD completion despite a valid rotation.
            rot=u@np.diag([1.,np.linalg.det(u@vh)])@vh
            err=float(np.max(np.abs(centered@rot-local)))
            maxerror=max(maxerror,err)
            if err>2e-5 or np.linalg.det(rot)<0: raise ValueError('WORLD_LOCAL_PAST_ALIGNMENT_UNRESOLVED')
            j=ix[int(sid)];start=int(metadata['start_frame'][j])
            verified=not bool(metadata['ambiguous_track_time'][j]) and start>=0
            records.append(dict(sample_id=int(sid),split=split,track=str(d['track'][i]),
                sequence=str(metadata['sequence'][j]),scene=str(metadata['scene_family'][j]),
                t0=start+7 if verified else -1,end_index=start+19 if verified else -1,
                verified_time=verified,cell=np.floor(world[-1]/.5).astype(int).tolist(),
                heading=heading(world),endpoint=d['gt'][i,-1].tolist()))
    guard.check()
    write_json(out,dict(records=records,provenance=dict(raw=str(raw_path.resolve()),raw_sha256=raw_expected_sha,
        metadata_sha256=sha256(mdpath),encoder_sha256=sha256(ENCODER),past_alignment_max=maxerror,
        adapter_source_SHA=sha256(__file__),resource_estimate_bytes=estimate,resource_observations=guard.observations,
        source_time_unit='verified metadata SOURCE indices, not claimed video frames',
        no_raw_future_read=True,orientation='existing focal-heading cached local GT, verified using observed past only')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw-world-npy',type=Path,required=True)
    ap.add_argument('--raw-sha256',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--estimated-peak-gib',type=float,required=True)
    a=ap.parse_args();prepare(a.raw_world_npy,a.output,a.raw_sha256,int(a.estimated_peak_gib*2**30))
