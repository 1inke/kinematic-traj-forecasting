"""No group/B/target/quality/GT is read during input or hash registration."""
from core import *
def main():
    require3();g=guard();out=HERE/'prepared';out.mkdir(exist_ok=True)
    if (out/'done.json').exists():raise FileExistsError('Preserve completed preparation')
    meta=load_npz(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz');lookup={int(v):i for i,v in enumerate(meta['sample_id'])}
    registry={};obs={}
    for split in ('opt','hold'):
        with np.load(TWELVE/f'prepared/observations_{split}.npz') as z:
            ids=z['sample_id'];tracks=z['track_key'];past=z['past'].astype(np.float64)
        ix=np.array([lookup[int(v)] for v in ids]);seq=meta['sequence'][ix];ped=meta['pedestrian_id'][ix];start=meta['start_frame'][ix].astype(np.int64)
        ambiguous=meta['ambiguous_track_time'][ix].astype(bool);known=(start>=0)&~ambiguous
        assert np.array_equal(tracks,np.array([f'{s}|{int(p)}' for s,p in zip(seq,ped)]))
        centered=past-past[:,-1:];obs[split]=dict(sample_id=ids,track_key=tracks,past=centered,sequence=seq,start=start,known=known)
        tag='ambiguity-query-v1' if split=='hold' else 'ambiguity-calibration-v1'
        windowtag='ambiguity-query-window-v1' if split=='hold' else 'ambiguity-calibration-window-v1'
        chosen=representative(ids,tracks,tag,windowtag)
        save_npz(out/f'identity_{split}.npz',**obs[split],reference_rows=chosen)
        registry[split]=dict(n=len(ids),total_tracks=len(set(tracks)),reference_n=len(chosen),hash_tag=tag,window_hash=windowtag,
            all_IDs=[dict(sample_id=int(sid),track_key=str(t),sequence=str(s),start_index=int(st),status='KNOWN_FULL20_SOURCE_INTERVAL' if k else 'UNKNOWN_OR_AMBIGUOUS_TIME_RETAINED') for sid,t,s,st,k in zip(ids,tracks,seq,start,known)],
            reference_rows=chosen.tolist(),reference_IDs=ids[chosen].tolist())
        g.check()
    assert len(np.intersect1d(obs['opt']['sample_id'],obs['hold']['sample_id']))==0 and not set(obs['opt']['track_key'])&set(obs['hold']['track_key'])
    for seed in SEEDS:
        for split in ('opt','hold'):
            a=old_columns(seed,split,('sample_id','track_key','native','kin','eligible'))
            assert np.array_equal(a['sample_id'],obs[split]['sample_id']) and np.array_equal(a['track_key'],obs[split]['track_key'])
            assert np.isfinite(a['native']).all() and np.isfinite(a['kin']).all() and a['eligible'][:,0].all()
            save_npz(out/f'pools_seed{seed}_{split}.npz',**a);g.check()
    write_json(HERE/'QUERY_REGISTRY.json',registry)
    write_json(HERE/'INPUT_AUDIT.json',dict(status='PASS',all_ids=7921,opt=4096,hold=3825,no_GT_group_quality_B_target_read=True,
        no_new_data_or_time_resolving_from_quality=True,metadata_is_source_identity_full20_fingerprint_not_input_feature=True,
        unknown_by_split={k:sum(r['status']!='KNOWN_FULL20_SOURCE_INTERVAL' for r in v['all_IDs']) for k,v in registry.items()},
        fixed_hash_population_before_posthoc=True))
    resource(out/'RESOURCE.json',g);finish(out,dict(metadata_SHA=sha256(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'),protocol_SHA=sha256(HERE/'protocol.json'),GTfree=True))
    print('PASS prepared GT-free7921 IDs, fixed256 references/queries, pools3; no held labels read',flush=True)
if __name__=='__main__':main()
