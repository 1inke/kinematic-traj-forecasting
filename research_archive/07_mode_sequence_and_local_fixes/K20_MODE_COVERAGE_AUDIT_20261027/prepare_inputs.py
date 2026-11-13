from common import *
from morphology import configuration,features
def main():
    g=guard(True);src=read(PARENT/'prepared/ready.json');meta_path=ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'
    meta=load(meta_path,('sample_id','sequence','pedestrian_id','scene_family','start_frame','ambiguous_track_time','loss_change_group'));lookup={int(k):i for i,k in enumerate(meta['sample_id'])};aud=[];required=[PARENT/'source_manifest.json',PARENT/'PRODUCTION_SOURCE_LOCK.json',PARENT/'prepared/ready.json',PARENT/'RESULT_ANALYSIS.json',meta_path,Path('<LOCAL_DOWNLOADS>/new_exp.md')]
    sets=[];tracks=[];all_native=[]
    for split in SPLITS:
        p=PARENT/f'prepared/{split}.npz';assert sha(p)==src['outputs'][split+'.npz'];x=load(p,('native','past','sample_id','track_key','group'));required.append(p)
        rows=np.array([lookup[int(k)] for k in x['sample_id']]);assert len(np.unique(x['sample_id']))==len(rows)
        assert np.array_equal(x['track_key'],np.array([f'{s}|{int(t)}' for s,t in zip(meta['sequence'][rows],meta['pedestrian_id'][rows])])) and np.array_equal(x['group'],meta['loss_change_group'][rows])
        x.update(sequence=meta['sequence'][rows],scene=meta['scene_family'][rows],target=meta['pedestrian_id'][rows],start_frame=meta['start_frame'][rows],ambiguous_source_time=meta['ambiguous_track_time'][rows])
        assert x['native'].shape==(len(rows),20,12,2) and x['past'].shape==(len(rows),8,2) and np.all(x['past'][:,-1]==0)
        assert np.isfinite(x['past']).all() and np.isfinite(x['native']).all();save(HERE/f'inputs/X_{split}.npz',**x);sets.append(set(x['sample_id']));tracks.append(set(x['track_key']))
        f=PARENT/f'populations/seed42/{split}';d=read(f/'done.json');gt=[];ids=[]
        required.extend([f/'done.json',f/'audit/B0_metrics.npz',f/'audit/B0_GROUPS.json',f/'audit/REVIEW.json'])
        for name in d['chunks']:
            q=f/name;assert sha(q)==d['outputs'][name];required.append(q);z=load(q,('gt','sample_id'));gt.append(z['gt']);ids.append(z['sample_id'])
        y=np.concatenate(gt);assert np.array_equal(np.concatenate(ids),x['sample_id']);a,ff,j,good=quality(x['native'],y);b=load(f/'audit/B0_metrics.npz')
        for key,value in (('ADE',a),('FDE',ff),('J',j)):assert np.max(abs(value-b[key]),initial=0)<=2e-12
        assert np.array_equal(good.any(1),b['coverage'])
        for seed in (43,44):
            other=PARENT/f'populations/seed{seed}/{split}/audit/B0_metrics.npz';required.append(other);assert np.array_equal(load(other,('coverage',))['coverage'],good.any(1))
        save(HERE/f'inputs/GT_{split}.npz',gt=y,sample_id=x['sample_id'])
        # Keep geometric definition exactly separate from new morphology.
        save(HERE/f'inputs/GEOM_{split}.npz',sample_id=b['sample_id'],distinct=b['distinct'],effective_good=b['effective_good'])
        aud.append(dict(population=split,n=len(rows),tracks=len(set(x['track_key'])),change=int((x['group']==2).sum()),native_good=int(good.any(1).sum()),native_change_good=int(good.any(1)[x['group']==2].sum()),ambiguous_source_start=int(x['ambiguous_source_time'].sum()),ADE=float(a.min(1).mean()),FDE=float(ff.min(1).mean()),J=float(j.min(1).mean()),reference_EXACT=True))
        all_native.append(x);g.check()
    assert sum(map(len,sets))==len(set.union(*sets))==9777
    for i in range(3):
        for k in range(i):assert not sets[i]&sets[k] and not tracks[i]&tracks[k]
    sp=PARENT/'prepared/split.npz';assert sha(sp)==src['outputs']['split.npz'];s=load(sp);required.append(sp);save(HERE/'inputs/train_val.npz',**s)
    train={k:v[s['train']] for k,v in all_native[0].items()};cfg=configuration(train);cfg.update(train_windows=len(s['train']),val_windows=len(s['val']),train_track_count=len(set(train['track_key'])),primary_population='hold',secondary='inner',primary_categories='coarse20',sample_independence='native once; whole sequence|target track, not slots or copied seeds')
    vf=features(all_native[0]['native'][s['val']],all_native[0]['past'][s['val']],cfg['slow_floor']);assert np.isfinite(vf['phi']).all();cfg['val_sanity_finite_no_retuning']=True
    write(HERE/'config/analysis_config.json',cfg)
    receipt=dict(status='PASS_SOURCE_ID_NATIVE_REPRODUCTION',populations=aud,unique_windows=9777,no_three_seed_replication=True,no_new_model_or_prediction=True,dt=.4,time_contract='inherited regular8+12 grid; raw absolute timestamps not carried in cache; ambiguous source starts retained, not claimed video-timestamp certification',p0_exact_zero=True,all_original_IDs_retained=True,GT_used_only_initial_native_reproduction_not_mode_fitting=True)
    write(HERE/'audit/alignment_checks.json',receipt);write(HERE/'audit/native_reproduction.json',receipt)
    inherited=read(PARENT/'source_manifest.json');files=inherited['files'].copy()
    def add(p):
        try:k=p.relative_to(ROOT).as_posix()
        except ValueError:k=p.as_posix()
        value=sha(p)
        if k in files:assert files[k]==value,k
        files[k]=value
    for p in required:add(p)
    for directory in ('inputs','config'):
        for p in (HERE/directory).glob('*'):
            if p.is_file() and p.name!='sources_manifest.json':add(p)
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):add(p)
    digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    write(HERE/'config/sources_manifest.json',dict(files=files,digest=digest,inherited_count=len(inherited['files']),inherited_digest=inherited['digest']))
    assert verify_sources()==digest;resource(HERE/'audit/PREPARE_RESOURCE.json',g);write(HERE/'config/CONFIG_LOCK.json',dict(config_SHA=sha(HERE/'config/analysis_config.json'),source=digest,at=now(),no_E2_GT_mode_matching_started=True));print('PASS INPUT',aud,'slowfloor',cfg['slow_floor'],flush=True)
if __name__=='__main__':main()
