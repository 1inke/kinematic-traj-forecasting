"""Explicit per-output teacher ID/row/mask sidecars; no new forward."""
from common import *

def main():
    r=read_json(HERE/'frozen_eval/checkpoint_replay_manifest.json')
    if r['status']!='SUCCESS_FROZEN22':raise RuntimeError('COMPLETE22_REQUIRED')
    source=read_json(HERE/'PREFLIGHT.json');evidence={row['path']:row for row in source['KIN_complete_cache_row_replay']};data=data_load();out=HERE/'frozen_eval/teacher_mapping';out.mkdir(exist_ok=True);rows=[]
    def good(a,gt):
        e=np.linalg.norm(a.astype(float)-gt[:,None],axis=-1);return (e.mean(-1)<=.2)&(e[:,:,-1]<=.4)
    for record in r['records']:
        key=record['job'];path=kin_path(record['split'],record['seed']);pinned=evidence[str(path)];ix=data['split'][record['split']]
        if sha256(path)!=pinned['file_sha256']:raise ValueError('TEACHER_FILE_CHANGED')
        with np.load(HERE/'frozen_eval/predictions'/f'{key}.npz',allow_pickle=False) as z:
            if not np.array_equal(z['sample_id'],data['ids'][ix]) or not np.array_equal(z['teacher_global_registry_row'],ix):raise ValueError('TEACHER_REGISTRY_MAPPING')
            original=np.load(path,mmap_mode='r')
            if not np.array_equal(original,z['teacher_kin']):raise ValueError('SAVED_TEACHER_ROW_NOT_ORIGINAL')
            ng=good(z['native'],z['gt']);kg=good(z['teacher_kin'],z['gt']);covered=ng.any(1);R=ng&covered[:,None];N=kg&~covered[:,None];filename=out/f'{key}.npz'
            np.savez_compressed(filename,sample_id=z['sample_id'],global_registry_row=ix,cache_row=np.arange(len(ix)),R_native_good_mask=R,N_failed_KIN_good_mask=N,B_native_good_mask=ng,B_KIN_good_mask=kg,teacher_file_sha256=np.array(pinned['file_sha256']),teacher_file_relative=np.array(path.relative_to(ROOT).as_posix()))
        rows.append(dict(job=key,path=filename.relative_to(HERE).as_posix(),sha256=sha256(filename),teacher_file=str(path),teacher_file_sha256=pinned['file_sha256'],row_ID_order_sha256=array_sha(data['ids'][ix]),mapping_kind='source full numeric replay plus exact saved cache rows and registered IDs, not historical lock',R_paths=int(R.sum()),N_paths=int(N.sum()),excluded_KIN_on_native_success=int((kg&covered[:,None]).sum()),student_normalization='raw prepared8focal-local positions plus last velocity; no fitted student scaler',current_teacher_scale_fit_IDs_sha256=array_sha(data['ids'][data['split']['FIT']]),historical_teacher_statistics_ID_manifest='UNKNOWN_NOT_RETROACTIVELY_BOUND'))
    verify_sources();write_json(out/'MANIFEST.json',dict(status='PASS_EXPLICIT22_TEACHER_ROW_MASK_CONTRACTS',rows=rows,script_sha256=sha256(__file__),no_model_or_training=True));print('22 explicit teacher cache-row/ID/R-N masks saved, original caches unchanged.')

if __name__=='__main__':main()
