"""Whole KIN20 pool physics diagnostic; not the sparse teacher subset gate."""
from common import *

def main():
    from f16.eval import motion
    data=data_load();pre=read_json(HERE/'PREFLIGHT.json');verified={x['path']:x for x in pre['KIN_complete_cache_row_replay']};rows=[];guard=Guard(read_json(HERE/'protocol.json')['resource']['estimate_bytes']);guard.check('startup')
    for seed in SEEDS:
        for split in ['FIT','inner']:
            guard.check();path=kin_path(split,seed);ix=data['split'][split]
            if sha256(path)!=verified[str(path)]['file_sha256']:raise ValueError('KIN_SOURCE_CHANGED')
            kin=np.load(path)
            for name,mask in [('overall',np.ones(len(ix),bool)),('stable',data['groups'][ix]==0),('transition',data['groups'][ix]==1),('change',data['groups'][ix]==2)]:
                native=motion(data['native'][ix[mask]]);tm=motion(kin[mask]);passed={k:tm[k]<=1.01*native[k] for k in tm};rows.append(dict(seed=seed,split=split,group=name,N=int(mask.sum()),native=native,KIN_all20_motion=tm,per_metric_pass=passed,all5_pass=all(passed.values())))
    verify_sources();write_json(HERE/'TEACHER_POOL_MOTION.json',dict(scope='whole cached KIN20 pool only; quality-selected variable-count teacher subsets and final student sets are DIFFERENT',rows=rows,no_teacher_filtering_or_replacement=True,no_new_training=True))
    print('Whole KIN pool diagnostic:',sum(r['all5_pass'] for r in rows),'/',len(rows),'group/seed/split five-motion passes; not sparse-teacher safety certificate.')

if __name__=='__main__':main()
