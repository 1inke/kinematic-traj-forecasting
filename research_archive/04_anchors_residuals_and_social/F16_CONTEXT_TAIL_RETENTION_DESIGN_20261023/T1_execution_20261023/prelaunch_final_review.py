"""Explicit CPU acceptance extension: production manifest expression and invalid CI denominators."""
from core import *

def main():
    digest=verify_sources();stats=module('_tail_final_stat_review',HERE/'statistical_analysis.py')
    tracks=np.array([f'track{i//3}' for i in range(9)]);groups=np.full(9,2);arrays={}
    for arm in (*ARMS,'native'):
        arrays[arm]=[dict(Jmin=np.ones(9),coverage=np.zeros(9,bool),minADE=np.ones(9)*.3,minFDE=np.ones(9)*.5) for _ in SEEDS]
    review=stats.bootstrap(arrays,tracks,groups,draws=20000)
    zero=[r for r in review['results'] if r['metric']=='reference_success_lost_rate']
    assert len(zero)==4 and all(r['point'] is None and all(r[tag]['invalid']==20000 and r[tag]['status']=='INSUFFICIENT' for tag in ('paired_track_seed','fixed_seed_track_only')) for r in zero)
    assert review['results'][0]['paired_track_seed']['status']=='INSUFFICIENT'
    write_json(HERE/'PREFLIGHT_ZERO_DENOMINATOR.json',review)
    tree=ast.parse((HERE/'training.py').read_text(encoding='utf-8'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write_json' and len(n.args)>1 and 'run_manifest.json' in ast.unparse(n.args[0])]
    assert len(calls)==1;expr=ast.Expression(calls[0].args[1]);rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            con=contract(arm,seed);scope=dict(globals(),arm=arm,s=seed,con=con,steps=592)
            actual=eval(compile(expr,str(HERE/'training.py'),'eval'),scope)
            assert actual['contract']==con and actual['epochs']==8 and actual['steps']==592 and actual['teacher_coordinate_loss'] is False
            for key in ('protocol','statistics','initial','orders','native_J','IDs'):assert len(con[key])==64
            rows.append(actual)
    assert verify_sources()==digest
    extension=read_json(HERE/'source_extension_prelaunch.json');assert extension['base_source_digest']==digest
    extension['extension_files'][Path(__file__).relative_to(ROOT).as_posix()]=sha256(__file__)
    for rel,value in extension['extension_files'].items():assert sha256(ROOT/rel)==value
    write_json(HERE/'source_extension_prelaunch.json',extension)
    write_json(HERE/'PRELAUNCH_FINAL_REVIEW.json',dict(status='PASS',base_source_digest=digest,extensions=len(extension['extension_files']),manifest_actual_production_AST_evaluated_six_contracts=rows,zero_denominator_20k_invalid_no_redraw=True,thin_track_INSUFFICIENT=True,all_old_and_locked_sources_unchanged=True,no_model_forward_optimizer=True))
    print('PRELAUNCH_FINAL_PASS',digest,'extensions',len(extension['extension_files']),flush=True)

if __name__=='__main__':main()
