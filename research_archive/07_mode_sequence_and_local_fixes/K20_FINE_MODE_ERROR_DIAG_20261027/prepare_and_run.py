from common import *
from fine_features import variants,FLOOR
import subprocess,traceback,time
def prepare():
    assert not (HERE/'manifest.json').exists(),'Do not overwrite an existing source lock'
    g=guard(True);oldmanifest=read(OLD/'config/sources_manifest.json')
    for rel,h in oldmanifest['files'].items():assert sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==h,rel
    seal=read(OLD/'completion_review/FINAL_PRESERVATION.json')
    for rel,h in seal['files'].items():assert sha(OLD/rel)==h,rel
    audit=[]
    for split,N,c,ng,cg in [('opt',4096,977,2415,396),('hold',3825,1023,1971,340),('inner',1856,497,932,166)]:
        x=xload(split);y=yload(split)
        assert len(x['sample_id'])==N and len(np.unique(x['sample_id']))==N
        assert np.array_equal(x['sample_id'],y['sample_id']) and np.all(x['past'][:,-1]==0)
        a,f,j,good=quality(x['native'],y['gt']);prior=load(OLD/f'E2_oracle_coverage/{split}/candidate_gt_metrics.npz')
        for key,value in [('ADE',a),('FDE',f),('J',j),('good',good)]:assert np.array_equal(prior[key],value)
        assert good.any(1).sum()==ng and good.any(1)[x['group']==2].sum()==cg and (x['group']==2).sum()==c
        audit.append(dict(split=split,N=N,change=c,native_good=ng,change_good=cg,tracks=len(np.unique(x['track_key'])),source_time_ambiguous=int(x['ambiguous_source_time'].sum()),native_quality_EXACT=True))
    protocol=dict(status='FROZEN_BEFORE_E3_GT_ASSOCIATION',variants=variants(),dt=.4,operational_floor=FLOOR,quality_ADE=.2,quality_FDE=.4,history3_coherence_min=.5,angle_boundary_degrees=1,speed_ratio_boundary=.01,low_speed_boundary_relative=.05,onset_smoothing='two displacements, t1 one displacement',END_bounds=[.2,.4],TIME_rates={'start':.75,'stop':1.25,'step':.025,'count':21,'pairs':441},TIME_definition='piecewise-linear original arc progress, knots at q_original4/8/12; third rate endpoint constrained; finite admissible grid exact, not continuous global optimum',TIME_zero_displacement='strict monotonic infeasible, retain Native',shape_resample_q_count=33,bootstrap=20000,seed=20261027,no_training=True,no_models=True,no_KIN_B1_B2=True,no_new_prediction_paths_saved=True,common_failure_denominator_hold_change=683,sensitivity_GT_blind=True)
    write(HERE/'configs/locked_protocol.json',protocol);write(HERE/'audit/INPUT_AUDIT.json',dict(status='PASS_EXACT_NATIVE_AND_INPUT_ID',populations=audit,prior_E1E2_source_count=len(oldmanifest['files']),prior_artifact_count=len(seal['files']),prior_final_receipt_SHA=sha(OLD/'completion_review/FINAL_PRESERVATION.json'),no_raw_video_timestamp_claim=True))
    resource(HERE/'audit/PREPARE_RESOURCE.json',g)
    files=oldmanifest['files'].copy()
    def add(p):
        try:rel=p.relative_to(ROOT).as_posix()
        except ValueError:rel=p.as_posix()
        v=sha(p)
        if rel in files:assert files[rel]==v,rel
        files[rel]=v
    for rel in seal['files']:add(OLD/rel)
    add(OLD/'completion_review/FINAL_PRESERVATION.json')
    add(Path('<LOCAL_DOWNLOADS>/K20_精细运动模式与轨迹误差归因_离线实验设计_20261027 (1).md'))
    for p in HERE.rglob('*'):
        if p.is_file() and (p.suffix in ('.py','.md') or p.parent.name in ('configs','preflight')):add(p)
    digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    write(HERE/'manifest.json',dict(files=files,digest=digest,inherited_count=len(oldmanifest['files']),inherited_digest=oldmanifest['digest'],at=now(),no_runtime_or_progress_control_in_scientific_source=True))
    verify_sources();write(HERE/'configs/SOURCE_LOCK.json',dict(at=now(),source=digest,protocol_SHA=sha(HERE/'configs/locked_protocol.json')))
def main():
    assert not (HERE/'execution_state.json').exists()
    state=dict(status='RUNNING',started=now(),no_training=True,active_worker=True);write(HERE/'execution_state.json',state)
    try:
        prepare()
        for script in ['e3_gt_free.py','e3_gt_associate.py','e4_error.py','e4_oracle.py','statistics_run.py']:
            verify_sources();mark('START_'+script)
            p=subprocess.run([sys.executable,'-B','-u',str(HERE/script)],cwd=HERE)
            assert p.returncode==0,script
        verify_sources();state.update(status='SUCCESS_REVIEW_REQUIRED',finished=now(),active_worker=False);write(HERE/'execution_state.json',state);write(HERE/'exit_status.json',state)
    except BaseException as e:
        state.update(status='FAILURE',finished=now(),active_worker=False,error=str(e),trace=traceback.format_exc());write(HERE/'execution_state.json',state);write(HERE/'FAILURE.json',state);write(HERE/'exit_status.json',state);raise
if __name__=='__main__':main()
