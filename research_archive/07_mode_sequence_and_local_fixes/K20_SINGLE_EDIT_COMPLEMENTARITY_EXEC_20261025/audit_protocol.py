"""Only AST-isolated kinematics source verification and synthetic arithmetic; no data/model."""
import ast,json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
DESIGN=ROOT/'docs/K20_SINGLE_EDIT_COMPLEMENTARITY_DESIGN_20261025'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def isolated(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1;scope={'np':np,'DT':.4}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope[name]

def without_origin(p):
    p=np.asarray(p,dtype=float);v=(p[:,:,1:]-p[:,:,:-1])/.4;a=(v[:,:,1:]-v[:,:,:-1])/.4;j=(a[:,:,1:]-a[:,:,:-1])/.4
    return [np.sqrt((x*x).sum(-1)) for x in (v,a,j)]

def stats(arr):
    s,a,j=arr
    return dict(speed_p95=float(np.quantile(s,.95)),accel_mean=float(a.mean()),accel_p95=float(np.quantile(a,.95)),jerk_mean=float(j.mean()),jerk_p95=float(np.quantile(j,.95)))

def main():
    sources=[ROOT/'CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k0/metrics.py',ROOT/'CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k0/config.py',ROOT/'CANDIDATE_SET_OVERNIGHT_REVIEW_20261019/review_saved.py',ROOT/'src/f16/eval.py',*DESIGN.glob('*.md')]
    before={str(p):sha(p) for p in sources}
    old=isolated(sources[0],'motion_scalars');old_stats=isolated(sources[0],'motion_stats')
    # motion_stats's globals explicitly link only the isolated mathematical function.
    old_stats.__globals__['motion_scalars']=old
    reviewed=isolated(sources[2],'motion');current=isolated(sources[3],'motion')
    text=(DESIGN/'02_IMPLEMENTATION_AND_ACCEPTANCE.md').read_text(encoding='utf-8-sig')
    assert '速度为预测相邻点11个差分，加速度10个、jerk9个' in text
    assert '若原定义与此不一致，停止并显式协调协议' in text
    fixtures=[]
    for name,p in [('zero',np.zeros((1,20,12,2))),('constant_future_offset',np.tile(np.array([1.,0.]),(1,20,12,1))),('constant_velocity',np.tile(np.stack((np.arange(1,13,dtype=float),np.zeros(12)),1),(1,20,1,1)))]:
        a=old(p);b=without_origin(p);assert [v.shape[2] for v in a]==[12,11,10] and [v.shape[2] for v in b]==[11,10,9]
        orig=old_stats(p);review=reviewed(p);now=current(p)
        assert all(orig[k]==review[k]==now[k] for k in orig)
        fixtures.append(dict(name=name,original_axes=[12,11,10],new_document_axes=[11,10,9],original=orig,new_document=stats(b),original_review_current_exact=True))
    after={str(p):sha(p) for p in sources};assert before==after
    result=dict(status='BLOCKED_PROTOCOL_MOTION_DEFINITION_BEFORE_E0',human_execution_authorized=True,contract_requires_stop_and_coordination=True,document_axes=[11,10,9],referenced_science_axes=[12,11,10],conflict_changes=['per_window_motion_eligible_mask','positive_action_labels','oracle_selection','pooled_motion_gate','native_feature_motion_channels'],source_SHA_before=before,source_SHA_after=after,synthetic_fixtures=fixtures,old_top_level_executed=False,old_source_modified=False,models_loaded=False,old_outputs_or_outer_CAL_loaded=False,E0_enumeration_started=False,E1_training_started=False,OS_task_or_automation_created=False,scientific_negative_result=False,proposed_resolution='Restore original12/11/10 through explicit prospective coordination amendment; preserve original design and original1.01 thresholds. Await specific direction, do not silently switch.')
    (HERE/'PROTOCOL_MOTION_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
