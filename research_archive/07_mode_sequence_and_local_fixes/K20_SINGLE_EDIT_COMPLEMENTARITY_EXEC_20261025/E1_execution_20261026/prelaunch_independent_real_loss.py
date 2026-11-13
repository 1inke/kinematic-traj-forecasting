"""Explicit prelaunch extension: actual nonzero opt model independent double loss/gradient, no optimizer update."""
import copy,json
import numpy as np
import torch
from env import *
from data_pipeline import OptData
from editor import ActionEditor,loss,numpy_loss_and_score_grad
from source_control import verify
def main():
    source=verify();pre=read_json(HERE/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['source_digest']==source
    guard=Guard(ESTIMATE);guard.check('startup');data=OptData(42)
    versions=[p for p in (HERE/'preflight_versions').iterdir() if (p/'PREFLIGHT.json').exists() and read_json(p/'PREFLIGHT.json')['source_digest']==source]
    if len(versions)!=1:raise ValueError('Unique matching actual preflight evidence required')
    rows=[]
    for arm in ARMS:
        path=versions[0]/f'{arm}_one_epoch_resume.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['contract']['source']==source and len(checkpoint['history'])==1 and checkpoint['history'][0]['windows']==513
        model=ActionEditor(data.mean,data.std).double();model.load_state_dict({k:v.double() if v.is_floating_point() else v for k,v in checkpoint['model'].items()})
        x,v,t=data.batch(np.arange(32),'cpu');x=x.double();scores=model(x);objective=loss(scores,v,t,arm);objective.backward();actual=[p.grad.clone() for p in model.parameters()]
        nv,ng=numpy_loss_and_score_grad(scores.detach().numpy(),v.numpy(),t.numpy(),arm);model.zero_grad(set_to_none=True);model(x).backward(torch.from_numpy(ng));reference=[p.grad.clone() for p in model.parameters()]
        error=max(float(abs(a-b).max()) for a,b in zip(actual,reference))
        if abs(nv-objective.item())>1e-10 or error>1e-10:raise ValueError('Independent actual loss/parameter-gradient failed')
        rows.append(dict(arm=arm,actual_opt_windows=32,actual_nonzero_513pref_checkpoint_SHA=sha256(path),double_scalar_error=abs(nv-objective.item()),all_parameter_gradient_maxerror=error,no_optimizer_update=True))
    assert verify()==source
    extension=dict(base_source_digest=source,extension_files={Path(__file__).relative_to(ROOT).as_posix():sha256(__file__)},purpose='additional actual nonzero E1 loss/gradient chain; scientific source unchanged')
    write_json(HERE/'source_extension_prelaunch.json',extension)
    out=dict(status='PASS',at=e0.now(),base_source_digest=source,source_count=len(read_json(HERE/'source_manifest.json')['files']),source_extension=extension,actual_independent_real_loss=rows,resource=guard.check(),E0_fully_passed=True,full_current_preflight_SHA=sha256(HERE/'PREFLIGHT.json'),no_old_files_deleted=True,no_old_F16_forward=True,no_formal_training_yet=True)
    write_json(HERE/'PRELAUNCH_REVIEW.json',out);print(json.dumps(dict(status='PASS',rows=rows,source_count=out['source_count'])))
if __name__=='__main__':main()
