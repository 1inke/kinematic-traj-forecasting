"""Actual full-model connectivity evidence before deciding protocol amendments.

Only 3 diagnostic updates per proposed arm, no formal/pilot8epoch model. The
diagnostic uses actual B0 width64 and clip5, explicitly NOT the conflicting
width96/clip1 prose; hence it is not an execution receipt for the supplied gate.
"""
import hashlib
import importlib.util
import time
from ud_common import *


def main():
    started=time.perf_counter()
    out=RUN/'alignment_check'
    out.mkdir(exist_ok=False)
    source=seal([DESIGN,*RUN.glob('*.py'),BASE/'source_manifest.json',BASE/'protocol.json',
                 BASE/'B0_seed42/initial.pt',BASE/'B0_seed42/final.pt',BASE/'C0/fit.npz'])
    verify(read_json(BASE/'source_manifest.json')['files'])
    manifest=read_json(BASE/'B0_FINAL_LOCK.json');verify(manifest['files'])
    spec=importlib.util.spec_from_file_location('ud_resource',ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
    rg=importlib.util.module_from_spec(spec);spec.loader.exec_module(rg)
    estimate=read_json(BASE/'MEMORY_ESTIMATE.json')['GPU_E']+128*1024**2
    guard=rg.Guard(estimate);guard.check(mode='startup',force=True)
    import numpy as np
    import torch
    import baseline_factory as bf
    from train_core import cpu_tree,exact
    from decoder_adapter import UDDecoder
    from motion_state_update import compute_past_kinematic_state_6d
    torch.set_num_threads(1)
    data=dict(np.load(BASE/'C0/fit.npz'))
    order=sorted(range(len(data['ids'])),key=lambda i:hashlib.sha256(('ud-connectivity-v1|'+str(data['ids'][i])).encode()).digest())[:8]
    selected=np.array(order)
    p=torch.tensor(data['past'][selected],device='cuda');y=torch.tensor(data['future'][selected],device='cuda')
    n=torch.tensor(data['num_valid'][selected],device='cuda')
    recorded=torch.load(BASE/'B0_seed42/initial.pt',map_location='cpu',weights_only=False)
    rows={}
    for arm in ('U-GRU','U-Delta','D-Gate','D-DiffGate'):
        model,cfg=bf.make_model()
        original=cpu_tree(model.state_dict())
        exact(original,recorded['model'],'FRESH_B0_INITIAL_PARAMS_BUFFERS')
        model.eval()
        with torch.no_grad():
            reference,logits,_,_=bf.prediction(model,p,n,y)
            reference=reference.clone();logits=logits.clone()
        # Separate constructor RNG does not change original initialization or global training RNG.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(20261102)
            decoder=UDDecoder(model.decoder,arm).to('cuda')
        model.decoder=decoder
        new_parameters={k:v for k,v in model.state_dict().items() if k not in original}
        for name,value in original.items():exact(value,model.state_dict()[name],name)
        model.eval()
        with torch.no_grad():
            candidate,score,_,_=bf.prediction(model,p,n,y)
        exact(reference,candidate,'ZERO_RESIDUAL_ORIGINAL_INITIAL_PATHS')
        exact(logits,score,'ZERO_RESIDUAL_ORIGINAL_INITIAL_SCORES')
        criterion=bf.ModeQueryLoss(cfg).cuda()
        optimizer,scheduler=bf.optimizer_scheduler(model,340,8)
        branch=model.decoder.update if hasattr(model.decoder,'update') else model.decoder.interaction
        steps=[]
        for step in (1,2,3):
            model.train()
            prediction,score,target,_=bf.prediction(model,p,n,y,augment=True)
            loss,_=bf.loss_value(prediction,target,score,criterion)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grads={name:float(param.grad.norm()) if param.grad is not None else None for name,param in branch.named_parameters()}
            if any(value is not None and not np.isfinite(value) for value in grads.values()):raise ValueError('NONFINITE_DIAGNOSTIC_GRAD')
            steps.append({'step':step,'loss':float(loss.detach()),'gradient_norms':grads,
                          'out_gradient_positive':grads['out.weight'] is not None and grads['out.weight']>0,
                          'all_internal_gradients_positive':all(v is not None and v>0 for k,v in grads.items() if k!='out.weight')})
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],5.)
            optimizer.step();scheduler.step();guard.check()
        widths={'context_input':model.decoder.context[1].in_features,
                'actual_context_output':model.decoder.context[-1].out_features,
                'mode_feature':model.decoder.mode_embedding.shape[-1]}
        params=sum(v.numel() for v in model.parameters())
        rows[arm]={'actual_dimensions':widths,'original_state_exact':True,
            'initial_paths_and_logits_exact':True,'total_parameters':params,
            'added_parameters':params-sum(v.numel() for v in original.values())+sum(v.numel() for v in recorded['model'].values() if False),
            'new_trainable_parameters':sum(v.numel() for k,v in model.named_parameters() if k not in original),
            'fits305000':params<=305000,'steps':steps,
            'literal_supplied_two_step_gate':steps[0]['out_gradient_positive'] and steps[1]['all_internal_gradients_positive'],
            'diagnostic_only_updates':3}
        # Count registered parameters directly, excluding buffers (not state_dict elements).
        rows[arm]['added_parameters']=sum(v.numel() for k,v in model.named_parameters() if k not in original)
        del model,decoder,branch,optimizer,scheduler
    verify(source);verify(read_json(BASE/'source_manifest.json')['files'])
    write_json(out/'ACTUAL_GRADIENT_CHECK.json',{'status':'CONTRACT_GATE_NOT_MET_NO_PRODUCTION',
        'at':now(),'rows':rows,'fixture_ids':data['ids'][selected].tolist(),
        'diagnostic_updates':12,'formal_models':0,'pilot_epochs':0,
        'GT_used_only_as_loss':True,'actual_B0_clip':5,'document_clip':1,
        'resource_estimate':estimate,'resource':guard.observations,'elapsed_seconds':time.perf_counter()-started,
        'source_before_after_unchanged':source,'old_64_sources_unchanged':True})
    write_json(RUN/'runtime_control.json',{'status':'STOP_CONTRACT_ALIGNMENT_WAIT_HUMAN',
        'at':now(),'actual_diagnostic_updates':12,'formal_models':0,'OS_worker_started':False,
        'reason':'Width96 vs actual64; clip1 vs B0clip5; literal Step1Wo>0/Step2allinternal>0 contradicted by exact zero original output heads; no science gate amended silently.',
        'no_old_process_queries':True,'no_delete':True})
    for arm,row in rows.items():
        print(arm,'params',row['total_parameters'],'added',row['added_parameters'],
              'Step1_Wo',row['steps'][0]['gradient_norms']['out.weight'],
              'Step2_allinternal',row['steps'][1]['all_internal_gradients_positive'],
              'Step3_allinternal',row['steps'][2]['all_internal_gradients_positive'],flush=True)


if __name__=='__main__':main()
