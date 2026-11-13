"""CPU synthetic numerical checks ONLY. No optimizer creation/steps/real GT."""
import ast, copy, inspect, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from joint_common import HERE,STAGE,read,protocol,authorize,sha
from joint_models import ConditionalVAE,parameter_inventory
from joint_losses import same_slot_accuracy,quality,expected_cardinality,loss
from independent_math import reconstructor

def fixture(dtype=torch.float64):
    # Self-contained observations/targets, not selected from any real results.
    gen=torch.Generator().manual_seed(90210)
    context=torch.randn(2,64,generator=gen,dtype=dtype)*.1
    past=torch.zeros(2,9,8,2,dtype=dtype)
    past[:,:,:,0]=torch.arange(8,dtype=dtype)*.12
    origin=past[:,0,-1].clone();matrix=torch.eye(2,dtype=dtype).expand(2,-1,-1).clone()
    b=dict(sampler_context=context,raw_past=past,origin=origin,matrix=matrix)
    future=origin[:,None]+torch.stack([torch.arange(1,13,dtype=dtype)*.12,torch.zeros(12,dtype=dtype)],-1)[None]
    return b,future

def active_faces(model,b,diagnostic,cal,activation_masks):
    y=diagnostic['paths'].double();past=b['raw_past'][:,0].double()
    obs=(past[:,-1]-past[:,-2])/.4
    v=(y-torch.cat([past[:,-1,None,None].expand(-1,20,1,-1),y[:,:,:-1]],2))/.4
    a=(v-torch.cat([obs[:,None,None].expand(-1,20,1,-1),v[:,:,:-1]],2))/.4
    masks=[v.norm(dim=-1)>cal['vmax'],a.norm(dim=-1)>cal['amax'],
           (v[:,:,0]-obs[:,None]).norm(dim=-1)>cal['joinmax'],
           v.norm(dim=-1)==0,a.norm(dim=-1)==0]
    # Raw Gaussian logvar clamp faces are checked independently of loss autograd.
    c=b['sampler_context'];prior=model.prior(c)[:,model.latent:]
    masks.extend([prior<=-8,prior>=8])
    for m in masks:activation_masks.append(m.detach().cpu().numpy().tobytes())
    return diagnostic['winner'].tolist(),activation_masks

def fixed_loss_with_faces(model,b,f,cal,spec,noise_state):
    masks=[];handles=[]
    for name,m in model.named_modules():
        if isinstance(m,torch.nn.ReLU):
            handles.append(m.register_forward_hook(lambda module,args,out:masks.append((out>0).detach().cpu().numpy().tobytes())))
    # Includes posterior clamp activity; forward sees the actual FIT training-style input only in this synthetic fixture.
    def post(module,args,out):
        lv=out[:,model.latent:]
        masks.extend([(lv<=-8).detach().cpu().numpy().tobytes(),(lv>=8).detach().cpu().numpy().tobytes()])
    handles.append(model.posterior[-1].register_forward_hook(post))
    try:
        torch.set_rng_state(noise_state);value,diagnostic=loss(model,'cvae','accuracy_dpp',b,f,cal,spec)
    finally:
        for h in handles:h.remove()
    return value,active_faces(model,b,diagnostic,cal,masks)

def numerical_tests():
    torch.set_num_threads(1);spec=protocol();cal=read(STAGE/'CALIBRATION.json');b,f=fixture()
    model=ConditionalVAE(42).double()
    with torch.no_grad():
        model.sampler[-1].weight.fill_(.003)
        model.sampler[-1].bias.copy_(torch.linspace(-.02,.02,320,dtype=torch.float64))
    y,a=model.generate(b);y1,_=model.generate(b)
    assert torch.equal(y,y1) and y.shape==(2,20,12,2)
    data=dict(context=np.broadcast_to(b['sampler_context'].numpy()[:,None],(2,9,64)),matrix=b['matrix'].numpy(),origin=b['origin'].numpy())
    reconstructed=reconstructor(model,'cvae')(a['z'].detach().numpy(),data,slice(None))
    decoder_error=float(np.max(abs(reconstructed-y.detach().numpy())))
    assert decoder_error<2e-12
    # Forward API intentionally has no GT; fail if a later edit adds one.
    assert list(inspect.signature(model.generate).parameters)==['bundle']
    changed=f+999.;assert torch.equal(y,model.generate(b)[0])
    local=model.to_local(f,b);world=model.to_world(local[:,None],b)[:,0]
    assert torch.max(abs(world-f))<1e-12
    score=quality(b['raw_past'][:,0],y,cal);ec=expected_cardinality(y,score,cal)
    yn=y.detach().numpy();pn=b['raw_past'][:,0].numpy()
    last=(pn[:,-1]-pn[:,-2])/.4
    vv=np.diff(np.concatenate([np.broadcast_to(pn[:,-1,None,None],(2,20,1,2)),yn],2),axis=2)/.4
    aa=np.diff(np.concatenate([np.broadcast_to(last[:,None,None],(2,20,1,2)),vv],2),axis=2)/.4
    nv=np.linalg.norm(vv,axis=-1);na=np.linalg.norm(aa,axis=-1);nj=np.linalg.norm(vv[:,:,0]-last[:,None],axis=-1)
    qnp=np.exp(-(np.maximum((nv-cal['vmax'])/cal['sigma_v'],0)**2).mean(-1)
                 -(np.maximum((na-cal['amax'])/cal['sigma_a'],0)**2).mean(-1)
                 -np.maximum((nj-cal['joinmax'])/cal['sigma_join'],0)**2)
    S=np.exp(-cal['k']*((yn[:,:,None]-yn[:,None])**2).sum((-1,-2)))
    r=np.sqrt(cal['epsilon_q']+(1-cal['epsilon_q'])*qnp);L=S*r[:,:,None]*r[:,None]
    eig=np.linalg.eigvalsh(L);ecn=(eig/(1+eig)).sum(1)
    qerror=float(np.max(abs(qnp-score.detach().numpy())));ecerror=float(np.max(abs(ecn-ec.detach().numpy())))
    assert max(qerror,ecerror)<2e-12
    # Independent same-slot scalar and analytic path gradient, away from zero.
    paths=torch.randn(2,20,12,2,generator=torch.Generator().manual_seed(19),dtype=torch.float64,requires_grad=True)
    acc,details=same_slot_accuracy(paths,f);dist=np.linalg.norm(paths.detach().numpy()-f[:,None].numpy(),axis=-1)
    w=np.argmin(dist.mean(-1)+.5*dist[:,:,-1],axis=1)
    ref=(dist.mean(-1)+.5*dist[:,:,-1])[np.arange(2),w].mean()
    grad=torch.autograd.grad(acc,paths)[0].numpy();manual=np.zeros_like(grad)
    for i in range(2):
        v=paths.detach().numpy()[i,w[i]]-f.numpy()[i];u=v/np.linalg.norm(v,axis=-1)[:,None]
        manual[i,w[i]]=u/(2*12);manual[i,w[i],-1]+=.5*u[-1]/2
    accuracy_error=float(np.max(abs(grad-manual)));assert abs(float(acc.detach())-ref)<1e-12 and accuracy_error<1e-12
    # Reject using different slots for ADE and FDE; fixed first-index tie.
    tie=torch.ones(1,20,12,2,dtype=torch.float64)
    _,d=same_slot_accuracy(tie,torch.zeros(1,12,2,dtype=torch.float64));assert d['winner'].item()==0
    assert 'min' not in inspect.signature(model.generate).parameters
    # Full CVAE+DSF parameter-chain directional FD, two pre-fixed directions/two eps.
    # Re-use identical reconstruction noise in EACH evaluation. No optimizer/update.
    params=[p for p in model.parameters() if p.requires_grad]
    gen=torch.Generator().manual_seed(771);noise_state=torch.get_rng_state()
    value,base_faces=fixed_loss_with_faces(model,b,f,cal,spec,noise_state)
    derivatives=torch.autograd.grad(value,params);fd=[]
    for direction in range(2):
        vectors=[torch.randn(p.shape,dtype=p.dtype,generator=gen) for p in params]
        norm=torch.sqrt(sum(v.square().sum() for v in vectors));vectors=[v/norm for v in vectors]
        analytic=float(sum((g*v).sum() for g,v in zip(derivatives,vectors)))
        base=[p.detach().clone() for p in params]
        for eps in (1e-4,1e-5):
            outputs=[];faces=[]
            for sign in (1.,-1.):
                with torch.no_grad():
                    for p,z,v in zip(params,base,vectors):p.copy_(z+sign*eps*v)
                val,face=fixed_loss_with_faces(model,b,f,cal,spec,noise_state)
                outputs.append(float(val.detach()));faces.append(face)
            with torch.no_grad():
                for p,z in zip(params,base):p.copy_(z)
            assert faces[0]==faces[1]==base_faces, 'FD_ACTIVE_FACE_SWITCH_DO_NOT_SEARCH_EPS'
            numeric=(outputs[0]-outputs[1])/(2*eps);error=abs(numeric-analytic)
            assert error<=1e-6+1e-3*abs(analytic),(direction,eps,error,analytic,numeric)
            fd.append(dict(direction=direction,eps=eps,abs_error=error,winner_faces=faces[0][0],
                           ReLU_quality_normzero_Gaussian_clamp_faces_EXACT=True))
    # q never enters inference; posterior never enters generate's execution graph.
    torch.set_rng_state(noise_state)
    g=torch.autograd.grad(model.generate(b)[0].sum(),list(model.posterior.parameters()),allow_unused=True)
    assert all(x is None for x in g)
    for action in ('preflight','train','evaluate'):
        try:authorize(action,True)
        except PermissionError:pass
        else:
            # On a later authorized production preflight control may be open;
            # no expensive action is ever dispatched by this numerical helper.
            if read(HERE/'control.json')['status']=='SCRIPTS_ONLY_DO_NOT_START':raise AssertionError('HOLD_NOT_ENFORCED')
    # Import-only regression of canonical evaluator in a fresh namespace; NO actual population evaluated.
    from joint_evaluate import canonical_api
    assert callable(canonical_api().evaluate)
    return dict(status='PASS_CPU_SYNTHETIC_FORWARD_BACKWARD_NO_OPTIMIZER_NO_REAL_DATA',
        numpy_q_error=qerror,numpy_EC_error=ecerror,accuracy_scalar_error=abs(float(acc.detach())-ref),
        analytic_accuracy_path_gradient_error=accuracy_error,full_CVAE_DSF_fixed_direction_FD=fd,
        no_future_prediction_API=True,no_posterior_gradient_in_generation=True,
        independent_CVAE_decoder64_error=decoder_error,canonical_evaluator_standalone_import=True,
        cvae_module_parameters=parameter_inventory(model),GPU_preflight_executed=False,
        optimizer_steps=0,real_training_updates=0)

def main():
    files=list(HERE.rglob('*.py'))
    for p in files:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    before=torch.cuda.is_initialized();result=numerical_tests()
    assert not before and not torch.cuda.is_initialized(),'CPU_CHECK_MUST_NOT_CREATE_CUDA_CONTEXT'
    # Production commands with --execute MUST fail before path/data/model access.
    for file in (HERE/'run_series.py',HERE/'01_existing_joint/run.py',HERE/'02_cvae_joint/run.py'):
        plan=subprocess.run([sys.executable,'-B',str(file),'plan'],capture_output=True,text=True,encoding='utf-8')
        assert plan.returncode==0 and json.loads(plan.stdout)['control']['allow_training'] is False
        denied=subprocess.run([sys.executable,'-B',str(file),'train','--execute','--execution','NEVER_CREATE','--preflight','MISSING'],capture_output=True,text=True,encoding='utf-8')
        assert denied.returncode!=0 and 'SCRIPTS_ONLY_HUMAN_HOLD' in denied.stderr
    pre=subprocess.run([sys.executable,'-B',str(HERE/'preflight_real.py'),'--execute','--output','NEVER_CREATE'],capture_output=True,text=True,encoding='utf-8')
    assert pre.returncode!=0 and 'SCRIPTS_ONLY_HUMAN_HOLD' in pre.stderr
    result.update(python_AST_files=len(files),three_default_entrypoints_plan_only=True,
        all_explicit_train_and_preflight_denied=True,no_OS_task_created=True,no_CUDA_context=True)
    result['source_files']={str(p.resolve()):sha(p) for p in files}
    # Do not overwrite a previous validation receipt after source refinements.
    output=HERE/'CODE_VALIDATION.json'
    version=2
    while output.exists():
        output=HERE/f'CODE_VALIDATION_V{version}.json';version+=1
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
