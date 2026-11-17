from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
from torch.nn import functional as F
from probe_io import ROOT,W,OLD,arrays,read,write,save_torch,seal,verify,sha,now
from probe_model import make_basis,score_head

def recipe(cfg):
    keys=('WINNER_FDE_WEIGHT','CLASSIFICATION_WEIGHT','ENDPOINT_LOSS_WEIGHT','SOFT_CLASSIFICATION','SOFT_TEMPERATURE','HARD_CLASSIFICATION_WEIGHT','DIVERSITY_WEIGHT','DIVERSITY_MARGIN','SHORT_HORIZON_WEIGHT','SHORT_HORIZON_STEPS')
    return dict(decoder={k:getattr(cfg.MODEL.DECODER,k) for k in keys},loss={k:getattr(cfg.LOSS,k) for k in ('START_WEIGHT','END_WEIGHT','WEIGHTED_PENALTY')},pred_time=cfg.PREDICTION.PRED_TIME,frequency=cfg.DATA.FREQUENCY)
def criterion_from(recipe,device,buffers):
    sys.path.insert(0,str(W/'FRONT_ENCODER_SERIES'))
    from locked_mode_query_loss import ModeQueryLoss
    cfg=SimpleNamespace(MODEL=SimpleNamespace(DECODER=SimpleNamespace(**recipe['decoder'])),LOSS=SimpleNamespace(**recipe['loss']),PREDICTION=SimpleNamespace(PRED_TIME=recipe['pred_time']),DATA=SimpleNamespace(FREQUENCY=recipe['frequency']))
    criterion=ModeQueryLoss(cfg).to(device);criterion.load_state_dict(buffers,strict=True);return criterion
def create_cache(device):
    cache_dir=ROOT/'cache';cache_dir.mkdir(exist_ok=True)
    if (cache_dir/'DONE.json').exists():verify(read(cache_dir/'DONE.json')['files']);return
    sys.path.insert(0,str(OLD));import baseline_factory as bf
    model,cfg=bf.make_model(device,42);ckpt=torch.load(OLD/'runs/E1C_E64_s42/final.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['ema'],strict=True);model.eval();model.requires_grad_(False)
    state_before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assets=dict(score_head={k:v.detach().cpu().clone() for k,v in model.decoder.score_head.state_dict().items()},criterion_recipe=recipe(cfg),loss_buffers=ckpt['loss_buffers'])
    save_torch(cache_dir/'frozen_assets.pt',assets)
    helper_path=Path('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116/code')
    sys.path.insert(0,str(helper_path));from eth_speed_math import maneuver_labels
    capture={}
    def mode_hook(m,i,o):capture['mode']=o[:,0].detach()
    def scale_hook(m,i,o):capture['scale']=F.softplus(o[:,0]).detach()+.25
    def delta_hook(m,i,o):capture['delta']=(.75*torch.tanh(o[:,0])).reshape(len(o),20,3,2).detach()
    def before_decoder(m,i):capture['offset']=i[1][:,0,-1].detach()
    hooks=[model.decoder.mode_mlp.register_forward_hook(mode_hook),model.decoder.proposal_scale.register_forward_hook(scale_hook),model.decoder.proposal_delta.register_forward_hook(delta_hook),model.decoder.register_forward_pre_hook(before_decoder)]
    checks=[]
    for split in ('fit','dev'):
        write(ROOT/'CURRENT.json',dict(at=now(),phase='BUILD_CACHE',split=split))
        data=arrays(OLD/f'C0/{split}.npz');saved=arrays(OLD/f'runs/E1C_E64_s42/{split}/predictions.npz')
        assert len(np.unique(saved['ids']))==len(saved['ids']) and set(saved['ids'].tolist())==set(data['ids'].tolist())
        lookup={key:i for i,key in enumerate(saved['ids'].tolist())};order=np.array([lookup[key] for key in data['ids'].tolist()]);saved_paths=saved['paths'][order];saved_logits=saved['logits'][order]
        for key in ('group','groups'):
            if key in saved:np.testing.assert_array_equal(saved[key][order],data['group'])
        if 'future' in saved:np.testing.assert_array_equal(saved['future'][order],data['future'])
        values={k:[] for k in ('mode','relative','scale','logits','y')};max_path=max_logit=0.
        for start in range(0,len(data['ids']),64):
            idx=slice(start,start+64);p=torch.as_tensor(data['past'][idx],device=device);num=torch.as_tensor(data['num_valid'][idx],device=device)
            with torch.no_grad():
                prediction,logits,_,transform=bf.prediction(model,p,num,future=None,augment=False)
                rel=capture['scale'][:,None,None]*(model.decoder.anchor_templates[None]+capture['delta'])
                reconstructed=torch.einsum('tp,nkpc->nktc',model.decoder.interpolation,rel)
                torch.testing.assert_close(reconstructed,prediction[:,0]-capture['offset'][:,None,None],atol=2e-5,rtol=2e-5)
                _,target,_,_,_,_=bf.prepare(p,torch.as_tensor(data['future'][idx],device=device),augment=False)
                y=target[:,0]-capture['offset'][:,None]
                world=bf.world_paths(prediction,transform).cpu().numpy()
                max_path=max(max_path,float(np.max(np.abs(world-saved_paths[idx]))));max_logit=max(max_logit,float(np.max(np.abs(logits[:,0].cpu().numpy()-saved_logits[idx]))))
                assert max_path<=1e-4 and max_logit<=1e-4,(split,max_path,max_logit)
                for k,v in [('mode',capture['mode']),('relative',rel),('scale',capture['scale']),('logits',logits[:,0]),('y',y)]:values[k].append(v.detach().cpu())
            if start%4096==0:print(f'CACHE {split} {start}/{len(data["ids"])}',flush=True)
        tensors={k:torch.cat(v) for k,v in values.items()}
        disp=data['past'][:,0,-1].astype(float)-data['past'][:,0,-4];heading_valid=np.linalg.norm(disp,axis=-1)>.03
        meta=dict(ids=data['ids'],group=data['group'],labels=maneuver_labels(data['past'],data['future']),heading_valid=heading_valid)
        save_torch(cache_dir/f'{split}.pt',dict(tensors=tensors,meta=meta))
        checks.append(dict(split=split,n=len(data['ids']),max_abs_world_path_difference=max_path,max_abs_logit_difference=max_logit,saved_prediction_alignment='PASS'))
    for h in hooks:h.remove()
    for k,v in model.state_dict().items():assert torch.equal(v.detach().cpu(),state_before[k]),k
    write(cache_dir/'VALIDATION.json',dict(status='PASS',at=now(),checks=checks,base_weights_unchanged=True,base_checkpoint_sha256=sha(OLD/'runs/E1C_E64_s42/final.pt'),encoder_frozen=True,no_cache_augmentation=True,future_not_passed_to_encoder=True))
    write(cache_dir/'DONE.json',dict(at=now(),files=seal([cache_dir/'fit.pt',cache_dir/'dev.pt',cache_dir/'frozen_assets.pt',cache_dir/'VALIDATION.json'])))
    del model,ckpt
    if device.startswith('cuda'):torch.cuda.empty_cache()

def load_cache(device):
    verify(read(ROOT/'cache/DONE.json')['files']);assets=torch.load(ROOT/'cache/frozen_assets.pt',map_location='cpu',weights_only=False)
    head=score_head().to(device);head.load_state_dict(assets['score_head']);head.eval();head.requires_grad_(False)
    criterion=criterion_from(assets['criterion_recipe'],device,assets['loss_buffers'])
    result={}
    for split in ('fit','dev'):
        d=torch.load(ROOT/f'cache/{split}.pt',map_location='cpu',weights_only=False)
        result[split]=dict(tensors={k:v.to(device) for k,v in d['tensors'].items()},meta=d['meta'])
    return result,head,criterion
