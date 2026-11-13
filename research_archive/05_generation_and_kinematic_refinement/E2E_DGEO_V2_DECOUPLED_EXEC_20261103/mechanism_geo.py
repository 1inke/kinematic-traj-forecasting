"""Bounded GT-free fixed128 mechanism diagnostics; saved quality is audited separately."""
from geo_common import *
from train_core_geo import torch,np,bf,cpu_tree,exact
from preflight import hash_indices


def calibrate():
    write_json(RUN/'MECHANISM_CONTRACT.json',dict(status='SEALED_BEFORE_SCIENTIFIC_PILOTS',
        dev_hash='dgeo-mechanism-v1',windows=128,seed=42,
        initial='Same exact fresh isolated construction, fixed hash128 dev inputs only; future not passed',
        diagnostics=['qk_offdiag_cos','negative_attention_fraction','gate_mean','target_update_norm','attention_fullweights'],
        unordered_model_indices_not_intents=True,signed_attention_not_probability=True,
        common_covered='Pairwise intersection method and control, and all four model intersections plus B0, same physical window IDs',
        unconditional_tail='AllK20 speed/acc/jerk/junction float64 original dt.4, p95/p99; descriptive only, not a replacement for conditional excess gate',
        original_gate='Exact prior compare_ud.pilot_screen including NOT_ESTIMABLE zeros; new all4 ADE protect and estimable-only tail reported separately not overwrite',
        no_subjective_gate_or_auto_multiseed=True))


@torch.no_grad()
def collect_mechanism(model,arm,fit,dev,guard,folder=None):
    before=cpu_tree(model.state_dict());model.eval();rule=read_json(RUN/'MECHANISM_CONTRACT.json')
    idx=hash_indices(dev['ids'],rule['dev_hash'],128)
    weights=[];updates=[];cos=[];gates=[]
    for start in range(0,len(idx),64):
        ii=idx[start:start+64]
        bf.prediction(model,torch.tensor(dev['past'][ii],device='cuda'),torch.tensor(dev['num_valid'][ii],device='cuda'))
        d=model.decoder.coordinator.diagnostics
        weights.append(d['attention_weights'].cpu().numpy());updates.append(d['update_norm'].cpu().numpy())
        cos.append(float(d['off_diag_cos_mean']));gates.append(float(d['gate_mean']));guard.check()
    weights=np.concatenate(weights);updates=np.concatenate(updates)
    out=Path(folder) if folder is not None else RUN/'mechanisms'/arm;out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'attention_fixed128.npz',ids=dev['ids'][idx],weights=weights,update_norm=updates)
    exact(before,cpu_tree(model.state_dict()),'MECHANISM_PARAMETERS_BUFFERS')
    write_json(out/'DIAGNOSIS.json',dict(arm=arm,n=128,off_diag_cos_mean=float(np.mean(cos)),
        negative_fraction=float((weights<0).mean()),positive_fraction=float((weights>0).mean()),
        row_sum_min=float(weights.sum(-1).min()),row_sum_max=float(weights.sum(-1).max()),
        gate_mean=float(np.mean(gates)),update_norm_mean=float(updates.mean()),
        no_model_weight_buffer_mutation=True,GT_forward=False,not_causal_proof=True))
    write_json(out/'done.json',dict(status='SUCCESS',files=seal(out.iterdir())))

