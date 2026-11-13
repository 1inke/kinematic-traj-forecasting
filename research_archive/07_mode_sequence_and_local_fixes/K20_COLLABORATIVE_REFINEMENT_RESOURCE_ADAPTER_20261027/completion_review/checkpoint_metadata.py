"""CPU tensor metadata only; does not import a model or run forward/backward."""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from core import *
import torch
torch.set_num_threads(1)

def main():
    g=guard(True);stats=load(BASE/'prepared/statistics.npz');buffer_keys=('mean','std','motion_scales','basis','basis_l1_bound');reports=[]
    for seed in SEEDS:
        initial=[]
        for arm in ARMS:
            f=BASE/f'runs/{arm}_seed{seed}';ct=contract(seed,arm);d=completed(f,ct);z=torch.load(f/'checkpoint.pt',map_location='cpu',weights_only=False);assert z['contract']==ct and z['state']['updates']==776 and z['state']['epoch']==8
            initial.append(torch.load(f/'initial.pt',map_location='cpu',weights_only=True));state=z['model'];assert all(torch.isfinite(v).all() for v in state.values());assert sum(v.numel() for k,v in state.items() if k not in buffer_keys)==17223
            for k in buffer_keys:assert torch.equal(state[k],initial[-1][k]) and np.array_equal(state[k].numpy(),stats[k])
            assert len(z['optimizer']['state'])==len([k for k in state if k not in buffer_keys]);assert z['state']['history']==read_json(f/'history.json')
            assert z['rng'];reports.append(dict(seed=seed,arm=arm,checkpoint_SHA=sha256(f/'checkpoint.pt'),all_buffers_to_initial_and_production_stats_EXACT=True,all_state_finite=True,parameters=17223,epochs=8,updates=776,optimizer_entries=len(z['optimizer']['state']),RNG_history_present_not_replayed=True));g.check()
        assert initial[0].keys()==initial[1].keys() and all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
    resource(BASE/'completion_review/CHECKPOINT_RESOURCE.json',g);write_json(BASE/'completion_review/CHECKPOINT_METADATA.json',dict(status='PASS_6_CPU_CHECKPOINTS',helper_SHA=sha256(Path(__file__)),records=reports,paired_initial_all_weights_buffers_EXACT=True,no_model_instantiation_forward_backward_optimizer_update=True,no_GPU_tensor_load=True,not_optimizer_RNG_replay=True))
    print('PASS CPU CHECKPOINT METADATA',flush=True)
if __name__=='__main__':main()
