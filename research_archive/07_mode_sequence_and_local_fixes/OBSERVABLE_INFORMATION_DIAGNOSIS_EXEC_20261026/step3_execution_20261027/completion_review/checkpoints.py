"""CPU metadata, paired init/buffers/orders. No model forward/training."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from core import *
from feature_blocks import read_stats
def main():
    g=guard('MLP');import torch
    records=[];lock=read_json(HERE/'ALL_PROBE_FINALS_LOCK.json')
    for seed in SEEDS:
        init=None;order=None
        for arm in ('C_base','C_token'):
            f=HERE/f'runs/{arm}_seed{seed}';c=read_json(f/'manifest.json');completed(f,c)
            assert c['source']==source() and c['stats_SHA']==sha256(HERE/f'prepared/stats_seed{seed}.npz') and c['table_SHA']==sha256(PREVIOUS/f'prepared/actions_seed{seed}.npz')
            initial=torch.load(f/'initial.pt',map_location='cpu',weights_only=False)
            final=torch.load(f/'final.pt',map_location='cpu',weights_only=False)
            assert final['contract']==initial['contract']==c and initial['step']==0 and final['step']==1024
            assert len(final['history'])==1024 and final['history']==read_json(f/'history.json')
            assert all(k in final for k in ('optimizer','rng','order','cursor'))
            assert all(k in final['rng'] for k in ('python','numpy','torch_cpu','torch_cuda','loader'))
            mean,std=read_stats(seed)
            for state in (initial,final):
                assert np.array_equal(state['model']['feature_mean'].numpy(),mean) and np.array_equal(state['model']['feature_std'].numpy(),std)
                assert all(torch.isfinite(v).all() for v in state['model'].values() if v.dtype.is_floating_point)
                mask=state['model']['active_mask'];assert mask.dtype==torch.bool and mask.shape==(1284,)
                expected=torch.ones(1284,dtype=torch.bool);expected[1092:]=False
                if arm=='C_base':expected[324:1092]=False
                assert torch.equal(mask,expected)
            params={k:v for k,v in initial['model'].items() if k.startswith('net.')};assert sum(v.numel() for v in params.values())==42208
            assert torch.count_nonzero(params['net.4.weight'])==0
            paired={k:v for k,v in initial['model'].items() if k!='active_mask'}
            rows=[r['batch_ID_rows'] for r in final['history']]
            if init is None:init=paired;order=rows
            else:
                assert init.keys()==paired.keys() and all(torch.equal(init[k],paired[k]) for k in init)
                assert order==rows
            assert len(rows)==1024 and all(len(r)==32 for r in rows)
            for epoch in range(8):assert sorted(sum(rows[epoch*128:(epoch+1)*128],[]))==list(range(4096))
            assert sha256(f/'final.pt')==lock['models'][f'{arm}_seed{seed}']
            records.append(dict(arm=arm,seed=seed,updates=1024,params=42208,loss_first=final['history'][0]['loss'],loss_last=final['history'][-1]['loss'],model_SHA=sha256(f/'final.pt'),paired_init_buffers_except_mask_EXACT=True,paired_training_rows_EXACT=True))
            del initial,final;g.check()
    write_json(HERE/'completion_review/CHECKPOINT_REVIEW.json',dict(status='PASS',records=records,no_model_forward_optimizer_or_training=True,helper_SHA=sha256(__file__)))
    resource_report(HERE/'completion_review/CHECKPOINT_RESOURCE.json',g);print('PASS six checkpoint contracts, paired initial weights/buffers and all6144 batch histories',flush=True)
if __name__=='__main__':main()
