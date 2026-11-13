"""CPU checkpoint metadata only: no model construction/forward/optimizer step."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import bootstrap
from base import *

def main():
    g=guard('MLP')
    import torch
    records=[]
    for part,arms,updates in [('step1',('I','S'),512),('step2',('M_full',),1024)]:
        for arm in arms:
            for seed in SEEDS:
                f=HERE/f'{part}/{arm}_seed{seed}';c=read_json(f/'manifest.json');completed(f,c)
                state=torch.load(f/'final.pt',map_location='cpu',weights_only=False)
                assert state['contract']==c and state['step']==updates and len(state['history'])==updates
                assert state['history']==read_json(f/'history.json')
                for key in ('model','optimizer','rng','order','cursor'):assert key in state
                for v in state['model'].values():
                    if v.dtype.is_floating_point:assert torch.isfinite(v).all()
                mean,std=stats(42 if part=='step1' else seed)
                assert np.array_equal(state['model']['feature_mean'].numpy(),mean)
                assert np.array_equal(state['model']['feature_std'].numpy(),std)
                records.append(dict(run=str(f.relative_to(HERE)),updates=updates,model_SHA=sha256(f/'final.pt'),embedded_contract_history_stat_finite_PASS=True))
                del state;g.check()
    lock=read_json(HERE/'ALL_PROBE_FINALS_LOCK.json')
    for name,h in lock['models'].items():
        f=HERE/'step2'/name
        p=f/('final.pt' if name.startswith('M_full') else 'final.joblib')
        assert sha256(p)==h
    write_json(HERE/'completion_review/CHECKPOINT_METADATA.json',dict(status='PASS',records=records,source_SHA=sha256(__file__),no_model_forward_optimizer_or_training=True))
    resource_report(HERE/'completion_review/CHECKPOINT_RESOURCE.json',g)
    print('PASS 9 neural checkpoint metadata; 6 tree final hashes; no model forward',flush=True)
if __name__=='__main__':main()
