"""Read-only verification of the sealed outputs, with a new standalone receipt."""
from pathlib import Path
import hashlib
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from dec_io import now, arrays, read, write, sha, verify


def close(x, y, tag=''):
    np.testing.assert_allclose(x, y, rtol=2e-12, atol=2e-12, err_msg=tag)


def main():
    start = time.monotonic()
    done = read(ROOT / 'DONE.json')
    result = read(ROOT / 'RESULT.json')
    worker = read(ROOT / 'standalone/WORKER_EXIT.json')
    assert worker['status'] == 'SUCCESS' and worker['exit_code'] == 0
    assert sha(ROOT / 'SOURCE_MANIFEST.json') == result['source_manifest_sha256']
    contracts = [ROOT / k for k in ('SOURCE_MANIFEST.json','DONE.json','cache/DONE.json')]
    checkpoints = sorted((ROOT / 'runs').glob('*/checkpoints/epoch*.json'))
    runs_done = sorted((ROOT / 'runs').glob('*/DONE.json'))
    assert len(checkpoints) == 360 and len(runs_done) == done['completed_runs'] == result['completed_runs'] == 18
    sealed = {}
    for p in contracts + checkpoints + runs_done:
        for key, digest in read(p)['files'].items():
            assert key not in sealed or sealed[key] == digest
            sealed[key] = digest
    verify(sealed)
    torch.set_num_threads(2)
    dev = torch.load(ROOT / 'cache/dev.pt', map_location='cpu', weights_only=False)
    y = dev['tensors']['y'].numpy().astype(np.float64)
    meta = dev['meta']
    d0 = arrays(ROOT / 'D0_dev_windows.npz')
    for key in ('ids','group','labels'):
        np.testing.assert_array_equal(d0[key],meta[key])
    social = d0['social']
    groups = dict(all=np.ones(len(y), bool), stable=meta['group']==0, transition=meta['group']==1,
                  change=meta['group']==2, decel_stop=np.isin(meta['labels'],['deceleration','stop']),
                  social=social, non_social=~social)
    sh = read(ROOT / 'NEIGHBOR_SHUFFLE.json')
    donor, count = np.array(sh['donor_row']), dev['tensors']['nbr_valid'].sum(-1).numpy()
    np.testing.assert_array_equal(count, d0['neighbor_count'])
    np.testing.assert_array_equal(np.sort(donor), np.arange(len(y)))
    np.testing.assert_array_equal(count, count[donor])
    assert np.all(donor[count>0] != np.flatnonzero(count>0))
    for name, mask in groups.items():
        a = result['D0']['dev'][name]
        s = mask & social
        assert a['n'] == mask.sum() and a['social_n'] == s.sum()
        close(a['social_rate'], s.sum()/mask.sum())
        close(a['c3_neighbors_per_candidate'],d0['c3_count'][mask].mean())
        close(a['c4_neighbors_per_candidate'],d0['c4_count'][mask].mean())
        close(a['same_count_fraction'],(d0['c3_count'][mask]==d0['c4_count'][mask,None]).mean())
        if s.any():
            close(a['candidate_sets_vary_among_social'],d0['candidate_sets_vary'][s].mean())
    def win(folder, expected, suffix=''):
        p = arrays(folder / f'predictions{suffix}_local.npz')
        w = arrays(folder / f'window_metrics{suffix}.npz')
        np.testing.assert_array_equal(p['ids'],meta['ids'])
        for k in ('ids','group','labels'):
            np.testing.assert_array_equal(w[k],meta[k])
        np.testing.assert_array_equal(w['social'],social)
        delta = p['paths'].astype(np.float64) - y[:,None]
        distance = np.hypot(delta[...,0],delta[...,1])
        ade = distance.sum(-1)/12
        end = distance[:,:,-1]
        shift = p['shift'].astype(np.float64)
        values = dict(minADE=ade.min(-1), minFDE=end.min(-1),
                      top1FDE=end[np.arange(len(y)),p['logits'].argmax(-1)],
                      uncovered=end.min(-1)>.4,mean_shift=np.hypot(shift[...,0],shift[...,1]).mean((1,2)))
        for k, v in values.items():
            close(v,w[k],folder.name+suffix+'/'+k)
        for g, mask in groups.items():
            b = expected[g]
            assert b['n']==mask.sum()
            for k, v in values.items():
                close(v[mask].mean(), b[k],g+'/'+k)
            close(np.percentile(values['minFDE'][mask],90),b['minFDE_p90'])
    win(ROOT/'A0',result['baseline'])
    populations = 1
    expected_orders = {}
    for s in range(3):
        gen = torch.Generator().manual_seed(20261117+s)
        expected_orders[str(s)] = [hashlib.sha256(torch.randperm(21719,generator=gen).numpy().astype('<i8').tobytes()).hexdigest() for _ in range(20)]
    for a, seeds in result['runs'].items():
        for s,v in seeds.items():
            folder = ROOT/'runs'/f'{a}_s{s}'
            assert read(folder/'RESULT.json') == v
            win(folder,v['clean'])
            populations += 1
            if a in ('C2','C3','C4'):
                win(folder,v['shuffled'],'_shuffled')
                populations += 1
            train = read(folder/'TRAINING.json')
            assert train['epochs']==20 and train['updates']==1700
            assert [b['order_hash'] for b in train['history']] == expected_orders[s]
            state = torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
            assert state['epoch']==20 and state['updates']==1700 and state['scheduler']['last_epoch']==1700
            assert v['future_shuffle']['status']=='BITWISE_EQUAL'
    d = result['decision']
    r = result['runs']
    base = result['baseline']
    def gain(a,b,what='minFDE',group='all',mode='clean'):
        v = np.array([r[a][s][mode][group][what] for s in ('0','1','2')])
        ref = np.array([base[group][what] if b=='A0' else r[b][s]['clean'][group][what] for s in ('0','1','2')])
        return (ref-v)/ref
    first = gain('A1','A0')
    assert first.mean()>=.008 and abs(first.mean()-.0129)<=.003
    assert gain('A1','A0','minADE').mean()>=-.002 and d['proxy']['passes']
    close(first,d['proxy']['P0']['minFDE']['by_seed'])
    close(first.std(ddof=1),d['proxy']['sigma'])
    thr = max(.005,2*first.std(ddof=1))
    close(thr,d['proxy']['threshold'])
    ref = min(('A1','A1L'),key=lambda a:np.mean([x['clean']['all']['minFDE'] for x in r[a].values()]))
    assert ref==d['reference']
    for a in ('C1','C2','C3','C4'):
        for k in ('minADE','minFDE'):
            g = gain(a,ref,k)
            close(g,d['comparisons_vs_R'][a][k]['by_seed'])
            close(g.mean(),d['comparisons_vs_R'][a][k]['mean'])
        assert gain(a,ref).mean()<thr
        checks=d['guards'][a]['checks']
        for key,value in [('top1_vs_A0_relative_harm',-gain(a,'A0','top1FDE').mean()),
                          ('stable_vs_R_relative_harm',-gain(a,ref,group='stable').mean()),
                          ('decel_stop_vs_R_relative_harm',-gain(a,ref,group='decel_stop').mean())]:
            close(value,checks[key]['value'])
            assert checks[key]['passes']==(value<=checks[key]['limit'])
    for key, a, b in [('Q3_vs_A0','C3','A0'),('Q4_vs_C4','C3','C4'),('Q5_vs_C2_descriptive','C3','C2')]:
        for k in ('minADE','minFDE'):
            close(gain(a,b,k),d[key][k]['by_seed'])
    social_gain = gain('C3','C1',group='social').mean()
    shuffled_gain = gain('C3','C1',group='social',mode='shuffled').mean()
    close(social_gain-shuffled_gain,d['mechanism']['social_relative_gain_drop'])
    assert social_gain<0 and not d['mechanism']['passes']
    assert done['verdict']==d['verdict']=='NO_INFO_GAIN' and d['protection_status']=='NO_EFFECTIVE_ARM'
    assert not d['eligible_to_advance'] and not result['B_stage_started']
    record = dict(at=now(),status='PASS',elapsed_seconds=time.monotonic()-start,sha_files_checked=len(sealed),
                  checkpoint_receipts=len(checkpoints),completed_runs=len(runs_done),optimizer_updates=30600,
                  populations_recomputed=populations,window_records=populations*len(y),
                  geometry_and_group_metrics='MATCH_FROM_SAVED_PREDICTIONS',D0_dev_summary='MATCH_FROM_DERIVED_ARRAYS',
                  batch_orders='MATCH_REBUILT_CPU_GENERATOR',matched_count_neighbor_permutation='PASS',
                  verdict='NO_INFO_GAIN',protection='NO_EFFECTIVE_ARM',new_training=False,new_inference=False,
                  future_shuffle_check='ORIGINAL_PER_RUN_RECEIPTS_CONFIRMED',source_done_sha256=sha(ROOT/'DONE.json'))
    write(ROOT/'INDEPENDENT_VERIFICATION.json',record)
    print(record)


if __name__ == '__main__':
    main()
