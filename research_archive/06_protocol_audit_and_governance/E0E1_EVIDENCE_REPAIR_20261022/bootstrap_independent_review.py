"""P1 second implementation: expand shared sampled tracks to window rows."""
from common import *

def main():
    verify_sources();data=data_load();ix=np.load(ROOT/'qoder_work/v1_1/cal_v11_index.npy');mask=data['groups'][ix]==2
    with (ROOT/'qoder_work/v1_1/cal_v11_windows.csv').open(encoding='utf-8-sig') as f:track=np.array([r['track'] for r in csv.DictReader(f)])[mask]
    values=np.unique(track);indices=[np.flatnonzero(track==v) for v in values];delta=np.stack([np.load(E0/f'delta_change_clean_seed{s}.npy') for s in SEEDS]);rng=np.random.default_rng(20261020);draw=rng.integers(len(values),size=(20000,len(values)));seeds=rng.integers(3,size=(20000,3));fixed=np.empty(20000);resampled=np.empty(20000)
    for i in range(20000):
        rows=np.concatenate([indices[j] for j in draw[i]]);per=delta[:,rows].mean(1);fixed[i]=per.mean();resampled[i]=per[seeds[i]].mean()
    fci=np.quantile(fixed,[.025,.975])*100;sci=np.quantile(resampled,[.025,.975])*100;old=read_json(HERE/'E0_STATISTICS_CORRECTED.json');diff=max(float(np.max(np.abs(fci-old['shared_track_fixed_seed_CI_pp']))),float(np.max(np.abs(sci-old['shared_track_paired_seed_CI_pp']))))
    if diff>2e-12:raise RuntimeError('SHARED_TRACK_CI_SECOND_IMPLEMENTATION_MISMATCH')
    verify_sources();write_json(HERE/'E0_BOOTSTRAP_INDEPENDENT_REVIEW.json',dict(status='PASS',script_sha256=sha256(__file__),draws=20000,n_tracks=len(values),n_windows=int(mask.sum()),shared_fixed_seed_CI_pp=fci,shared_paired_seed_CI_pp=sci,max_CI_diff_pp=diff,scope='Posthoc E0 coverage only, not new confirmatory inference or full W audit.'))
    print('E0 shared-track20k independent expanded-window CI check PASS;maxdiff',diff)

if __name__=='__main__':main()
