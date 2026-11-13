from core import *
from audit import run
from prepare import raw_features
def main():
    g=guard(True);h=verify();p=read_json(HERE/'PREFLIGHT_GPU.json');assert p['status']=='PASS' and p['source']==h;z=dataset('opt');st=load(HERE/'prepared/statistics.npz');split=load(HERE/'prepared/split.npz');x=z['raw'][split['train']].astype(float);mu=np.zeros(263);m2=np.zeros(263);count=0
    for row in x:
        for v in row:
            count+=1;delta=v-mu;mu+=delta/count;m2+=delta*(v-mu)
    sd=np.sqrt(m2/count);sd[sd<1e-6]=1;error=max(float(abs(mu-st['mean']).max()),float(abs(sd-st['std']).max()));assert error<=1e-11
    fresh=raw_features(z['past'][:128],z['tokens'][:128],z['native'][:128]);assert np.array_equal(fresh,z['raw'][:128]);alter=z['gt'][:128]+123;assert np.array_equal(fresh,raw_features(z['past'][:128],z['tokens'][:128],z['native'][:128]))
    assert not np.intersect1d(z['track_key'][split['train']],z['track_key'][split['val']]).size
    a=time.perf_counter();report=run(42,'opt',HERE/p['saved128_folder'],g);seconds=time.perf_counter()-a;assert verify()==h
    write_json(HERE/'PREFLIGHT_CPU.json',dict(status='PASS',source=h,train_only_fullstatistics_independent_Welford_maxerror=error,all128_rawinput_EXACT=True,GT_not_prediction_API=True,actual128_full_quality_clusters_motion=report,actual_CPU_seconds=seconds,no_Torch=True));resource(HERE/'PREFLIGHT_CPU_RESOURCE.json',g);print('PASS CPU',seconds,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as ex:failure('CPU_preflight',ex);raise
