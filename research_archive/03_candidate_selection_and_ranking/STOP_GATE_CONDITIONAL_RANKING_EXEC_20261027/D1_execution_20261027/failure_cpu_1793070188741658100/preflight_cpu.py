from audit_saved import *
import time,traceback,shutil
fb={'np':np};pure(OLD/'feature_builder.py',('build',),fb)
def main():
    start=time.perf_counter();h=source0();g=guard('CPU');reports=[];first=next(aux(42,'opt'))[1]
    for seed in SEEDS:
        st=load_npz(CACHE/f'stats_seed{seed}.npz');mu=st['window_mean'].mean(0);v=st['window_variance'].mean(0)+((st['window_mean']-mu)**2).mean(0);sd=np.sqrt(v);sd[sd<1e-6]=1
        err=max(float(abs(mu-st['mean64']).max()),float(abs(sd-st['std64']).max()));assert err<=1e-11
        assert np.array_equal(st['mean32'],st['mean64'].astype(np.float32)) and np.array_equal(st['std32'],st['std64'].astype(np.float32))
        reports.append(dict(seed=seed,all4096_equalwindow_independent_secondpass_stat_error=err,old_buffers_exact=True));g.check()
    obs=metadata('opt');mm=np.load(CACHE/'features_seed42.npy',mmap_mode='r');maximum=0.
    for i in range(128):
        ref,_=legacy['direct'](first['native'][i],first['kin'][i],first['gt'][i])
        for k in ('motion','quality','margin'):maximum=max(maximum,float(abs(ref[k]-first[k][i]).max()));assert maximum<=2e-12
        for k in ('eligible','rejects','B','target'):assert np.array_equal(ref[k],first[k][i])
        x=fb['build'](obs['past'][i],obs['tokens'][i],first['native'][i],first['kin'][i],first['motion'][i]);assert np.array_equal(x.astype(np.float32),mm[i])
        if i<8:
            changed,_=legacy['direct'](first['native'][i],first['kin'][i],first['gt'][i]+123.0)
            assert np.array_equal(changed['motion'],ref['motion']) and np.array_equal(changed['eligible'],ref['eligible'])
            assert np.array_equal(x,fb['build'](obs['past'][i],obs['tokens'][i],first['native'][i],first['kin'][i],changed['motion']))
    mm._mmap.close();gp=read_json(HERE/'PREFLIGHT_GPU.json');assert gp['status']=='PASS' and gp['source']==h
    a=time.perf_counter();report=audit(42,'opt',HERE/gp['saved128_folder']);seconds=time.perf_counter()-a
    assert verify_manifest(HERE)==h;write_json(HERE/'PREFLIGHT_CPU.json',dict(status='PASS',source=h,stats=reports,actual128_full400actions_literal_masks_labels_EXACT=True,max_geometry_error=maximum,all128_inputs_rawfloat32_exact=True,saved128=report,actual_CPU_save_audit_seconds=seconds,seconds=time.perf_counter()-start,no_Torch_model_forward=True))
    resource(HERE/'PREFLIGHT_CPU_RESOURCE.json',g);print('PASS CURRENT CPU PREFLIGHT',seconds,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as ex:
        out=HERE/('failure_cpu_'+str(time.time_ns()));out.mkdir();write_json(out/'FAILURE.json',dict(error=str(ex),traceback=traceback.format_exc(),preserve_all=True))
        for p in HERE.glob('*.py'):shutil.copyfile(p,out/p.name)
        raise
