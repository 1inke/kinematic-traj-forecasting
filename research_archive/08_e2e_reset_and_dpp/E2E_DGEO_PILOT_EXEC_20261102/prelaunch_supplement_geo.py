"""Actual full-epoch paired RNG, fixed128 initial mechanism and independent literal geometry gradient."""
import ast,time
from geo_common import *
from train_core_geo import *
from preflight import guard_module
from geo_modules import compute_anchor_geo_features_12d
from geo_math import geom
from mechanism_geo import calibrate,collect_mechanism


def main():
    start=time.perf_counter();torch.set_num_threads(1)
    for p in RUN.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    calibrate();fit=dict(np.load(BASE/'C0/fit.npz'));dev=dict(np.load(BASE/'C0/dev.npz'))
    rg=guard_module();guard=rg.Guard(read_json(RUN/'MEMORY_ESTIMATE.json')['GPU_E']);guard.check(force=True)
    # Same fixed non-cut geometry fixture; independent every-coordinate analytic derivatives.
    a=torch.linspace(.1,2.,8*20*3*2,dtype=torch.float64).reshape(8,20,3,2).requires_grad_(True)
    p=torch.linspace(-.7,.2,8*8*2,dtype=torch.float64).reshape(8,8,2).requires_grad_(True)
    w=torch.linspace(-.3,.7,8*20*12,dtype=torch.float64).reshape(8,20,12)
    g=compute_anchor_geo_features_12d(a,p);ga,gp=torch.autograd.grad((g*w).sum(),(a,p))
    # Separate functional expression uses no production geometry helper.
    ar=a.detach().clone().requires_grad_(True);pr=p.detach().clone().requires_grad_(True)
    ref=geom(ar,pr,torch);ra,rp=torch.autograd.grad((ref*w).sum(),(ar,pr))
    err=max(float((ga-ra).abs().max()),float((gp-rp).abs().max()))
    if err>1e-10:raise ValueError('GEOMETRY_ALL_COORDINATE_GRADIENT')
    report={};old=torch.load(BASE/'B0_seed42/epoch01.pt',map_location='cpu',weights_only=False)['rng']
    for arm in ARMS:
        t=Trainer(fit,np.arange(len(fit['ids'])),arm,check_B0_order=True)
        collect_mechanism(t.model,arm,fit,dev,guard,folder=RUN/'initial_mechanisms'/arm)
        for _ in range(340):t.step();guard.check()
        exact(bf.rng_state(t.generator),old,'REAL_FIRST_FULL_EPOCH_ORIGINAL_RNG_'+arm)
        report[arm]=dict(full_fit_windows=21719,updates=340,all_original_RNG_and_batchIDs_exact=True,
            initial_fixed128_mechanism='PASS',not_scientific_model=True)
        del t;print('GEO_REAL_PAIRED_EPOCH_PASS',arm,flush=True)
    verify(read_json(BASE/'source_manifest.json')['files']);verify(read_json(UD/'source_manifest.json')['files'])
    write_json(RUN/'PRELAUNCH_SUPPLEMENT.json',dict(status='PASS_GEO_ALL4_FULL_EPOCH_AND_INITIAL128',
        elapsed_seconds=time.perf_counter()-start,report=report,fixture_updates=1360,geometry_coordinate_gradient_max_error=err,resource=guard.observations))


if __name__=='__main__':main()
