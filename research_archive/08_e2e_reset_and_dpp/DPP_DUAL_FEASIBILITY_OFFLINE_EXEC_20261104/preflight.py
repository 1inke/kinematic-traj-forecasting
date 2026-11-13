"""Real saved128 preflight, must NOT run until later explicit approval."""
import argparse,time
from common import *
from scoring import *
from independent import score_loop,spectral_ec_loop,quality_loop
from prepare import hash128

def main(execute=False):
    if not authorize(execute,'preflight'):return
    t=time.perf_counter();source=read(RUN/'source_manifest.json');verify(source['files'])
    c=read(RUN/'CALIBRATION.json');reports={}
    out=RUN/'real_preflight';out.mkdir(exist_ok=False)
    for name in MODELS:
        p,m=checked_population(name,'fit');idx=hash128(p['ids'],'dpp-actual-preflight-v1')
        paths=p['paths'][idx].astype(np.float64);past=p['past'][idx].astype(np.float64);future=p['future'][idx].astype(np.float64)
        ts=time.perf_counter();s=kinematic(past,paths,c);q,valid=score_loop(past,paths,c)
        error=float(np.max(abs(q-s['qkin'])))
        if error>2e-12 or not np.array_equal(valid,s['kin_valid']):raise ValueError('ACTUAL_KIN_SCORE_AUDIT')
        d=dpp(paths,s['qkin'],c['k'],c['epsilon_q'])
        ei=spectral_ec_loop(paths,s['qkin'],c['k'],c['epsilon_q'])
        if np.max(abs(ei-d['ec']))>2e-12:raise ValueError('ACTUAL_DPP_SECOND_IMPLEMENTATION')
        ade,fde=quality_loop(paths,future)
        if np.max(abs(ade-m['ADE'][idx]))>2e-12 or np.max(abs(fde-m['FDE'][idx]))>2e-12:raise ValueError('ACTUAL_OLD_QUALITY')
        saved=out/(name+'.npz');np.savez_compressed(saved,ids=p['ids'][idx],qkin=s['qkin'],ec=d['ec'],valid=s['kin_valid'],ADE=ade,FDE=fde)
        with np.load(saved,allow_pickle=False) as f:
            if not np.array_equal(f['qkin'],s['qkin']) or not np.array_equal(f['ids'],p['ids'][idx]):raise ValueError('READBACK')
        reports[name]=dict(N=128,score_error=error,EC_error=float(np.max(abs(ei-d['ec']))),elapsed_seconds=time.perf_counter()-ts,saved_sha=sha(saved))
    verify(source['files']);write(RUN/'PREFLIGHT_REAL.json',dict(status='PASS_ACTUAL_SAVED128_KIN_DPP_QUALITY_READBACK',source_manifest=sha(RUN/'source_manifest.json'),reports=reports,elapsed_seconds=time.perf_counter()-t,no_training=True))
    print('REAL_STAGE_A_PREFLIGHT_PASS')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');main(p.parse_args().execute)

