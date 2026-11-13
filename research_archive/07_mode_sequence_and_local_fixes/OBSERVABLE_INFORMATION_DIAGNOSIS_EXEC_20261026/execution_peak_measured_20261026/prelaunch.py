from bootstrap import HERE,PARENT,run_original
from base import *
from evaluate_saved import evaluate,independent_motion,independent_check

def main():
    g=guard('AUDIT');pre=read_json(HERE/'PREFLIGHT_MLP.json');report=[]
    with np.load(HERE/'prepared/tiny.npz') as q:z={k:q[k] for k in q.files if k not in ('features','opt_row')}
    for arm in ('I','S'):
        p=HERE/pre['folder']/f'{arm}_saved128.npz'
        with np.load(p) as q:
            independent_check(z,q['scores'],q['chosen'],q['paths']);paths=q['paths'];rows,_=evaluate(paths,z['gt'],z['group'],z['native'],z['sample_id'],z['track_key'],include_clusters=False)
            for i,name in enumerate(('overall','stable','transition','change')):
                mask=np.ones(128,bool) if i==0 else z['group']==i-1
                if mask.any():
                    for actual,key in ((paths,'motion'),(z['native'],'native_motion')):
                        ref=independent_motion(actual[mask]);assert all(abs(ref[k]-rows[name][key][k])<=2e-12 for k in ref)
        report.append(dict(arm=arm,input_sha=sha256(p),all4groups_all5pooled_native_and_model_EXACT=True));g.check()
    write_json(HERE/'PREFLIGHT_SAVED_MOTION.json',dict(status='PASS',records=report,source=source(),no_Torch='torch' not in sys.modules,resource=g.check(),at=now()))
    run_original('prelaunch.py')

if __name__=='__main__':
    try:main()
    except BaseException as e:fail(e);raise
