"""Future explicit GPU preflight. THIS FILE IS NOT EXECUTED WHILE ON HOLD."""
import argparse, copy, math, time
from pathlib import Path
from joint_common import *

def main(output,execute=False):
    authorize('preflight',execute);package=verify_package();verify(package['dependencies'])
    out=Path(output).resolve()
    if not out.is_relative_to(HERE/'receipts'):raise ValueError('PREFLIGHT_OUTPUT_NAMESPACE')
    out.mkdir(parents=True,exist_ok=False)
    # Training imports and CUDA creation are AFTER authorization.
    import torch
    from joint_train import Trainer,predict
    from joint_evaluate import evaluate
    from independent_math import reconstructor
    from validate_scripts import numerical_tests
    if not torch.cuda.is_available():raise RuntimeError('ACTUAL_CUDA_REQUIRED')
    torch.set_num_threads(1);rg=resource_module()
    # Initial inventory, not available-memory reverse engineering or a hard cap.
    planned_E=4520878080+21719*12*2*4+512*1024**2
    guard=rg.Guard(planned_E);guard.check(force=True)
    started=time.perf_counter();data=cache('fit');future,_=labels('fit',data['ids'])
    synthetic=numerical_tests();results=[];bench=[];source=sha(HERE/'SCRIPT_MANIFEST.json')
    ids=data['ids'];subset=np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256(('joint-resume-v1|'+str(ids[i])).encode()).digest())[:513])
    small={k:v[subset] for k,v in data.items()};target=future[subset]
    for family in ('existing','cvae'):
        initials=[];orders=[]
        for arm in ('accuracy_only','accuracy_dpp'):
            one=Trainer(family,arm,42,small,target,source,epochs=2)
            initials.append(cpu(one.model.state_dict()))
            if family=='existing':
                with torch.no_grad():
                    native=one.model.generate(batch(data,slice(0,32),'cuda'))[0].cpu().numpy()
                if not np.array_equal(native,data['native'][:32]):raise ValueError('EXISTING_INITIAL_NATIVE_IDENTITY')
            # Real 513 FIT, 17 batches/epoch, exact split at19/34 updates.
            for _ in range(19):one.step();guard.check()
            middle=one.snapshot();cp=out/f'{family}_{arm}_interrupted19.pt';save_checkpoint(cp,middle)
            for _ in range(15):one.step();guard.check()
            expected=one.snapshot();one.check_frozen();orders.append(one.orders)
            two=Trainer(family,arm,42,small,target,source,epochs=2)
            two.restore(trusted_checkpoint(cp,sha(cp)))
            for _ in range(15):two.step();guard.check()
            exact(expected,two.snapshot());two.check_frozen()
            active={n for n,p in two.model.named_parameters() if p.requires_grad}
            if two.connected!=active:raise ValueError('PHASED_GRAD_CONNECTIVITY '+family+'/'+arm)
            # Bad contract actually rejected, not merely schema-inspected.
            rejected=[]
            for key in ('source_sha','protocol_sha','ids'):
                bad=copy.deepcopy(expected);bad['contract'][key]='BAD'
                try:two.restore(bad)
                except (ValueError,TypeError):rejected.append(key)
                else:raise ValueError('BAD_RESUME_ACCEPTED '+key)
            badpath=out/f'{family}_{arm}_badsha.txt';write(badpath,dict(not_a_checkpoint=True))
            try:trusted_checkpoint(cp,sha(badpath))
            except ValueError:rejected.append('checkpoint_SHA')
            else:raise ValueError('BAD_CHECKPOINT_ACCEPTED')
            sample={k:v[:128] for k,v in small.items()}
            t=time.perf_counter();paths,control=predict(two.model,sample,'cuda',guard)
            evaluate(out/f'saved_{family}_{arm}',paths,control,sample,'fit',guard=guard,fixture=True,reconstruct=reconstructor(two.model,family))
            saved_seconds=time.perf_counter()-t
            # Inference API has no future; a changed future tensor is not an argument.
            b=batch(small,slice(0,8),'cuda')
            with torch.no_grad():
                y0=two.model.generate(b)[0];unused_future=torch.full((8,12,2),999.,device='cuda')
                y1=two.model.generate(b)[0];exact(y0,y1)
            # Measure disposable fixture models only, never select/save them as science.
            measure=Trainer(family,arm,42,small,target,source,epochs=3)
            times=[]
            for j in range(36):
                torch.cuda.synchronize();t=time.perf_counter();measure.step();torch.cuda.synchronize()
                if j>=4:times.append(time.perf_counter()-t)
            bench.append(dict(family=family,arm=arm,warm4_measure32_max=max(times),mean=float(np.mean(times)),saved128_seconds=saved_seconds))
            results.append(dict(family=family,arm=arm,real513twoepochs34_exact_interrupt19=True,
                all_trainable_tensor_phase_connected=True,bad_readers_rejected=rejected,
                saved128_full_quality_cluster_motion=True,future_not_in_inference_API=True))
            del one,two,measure
        exact(initials[0],initials[1]);exact(orders[0],orders[1])
    # All receipts remain fixtures, no full FIT two-epoch claim.
    verify(package['files']);verify(package['dependencies']);guard.check(force=True)
    peak=rg.snapshot()['peak_private'];payload=read(OLD/'CACHE_RECEIPT.json')['splits']['fit']['bytes_payload']
    estimate=peak+payload+512*1024**2
    seconds=1.3*(sum(b['warm4_measure32_max']*13580*3 for b in bench)
        +sum(b['saved128_seconds']*(21719+4753)/128*3 for b in bench)+300)
    write(out/'PREFLIGHT.json',dict(status='PASS_REAL_JOINT_GPU_RESUME_SAVED_PREFLIGHT',at=now(),
        script_manifest_sha=source,source_files=package['files'],dependency_files=package['dependencies'],
        results=results,synthetic_checks=synthetic,fixture_updates_not_scientific=True,
        full_FIT_two_epochs=False,benchmark=bench,planning_seconds=seconds,
        ETA_seconds=[.7*seconds,1.5*seconds],GPU_E=estimate,
        E_inventory=dict(measured_private_peak=peak,FIT_cache_payload=payload,allocator_output_reserve=512*1024**2),
        CPU_E=2*1024**3,resource_observations=guard.observations,elapsed=time.perf_counter()-started))
    print('REAL_PREFLIGHT_PASS_NOT_FORMAL_TRAINING')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();main(a.output,a.execute)
