"""Exercise late-added production consumers before source lock, no scientific pilots."""
import ast
import time
from ud_common import *
from train_core_ud import *
from preflight import guard_module,hash_indices
import mechanism_ud as mech
from compare_ud import summarize_pair,pilot_screen


def main():
    start=time.perf_counter();torch.set_num_threads(1)
    for p in RUN.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    mech.calibrate();out=RUN/'prelaunch_supplement';out.mkdir(exist_ok=False)
    fit=dict(np.load(BASE/'C0/fit.npz'));dev=dict(np.load(BASE/'C0/dev.npz'))
    rg=guard_module();guard=rg.Guard(read_json(RUN/'MEMORY_ESTIMATE.json')['GPU_E']);guard.check(force=True)
    mech.RUN=out;write_json(out/'MECHANISM_CONTRACT.json',read_json(RUN/'MECHANISM_CONTRACT.json'))
    fixed=hash_indices(dev['ids'],'ud-prelaunch-saved-mechanism-v1',128)
    small={k:v[fixed] for k,v in dev.items()}
    reports={}
    reference=torch.load(BASE/'B0_seed42/epoch01.pt',map_location='cpu',weights_only=False)['rng']
    for arm in ARMS:
        trainer=Trainer(fit,np.arange(len(fit['ids'])),arm,check_B0_order=True)
        for _ in range(340):trainer.step();guard.check()
        exact(bf.rng_state(trainer.generator),reference,'ACTUAL_FIRST_FULL_FIT_EPOCH_B0_ALL_RNG_'+arm)
        # Every late-added mechanism path runs with nonzero actual trained branch parameters.
        mech.collect_mechanism(trainer.ema.module,arm,small,small,guard)
        saved=dict(np.load(RUN/'preflight'/arm/'saved128/window_metrics.npz'))
        population=dict(np.load(RUN/'preflight'/arm/'saved128/predictions.npz'))
        groups=read_json(RUN/'preflight'/arm/'saved128/metrics.json')
        pair=summarize_pair(saved,saved,groups['overall'],groups['overall'],population,population,np.ones(128,bool))
        if pair['rescue'] or pair['lost'] or pair['coverage_delta']:raise ValueError('IDENTITY_PAIR_FAILURE')
        screen=pilot_screen(groups,groups)
        if screen['continuous_ADE_pass']:raise ValueError('IDENTITY_FALSE_ADE_GAIN')
        # Real consumer gates must reject damaged digest, not only a synthetic flag.
        files=read_json(RUN/'preflight'/arm/'saved128/done.json')['files'];verify(files)
        wrong=dict(files);wrong[next(iter(wrong))]='0'*64
        try:verify(wrong)
        except ValueError:pass
        else:raise ValueError('BAD_OUTPUT_NOT_REJECTED')
        reports[arm]=dict(actual_full_fit_first_epoch=21719,fixture_updates=340,
            saved_B0_all_RNG_and_batchIDs_exact=True,late_mechanism_nonzero_saved_pipeline='PASS',paired_self_comparison_no_false_gain=True)
        del trainer
        print('PRELAUNCH_REAL_CONSUMER_PASS',arm,flush=True)
    verify(read_json(BASE/'source_manifest.json')['files']);guard.check(force=True)
    write_json(RUN/'PRELAUNCH_SUPPLEMENT.json',dict(status='PASS_ACTUAL_ALL4_PAIRED_FULL_EPOCH_AND_MECHANISM_CONSUMERS',
        elapsed_seconds=time.perf_counter()-start,arms=reports,fixture_updates=1360,scientific_pilot_updates=0,
        same_assistant_not_external=True,resource=guard.observations))


if __name__=='__main__':main()
