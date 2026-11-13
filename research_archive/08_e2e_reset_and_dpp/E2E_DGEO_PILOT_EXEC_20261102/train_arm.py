"""Exactly one fresh paired pilot. No dev selection or old model training."""
import argparse
import time
from geo_common import *
from train_core_geo import *
from evaluate_geo import save_population
from preflight import guard_module
from mechanism_geo import collect_mechanism


def main(arm):
    torch.set_num_threads(1);start=time.perf_counter()
    manifest=read_json(RUN/'source_manifest.json');verify(manifest['files'])
    if read_json(RUN/'PRELAUNCH_REVIEW.json')['status']!='PASS_AUTHORIZED_ALL4_PILOTS':raise ValueError('NOT_ADMITTED')
    rg=guard_module();guard=rg.Guard(read_json(RUN/'MEMORY_ESTIMATE.json')['GPU_E']);guard.check(force=True)
    fit=dict(np.load(BASE/'C0/fit.npz'));dev=dict(np.load(BASE/'C0/dev.npz'))
    trainer=Trainer(fit,np.arange(len(fit['ids'])),arm,check_B0_order=True)
    run=RUN/'runs'/arm;run.mkdir(parents=True,exist_ok=False)
    contract=dict(arm=arm,source_manifest=sha(RUN/'source_manifest.json'),protocol=sha(RUN/'protocol.json'),
        fit=sha(BASE/'C0/fit.npz'),seed=42,epochs=8,batch=64,clip=5,
        normalization='PAST_ONLY_HEADING_PER_AGENT_OBS_MEAN_NO_FITTED_STATS',
        selection='FIXED_FINAL_EPOCH8_EMA_0.995',fresh=True,paired_B0_initial_order_augmentation_RNG=True)
    write_json(run/'manifest.json',contract);save_checkpoint(run/'initial.pt',trainer.snapshot(contract))
    summaries=[]
    for epoch in range(1,9):
        t=time.perf_counter();losses=[]
        for _ in range(trainer.steps_per_epoch):
            row=trainer.step();losses.append(row['loss'])
            try:guard.check()
            except MemoryError:
                save_checkpoint(run/f'resource_stop_{trainer.updates}.pt',trainer.snapshot(contract));raise
            if trainer.updates%64==0:write_json(RUN/'progress'/f'{time.time_ns()}.json',dict(at=now(),stage='DGEO_TRAIN',arm=arm,epoch=epoch,update=trainer.updates))
        # Identical random streams prove augmentation/dropout/shuffle calls are paired, not just initial seed.
        old=torch.load(BASE/f'B0_seed42/epoch{epoch:02}.pt',map_location='cpu',weights_only=False)
        exact(bf.rng_state(trainer.generator),old['rng'],'EPOCH_B0_SHARED_ALL_RNG');del old
        save_checkpoint(run/f'epoch{epoch:02}.pt',trainer.snapshot(contract))
        summaries.append(dict(epoch=epoch,updates=trainer.updates,mean_train_loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-t))
        write_json(run/f'epoch{epoch:02}_history.json',summaries[-1]);print(arm,'EPOCH',epoch,flush=True)
    if trainer.updates!=2720:raise ValueError('WRONG_UPDATE_COUNT')
    save_checkpoint(run/'final.pt',trainer.snapshot(contract));write_csv(run/'history.csv',trainer.history)
    write_json(run/'batch_orders.json',dict(batch_IDs=trainer.batch_orders,steps=trainer.updates))
    write_json(run/'EPOCH_SUMMARY.json',summaries)
    write_json(run/'FINAL_LOCK.json',dict(at=now(),updates=2720,epochs=8,
        files=seal([run/'final.pt',run/'initial.pt',run/'manifest.json',run/'history.csv',run/'batch_orders.json']+sorted(run.glob('epoch*.pt')))))
    for split,data in [('fit',fit),('dev',dev)]:
        write_json(RUN/'progress'/f'{time.time_ns()}.json',dict(at=now(),stage='DGEO_FINAL_SAVE_INDEPENDENT_FULL_AUDIT',arm=arm,split=split))
        paths,logits=infer(trainer.ema.module,data,guard=guard)
        save_population(RUN/'populations'/arm/split,paths,logits,data,guard);del paths,logits
    collect_mechanism(trainer.ema.module,arm,fit,dev,guard)
    verify(manifest['files']);guard.check(force=True)
    write_json(run/'RESOURCE.json',dict(status='PASS_WARNING_FIRST_V3',snapshot=rg.snapshot(),observations=guard.observations))
    write_json(run/'done.json',dict(status='SUCCESS',arm=arm,at=now(),elapsed_seconds=time.perf_counter()-start,
        epochs=8,updates=2720,all_original_initial_weights_buffers_and_all_epoch_RNG_order_exact=True,
        files=seal([p for p in run.iterdir() if p.is_file()])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=ARMS);main(p.parse_args().arm)
