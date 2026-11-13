"""One fresh seed42, eight fixed epochs; final EMA only, no dev selection."""
import time
from common import *
from train_core import *
from preflight import guard_module
from evaluate_saved import save_population


def progress(**x):
    write_json(HERE/'progress'/f'{time.time_ns()}.json',dict(at=now(),**x))


def main():
    torch.set_num_threads(1)
    started=time.perf_counter()
    manifest=read_json(HERE/'source_manifest.json');verify(manifest['files'])
    pre=read_json(HERE/'PREFLIGHT.json')
    if pre['status']!='PASS_B0_ACTUAL_PREFLIGHT' or not pre['budget_admitted']:raise ValueError('NOT_ADMITTED')
    rg=guard_module();guard=rg.Guard(read_json(HERE/'MEMORY_ESTIMATE.json')['GPU_E'])
    guard.check(force=True)
    train=dict(np.load(HERE/'C0/fit.npz'))
    dev=dict(np.load(HERE/'C0/dev.npz'))
    trainer=Trainer(train,np.arange(len(train['ids'])),epochs=8)
    run=HERE/'B0_seed42';run.mkdir(exist_ok=False)
    contract={'source_manifest':sha(HERE/'source_manifest.json'),
        'protocol':sha(HERE/'protocol.json'),'fit':sha(HERE/'C0/fit.npz'),
        'evaluation_contract':sha(HERE/'C0/EVALUATION_CONTRACT.json'),
        'seed':42,'epochs':8,'batch':64,'selection':'FIXED_FINAL_EMA_0.995',
        'normalization':'PAST_ONLY_HEADING_PER_AGENT_OBS_MEAN_NO_FITTED_STATS',
        'no_pretrained_checkpoint_or_tokens':True}
    save_checkpoint(run/'initial.pt',trainer.snapshot(contract))
    write_json(run/'manifest.json',contract)
    epochs=[]
    for epoch in range(1,9):
        epoch_start=time.perf_counter();losses=[]
        for _ in range(trainer.steps_per_epoch):
            row=trainer.step();losses.append(row['loss'])
            try:guard.check()
            except MemoryError:
                save_checkpoint(run/f'resource_stop_update{trainer.updates}.pt',trainer.snapshot(contract))
                raise
            if trainer.updates%64==0:
                progress(stage='B0_TRAIN',epoch=epoch,update=trainer.updates,fit_count=len(train['ids']))
        save_checkpoint(run/f'epoch{epoch:02}.pt',trainer.snapshot(contract))
        row=dict(epoch=epoch,updates=trainer.updates,mean_train_loss=float(np.mean(losses)),
            lr=trainer.optimizer.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-epoch_start)
        epochs.append(row)
        write_json(run/f'epoch{epoch:02}_history.json',row)
        print('EPOCH',row,flush=True)
    if trainer.updates!=2720:raise ValueError('WRONG_UPDATE_COUNT')
    save_checkpoint(run/'final.pt',trainer.snapshot(contract))
    write_csv(run/'history.csv',trainer.history)
    write_json(run/'batch_orders.json',{'batch_IDs':trainer.batch_orders,'steps':trainer.updates})
    write_json(run/'EPOCH_SUMMARY.json',epochs)
    write_json(HERE/'B0_FINAL_LOCK.json',{'at':now(),'selection':'epoch8_EMA_not_dev_selected',
        'files':seal([run/'final.pt',run/'initial.pt',run/'history.csv',run/'batch_orders.json',run/'manifest.json']),
        'epoch_checkpoints':seal(sorted(run.glob('epoch*.pt'))),
        'science_updates':trainer.updates,'source_manifest':sha(HERE/'source_manifest.json')})
    rows={}
    for name,data in [('fit',train),('dev',dev)]:
        progress(stage='B0_FINAL_EVALUATE_SAVE_INDEPENDENT_AUDIT',split=name)
        paths,logits=infer(trainer.ema.module,data,guard=guard)
        rows[name]=save_population(HERE/'populations'/name,paths,logits,data,guard)
        del paths,logits
    verify(manifest['files'])
    guard.check(force=True)
    readings=guard.observations
    write_json(HERE/'SCIENCE_RESOURCE.json',{'status':'PASS_CURRENT_PRESSURE_GUARD',
        'policy':'warning_first_remaining_v3','snapshot':rg.snapshot(),'observations':readings,
        'not_memory_lock':True,'no_OS_pagefile_userkill':True})
    write_json(HERE/'SCIENCE_DONE.json',{'status':'SUCCESS','at':now(),
        'elapsed_seconds':time.perf_counter()-started,'fresh_models':1,'epochs':8,'updates':trainer.updates,
        'fixed_final_EMA':True,'all_outputs_independently_audited_in_save_pipeline':True,
        'no_U_D_V_or_formal_runs':True})
    print('SCIENCE_COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
