import time,traceback,csv
from run_common import *
from train_core import Trainer,infer

def main():
    with (RUN/'worker.lock').open('x') as f:f.write(str(os.getpid()))
    started=time.perf_counter();write(RUN/'WORKER_START.json',dict(at=now(),pid=os.getpid(),parent_pid=os.getppid(),stage='INITIAL_SOURCE_CHECK'))
    try:
        source_check();torch.set_num_threads(1);rg=rg_module();E=read(RUN/'MEMORY_ESTIMATE.json')['GPU_E'];guard=rg.Guard(E);guard.check(force=True)
        adapter,_=load_current('cuda');frozen=cpu(adapter.model.state_dict());cache=cache_read('fit');indices=np.arange(len(cache['ids']));finals={}
        for seed in (42,43,44):
            paired=[];orders=[]
            for arm in ('uniform','kin'):
                progress('TRAIN_DSF',arm=arm,seed=seed,epoch=0)
                folder=RUN/'runs'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
                tr=Trainer(adapter,cache,indices,arm,seed,epochs=20);paired.append(cpu(tr.sampler.state_dict()));write(folder/'contract.json',tr.contract);save_pt(folder/'initial.pt',tr.snapshot())
                for epoch in range(1,21):
                    for _ in range(679):tr.step();guard.check()
                    if tr.epoch!=epoch or tr.cursor!=len(indices):raise ValueError('EPOCH_COMPLETION')
                    save_pt(folder/f'epoch{epoch:02}.pt',tr.snapshot());write(folder/f'epoch{epoch:02}.json',dict(at=now(),epoch=epoch,updates=tr.updates,last_step=tr.history[-1]))
                    progress('TRAIN_DSF',arm=arm,seed=seed,epoch=epoch,updates=tr.updates)
                if tr.updates!=13580 or len(tr.order_history)!=20:raise ValueError('COMPLETE_RUN_UPDATES')
                exact(frozen,cpu(adapter.model.state_dict()),'FROZEN_AFTER_TRAIN_RUN')
                save_pt(folder/'final.pt',tr.snapshot());write(folder/'history.json',tr.history);write(folder/'epoch_orders.json',tr.order_history)
                orders.append(copy.deepcopy(tr.order_history));write(folder/'done.json',dict(status='SUCCESS_DSF_FIXED20',source=sha(RUN/'PRODUCTION_SOURCE_LOCK.json'),files=seal(folder.iterdir()),epochs=20,updates=13580));finals[f'{arm}_seed{seed}']=read(folder/'done.json')
            exact(paired[0],paired[1],'PAIRED_DSF_INITIAL_SEED'+str(seed));exact(orders[0],orders[1],'PAIRED_EPOCH_WINDOW_IDS_SEED'+str(seed))
        write(RUN/'ALL6_FINALS_LOCK.json',dict(at=now(),status='ALL6_DSF_FINAL_SHA_SEALED_BEFORE_DEV_LABELS',runs=finals,checkpoint_files={str(RUN/'runs'/name/'final.pt'):sha(RUN/'runs'/name/'final.pt') for name in finals},held_labels_opened=False))
        from evaluate import evaluate_population,independent_saved_review
        audits={};outputs=[]
        for split in ('fit','dev'):
            current=cache_read(split);native_folder=RUN/'populations/native'/split
            progress('SAVE_NATIVE_REFERENCE',split=split)
            evaluate_population(native_folder,current['native'].astype(np.float64),current['native_logits'],None,current,split,guard)
            audits['native/'+split]=independent_saved_review(adapter,current,native_folder,'native',guard)
            for seed in (42,43,44):
                for arm in ('uniform','kin'):
                    name=f'{arm}_seed{seed}';folder=RUN/'populations'/name/split;progress('INFER_SAVE_AUDIT',arm=arm,seed=seed,split=split)
                    state=torch.load(RUN/'runs'/name/'final.pt',map_location='cpu',weights_only=False);sampler=new_sampler(seed,'cuda');sampler.load_state_dict(state['sampler'],strict=True)
                    paths,logits,z=infer(adapter,sampler,current,guard);evaluate_population(folder,paths,logits,z,current,split,guard)
                    progress('INDEPENDENT_SAVED_REVIEW',arm=arm,seed=seed,split=split)
                    audits[name+'/'+split]=independent_saved_review(adapter,current,folder,arm,guard)
                    del paths,logits,z,sampler,state
            del current
        write(RUN/'ARITHMETIC_AUDIT.json',dict(status='PASS_ALL14_POPULATIONS_CANONICAL_AND_INDEPENDENT_SAVED',populations=audits,no_whole_encoder_or_gradient_replay=True))
        progress('PRIMARY_TRACK_SEED_BOOTSTRAP');import compare;compare.main();import protection_review;protection_review.main()
        progress('ACTUAL_ARRAY_PIPELINE_COST');import cost_profile;cost_profile.main()
        exact(frozen,cpu(adapter.model.state_dict()),'FROZEN_FINAL');source=source_check();guard.check(force=True)
        write(RUN/'RESOURCE.json',dict(policy='warning_first_v3',estimate=E,snapshot=rg.snapshot(),observations=guard.observations))
        for p in (RUN/'runs').rglob('done.json'):check_files(read(p)['files']);outputs.append(p);outputs.extend(map(Path,read(p)['files']))
        for p in (RUN/'populations').rglob('done.json'):check_files(read(p)['files']);outputs.append(p);outputs.extend(map(Path,read(p)['files']));outputs.append(p.parent/'independent_review.json')
        write(RUN/'PRESERVATION.json',dict(status='PASS_ALL_SOURCE_AND_SAVED_OUTPUT_SHA',source_files=source['files'],outputs=seal(set(outputs))))
        result=dict(status='SUCCESS_CURRENT_GENERATOR_DSF_UNIFORM_KIN_REVIEW_REQUIRED',at=now(),new_DSF_runs=6,DSF_epochs=120,DSF_updates=81480,frozen_generator_unchanged=True,scene_and_joint='NOT_TRIGGERED_MAP_ADMISSION',CVAE_pretraining='NOT_NEEDED',conditional_on_one_pretrained_seed42=True,formal_study_count=1,baseline_upgraded=False,no_automatic_extra_variants=True)
        write(RUN/'RESULT.json',result)
        lines=['# 当前生成器上的DSF单因素验证','', 'uniform与kin各三DSF种子、20轮；同一个已训练F16+h3冻结，原模型无更新。不是三个独立全流程模型。', '', '所有逐seed/组完整候选质量、原聚簇与独立pdist、五motion/junction、support交换、raw/logq/EC及控制见populations；固定family18和20000全draw见PRIMARY_COMPARISON.json。','', '场景图科学准入未完成，scene/joint未执行；非高斯query control不可解释原Gaussian radius。参数468914>原305000目标，成本只prepared-neighbor-array全模型链，非rawstream邻居搜索成本。', '', 'ARS DA2/3同助手非外审：EC或q提高不等覆盖收益；原成功lost/原五motion和major5%门不降低，不自动升级baseline。预训练基底固定一份、相关窗口/录像和反复dev探索限制均保留。', '', 'Saved算术与decoder重构不是整个encoder/所有梯度重放。所有代码模型日志证据保留，无删除。']
        with (RUN/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
        roots=[RUN/n for n in ('RESULT.json','RESULT_ANALYSIS.md','RESOURCE.json','PRESERVATION.json','ARITHMETIC_AUDIT.json','ALL6_FINALS_LOCK.json','PRIMARY_COMPARISON.json','BOOTSTRAP_DRAWS.npz','COST_ACTUAL_ARRAY_PIPELINE.json','PROTECTION_GATES.json','RESULT_NUMERIC_SUMMARY.md')]
        write(RUN/'FINAL_SEAL.json',dict(status='PASS_FINITE_GENERALIZED_DSF_UNIFORM_KIN_SAVED_REVIEW',files=seal(roots),sources=source['files']))
        write(RUN/'WORKER_SUCCESS.json',dict(status='SUCCESS',at=now(),elapsed_seconds=time.perf_counter()-started,files=seal([RUN/'FINAL_SEAL.json'])))
        write(RUN/'WORKER_EXIT.json',dict(status='SUCCESS',at=now(),pid=os.getpid(),exit_code=0));progress('COMPLETE')
    except BaseException as e:
        write(RUN/'WORKER_FAILURE.json',dict(status='FAILURE',at=now(),error=repr(e),traceback=traceback.format_exc(),retain_all_evidence=True))
        write(RUN/'WORKER_EXIT.json',dict(status='FAILURE',at=now(),pid=os.getpid(),exit_code=1));raise

if __name__=='__main__':main()
