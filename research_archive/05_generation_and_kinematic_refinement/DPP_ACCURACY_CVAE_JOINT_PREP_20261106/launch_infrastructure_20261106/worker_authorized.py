"""One finite OS-independent train->saved review->statistics worker.

Infrastructure extension only. Original prepared science code remains SHA-exact.
"""
from pathlib import Path
import sys,os,time,traceback,json,ast
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from joint_common import *

def mutable(path,value):
    p=Path(path);tmp=p.with_suffix('.new.json')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(tmp,p)

class ProgressOutput:
    def __init__(self,root,stream):self.root=root;self.stream=stream;self.buffer=''
    def write(self,value):
        self.stream.write(value);self.buffer+=value
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            try:
                if line.startswith(('existing_accuracy_','cvae_accuracy_')):
                    name,row=line.split(' ',1);record=ast.literal_eval(row)
                    mutable(self.root/'runtime_state.json',dict(status='RUNNING',at=now(),stage='TRAIN',
                        run=name,actual_epoch=record['epoch'],actual_updates=record['update'],loss=record['loss'],pid=os.getpid()))
            except (ValueError,SyntaxError,KeyError):pass # Other stdout is not scientific progress.
        return len(value)
    def flush(self):self.stream.flush()

def main(root):
    root=Path(root).resolve();start=time.perf_counter()
    ext=read(root/'INFRASTRUCTURE_LOCK.json');verify(ext['files'])
    original=sys.stdout;sys.stdout=ProgressOutput(root,original)
    try:
        write(root/'WORKER_START.json',dict(at=now(),pid=os.getpid(),parent_pid=os.getppid(),
            commandline=sys.argv,stage='INITIAL_SOURCE_CHECK',not_training_complete=True))
        mutable(root/'runtime_state.json',dict(status='RUNNING',at=now(),pid=os.getpid(),stage='INITIAL_SOURCE_CHECK'))
        import run_series
        run_series.train(root,ext['preflight'],family='all',execute=True)
        # This root marker is permitted ONLY after the source-verified 12-model lock.
        allfinal=read(root/'ALL12_FINALS_LOCK.json')
        if len(allfinal['runs'])!=12:raise ValueError('ALL12_NOT_LOCKED')
        mutable(root/'runtime_state.json',dict(status='RUNNING',at=now(),pid=os.getpid(),stage='SAVED_POPULATIONS_AUDIT_AND_FAMILY30'))
        run_series.evaluate_all(root,execute=True)
        verify(read(root/'FINAL_SCIENCE_SEAL.json')['files']);verify(ext['files'])
        from joint_statistics import values
        groups=[]
        for split in ('fit','dev'):
            nm=read(root/'populations/native'/split/'metrics.json')
            for name in run_names():
                p=root/'populations'/name/split;mm=read(p/'metrics.json');ss=read(p/'support.json')
                for g in ('overall','stable','transition','change'):
                    a,b,s=mm[g],nm[g],ss[g]
                    groups.append(dict(run=name,split=split,group=g,N=a['n'],good=a['good'],
                        native_good=b['good'],coverage=a['coverage'],native_coverage=b['coverage'],
                        minADE=a['minADE'],native_ADE=b['minADE'],minFDE=a['minFDE'],native_FDE=b['minFDE'],
                        rescue=s['rescue'],lost=s['lost'],lost_rate=s['lost_rate'],
                        motion=a['motion'],native_motion=b['motion']))
        primary=read(root/'PRIMARY_COMPARISON.json');gates=read(root/'PROTECTION_GATES.json')
        write(root/'ALL_GROUPS_REVIEW.json',dict(rows=groups,primary=primary,
            cost='NOT_MEASURED_RAW_STREAM_NO_DEPLOYMENT_PASS',same_assistant_ARS_not_external_review=True))
        report=['# 两方向联合训练：完整保存结果待人类阅读','',
            '固定12run/240epoch/162960更新，26人口。所有12final SHA锁后才开放dev评价；原科学源码未改。',
            '原F16+h3仅在existing方向冻结；CVAE为新prior/posterior/decoder/DSF联合训练，共用冻结观测context。',
            '', '|run|dev change good/960|native good|rescue|lost|ADE|FDE|', '|---|---:|---:|---:|---:|---:|---:|']
        for r in groups:
            if r['split']=='dev' and r['group']=='change':
                report.append(f"|{r['run']}|{r['good']}|{r['native_good']}|{r['rescue']}|{r['lost']}|{r['minADE']:.6f}|{r['minFDE']:.6f}|")
        report+=['','全部96seed/split/group和原五motion/保护门见ALL_GROUPS_REVIEW与PROTECTION_GATES。',
            '固定family30所有普通/校正/fixedseed CI完整保留，不用单个阳性替代原门。',
            'ARS DA2：加入监督不保证保留旧成功；CVAE重构与推理prior之间仍可能有差距，核所有救回/丢失与运动代价。',
            'ARS DA3：两结构的容量/可达集和共享旧观测encoder不同，不能声称纯结构因果、三独立whole-model泛化或SOTA；同助手检查非外审。',
            '成本为NOT_MEASURED_RAW_STREAM，不作部署PASS；场景/联合地图臂未执行，不当阴性。',
            '不自动升级baseline或补扫权重/LR/epoch/额外变体，所有模型/失败/日志/自动记录保留。']
        with (root/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
        write(root/'COMPLETION_REVIEW.json',dict(status='PASS_FINITE_12RUN_26POP_SAVED_SCIENCE_NOT_RAW_COST',at=now(),
            training_runs=12,epochs=240,updates=162960,group_rows=len(groups),
            all_saved_source_output_SHA_verified=True,production_science=sha(root/'FINAL_SCIENCE_SEAL.json'),
            infrastructure_files=ext['files'],scientific_thresholds_unchanged=True,
            no_full_model_or_gradient_replay_claim=True,elapsed_seconds=time.perf_counter()-start))
        roots=[root/n for n in ('FINAL_SCIENCE_SEAL.json','RESULT_ANALYSIS.md','ALL_GROUPS_REVIEW.json','COMPLETION_REVIEW.json',
            'PRIMARY_COMPARISON.json','BOOTSTRAP_DRAWS.npz','PROTECTION_GATES.json','RESOURCE.json','RESULT.json','ALL12_FINALS_LOCK.json')]
        write(root/'FINAL_WORKER_SEAL.json',dict(status='PASS_FINITE_SAVED_SCIENCE_COST_UNMEASURED',files=seal(roots),infrastructure=ext['files']))
        write(root/'WORKER_SUCCESS.json',dict(status='SUCCESS_SAVED_SCIENCE_COST_UNMEASURED',at=now(),elapsed_seconds=time.perf_counter()-start,
            seal_sha=sha(root/'FINAL_WORKER_SEAL.json'),new_formal_studies=2,baseline_upgraded=False))
        write(root/'WORKER_EXIT.json',dict(status='SUCCESS',exit_code=0,pid=os.getpid(),at=now()))
        mutable(root/'runtime_state.json',dict(status='COMPLETE_SAVED_SCIENCE_REVIEW_REQUIRED',at=now(),pid=os.getpid(),active_worker=False,
            raw_cost='NOT_MEASURED_NOT_DEPLOYMENT_PASS',no_automatic_extra_experiments=True))
    except BaseException as e:
        write(root/f'WORKER_FAILURE_{time.time_ns()}.json',dict(status='FAILURE',at=now(),error=repr(e),traceback=traceback.format_exc(),retain_all=True))
        write(root/f'WORKER_EXIT_FAILURE_{time.time_ns()}.json',dict(status='FAILURE',exit_code=1,pid=os.getpid(),at=now()))
        mutable(root/'runtime_state.json',dict(status='FAILURE_RETAIN_ALL',at=now(),pid=os.getpid(),active_worker=False,error=repr(e)))
        raise
    finally:sys.stdout=original

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
