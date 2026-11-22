"""One Colab-side study, from preflight to evidence and resumable checkpoints."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from advisor_decisions import FOLDS, SEEDS, confirm, run_path, screen
from advisor_model import ARMS, torch
from advisor_preflight import deterministic, structural
from prepare_inputs import digest
from run_b0 import verify_inputs, write_json

ROOT = Path(__file__).resolve().parent
ORDER = ("R2","R3","S2","S3","G1","G2","S4","S1","R1","G2-intra")


class Deadline(Exception):
    pass


def read(p):
    return json.loads(p.read_bytes())


def status(root,phase,**state):
    write_json(root/"CURRENT.json",dict(phase=phase,at_utc=datetime.now(timezone.utc).isoformat(),**state))


def usable(out, deep=False):
    p = out/"RESULT.json"
    if not p.is_file():
        return False
    val = read(p)
    if val["status"] != "PASS":
        return False
    return all((out/k).is_file() and (not deep or digest(out/k)==h) for k,h in val.get("artifacts",{}).items())


def launch(arm,fold,seed,out,bench):
    out.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,"-u","-B",str(ROOT/"advisor_train.py"),"--arm",arm,"--fold",fold,
               "--seed",str(seed),"--out",str(out)]
    if bench:
        command.append("--bench")
    env = {**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    with (out/"worker.log").open("a",encoding="utf-8") as log:
        result = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    return result


def stop_processes(active):
    for process in active:
        if process.poll() is None:
            process.terminate()
    for process in active:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def queue(root,stage,jobs,cap,admission,deadline,bench=False):
    todo = [j for j in jobs if not usable(j[-1],deep=not bench)]
    failed = [j[-1].name for j in todo if not bench and (j[-1]/"FAILURES.json").exists()
              and len(read(j[-1]/"FAILURES.json"))>=2]
    pending = [j for j in todo if j[-1].name not in failed]
    active = {}
    start = time.monotonic()
    count = 0
    last_progress = None
    try:
        while pending or active:
            now = time.monotonic()
            if now>=deadline:
                raise Deadline()
            while len(active)<cap and pending and now<admission:
                a,f,s,out = pending.pop(0)
                p = launch(a,f,s,out,bench)
                active[p]=(a,f,s,out)
            if not active:
                raise Deadline()
            progress = dict(planned=len(jobs),
                            completed_before=len(jobs)-len(pending)-len(active)-len(failed),
                            waiting=len(pending),
                            active=[dict(arm=a,fold=f,seed=s,child_pid=p.pid,run=out.name)
                                    for p,(a,f,s,out) in active.items()])
            if progress != last_progress:
                status(root,stage,**progress)
                last_progress = progress
            time.sleep(.5 if bench else 5)
            for p in list(active):
                rc = p.poll()
                if rc is None:
                    continue
                job = active.pop(p)
                out = job[-1]
                if rc!=0 or not usable(out):
                    history = out/"FAILURES.json"
                    old = read(history) if history.exists() else []
                    old.append(dict(exit_code=rc,at=datetime.now(timezone.utc).isoformat()))
                    write_json(history,old)
                    if not bench and len(old)<2:
                        pending.insert(0,job)
                    else:
                        failed.append(out.name)
                else:
                    count+=1
            if pending and not active and time.monotonic()>=admission:
                raise Deadline()
        status(root,stage,planned=len(jobs),completed_before=len(jobs)-len(failed),
               waiting=0,active=[],failures=failed)
    finally:
        stop_processes(active)
    return dict(seconds=time.monotonic()-start,completed=count,failed=failed)


def trial(root,arms,cap,tag,deadline):
    bench = ROOT/"_bench"/tag
    jobs = [(arm,"C0",42+i%3,bench/f"{arm}_{i}") for i,arm in enumerate(arms)]
    result = queue(root,tag,jobs,cap,deadline,deadline,bench=True)
    times = [read(j[-1]/"RESULT.json") for j in jobs if usable(j[-1])]
    result["measurements"] = times
    return result


def benchmarks(root,deadline,checks):
    records = {str(c):trial(root,["B0"]*c,c,f"PF6_B0_{c}",deadline) for c in range(1,7)}
    solo = records["1"]["measurements"][0]["seconds"]
    eligible = [(len(r["measurements"])/r["seconds"],int(k)) for k,r in records.items() if
                len(r["measurements"])==int(k) and
                max(m["seconds"] for m in r["measurements"])<=1.6*solo]
    cap = max(eligible)[1] if eligible else 1
    arms = {arm:trial(root,[arm],1,f"PF7_{arm}",deadline)["measurements"][0] for arm in ARMS}
    memory = int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free",
        "--format=csv,noheader,nounits"],text=True).splitlines()[0])*1024**2
    extra = max(m["peak_reserved_bytes"] for m in arms.values()) + 512*1024**2
    cap = min(cap,max(int(.90*memory)//extra,1))
    mixed = {}
    while cap>1:
        variants = ["G2","S4","R2"]
        tests = [variants[i%3] for i in range(cap)]
        result = trial(root,tests,cap,f"PF6_mixed_{cap}",deadline)
        mixed[str(cap)] = result
        if not result["failed"]:
            break
        cap-=1
    brief = {}
    for arm,a in arms.items():
        local = checks["CPU"][arm]
        increases = dict(parameters=a["parameters"]/arms["B0"]["parameters"]-1,
                         CPU_latency=local["CPU_single_sample_ms"]/checks["CPU"]["B0"]["CPU_single_sample_ms"]-1)
        brief[arm] = dict(parameters=a["parameters"],CPU_ms=local["CPU_single_sample_ms"],
                          L4_epoch_seconds=a["seconds"],L4_peak_reserved_bytes=a["peak_reserved_bytes"],
                          increases=increases,lightweight=increases["parameters"]<=.3 and increases["CPU_latency"]<=.5)
    write_json(root/"COST.json",dict(arms=brief))
    return dict(chosen=cap,B0=records,mixed=mixed,min_free_bytes=memory)


def build_report(out,admission,confirmation):
    markers = read(out/"stage1/RESULT.json")["markers"]
    active = [name for name,value in markers.items() if value]
    lines = ["# EXP-ADVISOR-3DIR", "", "C0 筛选和已准入的五折评价已完成。", "",
             "C0 达到的判据为 " + ("、".join(active) if active else "无") + "。",
             "五折比较臂为 " + ("、".join(admission) if admission else "无，已完成 B0 五折基准") + "。", ""]
    for arm,row in confirmation.items():
        result = "通过" if row["F-PASS"] else "未通过"
        lines.append(f"- {arm} 五折{result}，ETH 相对改善 {row['F-ETH']:+.2%}。")
    lines += ["", "[C0 结果](stage1/RESULT.json) · [五折结果](stage2/RESULT.json) · [预检](PREFLIGHT.json)", "",
              "C0 dev 已多次使用。ETH 折与 C0 dev 没有相同窗口。三个方向的证据限于定义的模型和训练预算。", ""]
    (out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def work(root, hours):
    input_hashes = verify_inputs(ROOT)
    fp = ROOT/"fivefold_inputs.manifest.json"
    for name,wanted in read(fp)["members"].items():
        assert digest(ROOT/name)==wanted, name
    root.mkdir(parents=True,exist_ok=True)
    device = torch.cuda.get_device_name(0)
    assert "L4" in device.split(), f"Expected NVIDIA L4, got {device}"
    print(f"Study runtime: {device}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}", flush=True)
    began = time.monotonic()
    admit_until = began + hours*3600
    deadline = admit_until + min(30*60,hours*3600*.1)
    checkpoints = root/"PREFLIGHT.json"
    status(root,"PREFLIGHT_PF1_PF3_PF4_PF5_PF8_PF9")
    code_files = [*sorted(ROOT.glob("advisor_*.py")), ROOT/"run_b0.py"]
    binding = dict(inputs=input_hashes,code={p.name:digest(p) for p in code_files},
                   fivefold_manifest=digest(fp))
    checks = read(checkpoints) if checkpoints.exists() else {}
    if checks:
        previous = checks["binding"]
        assert previous["inputs"] == binding["inputs"] and previous["fivefold_manifest"] == binding["fivefold_manifest"], (
            "Study evidence was produced from different inputs."
        )
        if previous != binding:
            mutable_runner = {"advisor_study.py", "run_b0.py"}
            previous_protocol = {name: h for name,h in previous["code"].items() if name not in mutable_runner}
            current_protocol = {name: h for name,h in binding["code"].items() if name not in mutable_runner}
            assert previous_protocol == current_protocol, "Study model, training or analysis source changed."
            print("Study runner updated; repeating structural preflight before resuming.", flush=True)
            checks = {}
    if not checks:
        checks = dict(binding=binding,PF1="VERIFIED", **deterministic(), CPU=structural("cpu"))
        checks["GPU"]=structural("cuda:0",windows=16,latency=False)
        write_json(checkpoints,checks)
    b0 = run_path(root,"B0","C0",42)
    if usable(b0, deep=True):
        saved = read(b0/"RESULT.json")["contract"]
        assert saved["torch"] == str(torch.__version__) and saved["cuda"] == torch.version.cuda, (
            "Resume with the same PyTorch and CUDA versions as the saved B0."
        )
        print("Verified completed B0 seed 42; reusing its 64-epoch result.", flush=True)
    jobs = [("B0","C0",42,b0)]
    ended = queue(root,"PF2_B0_S42",jobs,1,admit_until,deadline)
    if ended["failed"]:
        raise RuntimeError(f"B0 PF2 failed: {ended['failed']}")
    archive = read(ROOT/"reference/B0_s42_dev_metrics.json")["overall"]["minFDE"]
    measured = read(jobs[0][-1]/"RESULT.json")["dev"]["overall"]["minFDE"]
    checks["PF2"] = dict(archive10=archive,L4=measured,relative_delta=measured/archive-1,
                          within_1_5_percent=abs(measured/archive-1)<=.015,
                          controls="Three matched L4 B0 runs for all comparisons")
    write_json(checkpoints,checks)
    if "PF6" not in checks:
        checks["PF6"] = benchmarks(root,admit_until,checks)
    checks["status"] = "PASS"
    write_json(checkpoints,checks)
    run_phases(root,checks["PF6"]["chosen"],admit_until,deadline)


def run_phases(root,cap,admit_until,deadline,sensitivity=True):
    arms = ("B0",*ORDER)
    schedule=[(a,"C0",s,run_path(root,a,"C0",s)) for a in arms for s in SEEDS]
    stage1 = queue(root,"STAGE1_C0",schedule,cap,admit_until,deadline)
    if stage1["failed"]:
        raise RuntimeError(f"Stage 1 failed for: {stage1['failed']}")
    status(root,"STAGE1_SUMMARY",completed=len(schedule))
    admission = screen(root)
    stage2_out = [(a,f,s,run_path(root,a,f,s)) for f in FOLDS for a in ("B0",*admission) for s in SEEDS]
    for fold in FOLDS:
        batch = [j for j in stage2_out if j[1]==fold]
        ended = queue(root,f"STAGE2_{fold}",batch,cap,admit_until,deadline)
        if ended["failed"]:
            raise RuntimeError(f"Fold {fold} failed for: {ended['failed']}")
    result = confirm(root,admission)
    build_report(root,admission,result)
    runtime = root/"LOCAL_RUNTIME.json"
    if runtime.is_file():
        report = root/"REPORT.md"
        with report.open("a",encoding="utf-8") as stream:
            stream.write("\nC0 汇总包含 Colab L4 已完成的运行，以及本机 GPU 新完成或从 L4 检查点续跑的运行。"
                         "C0 的跨设备比较用于探索。各组环境见 RESULT.json 的 contract 字段，"
                         "跨设备续跑组还保留 CLOUD_CONTRACT.json。"
                         "本机并发和环境见 [LOCAL_RUNTIME.json](LOCAL_RUNTIME.json)。\n")
    # Optional sensitivity equations leave the screening and admission receipts untouched.
    if sensitivity and admit_until-time.monotonic()>40*60:
        sensitivity=[(a,"C0",s,root/"sensitivity"/f"{a}_s{s}")
                     for a in ("R2-half","S4-wide") for s in SEEDS]
        queue(root,"OPTIONAL_SENSITIVITY",sensitivity,cap,admit_until,deadline)
    runtime_files = ("PREFLIGHT.json","COST.json")
    if runtime.is_file():
        runtime_files = ("LOCAL_RUNTIME.json","cloud/PREFLIGHT.json","cloud/COST.json")
    files = (*runtime_files,"stage1/RESULT.json","stage1/MECH_R.json","stage1/G1_compression.json",
             "stage2/ADMISSION.json","stage2/RESULT.json","REPORT.md")
    write_json(root/"DONE.json",dict(status="PASS",stage1=len(schedule),stage2=len(stage2_out),
        files={name:digest(root/name) for name in files}))
    status(root,"DONE",stage1=len(schedule),stage2=len(stage2_out))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--hours",type=float,default=23.)
    args = parser.parse_args()
    try:
        work(args.out,args.hours)
    except Deadline:
        status(args.out,"TIME_LIMIT_RESUMABLE")
        raise SystemExit(75)
