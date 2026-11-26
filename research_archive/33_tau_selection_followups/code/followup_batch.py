"""Run one independent group of follow-ups with paired metrics and recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from tau_followups import (ANCHOR, BATCHES, E1, E2, E3, E5, SEEDS, SETUPS, data_paths, digest,
                           import_prior, load_data, read, validate_result, write_json)
from tracks_analysis import cluster_draws, interval, ratio, summarize_group

SCRIPT = Path(__file__).with_name("tau_followups.py")
METRICS = ("minFDE", "minADE", "Top1_FDE", "Top1_ADE")


def worker(args, gpu, command, *extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, "-u", str(SCRIPT), command, "--data", str(args.data), *extra],
                   env=env, check=True)


def evaluate(args, names, ids):
    runs = args.out/"runs"
    groups = None
    by_name = {}
    for name in names:
        metrics = {metric: [] for metric in METRICS}
        for seed in SEEDS:
            path = runs/f"{name}_s{seed}"
            assert validate_result(path, name, seed, ids)
            with np.load(path/"METRICS.npz", allow_pickle=False) as z:
                if groups is None:
                    groups = z["tracks"]
                np.testing.assert_array_equal(groups, z["tracks"])
                for key in metrics:
                    metrics[key].append(z[key])
        by_name[name] = {key: np.mean(v, axis=0) for key,v in metrics.items()}
    all_rows = np.ones(len(ids), dtype=bool)
    def pair(base, arm):
        return {key: summarize_group(by_name[base][key], by_name[arm][key]-by_name[base][key],
                                     groups, all_rows) for key in METRICS}
    def contrasts(base, delta):
        _, lookup, counts = cluster_draws(groups)
        b = np.bincount(lookup, weights=base)
        d = np.bincount(lookup, weights=delta)
        values = ratio(counts@d, counts@b)
        return dict(absolute_m=float(delta.mean()), relative=float(d.sum()/b.sum()),
                    ci95=interval(values, .95))
    def vector(name):
        return by_name[name]["minFDE"]
    means = {name:{key:float(value.mean()) for key,value in metrics.items()}
             for name,metrics in by_name.items()}
    if args.batch == "selection":
        e1 = {control:pair(f"{control}_100",ANCHOR) for control in E1 if control!="tau"}
        e1_gates = {name:row["minFDE"]["ci95"][1] <= 0 for name,row in e1.items()}
        e2 = {}
        _, codes, counts = cluster_draws(groups)
        def relative_gain(control, percent):
            base = vector(f"{control}_{percent}")
            delta = base-vector(f"tau_{percent}")
            b = np.bincount(codes, weights=base)
            d = np.bincount(codes, weights=delta)
            return float(d.sum()/b.sum()), ratio(counts@d, counts@b)
        for name in ("distance","top_p"):
            g25, d25 = relative_gain(name,25)
            g50, d50 = relative_gain(name,50)
            g100, d100 = relative_gain(name,100)
            e2[name] = dict(gain={25:g25,50:g50,100:g100},
                            ci95_gain={25:interval(d25,.95),50:interval(d50,.95),100:interval(d100,.95)},
                            change_25_vs_100=g25-g100,ci95_change=interval(d25-d100,.95),
                            meets_1pp=(g25-g100)>=.01)
        write_json(args.out/"E1_RESULT.json", dict(comparisons=e1, upper_ci_at_most_zero=e1_gates,
                                                    means={k:v for k,v in means.items() if k.endswith("_100")}))
        write_json(args.out/"E2_RESULT.json",dict(comparisons=e2,means=means))
    else:
        base, state, sparse, both = (f"encoder_{arm}" for arm in E3)
        factorial = dict(state_only=pair(base,state), sparse_only=pair(base,sparse),
                         state_given_sparse=pair(sparse,both), sparse_given_state=pair(state,both))
        interaction = contrasts(vector(base),vector(both)-vector(state)-vector(sparse)+vector(base))
        write_json(args.out/"E3_RESULT.json", dict(comparisons=factorial, minFDE_interaction=interaction,
                                                    means={k:v for k,v in means.items() if k.startswith("encoder_")}))
        grid = {name:ANCHOR if name=="r06_t24" else f"sens_{name}" for name in E5}
        gap = (max(means[k]["minFDE"] for k in grid.values()) -
               min(means[k]["minFDE"] for k in grid.values()))/means[ANCHOR]["minFDE"]
        write_json(args.out/"E5_RESULT.json", dict(means={name:means[k] for name,k in grid.items()},
                    comparisons={name:pair(ANCHOR,k) for name,k in grid.items() if k!=ANCHOR},
                    relative_range=gap, below_1_percent=gap<.01))


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if args.l4:
        assert len(devices) == 1 and "L4" in devices[0].split(), devices
    else:
        assert devices == ["Tesla T4", "Tesla T4"], devices
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        checks = [pool.submit(worker,args,i,"preflight","--out",str(args.out/f"preflight_{i}"))
                  for i in range(len(devices))]
        for check in checks:
            check.result()
    print(f"GPU_PREFLIGHT_PASS {devices}", flush=True)
    elapsed = args.budget_hours * 3600
    deadline = time.time() + elapsed
    names = BATCHES[args.batch]
    dev = load_data(*data_paths(args.data,"C0",True))
    ids = dev["ids"]
    tasks = []
    for seed in SEEDS:
        for name in names:
            out = args.out/"runs"/f"{name}_s{seed}"
            if validate_result(out,name,seed,ids):
                assert read(out/"RESULT.json")["contract"]["protocol"]==args.protocol
                continue
            if name == ANCHOR and args.prior is not None:
                import_prior(args.prior,out,args.data,seed,args.original_assets,args.protocol)
            else:
                tasks.append((seed,name,out))
    write_json(args.out/"PLAN.json",dict(batch=args.batch,protocol=args.protocol,gpus=devices,
                                       tasks=[out.name for _,_,out in tasks]))
    lanes = (0,)*6 if args.l4 else (0,1,0,1)
    tasks.sort(key=lambda task: -SETUPS[task[1]]["pct"])
    def lane(gpu, assigned):
        for seed,name,out in assigned:
            if time.time() >= deadline:
                return
            print(f"START {out.name} physical_gpu={gpu}",flush=True)
            worker(args,gpu,"train","--name",name,"--seed",str(seed),"--out",str(out),
                   "--protocol",args.protocol,"--deadline",str(deadline))
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        jobs = [pool.submit(lane,gpu,tasks[i::len(lanes)]) for i,gpu in enumerate(lanes)]
        for job in jobs:
            job.result()
    complete = all(validate_result(args.out/"runs"/f"{name}_s{seed}",name,seed,ids)
                   for name in names for seed in SEEDS)
    if complete:
        evaluate(args,names,ids)
    status = "COMPLETE" if complete else "TIME_LIMIT_RESUMABLE"
    write_json(args.out/"STATUS.json",dict(status=status,batch=args.batch,protocol=args.protocol))
    print(f"FOLLOWUP_STATUS {args.batch} {status}",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--batch",choices=BATCHES,required=True)
    p.add_argument("--protocol",required=True)
    p.add_argument("--original-assets",required=True)
    p.add_argument("--prior",type=Path)
    p.add_argument("--budget-hours",type=float,default=7.75)
    p.add_argument("--l4",action="store_true")
    run(p.parse_args())
