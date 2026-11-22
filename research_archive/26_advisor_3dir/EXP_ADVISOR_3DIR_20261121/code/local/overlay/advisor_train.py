"""Train and evaluate one declared advisor run. No scheduler side effects on import."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from advisor_eval import save_evaluation
from advisor_model import ARMS, AdvisorDecoder, attach, advisor_loss
from prepare_inputs import digest
from run_b0 import checkpoint, latest, load_data, publish, write_json

ROOT = Path(__file__).resolve().parent
BASE = ROOT/"qoder_work/NEXT_PHASE_EXEC_20261114"
sys.path.insert(0,str(BASE))
import common
common.ASTRA = ROOT/"ASTRA_ACCURACY_Q20"
import train_core as core

CLOUD_TRAINER_SHA = "7768beb8b2749d52edff7e5617ac76f62025acf2912f797d3832d7ec7f2ae52f"


class Trainer(core.Trainer):
    def __init__(self, data, arm, seed, device, epochs):
        super().__init__(data,np.arange(len(data["ids"])),batch=64,epochs=epochs,
                         device=device,seed=seed,order_seed=seed,arm="B0")
        self.advisor = arm
        if arm=="S1":
            self.loss_fn.winner_fde_weight = 1.
        elif arm!="B0":
            attach(self.model,arm)
            self.ema = core.ModelEMA(self.model,.995)
            self.optimizer,self.scheduler = core.optimizer_scheduler(self.model,self.steps_per_epoch,epochs)

    def step(self):
        if self.order is None or self.cursor==len(self.order):
            if self.epoch>=self.epochs:
                raise StopIteration
            self.order = self.indices[torch.randperm(len(self.indices),generator=self.generator).numpy()]
            self.cursor,self.epoch = 0,self.epoch+1
        ids = self.order[self.cursor:self.cursor+self.batch]
        self.model.train()
        past,future,tokens = (torch.as_tensor(self.data[k][ids],device=self.device)
                              for k in ("past","future","num_valid"))
        pred,logits,target,_ = core.prediction(self.model,past,tokens,future,augment=True)
        loss,detail,matched = advisor_loss(core,self.model,self.advisor,pred,target,logits,self.loss_fn)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is nonfinite.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                               5.,error_if_nonfinite=True)
        self.optimizer.step()
        self.ema.update(self.model)
        self.scheduler.step()
        self.updates += 1
        self.cursor += len(ids)
        self.batch_orders.append(self.data["ids"][ids].tolist())
        row = dict(epoch=self.epoch, update=self.updates, cursor=self.cursor, loss=float(loss.detach()),
                   regression=float(detail["regression"]), classification=float(detail["classification"]),
                   grad_norm=float(grad), lr=self.optimizer.param_groups[0]["lr"])
        if matched is not None:
            row["emta_matches"] = int(matched.sum())
            row["emta_examples"] = len(matched)
        self.history.append(row)
        return row


@torch.no_grad()
def infer(model,data):
    model.eval()
    chunks = []
    device = next(model.parameters()).device
    for i in range(0,len(data["ids"]),64):
        sl = slice(i,i+64)
        p,n = (torch.as_tensor(data[k][sl],device=device) for k in ("past","num_valid"))
        out,z,_,tf = core.prediction(model,p,n)
        a = model.decoder.aux if isinstance(model.decoder,AdvisorDecoder) else None
        paths = core.world_paths(out,tf).double().cpu().numpy()
        first = (core.world_paths(a[0],tf).double().cpu().numpy(),a[1][:,0].cpu().numpy()) if a else (None,None)
        chunks.append((paths,z[:,0].cpu().numpy(),*first))
    return tuple(np.concatenate(col) if col[0] is not None else None for col in zip(*chunks))


def inputs(fold):
    if fold=="C0":
        return [(BASE/f"C0/{name}.npz",num) for name,num in (("fit",21719),("dev",4753))]
    audit = json.loads((ROOT/"fivefold/source_audit.json").read_bytes())
    assert audit["recording_isolation_passed"]
    population = audit["folds"][fold]["splits"]
    return [(ROOT/f"fivefold/{fold}/{name}.npz",population[name]["windows"]) for name in ("train","test")]


def evaluate_run(out, trainer, arm, fit, dev):
    with tempfile.TemporaryDirectory(prefix="advisor_final_") as tmp:
        weights = Path(tmp)/"final_ema.pt"
        torch.save(core.cpu_tree(trainer.ema.module.state_dict()),weights)
        weights_sha = publish(weights,out/weights.name)
    result = save_evaluation(out,arm,lambda d:infer(trainer.ema.module,d),fit,dev)
    result["artifacts"]["final_ema.pt"] = weights_sha
    return result


def compatible_colab(saved, current):
    """Require the full original protocol, allowing only the known runtime transition."""
    if current.get("gpu_name") is None:
        return False
    legacy = {key: value for key, value in current.items() if key != "gpu_name"}
    legacy.update(torch="2.14.1+cu130", cuda="13.0",
                  source={**current["source"], "advisor_train.py": CLOUD_TRAINER_SHA})
    return saved == legacy


class ColabCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.multiarray" and name == "_reconstruct":
            module = "numpy.core.multiarray"
        return super().find_class(module, name)


def restore(out, trainer, contract, local_gpu):
    if not local_gpu:
        return latest(out,core,trainer,contract)
    loader = SimpleNamespace(__name__="colab_checkpoint_pickle", Unpickler=ColabCheckpointUnpickler)
    return latest(out,core,trainer,contract,
                  accept_contract=lambda saved: compatible_colab(saved,contract), pickle_module=loader)


def main(args):
    torch.set_num_threads(1)
    sources = inputs(args.fold)
    fit,dev = (load_data(p,n) for p,n in sources)
    if args.smoke:
        fit,dev = ({k: v[:8] for k,v in d.items()} for d in (fit,dev))
    epochs = 2 if args.smoke else (1 if args.bench else 64)
    device = "cpu" if args.smoke else "cuda:0"
    local_gpu = not args.smoke and os.environ.get("ADVISOR_LOCAL_GPU") == "1"
    if not args.smoke:
        assert torch.cuda.is_available(), "CUDA GPU required"
        gpu_name = torch.cuda.get_device_name(0)
        assert local_gpu or "L4" in gpu_name.split(), f"Expected NVIDIA L4, got {gpu_name}"
    checkpoint_interval = 1 if args.smoke else 8
    args.out.mkdir(parents=True,exist_ok=True)
    contract = dict(arm=args.arm,seed=args.seed,order_seed=args.seed,fold=args.fold,epochs=epochs,
                    device=device,torch=str(torch.__version__),cuda=torch.version.cuda,batch=64,gradient_clip=5,
                    training_inputs={p.name:digest(p) for p,_ in sources},
                    source={name:digest(ROOT/name) for name in ("advisor_model.py","advisor_train.py","advisor_eval.py")})
    contract["baseline_source"] = digest(ROOT/"baseline_sources.manifest.json")
    if local_gpu:
        contract["gpu_name"] = gpu_name
    spec = args.out/"CONTRACT.json"
    if spec.exists():
        saved = json.loads(spec.read_bytes())
        if saved != contract:
            assert compatible_colab(saved, contract), f"Run contract changed: {spec}"
            write_json(args.out/"CLOUD_CONTRACT.json",saved)
            write_json(spec,contract)
    else:
        write_json(spec,contract)
    final = args.out/"RESULT.json"
    if final.exists():
        r = json.loads(final.read_bytes())
        assert r["contract"]==contract
        if all((args.out/n).is_file() and digest(args.out/n)==h for n,h in r.get("artifacts",{}).items()):
            return
    trainer = Trainer(fit,args.arm,args.seed,device,epochs)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    start = 0 if args.bench else restore(args.out,trainer,contract,local_gpu)
    write_json(args.out/"CURRENT.json",dict(status="TRAINING",pid=os.getpid(),epoch=start,updates=trainer.updates))
    for epoch in range(start+1,epochs+1):
        t = time.monotonic()
        stats = [trainer.step() for _ in range(trainer.steps_per_epoch)]
        row = dict(epoch=epoch,seconds=time.monotonic()-t,updates=trainer.updates,
                   mean_loss=float(np.mean([x["loss"] for x in stats])),
                   emta_matches=sum(x.get("emta_matches",0) for x in stats),
                   emta_examples=sum(x.get("emta_examples",0) for x in stats))
        write_json(args.out/f"epoch_{epoch:03d}_timing.json",row)
        write_json(args.out/"CURRENT.json",dict(status="TRAINING",pid=os.getpid(),**row))
        if not args.bench and (epoch%checkpoint_interval==0 or epoch==epochs):
            checkpoint(args.out,core,trainer,contract)
        print(f"{args.arm} {args.fold} s{args.seed}: {row}",flush=True)
        if args.stop_at and epoch==args.stop_at:
            return
    result = dict(status="PASS",contract=contract,epochs=epochs,updates=trainer.updates,
                  seconds=time.monotonic()-begin,parameters=sum(x.numel() for x in trainer.model.parameters()),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved() if not args.smoke else 0)
    if (args.out/"CLOUD_CONTRACT.json").exists():
        result["resumed_from_colab"] = True
    if not args.bench:
        result.update(evaluate_run(args.out,trainer,args.arm,fit,dev))
    write_json(final,result)
    write_json(args.out/"CURRENT.json",dict(status="COMPLETE",epoch=epochs,updates=trainer.updates))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=(*ARMS,"R2-half","S4-wide"),required=True)
    parser.add_argument("--fold",choices=("C0","eth","hotel","univ","zara01","zara02"),default="C0")
    parser.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    parser.add_argument("--out",type=Path,required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke",action="store_true")
    mode.add_argument("--bench",action="store_true")
    parser.add_argument("--stop-at",type=int)
    main(parser.parse_args())

