"""Train one QRANK arm using the archive-26 baseline and durable checkpoints."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from advisor_train import core, inputs, load_data
from prepare_inputs import digest
from qrank_eval import evaluate
from qrank_model import ARMS, attach, ml_loss, observed_speed
from run_b0 import checkpoint, latest, publish, write_json


ROOT = Path(__file__).resolve().parent


class Trainer(core.Trainer):
    def __init__(self, data, arm, seed, device, epochs, use_kin=True, batch=64):
        super().__init__(data,np.arange(len(data["ids"])),batch=batch,epochs=epochs,device=device,seed=seed,order_seed=seed)
        self.qrank_arm = arm
        self.supervise_quality = True
        if arm!="B0":
            attach(self.model, arm)
            decoder = self.model.decoder
            decoder.use_kin = use_kin
            if decoder.order_kind=="rand":
                decoder.train_generator = torch.Generator(device=device).manual_seed(seed+1000)
            self.ema = core.ModelEMA(self.model,.995)
            self.optimizer,self.scheduler = core.optimizer_scheduler(self.model,self.steps_per_epoch,epochs)
        quality = tuple(self.model.decoder.quality.parameters()) if arm!="B0" else ()
        self.quality_parameters = quality
        head = {id(p) for p in quality}
        self.trunk_parameters = [p for p in self.model.parameters() if p.requires_grad and id(p) not in head]

    def snapshot(self, contract):
        state = super().snapshot(contract)
        g = getattr(self.model.decoder,"train_generator",None)
        if g is not None:
            state["rank_rng"] = g.get_state().cpu()
        return state

    def load(self,state,contract):
        super().load(state,contract)
        g = getattr(self.model.decoder,"train_generator",None)
        if g is not None:
            g.set_state(state["rank_rng"].cpu())

    def step(self):
        if self.order is None or self.cursor==len(self.order):
            if self.epoch>=self.epochs:
                raise StopIteration
            self.order = self.indices[torch.randperm(len(self.indices),generator=self.generator).numpy()]
            self.cursor,self.epoch = 0,self.epoch+1
        ids = self.order[self.cursor:self.cursor+self.batch]
        self.model.train()
        p,f,n = (torch.as_tensor(self.data[k][ids],device=self.device) for k in ("past","future","num_valid"))
        paths,logits,target,_ = core.prediction(self.model,p,n,f,augment=True)
        if self.qrank_arm=="B0":
            loss,detail = core.loss_value(paths,target,logits,self.loss_fn)
            layers = torch.tensor([[float(loss.detach()),float(detail["regression"]),
                                    float(detail["classification"]),0.]],device=self.device)
            hinge, hits = loss.new_zeros(()),None
        else:
            loss,hits,layers,hinge = ml_loss(core,self.model,self.qrank_arm,target,self.loss_fn,self.supervise_quality)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite QRANK loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.trunk_parameters,5.,error_if_nonfinite=True)
        if self.quality_parameters:
            torch.nn.utils.clip_grad_norm_(self.quality_parameters,5.,error_if_nonfinite=True)
        self.optimizer.step()
        self.ema.update(self.model)
        self.scheduler.step()
        self.updates += 1
        self.cursor += len(ids)
        self.batch_orders.append(self.data["ids"][ids].tolist())
        row = dict(loss=float(loss.detach()),grad_norm=float(grad),hinge=float(hinge),layers=layers.detach().tolist())
        if hits is not None:
            speed = observed_speed(self.model.decoder.past)
            slow,fast = speed<=1.04,speed>1.3
            mask = torch.stack((torch.ones_like(slow),slow,~(slow|fast),fast))
            row["matched"] = torch.stack(((mask&hits.any(1)).sum(1),mask.sum(1)),1).tolist()
        self.history.append(row)
        return row


def emit_log(out, epochs):
    with tempfile.TemporaryDirectory(prefix="qrank_log_") as temp:
        path = Path(temp)/"training_log.jsonl"
        path.write_text("".join(json.dumps(row)+"\n" for row in epochs),encoding="utf-8")
        return publish(path,out/path.name)


def main(args):
    torch.set_num_threads(1)
    if not args.smoke:
        assert torch.cuda.is_available() and "L4" in torch.cuda.get_device_name(0).split()
    device = "cpu" if args.smoke else "cuda:0"
    sources = inputs(args.fold)
    fit,dev = (load_data(p,n) for p,n in sources)
    if args.smoke:
        fit,dev = ({k:v[:12] for k,v in d.items()} for d in (fit,dev))
    epochs = 2 if args.smoke else 1 if args.bench else 64
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    use_kin = not args.drop_kin
    contract = dict(arm=args.arm,fold=args.fold,seed=args.seed,order_seed=args.seed,epochs=epochs,
                    batch=64,gradient_clip=5,ema=.995,device=device,use_kin=use_kin,torch=str(torch.__version__),
                    cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if not args.smoke else None,
                    data={p.name:digest(p) for p,_ in sources},
                    code={p.name:digest(p) for p in sorted(ROOT.glob("*.py"))})
    spec = out/"CONTRACT.json"
    if spec.exists() and json.loads(spec.read_bytes())!=contract:
        raise ValueError(f"Runtime or source contract changed: {spec}")
    write_json(spec,contract)
    t = Trainer(fit,args.arm,args.seed,device,epochs,use_kin)
    if not args.smoke:
        torch.cuda.reset_peak_memory_stats()
    start = 0 if args.bench else latest(out,core,t,contract)
    began = time.monotonic()
    history = [json.loads(p.read_bytes()) for p in sorted(out.glob("epoch_*_timing.json"))
               if int(p.name.split("_")[1])<=start]
    for epoch in range(start+1,epochs+1):
        b = time.monotonic()
        rows = [t.step() for _ in range(t.steps_per_epoch)]
        metrics = {k:float(np.mean([r[k] for r in rows])) for k in ("loss","grad_norm","hinge")}
        metrics["layers"] = np.mean([r["layers"] for r in rows],axis=0).tolist()
        if "matched" in rows[0]:
            metrics["emta_counts"] = dict(zip(("overall","slow","mid","fast"),np.sum([r["matched"] for r in rows],axis=0).tolist()))
        r = dict(epoch=epoch,seconds=time.monotonic()-b,updates=t.updates,**metrics)
        write_json(out/f"epoch_{epoch:03d}_timing.json",r)
        history.append(r)
        if not args.bench and (args.smoke or epoch%8==0 or epoch==epochs):
            checkpoint(out,core,t,contract)
        write_json(out/"CURRENT.json",dict(status="TRAINING",pid=os.getpid(),**r))
        print(f"{args.arm} {args.fold} s{args.seed}: {r}",flush=True)
        if args.stop_at==epoch:
            return
    is_deep = args.arm!="B0"
    parameters = sum(p.numel() for p in t.model.parameters())
    unused = 0
    if is_deep and t.model.decoder.order_kind not in ("risk","qual"):
        unused += sum(p.numel() for p in t.model.decoder.quality.parameters())
    if is_deep and t.model.decoder.order_kind=="none":
        unused += t.model.decoder.rank.numel()
    result = dict(status="PASS",contract=contract,epochs=epochs,updates=t.updates,seconds=time.monotonic()-began,
                  total_training_seconds=sum(r["seconds"] for r in history),parameters=parameters,
                  inference_parameters=parameters-unused,peak_reserved_bytes=torch.cuda.max_memory_reserved() if not args.smoke else 0,
                  final_epoch=history[-1])
    if not args.bench:
        result.update(evaluate(core,t.ema.module,dev,out,args.fold))
        with tempfile.TemporaryDirectory(prefix="qrank_weights_") as tmp:
            path = Path(tmp)/"final_ema.pt"
            torch.save(core.cpu_tree(t.ema.module.state_dict()),path)
            result["artifacts"][path.name] = publish(path,out/path.name)
        result["artifacts"]["training_log.jsonl"] = emit_log(out,history)
    write_json(out/"RESULT.json",result)
    write_json(out/"CURRENT.json",dict(status="COMPLETE",epoch=epochs,updates=t.updates))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=("B0",*ARMS),required=True)
    p.add_argument("--fold",choices=("C0","eth","hotel","univ","zara01","zara02"),default="C0")
    p.add_argument("--seed",type=int,choices=range(42,47),required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--drop-kin",action="store_true")
    p.add_argument("--bench",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--stop-at",type=int)
    main(p.parse_args())
