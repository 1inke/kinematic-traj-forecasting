"""Inherit the exact B0 step/snapshot/load; attach factor before AdamW/EMA use."""
from geo_common import *
import numpy as np
import baseline_factory as bf
import torch
from train_core import Trainer as B0Trainer,cpu_tree,exact,save_checkpoint,infer
from geo_modules import DGeoDecoder


class Trainer(B0Trainer):
    def __init__(self,data,indices,arm,batch=64,epochs=8,device='cuda',seed=42,check_B0_order=False):
        super().__init__(data,indices,batch,epochs,device,seed)
        shared=cpu_tree(self.model.state_dict())
        before=bf.rng_state(self.generator)
        devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(20261102+seed)
            self.model.decoder=DGeoDecoder(self.model.decoder,arm).to(device)
        self.ema=bf.ModelEMA(self.model,.995)
        self.optimizer,self.scheduler=bf.optimizer_scheduler(self.model,self.steps_per_epoch,epochs)
        for name,value in shared.items():exact(value,self.model.state_dict()[name],name)
        exact(before,bf.rng_state(self.generator),'ISOLATED_NEW_MODULE_CONSTRUCTOR_RNG')
        self.arm=arm
        self.expected_orders=read_json(BASE/'B0_seed42/batch_orders.json')['batch_IDs'] if check_B0_order else None
        if check_B0_order:
            if seed!=42 or batch!=64 or epochs!=8:raise ValueError('WRONG_SAVED_B0_PAIRING_SCOPE')
            initial=torch.load(BASE/'B0_seed42/initial.pt',map_location='cpu',weights_only=False)
            exact(shared,initial['model'],'ALL_ORIGINAL_B0_WEIGHTS_BUFFERS')
            exact(bf.rng_state(self.generator),initial['rng'],'INITIAL_SHARED_ALL_RNG')

    def step(self):
        row=super().step()
        if self.expected_orders is not None:
            exact(self.batch_orders[-1],self.expected_orders[self.updates-1],'PRODUCTION_B0_PAIRED_BATCH_ID_ORDER')
        return row
