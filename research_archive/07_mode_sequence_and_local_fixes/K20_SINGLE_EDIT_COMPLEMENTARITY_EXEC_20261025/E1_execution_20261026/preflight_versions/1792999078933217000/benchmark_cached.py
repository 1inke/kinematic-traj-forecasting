"""Cached editor cost only; never mislabeled as full F16+KIN+400-check end-to-end."""
import time,numpy as np,torch
from env import *
from data_pipeline import OptData
from train_paired import initialize
def run():
    data=OptData(42);rows=[]
    for device,batch in (('cpu',1),('cuda',1),('cuda',32)):
        model,_,_=initialize(42,data,device);model.load_state_dict(torch.load(HERE/'runs/S_seed42/checkpoint.pt',map_location='cpu',weights_only=False)['model']);model.eval();x,v,t=data.batch(np.arange(batch),device)
        samples=[]
        with torch.no_grad():
            for i in range(550):
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();model(x)
                if device=='cuda':torch.cuda.synchronize()
                if i>=50:samples.append(time.perf_counter()-start)
        rows.append(dict(device=device,batch=batch,warm50_measure500=True,seconds_mean=float(np.mean(samples)),seconds_p50=float(np.median(samples)),scope='cached observable401action features -> editor scores only',CPU_threads=1))
    result=dict(status='CACHED_ONLY_NOT_END_TO_END',rows=rows,editor_parameters=11488,total_unique_parameters='PENDING_REAL_F16_KIN_INVENTORY',budget=305000,budget_pass=False,real_end_to_end_status='PENDING_EXPLICIT_RAW_OBSERVATION_PIPELINE_BENCHMARK',real_end_to_end_required_components=['F16 encoder/decoder including neighbours and normalization','KIN generation','400 GT-free motion checks and features','transfer and editor'],no_claim_of_deployability=True)
    write_json(HERE/'COST_CACHED.json',result);return result
