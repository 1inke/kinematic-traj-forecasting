from train import *
def main():
    g=guard();f=HERE/'cost';f.mkdir(exist_ok=True);d=Data();reports=[]
    for arm in ARMS:
        for dev,batch in (('cpu',1),('cpu',32),('cuda',1),('cuda',32)):
            m,o,r=initialize(42,arm,d,dev);ck=torch.load(HERE/f'runs/{arm}_seed42/checkpoint.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);m.eval();x,n,y=d.batch(np.arange(batch),dev);values=[]
            with torch.no_grad():
                for i in range(550):
                    if dev=='cuda':torch.cuda.synchronize()
                    a=time.perf_counter();m(x,n)
                    if dev=='cuda':torch.cuda.synchronize()
                    if i>=50:values.append(time.perf_counter()-a)
            reports.append(dict(arm=arm,device=dev,batch=batch,CPU_threads=1,warm=50,measured=500,mean_seconds=float(np.mean(values)),p95_seconds=float(np.quantile(values,.95)),parameters=sum(q.numel() for q in m.parameters())));g.check()
    write_json(f/'COST_MODULE.json',dict(source=source(),records=reports,raw_end_to_end='PENDING_NOT_MEASURED',not_full_F16_native_latency=True,whole_unique_parameter_count='PENDING_UPSTREAM_CHECKPOINT_METADATA',parameter_budget_305000='NOT_CERTIFIED',no_KIN=True,no_lightweight_or_deployment_claim=True));resource(f/'RESOURCE.json',g);finish(f,dict(source=source(),cost='module_only'))
if __name__=='__main__':main()
