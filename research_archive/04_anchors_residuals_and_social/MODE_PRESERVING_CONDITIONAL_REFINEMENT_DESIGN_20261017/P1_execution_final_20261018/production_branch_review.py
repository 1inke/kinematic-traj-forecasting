"""Extra current-source production-branch regression, no scientific changes."""
import common as c,model as m,objectives as o,metrics as v,train
import numpy as np,torch,ast,time
def main():
    import io_retry
    io_retry.install()
    retry_review=io_retry.fixture(c.H/'write_retry_fixture');c.write('WRITE_RETRY_REVIEW.json',retry_review)
    start=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS';files=c.read(c.H/'source_manifest.json')['files'];c.verify(files);data=c.load();st=c.npz(c.H/'statistics_position.npz');labels=c.npz(c.H/'responsibility_labels.npz');ix=data['split']['FIT'];assert np.array_equal(labels['sample_id'],data['ids'][ix]);fixed=[]
    for first in range(0,len(ix),256):
        ii=ix[first:first+256];base=m.inputs(data,ii)[1];gt=torch.from_numpy(data['gt'][ii]).to(m.DEVICE);fixed.append(o.quality(base,gt).argmin(1).cpu().numpy())
    assert np.array_equal(np.concatenate(fixed),labels['owner']),'TORCH_CPU_NATIVE_OWNER_MISMATCH'
    # Execute verbatim actual train.main manifest expression, all12 seed/arm configurations.
    tree=ast.parse((c.H/'train.py').read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main');node=next(x.value for x in ast.walk(main) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='manifest' for t in x.targets));expr=compile(ast.Expression(body=node),str(c.H/'train.py'),'eval')
    manifests=[]
    for seed in m.SEEDS:
        for cell in m.CELLS:
            source=pre['source_digest'];man=eval(expr,dict(c=c,data=data,seed=seed,cell=cell,source=source))
            assert man['statistics_SHA']==c.sha(c.H/'statistics_position.npz') and man['owner_labels_SHA']==c.sha(c.H/'responsibility_labels.npz')
            assert man['anchor']==(cell in ('C','D')) and man['owner_rule']==('fixed' if cell in ('B','D') else 'dynamic');manifests.append(man)
    jobs=[]
    for cell in m.CELLS:
        m.seed(42);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);m.step(net,opt,data,ix[:256],cell)
        jj=data['split']['outer'][:128];row=train.save_job('production_branch_smoke/'+cell,net,data,jj,cell,pre['source_digest']);c.valid_done(row,pre['source_digest']);v.validate(c.npz(row['output']),c.read(row['metrics']),c.npz(row['aux']),'integrated_acceleration');jobs.append(row)
    row=train.save_job('production_branch_smoke/native',None,data,data['split']['outer'][:128],'native',pre['source_digest']);v.validate(c.npz(row['output']),c.read(row['metrics']),c.npz(row['aux']),'native20');jobs.append(row)
    c.verify(files);c.write('PRODUCTION_BRANCH_REVIEW.json',dict(status='PASS',source_digest=pre['source_digest'],source_count=len(files),helper_SHA=c.sha(__file__),fullFIT_native_owner_Torch_CPU_EXACT=True,actual_train_manifest_expression_all12=manifests,actual_train_save_job_all4_and_native=jobs,not_scientific_runs=True,elapsed_seconds=time.perf_counter()-start,resource=c.RESOURCE))
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_PRODUCTION_BRANCH_REVIEW');raise


