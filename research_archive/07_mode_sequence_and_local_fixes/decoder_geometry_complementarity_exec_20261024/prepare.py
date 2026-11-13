"""Actual saved128 fixtures + dependency/source admission before sole worker."""
from common import *
import time,urllib.request
from map_inventory import inventory
def fixtures():
    p=np.zeros((4,20,12,2));y=np.zeros((4,12,2));e=geometry(p,y)
    label,distinct,g=clusters(p,e['slot_good']);assert (distinct==1).all() and (g==1).all()
    assert all(np.array_equal(v,part) for v,part in zip(partitions(e['covered'],e['covered']).values(),[np.ones(4,bool),np.zeros(4,bool),np.zeros(4,bool),np.zeros(4,bool)]))
    p[0,1:,:,0]=.15;p[0,0,:,0]=-.15;_,_,g=clusters(p,geometry(p,y)['slot_good']);assert g[0]==2
    # More good clusters can leave covered-window count completely unchanged.
    assert geometry(p,y)['covered'].sum()==4
    masks=partitions(np.array([1,1,0,0],bool),np.array([1,0,1,0],bool));assert all(v.sum()==1 for v in masks.values())
    bad=np.full((1,20,12,2),0.);bad[0,0,-1,0]=.5;bad[0,1:,:,0]=.3;bad[0,1:,-1]=0
    assert not geometry(bad,y[:1])['covered'][0]
    return dict(status='PASS',identical_sets=True,copies_one_cluster=True,more_clusters_same_coverage=True,rescue_and_lost=True,disjoint_slot_minima_rejected=True)
def run():
    guard=Guard(E);guard.check('startup');test=fixtures();inv=inventory()
    wheel=next((HERE/'dependency_wheels').glob('pyarrow-21.0.0-*.whl'))
    import pyarrow
    assert pyarrow.__version__=='21.0.0' and Path(pyarrow.__file__).resolve().is_relative_to(HERE/'runtime_deps')
    package=json.load(urllib.request.urlopen('https://pypi.org/pypi/pyarrow/21.0.0/json',timeout=30))
    info=next(v for v in package['urls'] if v['filename']==wheel.name);assert info['digests']['sha256']==sha(wheel)
    write(HERE/'DEPENDENCY_PROVENANCE.json',dict(package='pyarrow',version='21.0.0',wheel=wheel.name,sha256=sha(wheel),official_url=info['url'],metadata='https://pypi.org/pypi/pyarrow/21.0.0/json',isolated_target=str(HERE/'runtime_deps'),old_environment_modified=False))
    files={}
    for path in [PREVIOUS/'series_source_manifest.json']+list((PREVIOUS/'direction_C_set_mechanisms').glob('C2_execution_20261024/source_manifest.json')):
        for k,v in read(path)['files'].items():
            assert k not in files or files[k]['sha256']==v['sha256'];assert sha(v['path'])==v['sha256'];files[k]=v.copy()
    inherited=len(files)
    inputs=list(DESIGN.glob('*.md'))+list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[HERE/'protocol.json',HERE/'q1_map_inventory/MAP_INVENTORY.json',HERE/'DEPENDENCY_PROVENANCE.json',wheel,PREVIOUS/'completion_review/REVIEW.json',PREVIOUS/'RESULT_ANALYSIS.json',PREVIOUS/'direction_B_scene_prior/records_20261024.json']
    e=PREVIOUS/'direction_C_set_mechanisms/C2_execution_20261024'
    for arm in ('C2_isomlp','C2_setmix'):
        for seed in SEEDS:
            run=e/f'{arm}_seed{seed}';done=read(run/'done.json');assert done['status']=='SUCCESS'
            for n,h in done['outputs'].items():assert sha(run/n)==h;inputs.append(run/n)
            inputs.append(run/'done.json')
    # Native/old A saved reference is immutable, not an extra model candidate pool.
    for s in ('FIT','inner'):
        inputs.append(OLD/f'native_{s}.npz')
        for seed in SEEDS:inputs.append(OLD/f'runs/A_seed{seed}/predictions_{s}.npz')
    for r in inv['records']:
        inputs.extend(Path(v['path']) for v in r['image_sha'])
        if r['source_and_license']['homography_source']:inputs.append(Path(r['source_and_license']['homography_source']))
    for p in inputs:
        key=str(p.resolve());val=dict(path=key,sha256=sha(p))
        if key in files:assert files[key]['sha256']==val['sha256']
        files.setdefault(key,val)
    # Full actual fixed savedFIT8x128, no model/GPU/optimizer.
    source=e/'C2_setmix_seed42/predictions_FIT.npz';z=load(source);start=time.perf_counter();errors=[]
    for begin in range(0,1024,128):
        guard.check();p=z['paths'][begin:begin+128];gt=z['gt'][begin:begin+128];ee=geometry(p,gt)
        a=clusters(p,ee['slot_good']);b=clusters(p,ee['slot_good'],True)
        assert all(np.array_equal(x,y) for x,y in zip(a,b))
        import pyarrow as pa,pyarrow.parquet as pq
        testpath=HERE/'preflight_saved128.parquet';pq.write_table(pa.table(dict(sample_id=z['sample_id'][begin:begin+128],covered=ee['covered'],good_clusters=a[2])),testpath)
        back=pq.read_table(testpath).to_pydict();assert back['covered']==ee['covered'].tolist() and back['sample_id']==z['sample_id'][begin:begin+128].tolist()
    seconds=time.perf_counter()-start;guard.check()
    write(HERE/'source_manifest.json',dict(files=files,digest=digest(files),inherited_count=inherited,count=len(files),old_digest_replaced=False))
    planning=seconds/1024*(18909+1856)*6+180
    write(HERE/'PREFLIGHT.json',dict(status='PASS_ACTUAL_SAVED_CPU8X128',fixtures=test,source_digest=verify_manifest(),source_count=len(files),scope='NO_MODEL_FORWARD_BACKWARD_OPTIMIZER',map_admission=inv['status'],actual1024_seconds=seconds,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],estimated_peak_bytes=E,resource_observations=guard.observations,full_Q0_already_executed=False))
    print(json.dumps(dict(status='PASS',source_count=len(files),planning_seconds=planning,map=inv['status'])))
if __name__=='__main__':run()
