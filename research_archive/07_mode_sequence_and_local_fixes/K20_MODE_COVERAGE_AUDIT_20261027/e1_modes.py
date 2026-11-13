from common import *
from morphology import assign,variants
def run():
    g=guard(True);cfg=read(HERE/'config/analysis_config.json');lock=read(HERE/'config/CONFIG_LOCK.json');assert sha(HERE/'config/analysis_config.json')==lock['config_SHA'];output=HERE/'E1_mode_distribution';record=[];opened=[]
    for split in SPLITS:
        file=HERE/f'inputs/X_{split}.npz';x=load(file);opened.append(file.relative_to(HERE).as_posix());assert not any('gt' in k.lower() for k in x)
        for variant in variants():
            name=variant[0];z=assign(x,cfg,variant);folder=output/split/name
            save(folder/'modes.npz',sample_id=x['sample_id'],slot=np.arange(20),**z)
            record.extend(dict(population=split,variant=name,sample_id=int(sid),track=x['track_key'][i],scene=x['scene'][i],sequence=x['sequence'][i],group=int(x['group'][i]),M=int(z['M'][i]),Meff=float(z['Meff'][i]),Pmax=float(z['Pmax'][i]),same_pair=float(z['same_pair'][i]),near_phi_pair=float(z['near_phi_pair'][i]),low_fraction=float(z['low'][i].mean()),undefined_fraction=float((z['mode'][i]==1).mean()) if not name.startswith('cluster') else float((~z['turn_defined'][i]&~z['low'][i]).mean()),structural_candidates=int(z['structural'][i].sum())) for i,sid in enumerate(x['sample_id']))
            g.check()
        mark('E1_FROZEN_MODE_FEATURES',population=split,windows=len(x['sample_id']));print('E1',split,flush=True)
    table(output/'window_mode_summary.csv',record);write(output/'GT_FREE_ACCESS_AUDIT.json',dict(status='PASS',parsed_data_files=opened,config=lock['config_SHA'],GT_prediction_or_E2_file_opened=False,no_Torch='torch' not in sys.modules,no_model_forward=True,no_population_exclusion=True))
    resource(output/'RESOURCE.json',g);finish(output,config_SHA=lock['config_SHA'])
    write(HERE/'E1_LOCK.json',dict(config_SHA=lock['config_SHA'],done_SHA=sha(output/'done.json'),at=now(),source=lock['source'],E2_not_started=True));print('PASS E1 LOCK',flush=True)
if __name__=='__main__':run()
