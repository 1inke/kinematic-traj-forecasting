import numpy as np
import pytest
from audit.verify import audit_file
from f16.eval import evaluate
from f16.io import sha256,write_json

def test_saved_GT_and_track_cannot_be_forged(tmp_path):
    ids=np.arange(3);native=np.zeros((3,20,12,2));gt=np.ones((3,12,2));groups=np.arange(3);tracks=np.array(['s::0','s::1','s::2'])
    cache=tmp_path/'cache.npz';labels=tmp_path/'labels.npz';md=tmp_path/'metadata.npz';manifest=tmp_path/'manifest.json';pred=tmp_path/'output.npz';result=tmp_path/'metrics.json'
    np.savez(cache,sample_id=ids,candidate=native.reshape(3,20,24),change_group=groups);np.savez(labels,sample_id=ids,future=gt);np.savez(md,sample_id=ids,sequence=['s']*3,pedestrian_id=ids)
    write_json(manifest,{'files':{p.name:sha256(p) for p in [cache,labels,md]}})
    np.savez(pred,paths=native,native=native,gt=gt,groups=groups,sample_id=ids,track=tracks);metrics,_=evaluate(native,gt,groups,native,ids,tracks);write_json(result,metrics)
    options=dict(frozen_cache=cache,frozen_labels=labels,frozen_metadata=md,frozen_manifest=manifest,root=tmp_path)
    assert audit_file(pred,result,**options)['frozen_labels_verified']
    np.savez(pred,paths=native,native=native,gt=gt+1,groups=groups,sample_id=ids,track=tracks)
    with pytest.raises(ValueError):audit_file(pred,result,**options)
    np.savez(pred,paths=native,native=native,gt=gt,groups=groups,sample_id=ids,track=['fake']*3)
    with pytest.raises(ValueError):audit_file(pred,result,**options)
