"""Infrastructure-only nested-sequence comparison repair; parent bytes preserved."""
import importlib.util
from pathlib import Path
OUT=Path(__file__).resolve().parent
PARENT=H=OUT.parent
FIRST=PARENT/'audit_recovery_20261015'
spec=importlib.util.spec_from_file_location('previous_readonly_recovery_common',FIRST/'common.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
for name in ('ROOT','read','write','sha','digest','verify','stage','npz','errors','motion','clusters','masks','load_data','load','close','valid_done','failure','GROUPS','PORTS','SIZES','RESOURCE','guard','policy','POLICY'):
    globals()[name]=getattr(prior,name)
prior.legacy.H=OUT;policy.H=OUT
# Same numerical estimate; reload this child's complete provenance JSON.
_read=prior._read;_config=_read('resource_policy.json');_estimate=_read('MEMORY_ESTIMATE.json')
policy.read=lambda name: _config if name=='resource_policy.json' else _estimate if name==_config['estimate_file'] else _read(name)

def compare(a,b):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys()
        return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(list,tuple)):
        assert isinstance(b,(list,tuple,prior.legacy.np.ndarray)) and len(a)==len(b)
        return max([compare(x,y) for x,y in zip(a,b)] or [0.])
    if a is None:assert b is None;return 0.
    if isinstance(a,(str,bool)):assert a==b;return 0.
    return close(a,b)  # Original2e-12, not a new scientific tolerance.
