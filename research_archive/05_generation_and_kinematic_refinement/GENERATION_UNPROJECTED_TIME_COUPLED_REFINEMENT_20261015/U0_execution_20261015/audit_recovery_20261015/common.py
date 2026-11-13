"""Read-only parent arithmetic; all writes/guards redirected to this recovery."""
import importlib.util,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
H=PARENT=OUT.parent
POLICY=PARENT/'audit_memory_policy_20261015'
spec=importlib.util.spec_from_file_location('recovery_memory_policy',POLICY/'memory_policy.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
policy.H=OUT
spec=importlib.util.spec_from_file_location('readonly_u0_common',PARENT/'common.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
# Config is immutable during an execution. Cache identical parsed JSON rather
# than opening two files for every candidate clustering resource measurement.
_read=policy.read
_config=_read('resource_policy.json');_estimate=_read('MEMORY_ESTIMATE.json')
policy.read=lambda name: _config if name=='resource_policy.json' else _estimate if name==_config['estimate_file'] else _read(name)
legacy.H=OUT;legacy.guard=policy.guard;legacy.ns['guard']=policy.guard;legacy.RESOURCE=policy.RESOURCE
for name in ('ROOT','read','write','sha','digest','verify','stage','npz','errors','motion','clusters','masks','load_data','load','close','compare','valid_done','failure','GROUPS','PORTS','SIZES'):
    globals()[name]=getattr(legacy,name)
guard=policy.guard;RESOURCE=policy.RESOURCE
