"""Recovery-only adapter; parent locked common.py bytes are never edited.

The reviewed parent top-level only defines helpers/loads four read-only arithmetic
functions. No audit, training, resource poll, worker or dataset loading is launched.
All future helper writes/stages/cluster guards are redirected into this child.
"""
import importlib.util
from pathlib import Path
import memory_policy as policy
H=Path(__file__).resolve().parent
PARENT=H.parent
spec=importlib.util.spec_from_file_location('u0_parent_readonly_resource_adapter',PARENT/'common.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
legacy.H=H;legacy.guard=policy.guard;legacy.ns['guard']=policy.guard
legacy.RESOURCE=policy.RESOURCE
for name in ('ROOT','read','write','sha','digest','verify','stage','npz','errors','motion','clusters','masks','load_data','load','close','compare','valid_done','failure','GROUPS','PORTS','SIZES'):globals()[name]=getattr(legacy,name)
guard=policy.guard;RESOURCE=policy.RESOURCE

def locks():raise RuntimeError('Recovery source manifest must inherit parent1916+completed output SHA before audit; no parent locks reuse')
