"""Full preflight with explicit constant-batch and legacy schema adapters."""
import preflight
from common import *
from stable_forward_v2 import predict
from reconcile_v3 import reconcile
from preflight_v2 import verify_v2

def verify_v3():
    verify_v2();value=read_json(HERE/'IMPLEMENTATION_TRANSITION_V3.json')
    for name,digest in value['files'].items():
        if sha256(ROOT/name)!=digest:raise RuntimeError('V3_IMPLEMENTATION_SOURCE_CHANGED')

if __name__=='__main__':
    verify_v3();preflight.predict=predict;preflight.reconcile=reconcile;preflight.main();verify_v3()
