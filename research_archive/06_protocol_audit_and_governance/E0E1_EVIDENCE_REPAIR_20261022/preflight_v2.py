"""Run the complete original preflight with the declared numerical adapter."""
import preflight
from common import *
from stable_forward_v2 import predict

def verify_v2():
    value=read_json(HERE/'IMPLEMENTATION_TRANSITION_V2.json')
    for name,digest in value['files'].items():
        if sha256(ROOT/name)!=digest:raise RuntimeError('V2_IMPLEMENTATION_SOURCE_CHANGED')

if __name__=='__main__':
    verify_v2();preflight.predict=predict;preflight.main();verify_v2()
