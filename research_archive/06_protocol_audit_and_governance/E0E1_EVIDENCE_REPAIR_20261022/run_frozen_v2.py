"""Declared constant-compute repair, frozen population evaluation only."""
import run_frozen
from stable_forward_v2 import predict
from preflight_v2 import verify_v2
from common import *

if __name__=='__main__':
    import traceback
    try:
        verify_v2();run_frozen.predict=predict;run_frozen.main();verify_v2()
    except Exception:
        write_json(HERE/'FAILURE.json',dict(stage='CURRENT_V2_FROZEN_EVALUATION_FAILURE',traceback=traceback.format_exc()));progress('STOP_CURRENT_FAILURE');raise
