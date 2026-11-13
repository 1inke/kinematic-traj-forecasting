"""Only repair approved numeric execution/schema handling; no new science."""
import run_frozen
from common import *
from stable_forward_v2 import predict
from reconcile_v3 import reconcile
from preflight_v3 import verify_v3

if __name__=='__main__':
    import traceback
    try:
        verify_v3();run_frozen.predict=predict;run_frozen.reconcile=reconcile;run_frozen.main();verify_v3()
    except Exception:
        write_json(HERE/'FAILURE.json',dict(stage='CURRENT_V3_FROZEN_EVALUATION_FAILURE',traceback=traceback.format_exc()));progress('STOP_CURRENT_FAILURE');raise
