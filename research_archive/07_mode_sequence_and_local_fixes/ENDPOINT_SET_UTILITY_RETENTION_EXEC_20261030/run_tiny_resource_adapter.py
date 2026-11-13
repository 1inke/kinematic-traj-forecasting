"""Bind production to the latest actual preflight inventory, without overwriting v1 receipt."""
from pathlib import Path
import common as c
import run_tiny

if __name__=='__main__':
    original=c.json_read
    latest=c.ROOT/'MEMORY_ESTIMATE_v3.json'
    if not latest.exists():raise RuntimeError('Current complete preflight inventory missing; no model startup')
    c.write(c.ROOT/'TINY_RESOURCE_ADAPTER.json',dict(status='EXPLICIT_CURRENT_PREFLIGHT_INVENTORY_BINDING',
        files={str(c.ROOT/'run_tiny_resource_adapter.py'):c.sha(__file__),str(c.ROOT/'run_tiny.py'):c.sha(c.ROOT/'run_tiny.py'),str(latest):c.sha(latest)},
        repair='Read latest independently measured resource receipt. Old resource receipt preserved; no training math/hyperparameter change.'))
    def current_read(path):
        return original(latest if Path(path).resolve()==(c.ROOT/'MEMORY_ESTIMATE.json').resolve() else path)
    c.json_read=current_read
    run_tiny.main()
