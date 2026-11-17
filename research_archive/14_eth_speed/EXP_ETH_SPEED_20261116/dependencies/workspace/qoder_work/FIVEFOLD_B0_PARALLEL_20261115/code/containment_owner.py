"""Preflight owner intentionally terminated to verify descendant cleanup."""
import json
import os
from pathlib import Path
import sys
import time
from job_control import ChildJob

directory = Path(sys.argv[1])
job = ChildJob()
payload = "import os,time,pathlib; pathlib.Path(" + repr(str(directory / 'actual_child_pid')) + ").write_text(str(os.getpid())); time.sleep(120)"
child = job.popen([sys.argv[2], '-B', '-c', payload], stdout=sys.stdout, stderr=sys.stderr)
(directory / 'OWNER.json').write_text(json.dumps(dict(owner=os.getpid(), redirector=child.pid)), encoding='utf-8')
while True:
    time.sleep(1)
