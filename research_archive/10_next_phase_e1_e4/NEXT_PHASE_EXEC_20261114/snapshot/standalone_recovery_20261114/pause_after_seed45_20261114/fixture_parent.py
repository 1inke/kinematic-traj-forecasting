import subprocess,sys,time
from pathlib import Path
p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(2)"])
Path(sys.argv[1]).write_text(str(p.pid))
p.wait()
time.sleep(20)
