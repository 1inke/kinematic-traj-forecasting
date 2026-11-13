"""Light coordinator performs warning-v3 admission BEFORE Popen of GPU preflight."""
import subprocess
import sys
import contracts_discrete as c

def main():
    E=4741091328+64*1024**2+128*1024**2;g=c.resource(E);g.check(mode='startup',force=True)
    p=subprocess.Popen([sys.executable,'-B',str(c.ROOT/'preflight_discrete.py')],cwd=c.PROJECT)
    code=p.wait()
    c.write(c.ROOT/'RESOURCE_PREFLIGHT_ADMISSION.json',dict(policy='warning_first_v3_actual_prePopen',estimate=E,observations=g.observations,child_exit=code))
    if code:raise SystemExit(code)
if __name__=='__main__':main()
