from core import *
import subprocess
def main():
    records=[]
    for name,cpu in (('preflight_gpu.py',False),('preflight_cpu.py',True)):
        g=guard(cpu=cpu,startup=True);receipt=HERE/(name+'.admission.json');resource(receipt,g)
        p=subprocess.run([sys.executable,'-B','-u',str(HERE/name)],cwd=HERE,creationflags=subprocess.CREATE_NO_WINDOW);assert p.returncode==0,'Actual current preflight failed '+name;records.append(dict(script=name,pre_Popen_startup_actual_PASS=True,receipt_SHA=sha256(receipt)));print('DONE',name,flush=True)
    write_json(HERE/'PREFLIGHT_REAL_STAGE_ADMISSION.json',dict(status='PASS',source=source(),records=records,resource_formula_unmodified=True,no_Torch_in_coordinator='torch' not in sys.modules));assert 'torch' not in sys.modules
if __name__=='__main__':
    try:main()
    except BaseException as ex:failure('actual_preflight_admission_or_child',ex);raise

