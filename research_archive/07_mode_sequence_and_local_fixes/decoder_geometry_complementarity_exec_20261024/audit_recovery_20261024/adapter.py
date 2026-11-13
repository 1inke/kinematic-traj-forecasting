"""Exact original audit except one first-velocity cast; versioned output only."""
import sys,ast
from pathlib import Path
sys.dont_write_bytecode=True
CHILD=Path(__file__).resolve().parent;PARENT=CHILD.parent
sys.path.insert(0,str(PARENT))
from common import *
def compiled():
    old=(PARENT/'audit_saved.py').read_text(encoding='utf-8')
    before='v[:,:,0]=p[:,:,0]/.4';after='v[:,:,0]=p[:,:,0].astype(float)/.4'
    assert old.count(before)==1
    fixed=old.replace(before,after)
    ot=ast.parse(old);ft=ast.parse(fixed)
    # Reversing exactly this assignment returns the original full AST.
    assert ast.dump(ast.parse(fixed.replace(after,before)),include_attributes=False)==ast.dump(ot,include_attributes=False)
    ns={'__name__':'_q0_repaired_audit','__file__':str(PARENT/'audit_saved.py')}
    exec(compile(ft,str(CHILD/'compiled_audit_saved.py'),'exec'),ns)
    def write_child(path,value):
        assert Path(path)==OUT/'REVIEW.json','Unexpected audit write destination'
        write(CHILD/'REVIEW.json',value)
    ns['write']=write_child
    return ns,dict(original_file=str(PARENT/'audit_saved.py'),original_SHA=sha(PARENT/'audit_saved.py'),
                   changed_expression=before+' -> '+after,other_AST_exact=True,compiled_source_SHA=hashlib.sha256(fixed.encode()).hexdigest(),tolerance=2e-12,output_only_in_child=True)
def audit():
    ns,equivalent=compiled();write(CHILD/'SOURCE_EQUIVALENCE.json',equivalent);return ns['audit']()
