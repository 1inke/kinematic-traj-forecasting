"""Second bounded counter regression: include composite einsum contractions.

V1 output is retained as explicitly incomplete. Original scientific models,
predictions and training contract unchanged. Import V1 run unchanged; replace
only counter and versioned output adapter, no training or optimizer.
"""
import sys,math
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import profile_cost_supplement as v1
import torch
from common import sha256
from f16.io import write_json
Parent=v1.Cost
class Cost(Parent):
    def __init__(self):super().__init__();self.seen=set()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        name=str(func._overloadpacket);self.seen.add(name)
        if name=='aten.einsum':
            eq,operands=args[:2];left,right=eq.split('->');inputs=left.split(',')
            assert len(inputs)==2 and len(operands)==2 and '.' not in eq,('UNSUPPORTED_EINSUM',eq)
            sizes={}
            for sub,t in zip(inputs,operands):
                assert len(sub)==t.ndim
                for label,n in zip(sub,t.shape):
                    assert label not in sizes or sizes[label] in (1,n) or n==1
                    sizes[label]=max(sizes.get(label,1),n)
            assert set(sizes)-set(right),('POINTWISE_NOT_MAC',eq)
            cost=int(math.prod(sizes.values()));self.mac[name]+=cost;self.calls[name]+=1
            self.shapes.append(dict(operator=name,equation=eq,MAC=cost,input_shapes=[list(x.shape) for x in operands]))
            return func(*args,**(kwargs or {}))
        return super().__torch_dispatch__(func,types,args,kwargs)
def emit(path,result):
    result.update(status='PASS_PROFILE_ONLY_CORRECTION_V2',prior_supplement_incomplete=True,
        prior_supplement_SHA=sha256(Path(__file__).with_name('PROFILE_COST_SUPPLEMENT.json')),
        second_regression='CPU2x3x4 times2x4x5 einsum requires120MAC; V1gave0, V2gives120; original V1files retained',
        helper_SHA=sha256(__file__),parent_helper_SHA=sha256(v1.__file__),
        repaired_counter='top-level linear/matmul/bmm/fusedTransformer plus explicit two-operand einsum union-dimension contraction; no nested double count')
    for k,a in result['arms'].items():assert 'aten.einsum' in a['operator_MAC']
    write_json(Path(path).with_name('PROFILE_COST_SUPPLEMENT_V2.json'),result)
if __name__=='__main__':
    with torch.inference_mode(),Cost() as c:torch.einsum('bik,bkj->bij',torch.ones(2,3,4),torch.ones(2,4,5))
    assert sum(c.mac.values())==120 and not c.unknown
    v1.Cost=Cost;v1.write_json=emit;v1.run()
