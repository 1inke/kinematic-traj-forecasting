"""Independent inference process: deliberately never loads GT/labels."""
import sys
import numpy as np
import contracts as c
import torch
from model import ParameterRegressor
from train import infer
from inference import predict_paths
def main():
    inp,out=sys.argv[1:];z=c.load_npz(inp)
    torch.manual_seed(42);m=ParameterRegressor().cuda()
    with torch.no_grad():m.layers[-1].bias.copy_(torch.tensor([.2,.1],device='cuda'))
    u=infer(m,z['z'],'cuda:0');p=predict_paths(z['past'],z['native'],u,'END')
    c.npz_new(out,u=u,**{k:v for k,v in p.items() if k!='continuous_param'})
if __name__=='__main__':main()
