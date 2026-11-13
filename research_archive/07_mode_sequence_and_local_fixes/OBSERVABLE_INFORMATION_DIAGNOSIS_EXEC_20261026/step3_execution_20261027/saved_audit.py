"""Reuse unchanged independent saved geometry/pdist/motion science, new root only."""
import core,sys
from core import *
sys.modules['base']=core
ev=module('_step3_saved_geometry',PREVIOUS.parent/'execution_staged_20261026/evaluate_saved.py')
if __name__=='__main__':ev.audit(sys.argv[1],int(sys.argv[2]),sys.argv[3])
