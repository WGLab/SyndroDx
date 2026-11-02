"""Immutable step1000 encoder; clinical images and feature tensors remain local."""
import sys,torch
from common import PROJECT,STAGE1,load,digest
def build_vision():
    sys.path.insert(0,str(STAGE1))
    from model import backbone
    # model.py may already refer to the alignment module; load encoder explicitly.
    import importlib.util
    spec=importlib.util.spec_from_file_location('locked_vision',STAGE1/'model.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    selection=load(PROJECT/'arcface_transfer_evaluation_20261028/selection.json')
    assert selection['step']==1000 and digest(selection['path'])==selection['sha256']
    model=m.backbone();s=torch.load(selection['path'],map_location='cpu',weights_only=False)
    model.load_state_dict(s['backbone'],strict=True);return model.eval().requires_grad_(False)
