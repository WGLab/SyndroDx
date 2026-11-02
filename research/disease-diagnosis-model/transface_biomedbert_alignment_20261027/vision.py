"""Load the development-selected, pre-GMDB TransFace checkpoint unchanged."""
import importlib.util
import sys
import torch
from common import STAGE1,GMDB_STAGE,digest,load

def build_vision():
    sys.path.insert(0,str(STAGE1))
    spec=importlib.util.spec_from_file_location('alignment_transface_source',STAGE1/'model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.backbone()
    receipt=load(GMDB_STAGE/'selected/identity_checkpoint.json')
    if receipt['step']!=2500 or digest(receipt['path'])!=receipt['sha256']:
        raise RuntimeError('selected_visual_checkpoint_changed')
    state=torch.load(receipt['path'],map_location='cpu',weights_only=False,mmap=True)
    weights={k.removeprefix('backbone.'):v for k,v in state['student'].items() if k.startswith('backbone.')}
    model.load_state_dict(weights,strict=True)
    model.requires_grad_(False);model.eval()
    return model
