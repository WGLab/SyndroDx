"""One fixed GMDB test pass after development selection; private features stay local."""
import ast
import csv
import importlib.util
import json
import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from common import WORK,SHARED,load,save,digest,now,verify_code
from data import verify_data
from disease_model import DiseaseModel

class EvaluationImages(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        if digest(row['crop_path'])!=row['crop_sha256']:raise RuntimeError('evaluation_source_changed')
        im=cv2.imread(row['crop_path'],cv2.IMREAD_COLOR)
        if im is None:raise RuntimeError('evaluation_image_decode_failed')
        im=cv2.cvtColor(cv2.resize(im,(112,112)),cv2.COLOR_BGR2RGB)
        return torch.from_numpy(im.copy()).permute(2,0,1).float().div_(127.5).sub_(1.)

def metric_functions():
    pinned=load(WORK/'metric_code_receipt.json')
    for name,sha in pinned.items():
        if digest(SHARED/name)!=sha:raise RuntimeError('frozen_gmdb_metric_code_changed')
    spec=importlib.util.spec_from_file_location('fixed_gmdb_metrics',SHARED/'metrics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tree=ast.parse((SHARED/'evaluate_checkpoint.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_metrics')
    scope=dict(np=np,evaluate=module.evaluate,TOP_K=module.TOP_K)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SHARED/'evaluate_checkpoint.py'),'exec'),scope)
    return scope['run_metrics']

def main(arm):
    os.umask(0o077);torch.set_num_threads(4);cv2.setNumThreads(1);torch.cuda.set_device(0)
    verify_code();receipt=verify_data();selection=load(WORK/'selected/models.json')
    if selection.get('state')!='frozen_before_test':raise RuntimeError('development_selection_required')
    chosen=selection['full_label'][arm];folder=WORK/'evaluation'/arm;folder.mkdir(parents=True,exist_ok=True)
    if load(folder/'result.json').get('state')=='complete':return
    reference=load(SHARED/'protocol.json')
    if digest(SHARED/'inputs.private.jsonl')!=reference['inputs_sha256']:raise RuntimeError('fixed_evaluation_inputs_changed')
    tables={}
    for name,sha in reference['split_sha256'].items():
        path=SHARED/'split_snapshot'/(name+'.csv')
        if digest(path)!=sha:raise RuntimeError('fixed_evaluation_split_changed')
        with path.open(newline='') as stream:tables[name]=list(csv.DictReader(stream))
    with (SHARED/'inputs.private.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
    model=DiseaseModel(receipt['diseases'],arm,load(WORK/'selected/identity_checkpoint.json'))
    if digest(chosen['path'])!=chosen['sha256']:raise RuntimeError('selected_adaptation_changed')
    state=torch.load(chosen['path'],weights_only=False,map_location='cpu',mmap=True)
    if state['signature']['case']!=chosen['case']:raise RuntimeError('selected_case_mismatch')
    model.load_adaptation(state['adaptation']);del state;model.cuda().eval()
    loader=DataLoader(EvaluationImages(rows),batch_size=128,num_workers=2,pin_memory=True)
    path=folder/'features.private.npy';tmp=path.with_suffix('.tmp')
    features=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(rows),512));offset=0
    with torch.no_grad():
        for images in loader:
            with torch.autocast('cuda',dtype=torch.bfloat16):values=model.embed(images.cuda(non_blocking=True))
            if not torch.isfinite(values).all():raise FloatingPointError('nonfinite_test_embeddings')
            values=values.float().cpu().numpy();features[offset:offset+len(values)]=values;offset+=len(values)
            save(folder/'status.json',dict(state='extracting',processed_images=offset,total_images=len(rows),updated_at_utc=now()))
    features.flush();del features,model;torch.cuda.empty_cache();tmp.replace(path)
    save(folder/'status.json',dict(state='scoring_fixed_splits',updated_at_utc=now()))
    metrics=metric_functions()(arm,np.load(path,mmap_mode='r'),rows,tables)
    save(folder/'result.json',dict(state='complete',arm=arm,selected_case=chosen['case'],selected_epoch=chosen['epoch'],
        checkpoint_sha256=chosen['sha256'],metrics=metrics,exploratory_gmdb=True,clinical_final_test_used=False,updated_at_utc=now()))
    save(folder/'status.json',dict(state='complete',updated_at_utc=now()))

if __name__=='__main__':
    try:main(sys.argv[1])
    except Exception as error:
        save(WORK/'evaluation'/sys.argv[1]/'failure.json',dict(error_type=type(error).__name__,oom=isinstance(error,torch.cuda.OutOfMemoryError),updated_at_utc=now()));raise
