"""Local frozen encoders and faithful GMDB syndrome-macro retrieval evaluation."""
import csv
import gc
import json
import logging
import os
from pathlib import Path
from settings import *
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms.functional import pil_to_tensor
from omegaconf import OmegaConf
from metrics import evaluate,TOP_K
from common import install_rope_dtype_compatibility,notify


class Faces(Dataset):
    def __init__(self,rows,cfg,offset):
        self.rows=rows;self.cfg=cfg;self.offset=offset
    def __len__(self):return len(self.rows)-self.offset
    def __getitem__(self,index):
        position=index+self.offset;row=self.rows[position]
        try:
            if digest(row['crop_path'])!=row['crop_sha256']:raise RuntimeError('crop_changed')
            size=int(self.cfg.crops.global_crops_size)
            mean=tuple(self.cfg.crops.rgb_mean)
            with Image.open(row['crop_path']) as source:image=source.convert('RGB')
            ratio=size/max(image.size)
            resized=image.resize((max(1,round(image.width*ratio)),max(1,round(image.height*ratio))),Image.Resampling.BICUBIC)
            canvas=Image.new('RGB',(size,size),tuple(round(m*255) for m in mean))
            canvas.paste(resized,((size-resized.width)//2,(size-resized.height)//2))
            value=pil_to_tensor(canvas).float().div_(255)
            value=(value-torch.tensor(mean)[:,None,None])/torch.tensor(list(self.cfg.crops.rgb_std))[:,None,None]
            image.close();resized.close();canvas.close()
            return position,value
        except Exception:
            raise RuntimeError('verified_gmdb_image_preprocessing_failed') from None


def encoder(cfg,kind,step):
    install_rope_dtype_compatibility()
    from dinov3.models import build_model_from_cfg
    with torch.device('meta'):model,_=build_model_from_cfg(cfg,only_teacher=True)
    model.to_empty(device='cuda');model.rope_embed.dtype=torch.float32;model.init_weights()
    if kind=='original_dinov3':
        from weights import OfficialBackbone
        OfficialBackbone(V1/'assets/dinov3-vit7b16').load(model)
    else:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        reader=dcp.FileSystemReader(WORK/'checkpoint_snapshot/ckpt'/str(step-1))
        metadata=reader.read_metadata().state_dict_metadata
        state=model.state_dict();payload={};keys={}
        for name,tensor in state.items():
            key='model.teacher.backbone.'+name
            if key not in metadata or tuple(metadata[key].size)!=tuple(tensor.shape):
                raise RuntimeError('teacher_backbone_tensor_missing_or_wrong_shape')
            keys[name]=key;payload[key]=tensor
        dcp.load(payload,storage_reader=reader,planner=DefaultLoadPlanner(allow_partial_load=True))
        model.load_state_dict({name:payload[key] for name,key in keys.items()},strict=True)
        save(WORK/'teacher_load_receipt.json',dict(state='passed',checkpoint_step=step,
            all_backbone_keys_covered=True,tensors=len(keys),teacher_not_student=True,updated_at_utc=now()))
    return model.eval().requires_grad_(False)


def extract_once(model,rows,cfg,kind,batch,protocol):
    folder=WORK/'features';folder.mkdir(exist_ok=True)
    path=folder/(kind+'.private.npy');progress=load(folder/(kind+'.progress.json'))
    if progress and progress.get('inputs_sha256')!=protocol['inputs_sha256']:
        raise RuntimeError('feature_cache_inputs_mismatch')
    completed=int(progress.get('completed',0))
    if completed and not path.exists():raise RuntimeError('feature_progress_without_array')
    output=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(len(rows),model.embed_dim))
    if completed==len(rows):return path
    loader=DataLoader(Faces(rows,cfg,completed),batch_size=batch,shuffle=False,
        num_workers=8,pin_memory=True,prefetch_factor=2)
    with torch.inference_mode():
        for indices,inputs in loader:
            inputs=inputs.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                value=model(inputs,is_training=True)['x_norm_clstoken'].float()
            value=torch.nn.functional.normalize(value,dim=-1)
            if not bool(torch.isfinite(value).all()):raise RuntimeError('nonfinite_gmdb_features')
            output[indices.numpy()]=value.cpu().numpy();output.flush()
            completed=int(indices[-1])+1
            state=dict(state='complete' if completed==len(rows) else 'extracting',completed=completed,total=len(rows),
                inputs_sha256=protocol['inputs_sha256'],batch_size=batch,updated_at_utc=now())
            save(folder/(kind+'.progress.json'),state)
            save(WORK/'status.json',dict(state,phase='frozen_feature_extraction',model=kind,checkpoint_step=protocol['checkpoint_step']))
            del inputs,value
    return path


def run_metrics(kind,features,rows,tables):
    index={r['image_id']:i for i,r in enumerate(rows)}
    fg=tables['gmdb_frequent_gallery_images'];fq=tables['gmdb_frequent_test_images']
    rg=tables['gmdb_rare_gallery_images'];rq=tables['gmdb_rare_test_images']
    results={}
    for strict in (False,True):
        name='patient_disjoint_query_supplement' if strict else 'provided_split_literal'
        results[name]={}
        for case in ['frequent_to_frequent','rare_to_rare','frequent_to_combined','rare_to_combined']:
            reports=[]
            for split in ([None] if case=='frequent_to_frequent' else range(10)):
                rare_g=[r for r in rg if int(r['split'])==split] if split is not None else []
                rare_q=[r for r in rq if int(r['split'])==split] if split is not None else []
                gallery=fg if case=='frequent_to_frequent' else rare_g if case=='rare_to_rare' else fg+rare_g
                queries=fq if case.startswith('frequent') else rare_q
                reports.append(dict(split=split,**evaluate(gallery,queries,features,index,strict=strict)))
            results[name][case]=dict(folds=reports,
                macro={f'top{k}':float(np.mean([r['macro'][f'top{k}'] for r in reports])) for k in TOP_K},
                macro_split_std={f'top{k}':float(np.std([r['macro'][f'top{k}'] for r in reports])) for k in TOP_K})
    return results


def main():
    os.umask(0o077);logging.disable(logging.INFO);torch.set_num_threads(8);torch.cuda.set_device(0)
    if torch.cuda.device_count()!=1:raise RuntimeError('one_independent_evaluation_gpu_required')
    protocol=load(WORK/'protocol.json')
    if protocol.get('state')!='prepared':raise RuntimeError('frozen_protocol_required')
    if digest(WORK/'inputs.private.jsonl')!=protocol['inputs_sha256']:raise RuntimeError('input_manifest_changed')
    for name,sha in protocol['split_sha256'].items():
        if digest(WORK/'split_snapshot'/(name+'.csv'))!=sha:raise RuntimeError('official_split_changed')
    ckpt=WORK/'checkpoint_snapshot/ckpt'/str(protocol['checkpoint_step']-1)
    receipt=load(WORK/'checkpoint_snapshot/receipt.json')
    if digest(ckpt/'.metadata')!=receipt['metadata_sha256']:raise RuntimeError('checkpoint_metadata_changed')
    for name,size in receipt['checkpoint_file_sizes'].items():
        if (ckpt/name).stat().st_size!=size:raise RuntimeError('checkpoint_file_size_changed')
    with (WORK/'inputs.private.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
    tables={}
    for name in TABLES:
        with (WORK/'split_snapshot'/(name+'.csv')).open(newline='') as stream:tables[name]=list(csv.DictReader(stream))
    cfg=OmegaConf.load(WORK/'checkpoint_snapshot/config.yaml')
    completed=load(WORK/'results.json');results=completed.get('models',{})
    for kind in protocol['models']:
        if kind in results:continue
        save(WORK/'status.json',dict(state='loading_encoder',model=kind,checkpoint_step=protocol['checkpoint_step'],updated_at_utc=now()))
        model=encoder(cfg,kind,protocol['checkpoint_step'])
        path=None
        for batch in (64,32,16,8,4):
            try:
                path=extract_once(model,rows,cfg,kind,batch,protocol);break
            except torch.OutOfMemoryError:
                save(WORK/'status.json',dict(state='retrying_smaller_inference_batch',batch=batch,model=kind,updated_at_utc=now()))
            gc.collect();torch.cuda.empty_cache()
        if path is None:raise RuntimeError('evaluation_gpu_capacity_insufficient')
        del model;gc.collect();torch.cuda.empty_cache()
        save(WORK/'status.json',dict(state='computing_official_metrics',model=kind,updated_at_utc=now()))
        features=np.load(path,mmap_mode='r',allow_pickle=False)
        if not np.isfinite(features).all():raise RuntimeError('invalid_completed_feature_cache')
        results[kind]=run_metrics(kind,features,rows,tables)
        save(WORK/'results.json',dict(state='in_progress',checkpoint_step=protocol['checkpoint_step'],models=results,updated_at_utc=now()))
    deltas={}
    for variant in ['provided_split_literal','patient_disjoint_query_supplement']:
        deltas[variant]={case:{k:results['craniofacial_cpt'][variant][case]['macro'][k]-results['original_dinov3'][variant][case]['macro'][k]
            for k in results['original_dinov3'][variant][case]['macro']} for case in results['original_dinov3'][variant]}
    save(WORK/'results.json',dict(state='complete',checkpoint_step=protocol['checkpoint_step'],protocol=str(WORK/'protocol.json'),
        models=results,cpt_minus_original=deltas,exploratory=True,provided_split_has_known_self_matches=True,
        matched_gmdb_supervised_training=False,updated_at_utc=now()))
    save(WORK/'status.json',dict(state='complete',checkpoint_step=protocol['checkpoint_step'],updated_at_utc=now()))
    notify('GMDB frozen checkpoint evaluation complete','GMDB checkpoint evaluation completed',
        'The frozen original DINOv3 and continued-pretraining encoder were evaluated with the fixed local GMDB split files. Aggregate reports and the split-overlap audit remain on HPC. Active pretraining was not interrupted.')

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(WORK/'status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise
