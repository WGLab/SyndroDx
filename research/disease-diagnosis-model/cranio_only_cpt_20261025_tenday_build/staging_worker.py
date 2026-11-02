"""Real pooled-patch feature diagnostics while a distributed pool assembles."""
import gc
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from omegaconf import OmegaConf
from configuration import WORK,BASE,V1
from common import load,save,digest,now,install_rope_dtype_compatibility
from pool_work import claim,finish
from evaluation_worker import fit_image
from weights import OfficialBackbone
from dinov3.models import build_model_from_cfg


def encoder(cfg,checkpoint):
    install_rope_dtype_compatibility()
    model,_=build_model_from_cfg(cfg,only_teacher=True)
    model.to_empty(device='cuda');model.rope_embed.dtype=torch.float32
    model.init_weights()
    if checkpoint is None:
        OfficialBackbone(V1/'assets/dinov3-vit7b16').load(model)
    else:
        reader=dcp.FileSystemReader(checkpoint)
        metadata=reader.read_metadata().state_dict_metadata
        state=model.state_dict();payload={};names={}
        for name,tensor in state.items():
            choices=['model.teacher.backbone.'+name,'model.model_ema.backbone.'+name]
            key=next((k for k in choices if k in metadata),None)
            if key is None:raise RuntimeError('required_teacher_backbone_tensor_missing')
            names[name]=key;payload[key]=tensor
        dcp.load(payload,storage_reader=reader,planner=DefaultLoadPlanner(allow_partial_load=True))
        model.load_state_dict({name:payload[key] for name,key in names.items()},strict=True)
    return model.eval()


def stop_requested():
    local=os.environ.get('STAGING_STOP_FILE')
    return ((local and Path(local).exists()) or (WORK/'pool_done.json').exists() or
            ((WORK/'join_requested.json').exists() and os.environ.get('POOL_PREWORK_IGNORE_GLOBAL_JOIN')!='1'))


def main():
    owner=os.environ.get('STAGING_OWNER',os.environ['SLURM_JOB_ID']);folder=WORK/'staging'
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=True
    if torch.cuda.device_count()!=1:raise RuntimeError('one_bound_gpu_required')
    manifest=load(folder/'manifest.json')
    with (folder/'inputs.private.jsonl').open() as stream:records=[json.loads(line) for line in stream]
    cfg=OmegaConf.load(WORK/'source_snapshot/config.yaml')
    model=None;current_case=None;reference=None
    while not stop_requested():
        task=claim(owner,current_case)
        if task is None:
            save(folder/f'lease_{owner}.diagnostic.json',dict(state='all_available_tasks_claimed_or_complete',updated_at_utc=now()))
            return
        key,spec=task;complete=False;quarantined=False
        try:
            if current_case!=spec['case']:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                model=encoder(cfg,manifest['cases'][spec['case']]['checkpoint'])
                current_case=spec['case']
                reference=np.load(manifest['cases'][current_case]['reference'],mmap_mode='r',allow_pickle=False)
            target=folder/(key+'.patch_mean.private.npy')
            temporary=folder/(key+'.patch_mean.partial.private.npy')
            progress=load(folder/(key+'.progress.json'))
            count=spec['stop']-spec['start'];position=progress.get('completed',0) if temporary.exists() else 0
            output=np.lib.format.open_memmap(temporary,mode='r+' if temporary.exists() else 'w+',dtype=np.float32,shape=(count,model.embed_dim))
            while position<count and not stop_requested():
                indices=list(range(spec['start']+position,min(spec['start']+position+16,spec['stop'])))
                if any(digest(records[i]['crop_path'])!=records[i]['crop_sha256'] for i in indices):
                    raise RuntimeError('released_diagnostic_image_modified')
                inputs=torch.stack([fit_image(records[i]['crop_path'],cfg) for i in indices]).cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    value=model(inputs,is_training=True)
                cls=torch.nn.functional.normalize(value['x_norm_clstoken'].float(),dim=-1)
                expected=torch.from_numpy(np.asarray(reference[indices]).copy()).cuda()
                cosine=torch.sum(cls*expected,dim=-1).min()
                patch=torch.nn.functional.normalize(value['x_norm_patchtokens'].float().mean(1),dim=-1)
                if not bool(torch.isfinite(patch).all()) or float(cosine)<.999:
                    save(folder/f'lease_{owner}.verification_failure.json',dict(state='failed',case=int(current_case),minimum_reference_cosine=float(cosine),patch_features_finite=bool(torch.isfinite(patch).all()),updated_at_utc=now()))
                    raise RuntimeError('diagnostic_encoder_reference_or_feature_failure')
                output[position:position+len(indices)]=patch.cpu().numpy();position+=len(indices)
                output.flush()
                save(folder/(key+'.progress.json'),dict(completed=position,total=count,updated_at_utc=now()))
                save(folder/f'lease_{owner}.diagnostic.json',dict(state='extracting_patch_features',checkpoint=int(current_case),completed_in_task=position,updated_at_utc=now()))
                del inputs,value,cls,expected,patch
            output.flush();del output
            if position==count:
                temporary.replace(target);complete=True
        except RuntimeError as error:
            if str(error)!='diagnostic_encoder_reference_or_feature_failure':raise
            # Keep the strict threshold. Do not admit or repeatedly retry this
            # chunk; continue useful work on the independent remaining tasks.
            quarantined=True
        finally:
            finish(key,owner,complete=complete,quarantined=quarantined)
    save(folder/f'lease_{owner}.diagnostic.json',dict(state='yielded_for_shared_training',updated_at_utc=now()))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(WORK/'staging'/('lease_'+os.environ['SLURM_JOB_ID']+'.failure.json'),dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise
