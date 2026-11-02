"""Full-state resume checks and deterministic sample-index continuation."""
import math
import torch


def loader_options(options,completed_steps,source_global_batch=512,target_world=20):
    options=dict(options)
    if options.get('sampler_advance')!=completed_steps*options['batch_size']:
        raise RuntimeError('resume_sampler_offset_mismatch')
    # Upstream adds start_iter to the seed AND advances the sampler. Preserve
    # the original permutation while retaining its correct sample offset.
    options['seed']-=completed_steps
    consumed=completed_steps*source_global_batch
    if consumed%target_world:raise RuntimeError('nonintegral_resume_global_sample_offset')
    options['sampler_advance']=consumed//target_world
    return options


def restore_full_state(path,*,model,optimizer,loader,completed_steps,**kwargs):
    if optimizer is None:raise RuntimeError('full_optimizer_required_for_resume')
    kwargs['strict_loading']=True
    iteration=loader(path,model=model,optimizer=optimizer,**kwargs)
    if iteration!=completed_steps-1:raise RuntimeError('checkpoint_iteration_mismatch')
    values=[]
    for state in optimizer.state.values():
        if 'step' in state:
            step=state['step']
            if hasattr(step,'to_local'):step=step.to_local()
            value=float(step.item() if isinstance(step,torch.Tensor) else step)
            if not math.isfinite(value) or not 0<value<=completed_steps:
                raise RuntimeError('invalid_restored_optimizer_step')
            values.append(value)
    if not values or max(values)!=completed_steps:
        raise RuntimeError('optimizer_state_was_not_restored')
    return iteration,dict(optimizer_state_entries=len(values),minimum_optimizer_step=min(values),
                           maximum_optimizer_step=max(values),strict_loading=True)



def compare_resharded_features(source,target,source_world,target_world):
    import numpy as np
    before=[np.load(source/f'rank_{i:02d}.private.npy',allow_pickle=False) for i in range(source_world)]
    after=[np.load(target/f'rank_{i:02d}.private.npy',allow_pickle=False) for i in range(target_world)]
    n=sum(len(x) for x in before)
    if n!=sum(len(x) for x in after) or not n:raise RuntimeError('restored_feature_count_mismatch')
    dim=before[0].shape[1]
    full=np.empty((n,dim),dtype=np.float32)
    for i,shard in enumerate(before):
        if full[i::source_world].shape!=shard.shape:raise RuntimeError('source_feature_shard_mismatch')
        full[i::source_world]=shard
    minimum=1.
    for i,shard in enumerate(after):
        expected=full[i::target_world]
        if expected.shape!=shard.shape:raise RuntimeError('restored_feature_shape_mismatch')
        if len(shard):minimum=min(minimum,float(np.min(np.sum(expected*shard,axis=1))))
    return minimum
