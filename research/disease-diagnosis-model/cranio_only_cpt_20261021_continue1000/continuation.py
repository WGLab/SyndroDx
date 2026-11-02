"""Full-state resume checks and deterministic sample-index continuation."""
import math
import torch


def loader_options(options,completed_steps):
    options=dict(options)
    if options.get('sampler_advance')!=completed_steps*options['batch_size']:
        raise RuntimeError('resume_sampler_offset_mismatch')
    # Upstream adds start_iter to the seed AND advances the sampler. Preserve
    # the original permutation while retaining its correct sample offset.
    options['seed']-=completed_steps
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

