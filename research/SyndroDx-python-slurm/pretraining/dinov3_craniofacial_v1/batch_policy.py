"""Keep the original sample budget and schedule when batch/world size changes."""
import math


def progress(policy, iteration):
    if not policy:
        return float(iteration)
    return policy['anchor_progress'] + (iteration - policy['anchor_iteration']) * (
        policy['global_batch_size'] / policy['base_global_batch_size']
    )


def rebase(previous, start, batch, world, base_total, base_global):
    if min(batch, world, base_total, base_global) <= 0 or start < 0:
        raise ValueError('invalid_execution_budget')
    anchor = progress(previous, start)
    remaining = max(0., base_total - anchor)
    steps = math.ceil(remaining * base_global / (batch * world))
    return dict(state='active', anchor_iteration=start, anchor_progress=anchor,
                base_total_iterations=base_total, base_global_batch_size=base_global,
                global_batch_size=batch * world, batch_size_per_gpu=batch,
                world_size=world, total_iterations=start + steps)


def schedule_index(policy, iteration, base_total):
    return min(base_total - 1, max(0, math.floor(progress(policy, iteration))))


def sampler_advance(policy, iteration, world):
    return round(progress(policy, iteration) * policy['base_global_batch_size'] / world)


def select_batch(receipts, baseline, min_gain=1.05, max_memory=72.):
    """Require finite full-model steps on all ranks and memory headroom."""
    candidates = [v for v in receipts if v.get('state') == 'passed'
                  and v.get('all_ranks_passed')
                  and v.get('peak_reserved_GiB', float('inf')) < max_memory
                  and v.get('median_step_seconds', 0) > 0]
    base = next((v for v in candidates if v['batch'] == baseline), None)
    if base is None:
        return baseline
    base_rate = baseline / base['median_step_seconds']
    faster = [v for v in candidates if v['batch'] > baseline
              and v['batch'] / v['median_step_seconds'] >= min_gain * base_rate]
    return max(faster, key=lambda v: v['batch'] / v['median_step_seconds'])['batch'] if faster else baseline
