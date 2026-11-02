"""Explicit identity-CPT stopping rules based on aggregates, never GMDB test."""
import datetime
import math
import statistics

def decide(rows,reports,policy,current_time=None):
    if not rows:return dict(advance=False,reason='awaiting_training_metrics')
    rows=sorted({r['completed_steps']:r for r in rows}.values(),key=lambda r:r['completed_steps'])
    latest=rows[-1];step=latest['completed_steps']
    if any(not math.isfinite(r['loss']) for r in rows):return dict(advance=False,reason='numerical_failure_requires_recovery')
    current_time=current_time or datetime.datetime.now(datetime.timezone.utc)
    start=datetime.datetime.fromisoformat(rows[0]['updated_at_utc'])
    days=(current_time-start).total_seconds()/86400
    if days>=policy['max_wall_days']:
        return dict(advance=True,reason='wall_budget_reached_not_claimed_converged',completed_steps=step,elapsed_days=days)
    if step<policy['min_steps']:return dict(advance=False,reason='minimum_training_not_reached',completed_steps=step)
    width=policy['window_steps'];end=(step//width)*width;means=[]
    compared=[r for r in rows if end-policy['windows']*width<r['completed_steps']<=end]
    if len({r['world_size'] for r in compared})>1:
        return dict(advance=False,reason='world_size_changed_in_loss_window')
    for i in range(policy['windows']-1,-1,-1):
        low=end-(i+1)*width;high=end-i*width
        window=[r for r in rows if low<r['completed_steps']<=high]
        if len(window)<width//20:return dict(advance=False,reason='insufficient_logged_window_coverage')
        if len({r['world_size'] for r in window})>1:return dict(advance=False,reason='world_size_changed_in_loss_window')
        means.append(statistics.mean(r['loss'] for r in window))
    loss_flat=all(abs(b-a)<=max(policy['absolute_loss_tolerance'],policy['relative_loss_tolerance']*abs(a)) for a,b in zip(means,means[1:]))
    values=sorted([r for r in reports if r.get('state')=='complete' and r['completed_steps']<=step],key=lambda r:r['completed_steps'])
    n=policy['validation_patience'];validation_flat=False
    if len(values)>n and step-values[-1]['completed_steps']<=500:
        old=max(r['metrics']['disease']['ridge']['top1'] for r in values[:-n])
        recent=max(r['metrics']['disease']['ridge']['top1'] for r in values[-n:])
        validation_flat=recent<=old+policy['validation_min_gain']
    return dict(advance=loss_flat and validation_flat,reason='loss_and_development_plateau' if loss_flat and validation_flat else 'continue_training',
        completed_steps=step,loss_window_means=means,loss_plateau=loss_flat,development_plateau=validation_flat,elapsed_days=days)

def choose_checkpoint(candidates):
    eligible=[c for c in candidates if c.get('step',0)>0 and c.get('complete') and math.isfinite(c.get('score',float('nan')))]
    if not eligible:return None
    return max(eligible,key=lambda c:(c['score'],-c['step']))
