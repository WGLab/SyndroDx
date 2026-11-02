"""Operational SSL loss plateau monitoring, not a representation-quality claim."""
import json
import math
import statistics
from pathlib import Path
from common import load,save,now

COMPONENTS=('dino_local_crops_loss','dino_global_crops_loss','ibot_loss')
POLICY=dict(window_steps=1000,half_window_steps=500,total_relative_change=.0025,
            component_relative_change=.005,recent_total_cv=.01,checks_required=3,
            check_interval_steps=250,minimum_new_steps_same_world=1000)


def read_records(path):
    records={}
    if not Path(path).exists():return []
    with Path(path).open() as stream:
        for line in stream:
            try:value=json.loads(line)
            except ValueError:continue
            if isinstance(value,dict) and isinstance(value.get('iteration'),int):records[value['iteration']]=value
    return [records[k] for k in sorted(records)]


def assess(records,step,resume_step,previous=None):
    previous=previous or {}
    output=dict(state='warming_convergence_window',completed_steps=step,plateau=False,
        consecutive_stable_checks=0,policy=POLICY,not_a_representation_quality_claim=True)
    if step-resume_step<POLICY['minimum_new_steps_same_world']:return output
    # Official metrics are recorded every10 steps; reject incomplete windows.
    older=[r for r in records if step-1000<=r['iteration']<step-500]
    recent=[r for r in records if step-500<=r['iteration']<step]
    if min(len(older),len(recent))<45:
        output['state']='incomplete_convergence_window';return output
    if any(b['iteration']-a['iteration']>20 for a,b in zip(older+recent,(older+recent)[1:])):
        output['state']='incomplete_convergence_window';return output
    values={}
    for name in ('total_loss',*COMPONENTS):
        for window in (older,recent):
            if any(not isinstance(r.get(name),(int,float)) or not math.isfinite(r[name]) for r in window):
                output['state']='invalid_convergence_metrics';return output
        before=statistics.mean(r[name] for r in older);after=statistics.mean(r[name] for r in recent)
        values[name]=dict(older_mean=before,recent_mean=after,
            absolute_relative_change=abs(after-before)/max(abs(before),1e-8))
    lr=[r.get('lr') for r in older+recent]
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in lr) or max(lr)-min(lr)>max(abs(lr[0])*1e-9,1e-15):
        output['state']='learning_rate_changed_within_window';return output
    cv=statistics.pstdev(r['total_loss'] for r in recent)/max(abs(values['total_loss']['recent_mean']),1e-8)
    stable=(values['total_loss']['absolute_relative_change']<POLICY['total_relative_change'] and
        all(values[name]['absolute_relative_change']<POLICY['component_relative_change'] for name in COMPONENTS) and
        cv<POLICY['recent_total_cv'])
    old_step=previous.get('completed_steps',-1)
    count=(previous.get('consecutive_stable_checks',0) if step-old_step==POLICY['check_interval_steps'] else 0)+1 if stable else 0
    output.update(state='operational_loss_plateau' if count>=POLICY['checks_required'] else 'stable_loss_check' if stable else 'loss_still_changing',
        plateau=count>=POLICY['checks_required'],consecutive_stable_checks=count,windows=values,recent_total_cv=cv)
    return output


def check(folder,step,resume_step):
    previous=load(folder/'convergence/latest.json')
    # The same checkpoint cannot count as two independent stable checks.
    if previous.get('completed_steps')==step:return previous
    report=assess(read_records(folder/'training_metrics.json'),step,resume_step,previous)
    report['updated_at_utc']=now()
    save(folder/'convergence'/f'step_{step:06d}.json',report)
    save(folder/'convergence/latest.json',report)
    return report
