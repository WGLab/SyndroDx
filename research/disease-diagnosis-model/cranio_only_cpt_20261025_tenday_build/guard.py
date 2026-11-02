"""Predeclared development guard; its tolerance is not a clinical margin."""
import math
KEYS=('disease_retrieval_top1','disease_ridge_top1','structural_retrieval_top1','structural_ridge_top1')
EPSILON=1e-12
LARGE_DECLINE=.02


def decide(current,original,previous=None,*,comparisons=None,previous_supported=()):
    for label,values in [('current',current),('original',original)]:
        for key in KEYS:
            v=values.get(key)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1:
                return dict(continue_training=False,state='paused_invalid_validation',reason=label+'_'+key)
    warnings=[];supported=[];stop=[]
    for key in KEYS:
        delta=current[key]-original[key]
        category,method=key.split('_')[:2]
        comparison=(comparisons or {}).get(category,{}).get(method,{})
        ci=comparison.get('paired_patient_bootstrap_95_interval')
        if not isinstance(ci,list) or len(ci)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in ci) or ci[0]>ci[1]:
            return dict(continue_training=False,state='paused_invalid_validation',reason='missing_or_invalid_interval_'+key)
        if delta < -EPSILON:
            warnings.append(dict(metric=key,delta=delta))
            if delta < -.10:stop.append(key)
            if ci[1]<0:
                supported.append(key)
                if delta < -LARGE_DECLINE or key in previous_supported:stop.append(key)
    return dict(continue_training=not stop,state='paused_validation_regression' if stop else 'warning_validation_change' if warnings else 'passed',
                declines=warnings,supported_declines=supported,stopping_metrics=stop,
                tolerance_is_development_heuristic=True)
