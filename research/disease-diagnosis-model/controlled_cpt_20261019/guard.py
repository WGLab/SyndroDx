"""Fail-closed metric gate; pure Python, independently testable."""
import math

KEYS = ('disease_retrieval_top1', 'disease_ridge_top1',
        'structural_retrieval_top1', 'structural_ridge_top1')
EPSILON = 1e-12  # Floating-point equality only, not a clinical decline tolerance.


def decide(current, original, previous):
    for name, values in [('current', current), ('original', original), ('previous', previous)]:
        for key in KEYS:
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                return dict(continue_training=False, state='paused_invalid_validation',
                            reason=name+'_'+key+'_missing_or_invalid')
    declines = []
    for key in KEYS:
        for label, reference in [('original', original), ('last_accepted', previous)]:
            if current[key] < reference[key] - EPSILON:
                declines.append(dict(metric=key, compared_with=label,
                                     delta=current[key]-reference[key]))
    return dict(continue_training=not declines,
                state='paused_validation_regression' if declines else 'passed',
                declines=declines)
