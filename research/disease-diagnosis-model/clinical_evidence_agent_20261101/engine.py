"""Bounded evidence-acquisition policy, not a diagnostic LLM or a trained model.

The runner has no network, image loading, shell execution or clinical text tools.
It consumes typed, reviewed observations and preserves the baseline ranker.
Public assertions alone never enter the ranker's patient observation list.
"""
from copy import deepcopy
from datetime import datetime
import math
import re


def timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result if result.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def observation_reasons(evidence, cutoff):
    """Eligibility for an observation is separate from pathogenicity/diagnosis."""
    reasons = []
    if evidence.get('kind') not in {'hpo', 'variant'}:
        reasons.append('unsupported_observation_kind')
    if evidence.get('review_status') != 'verified':
        reasons.append('source_observation_not_verified')
    if evidence.get('experiencer') != 'patient':
        reasons.append('experiencer_unverified')
    if evidence.get('source_kind') != 'clinical_record':
        reasons.append('not_a_patient_clinical_record')
    if not evidence.get('source_ref') or not re.fullmatch(r'[0-9a-f]{64}', evidence.get('source_sha256', '')):
        reasons.append('missing_source_provenance')
    if not evidence.get('reviewer_ref'):
        reasons.append('missing_review_provenance')
    c = timestamp(cutoff)
    available = timestamp(evidence.get('available_at'))
    event = timestamp(evidence.get('event_at'))
    if c is None:
        reasons.append('missing_decision_time')
    if available is None or event is None:
        reasons.append('missing_observation_or_availability_time')
    elif event > available:
        reasons.append('inconsistent_observation_time')
    elif c is not None and available > c:
        reasons.append('not_available_at_decision_time')
    kind, assertion = evidence.get('kind'), evidence.get('assertion')
    allowed = {'hpo': {'present', 'absent'}, 'variant': {'detected', 'not_detected'}}
    if assertion not in allowed.get(kind, set()):
        reasons.append('assertion_unresolved')
    if kind == 'hpo' and not re.fullmatch(r'HP:\d{7}', evidence.get('concept_id', '')):
        reasons.append('invalid_hpo_identifier')
    if kind == 'variant':
        if not evidence.get('concept_id'):
            reasons.append('missing_variant_identifier')
        if not evidence.get('identity_verified'):
            reasons.append('variant_identity_unverified')
        if not evidence.get('assay_scope_verified'):
            reasons.append('variant_assay_scope_unverified')
    return sorted(set(reasons))


def partition_evidence(evidence, cutoff):
    """Quarantine contradictions rather than choose the more convenient polarity."""
    accepted, pending = [], []
    for item in evidence:
        reasons = observation_reasons(item, cutoff)
        if reasons:
            pending.append({'evidence': deepcopy(item), 'reasons': reasons})
        else:
            accepted.append(deepcopy(item))
    polarities = {}
    for item in accepted:
        key = (item['kind'], item.get('concept_id'))
        polarities.setdefault(key, set()).add(item['assertion'])
    conflicts = {k for k, values in polarities.items() if len(values) > 1}
    clean = []
    seen = set()
    for item in accepted:
        key = (item['kind'], item.get('concept_id'))
        if key in conflicts:
            pending.append({'evidence': item, 'reasons': ['conflicting_patient_assertions']})
        elif key not in seen:
            clean.append(item)
            seen.add(key)
    return clean, pending


def entropy(values):
    return -sum(p * math.log(p) for p in values if p > 0)


def choose_hpo_question(scores, profiles, excluded):
    """Expected information gain ONLY when supplied frequencies are complete.

    Values are planning weights, not calibrated clinical probabilities. Missing
    disease-term frequencies are never replaced by zero. Profiles must be
    versioned independently by the caller; synthetic replay uses toy profiles.
    """
    if not scores or not profiles:
        return None
    diseases = sorted(scores)
    vals = [float(scores[d]) for d in diseases]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError('non_finite_ranker_score')
    weights = [math.exp(v - max(vals)) for v in vals]
    total = sum(weights)
    prior = [w / total for w in weights]
    options = []
    for term, frequencies in profiles.items():
        if term in excluded or not re.fullmatch(r'HP:\d{7}', term):
            continue
        if any(d not in frequencies for d in diseases):
            continue
        f = [float(frequencies[d]) for d in diseases]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in f):
            continue
        yes = sum(p * q for p, q in zip(prior, f))
        if yes <= 0 or yes >= 1:
            continue
        post_yes = [p * q / yes for p, q in zip(prior, f)]
        post_no = [p * (1 - q) / (1 - yes) for p, q in zip(prior, f)]
        gain = entropy(prior) - yes * entropy(post_yes) - (1 - yes) * entropy(post_no)
        if gain > 1e-12:
            options.append((gain, term))
    if not options:
        return None
    gain, term = sorted(options, key=lambda x: (-x[0], x[1]))[0]
    return {'tool': 'request_hpo_observation', 'concept_id': term,
            'planning_information_gain': gain, 'calibrated_probability': False}


def run(case, ranker=None, resolver=None, profiles=None, max_actions=5, purpose='development', question_selector=None):
    """Callbacks must be trusted HPC-local code; no tools are selected by text.

    Actual scores require an explicit ranker adapter. Without it, a real-data
    run only reports evidence readiness; it never fabricates a differential.
    A resolver returns real reviewed records or None. Missing answers remain
    unknown. The supplied case must carry a group-level split audit.
    """
    if not isinstance(max_actions, int) or not 0 <= max_actions <= 20:
        raise ValueError('invalid_action_budget')
    if purpose not in {'development', 'evaluation'}:
        raise ValueError('invalid_purpose')
    if not case.get('split_audit_passed') or not case.get('group_id'):
        raise ValueError('missing_split_audit')
    if purpose == 'development' and case.get('disposition') != 'train_existing_split':
        raise ValueError('protected_or_nontraining_patient')
    evidence = deepcopy(case.get('evidence', []))
    if not case.get('patient_key') or any(e.get('patient_key') != case['patient_key'] or not e.get('evidence_id') for e in evidence):
        raise ValueError('missing_or_mismatched_patient_evidence')
    if len({e['evidence_id'] for e in evidence}) != len(evidence):
        raise ValueError('duplicate_evidence_identifier')
    trace, asked = [], set()
    profiles = profiles or {}
    scores = {}
    reason = 'action_budget_exhausted'
    for step in range(max_actions + 1):
        accepted, pending = partition_evidence(evidence, case.get('decision_time'))
        if ranker is not None:
            scores = ranker(deepcopy(accepted))
            if not isinstance(scores, dict) or any(not math.isfinite(float(v)) for v in scores.values()):
                raise ValueError('invalid_ranker_output')
        if step == max_actions:
            break
        if timestamp(case.get('decision_time')) is None:
            action = {'tool': 'request_decision_time'}
            trace.append({'action': action, 'outcome': 'requires_case_review'})
            reason = 'missing_decision_time'
            break
        todo = next((p for p in pending if p['evidence'].get('evidence_id') not in asked), None)
        if todo:
            action = {'tool': 'request_evidence_verification',
                      'evidence_id': todo['evidence'].get('evidence_id'), 'reasons': todo['reasons']}
            key = action['evidence_id']
        else:
            excluded = asked | {e.get('concept_id') for e in evidence}
            action = question_selector(scores, excluded) if question_selector else choose_hpo_question(scores, profiles, excluded)
            if action is None:
                reason = 'no_verified_information_action_available'
                break
            if action.get('tool') != 'request_hpo_observation' or not re.fullmatch(r'HP:\d{7}', action.get('concept_id', '')) or action['concept_id'] in excluded:
                raise ValueError('invalid_or_repeated_question_action')
            key = action['concept_id']
        asked.add(key)
        answer = resolver(deepcopy(action)) if resolver is not None else None
        outcome = 'unavailable_unknown'
        if answer is not None:
            correct_target = (answer.get('concept_id') == action.get('concept_id') and answer.get('kind') == 'hpo') if action['tool'] == 'request_hpo_observation' else answer.get('evidence_id') == action.get('evidence_id')
            same_patient = answer.get('patient_key') == case.get('patient_key') and bool(case.get('patient_key'))
            id_collision = any(e.get('evidence_id') == answer.get('evidence_id') and
                               (e.get('kind'), e.get('concept_id')) != (answer.get('kind'), answer.get('concept_id'))
                               for e in evidence)
            if not answer.get('evidence_id') or id_collision or not correct_target or not same_patient or observation_reasons(answer, case.get('decision_time')):
                outcome = 'rejected_unverified_or_wrong_target'
            else:
                evidence = [e for e in evidence if e.get('evidence_id') != answer.get('evidence_id')]
                evidence.append(deepcopy(answer))
                outcome = 'verified_observation_added'
        trace.append({'action': action, 'outcome': outcome})
    accepted, pending = partition_evidence(evidence, case.get('decision_time'))
    return {'state': 'research_evidence_replay', 'stop_reason': reason,
            'differential': sorted(scores.items(), key=lambda kv: (-float(kv[1]), kv[0])),
            'score_semantics': 'adapter_scores_not_calibrated_clinical_probabilities',
            'accepted_observation_ids': [x['evidence_id'] for x in accepted],
            'pending_evidence': pending, 'trace': trace,
            'diagnosis_confirmed': False, 'training_admitted': False,
            'clinical_recommendations_generated': False}
