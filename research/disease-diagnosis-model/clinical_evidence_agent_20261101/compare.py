"""Matched endpoint comparison. Patient-level inputs stay local; stdout is aggregate.

This evaluates exported rankings. It does not generate genetic scores or expose
gold labels to the agent. All arms must share candidate universe, case decision
time, query budget and eligible patient groups. Research truth must be adjudicated.
"""
import argparse
import collections
import random
from common import *
from engine import timestamp


def evaluate(records, train_groups, expected_arms, budget, bootstrap=2000):
    if len(set(expected_arms)) != len(expected_arms) or len(expected_arms) < 2 or budget < 0:
        raise ValueError('invalid_comparison_protocol')
    if not records:
        raise ValueError('no_verified_evaluation_cases')
    by_arm = {arm: {} for arm in expected_arms}
    for r in records:
        if r['arm'] not in by_arm:
            raise ValueError('unexpected_arm')
        if not r.get('patient_key') or not r.get('group_id') or r['group_id'] in train_groups:
            raise ValueError('training_group_overlap_or_missing_identity')
        if not r.get('split_audit_passed') or not r.get('truth_adjudicated') or not r.get('truth_source_sha256'):
            raise ValueError('unverified_truth_or_split')
        if r.get('disposition') not in {'validation', 'test', 'protected'}:
            raise ValueError('not_an_evaluation_patient')
        if r.get('budget') != budget or not 0 <= r.get('actions_used', -1) <= budget:
            raise ValueError('unequal_budget_or_invalid_action_count')
        if timestamp(r.get('decision_time')) is None:
            raise ValueError('unresolved_decision_time')
        universe = r.get('candidate_universe', [])
        ranking = r.get('ranking', [])
        if not universe or len(set(universe)) != len(universe) or len(set(ranking)) != len(ranking) or not set(ranking) <= set(universe):
            raise ValueError('invalid_disease_candidate_ranking')
        if not r.get('truth_ids'):
            raise ValueError('missing_gold_truth')
        if r['patient_key'] in by_arm[r['arm']]:
            raise ValueError('duplicate_patient_in_arm')
        by_arm[r['arm']][r['patient_key']] = r
    reference = by_arm[expected_arms[0]]
    if any(set(rows) != set(reference) for rows in by_arm.values()):
        raise ValueError('unmatched_patient_cohorts')
    for arm, rows in by_arm.items():
        for key, r in rows.items():
            baseline = reference[key]
            for name in ['group_id', 'truth_ids', 'truth_source_sha256', 'candidate_universe', 'decision_time', 'disposition', 'initial_information_sha256', 'available_information_sha256']:
                if not r.get(name) or r[name] != baseline.get(name):
                    raise ValueError('unmatched_evaluation_context')
    keys = sorted(reference)
    groups = collections.defaultdict(list)
    for i, key in enumerate(keys):
        groups[reference[key]['group_id']].append(i)
    hits, metrics = {}, {}
    for arm, rows in by_arm.items():
        hits[arm] = {k: [int(bool(set(rows[p]['ranking'][:k]) & set(rows[p]['truth_ids']))) for p in keys] for k in [1, 5]}
        metrics[arm] = {f'top{k}_patient_micro': sum(v) / len(v) for k, v in hits[arm].items()}
        # For macro statistics, each patient's adjudicated disease set is a stratum.
        disease_groups = collections.defaultdict(list)
        for i, p in enumerate(keys):
            disease_groups[tuple(sorted(rows[p]['truth_ids']))].append(i)
        for k, v in hits[arm].items():
            metrics[arm][f'top{k}_adjudicated_disease_set_macro'] = sum(sum(v[i] for i in ix) / len(ix) for ix in disease_groups.values()) / len(disease_groups)
        metrics[arm]['mean_actions_used'] = sum(rows[p]['actions_used'] for p in keys) / len(keys)
    rng = random.Random(20261101)
    group_indices = list(groups.values())
    draws = [[i for _ in group_indices for i in rng.choice(group_indices)] for _ in range(bootstrap)]
    differences = {}
    base = expected_arms[0]
    for arm in expected_arms[1:]:
        differences[arm] = {}
        for k in [1, 5]:
            delta = [b - a for a, b in zip(hits[base][k], hits[arm][k])]
            samples = sorted(sum(delta[i] for i in ix) / len(ix) for ix in draws)
            differences[arm][f'top{k}'] = {'delta_patient_micro': sum(delta) / len(delta),
                'paired_patient_group_bootstrap_ci95': [samples[int(.025 * (len(samples) - 1))], samples[int(.975 * (len(samples) - 1))]] if samples else None}
    return {'state': 'matched_endpoint_comparison', 'patients': len(keys), 'patient_groups': len(groups),
            'arms': metrics, 'reference_arm': base, 'differences_from_reference': differences,
            'truth_outside_candidate_universe_patients': sum(not set(reference[p]['truth_ids']) & set(reference[p]['candidate_universe']) for p in keys),
            'out_of_universe_truth_counted_as_miss': True, 'training_group_overlap': 0,
            'interval_interpretation': 'conditional on frozen models/policy; not adjusted for prior development selection',
            'patient_records_in_output': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = load(args.protocol)
    result = evaluate(list(jsonl(args.records)), set(protocol['train_group_ids']), protocol['arms'], protocol['action_budget'])
    result.update(input_sha256=sha(args.records), protocol_sha256=sha(args.protocol), updated_at_utc=now())
    save(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    guarded_main(main)
