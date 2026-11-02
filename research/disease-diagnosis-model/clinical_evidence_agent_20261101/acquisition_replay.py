"""Retrospective masked-HPO replay; not a clinical timeline or an LLM trial.

Only the answer resolver sees hidden recorded findings. Unknown consumes one
action and never becomes a negative. All records and traces stay on HPC.
"""
import collections
import fcntl
import traceback
import numpy as np
import torch
from common import *
from frozen_v1 import FrozenV1
from hpo_questions import select

WORK = ROOT / 'acquisition_replay_v1'


def observations(ids, states):
    values = collections.defaultdict(set)
    for i, s in zip(ids, states):
        if int(i) and int(s) in (0, 1):
            values[int(i)].add(int(s))
    return {i: next(iter(s)) for i, s in values.items() if len(s) == 1}


def initial_observations(recorded, k, seed, row):
    positives = sorted(i for i, s in recorded.items() if s == 0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, row]))
    positives = list(rng.permutation(positives))[:k]
    return {i: s for i, s in recorded.items() if s == 1 or i in positives}


def answer(recorded, known, asked, term):
    if term is None:
        return 'no_question'
    if term in asked or term in known:
        raise ValueError('repeated_question')
    asked.add(term)
    if term not in recorded:
        return 'unknown'
    known[term] = recorded[term]
    return 'present' if recorded[term] == 0 else 'absent'


def pack(records, inverse):
    ids = np.zeros((len(records), 128), np.int64)
    states = np.zeros_like(ids)
    for row, record in enumerate(records):
        items = sorted(record.items(), key=lambda x: (x[1], inverse[x[0]]))
        if len(items) > 128:
            raise ValueError('too_many_observations')
        for col, (term, state) in enumerate(items):
            ids[row, col], states[row, col] = term, state
    return ids, states


def select_question(policy, probabilities, labels, excluded, pool, fixed_order,
                    associations, source_hash, term_index, inverse, rng):
    if policy == 'adaptive':
        scores = {label: float(np.log(max(float(p), 1e-30)))
                  for label, p in zip(labels, probabilities)}
        action = select(scores, {inverse[t] for t in excluded}, associations, source_hash)
        return term_index[action['concept_id']] if action else None
    if policy == 'fixed':
        return next((term for term in fixed_order if term not in excluded), None)
    if policy == 'random':
        remaining = [term for term in pool if term not in excluded]
        return int(rng.choice(remaining)) if remaining else None
    raise ValueError('unsupported_policy')


def hit_matrix(probabilities, labels):
    order = np.argsort(-probabilities, axis=1, kind='stable')
    return np.stack([(order[:, :k] == labels[:, None]).any(1) for k in (1, 5)], -1)


def metrics(hits, labels):
    # Repeated masks are averaged per patient, never counted as extra patients.
    h = hits.mean(0) if hits.ndim == 3 else hits
    return {f'top{k}_{kind}': float(value) for j, k in enumerate((1, 5))
            for kind, value in [('micro', h[:, j].mean()),
                                ('macro', np.mean([h[labels == d, j].mean() for d in np.unique(labels)]))]}


def paired(a, b, labels):
    delta = a.astype(float).mean(0) - b.astype(float).mean(0)
    strata = np.stack([delta[labels == d].mean(0) for d in np.unique(labels)])
    rng = np.random.default_rng(20261102)
    samples = strata[rng.integers(0, len(strata), (2000, len(strata)))].mean(1)
    return {f'top{k}': {'delta_macro': float(strata[:, j].mean()),
                        'paired_disease_bootstrap_ci95': np.quantile(samples[:, j], [.025, .975]).tolist()}
            for j, k in enumerate((1, 5))}


def infer(adapter, features, records, inverse, mode):
    ids, states = pack(records, inverse)
    pieces = [adapter.probabilities(features[start:start+32], ids[start:start+32],
                                    states[start:start+32], mode)[0]
              for start in range(0, len(features), 32)]
    return np.concatenate(pieces)


def main():
    WORK.mkdir(mode=0o700, exist_ok=True)
    lock = (WORK / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (WORK / 'results.json').exists():
        print(json.dumps({'state': 'already_complete', 'result': str(WORK / 'results.json')}))
        return
    protocol = load(WORK / 'protocol.json')
    torch.set_num_threads(2)
    source = PROJECT / 'protected_visual_fusion_20261101/private/gmdb.npz'
    receipt = load(PROJECT / 'protected_visual_fusion_20261101/input_receipt.json')
    if sha(source) != receipt['gmdb_sha256'] or receipt['patient_overlap'] != 0:
        raise ValueError('unverified_input')
    with np.load(source) as data:
        train, val = data['train'], data['validation']
        if set(train) & set(val) or len(set(val)) != len(val) or len(val) != 320:
            raise ValueError('split_guard_failed')
        features = data['features'][val].copy()
        recorded = [observations(i, s) for i, s in zip(data['ids'][val], data['states'][val])]
        train_records = [observations(i, s) for i, s in zip(data['ids'][train], data['states'][train])]
    # Labels remain solely in the evaluator; no selector or resolver receives them.
    with np.load(source) as data:
        truth = data['labels'][val].copy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save(WORK / 'status.json', {'state': 'loading_frozen_model', 'device': device, 'updated_at_utc': now()})
    adapter = FrozenV1(device)
    inverse = {v: k for k, v in adapter.terms.items()}
    association_path = ROOT / 'assets/class_hpo_associations.json'
    question_receipt = load(ROOT / 'hpo_question_receipt.json')
    if sha(association_path) != question_receipt['output_sha256']:
        raise ValueError('association_hash_mismatch')
    associations = load(association_path)['associations']
    pool = sorted({adapter.terms[t] for terms in associations.values() for t in terms})
    freq = collections.Counter(t for record in train_records for t, s in record.items() if s == 0)
    fixed_order = sorted(pool, key=lambda t: (-freq[t], inverse[t]))
    (WORK / 'private').mkdir(mode=0o700, exist_ok=True)
    np.savez_compressed(WORK / 'private/split_and_fixed_order.npz', train=train, validation=val, fixed_order=fixed_order)
    modes, policies = protocol['modes'], protocol['policies']
    masks, initial_counts, budget = protocol['mask_seeds'], protocol['initial_positive_hpo'], protocol['action_budget']
    full_hits = {mode: hit_matrix(infer(adapter, features, recorded, inverse, mode), truth) for mode in modes}
    curves, answer_stats, done = {}, {}, 0
    total = len(modes) * len(policies) * len(masks) * len(initial_counts)
    for initial_count in initial_counts:
        for seed in masks:
            initial = [initial_observations(r, initial_count, seed, row) for row, r in enumerate(recorded)]
            for mode in modes:
                baseline_prob = infer(adapter, features, initial, inverse, mode)
                for policy in policies:
                    key = f'k{initial_count}__seed{seed}__{mode}__{policy}'
                    known = [dict(r) for r in initial]
                    asked = [set() for _ in recorded]
                    rngs = [np.random.default_rng(np.random.SeedSequence([seed, row, 900])) for row in range(len(recorded))]
                    scores, hits, steps, actions = baseline_prob.copy(), [], [], []
                    hits.append(hit_matrix(scores, truth))
                    for step in range(1, budget + 1):
                        counts, row_actions = collections.Counter(), []
                        for row in range(len(recorded)):
                            term = select_question(policy, scores[row], adapter.labels, asked[row] | set(known[row]),
                                pool, fixed_order, associations, question_receipt['output_sha256'],
                                adapter.terms, inverse, rngs[row])
                            outcome = answer(recorded[row], known[row], asked[row], term)
                            counts[outcome] += 1
                            row_actions.append([term or 0, {'no_question': -1, 'unknown': 2, 'present': 0, 'absent': 1}[outcome]])
                        scores = infer(adapter, features, known, inverse, mode)
                        hits.append(hit_matrix(scores, truth))
                        steps.append(dict(counts)); actions.append(row_actions)
                    curves[key], answer_stats[key] = np.stack(hits), steps
                    np.savez_compressed(WORK / 'private' / (key + '.npz'),
                                        hits=curves[key], actions=np.array(actions), final_probabilities=scores)
                    done += 1
                    save(WORK / 'status.json', {'state': 'running', 'completed_rollouts': done,
                        'total_rollouts': total, 'device': device, 'updated_at_utc': now()})
    results, differences = {}, {}
    for k in initial_counts:
        results[str(k)], differences[str(k)] = {}, {}
        def gather(mode, policy, step):
            return np.stack([curves[f'k{k}__seed{s}__{mode}__{policy}'][step] for s in masks])
        for mode in modes:
            results[str(k)][mode] = {'initial': metrics(gather(mode, 'adaptive', 0), truth), 'policies': {}}
            differences[str(k)][mode] = {}
            for policy in policies:
                steps = []
                for step in range(budget + 1):
                    stats = metrics(gather(mode, policy, step), truth)
                    if step:
                        totals = collections.Counter()
                        for seed in masks:
                            for counts in answer_stats[f'k{k}__seed{seed}__{mode}__{policy}'][:step]:
                                totals.update(counts)
                        questions = sum(v for outcome, v in totals.items() if outcome != 'no_question')
                        stats.update(questions=questions, recorded_answers=totals['present']+totals['absent'],
                                     unknown_answers=totals['unknown'],
                                     recorded_answer_rate=(totals['present']+totals['absent'])/max(questions, 1),
                                     counts_aggregate_over_masks=True)
                    steps.append(stats)
                results[str(k)][mode]['policies'][policy] = steps
            for step in [1, 3, 5]:
                active = gather(mode, 'adaptive', step)
                differences[str(k)][mode][str(step)] = {
                    'adaptive_minus_no_questions': paired(active, gather(mode, 'adaptive', 0), truth),
                    **{f'adaptive_minus_{p}': paired(active, gather(mode, p, step), truth) for p in ['random', 'fixed']}}
        differences[str(k)]['adaptive_face_minus_hpo'] = {
            str(step): paired(gather('face_hpo', 'adaptive', step), gather('hpo_only', 'adaptive', step), truth)
            for step in [0, 1, 3, 5]}
    output = {'state': 'complete', 'purpose': 'retrospective_masked_HPO_development_simulation',
        'patients': len(recorded), 'diseases': len(np.unique(truth)), 'rollouts': done,
        'full_recorded_HPO_reference': {m: metrics(h, truth) for m, h in full_hits.items()},
        'initial_HPO_coverage': {str(k): {'at_least_k_positives': sum(sum(s == 0 for s in r.values()) >= k for r in recorded)} for k in initial_counts},
        'results': results, 'paired_comparisons': differences,
        'interval_scope': 'Disease bootstrap after averaging repeated masks per patient; reused development set, conditional on fixed policy, no multiplicity correction.',
        'clinical_timing_simulated': False, 'clinical_efficacy_established': False,
        'masked_annotations_not_clinically_reverified': True, 'unknown_imputed_as_negative': False,
        'ground_truth_used_by_policy': False, 'protected_test_accessed': False,
        'variant_evidence_used_for_ranking': False, 'LLM_controller_used': False,
        'checkpoint_updated': False, 'patient_records_in_output': False,
        'source_sha256': sha(source), 'protocol_sha256': sha(WORK / 'protocol.json'),
        'script_sha256': sha(Path(__file__)), 'model_manifest_sha256': sha(adapter.manifest_path),
        'public_association_sha256': sha(association_path), 'updated_at_utc': now()}
    save(WORK / 'results.json', output)
    save(WORK / 'status.json', {'state': 'complete', 'completed_rollouts': done, 'total_rollouts': total, 'updated_at_utc': now()})
    status = load(ROOT / 'status.json')
    status.update(acquisition_replay='complete_retrospective_development_simulation',
                  acquisition_results=str(WORK / 'results.json'), updated_at_utc=now())
    status['pending'] = [x for x in status['pending'] if x != 'agent information-acquisition benefit experiment']
    status['pending'].insert(0, 'independent validation of information-acquisition benefit')
    status['diagnostic_metrics_available'] = 'GMDB adapter check and masked-HPO development simulation; no clinical agent benefit established'
    save(ROOT / 'status.json', status)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('local_mail_runtime', PROJECT / 'transface_objective_comparison_20261028/matched_runtime.py')
        runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
        body = json.dumps({'scope': output['purpose'], 'patients': len(recorded),
                          'k3_face_hpo': results['3']['face_hpo'], 'result_path': str(WORK / 'results.json')}, indent=2)
        runtime.notify('agent masked-HPO replay completed', body)
        save(WORK / 'notification.json', {'state': 'local_notifier_invoked', 'updated_at_utc': now()})
    except Exception as e:
        save(WORK / 'notification.json', {'state': 'notifier_failed', 'error_type': type(e).__name__, 'updated_at_utc': now()})
    print(json.dumps({'state': 'complete', 'patients': len(recorded), 'rollouts': done}))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(WORK / 'status.json', {'state': 'failed', 'error_type': type(e).__name__,
             'locations': [{'file': Path(f.filename).name, 'line': f.lineno} for f in traceback.extract_tb(e.__traceback__)],
             'updated_at_utc': now()})
        raise SystemExit(1) from None
