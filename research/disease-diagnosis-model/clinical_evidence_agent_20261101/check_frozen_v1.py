"""Local adapter parity/coverage check on existing GMDB development arrays.

This is an integration check, not a new independent clinical test. No image,
patient row, feature, individual HPO list or ranking is printed.
"""
import collections
import numpy as np
import torch
from common import *
from frozen_v1 import FrozenV1


def metrics(probabilities, labels):
    order = np.argsort(-probabilities, axis=1, kind='stable')
    result = {}
    for k in [1, 5]:
        hit = (order[:, :k] == labels[:, None]).any(1)
        result[f'top{k}_patient_micro'] = float(hit.mean())
        result[f'top{k}_disease_macro'] = float(np.mean([hit[labels == c].mean() for c in np.unique(labels)]))
    return result


def main():
    torch.set_num_threads(2)
    source = PROJECT / 'protected_visual_fusion_20261101/private/gmdb.npz'
    receipt = load(PROJECT / 'protected_visual_fusion_20261101/input_receipt.json')
    if sha(source) != receipt['gmdb_sha256'] or receipt['patient_overlap'] != 0:
        raise ValueError('unverified_development_arrays')
    with np.load(source) as d:
        train, val = d['train'].copy(), d['validation'].copy()
        if set(train) & set(val) or len(val) != 320:
            raise ValueError('unexpected_split')
        features, ids, states, truth = [d[k][val].copy() for k in ['features', 'ids', 'states', 'labels']]
    save(ROOT / 'adapter_check_status.json', {'state': 'verifying_locked_model_assets', 'updated_at_utc': now()})
    adapter = FrozenV1(device='cpu')
    outputs, routing = {}, {}
    for mode in ['face_only', 'hpo_only', 'face_hpo']:
        scores, routes = [], []
        for start in range(0, len(features), 16):
            score, route = adapter.probabilities(features[start:start+16], ids[start:start+16], states[start:start+16], mode)
            scores.append(score)
            routes.extend(route)
        outputs[mode] = np.concatenate(scores)
        routing[mode] = dict(collections.Counter(routes))
        save(ROOT / 'adapter_check_status.json', {'state': 'checking_existing_development_inference', 'completed_modes': len(outputs), 'updated_at_utc': now()})
    # Direct call to the independently archived model verifies no hidden fusion change.
    select = np.flatnonzero(np.any(ids != 0, axis=1) & np.any(features != 0, axis=1) & ~np.any(states == 2, axis=1))[:8]
    if not len(select):
        raise ValueError('no_eligible_parity_cases')
    with torch.inference_mode():
        direct, _ = adapter.models['reference_11'](torch.from_numpy(features[select]), torch.from_numpy(ids[select]), torch.from_numpy(states[select]))
        direct = direct.float().softmax(-1).numpy()
    routed, _ = adapter.probabilities(features[select], ids[select], states[select])
    error = float(np.max(np.abs(direct - routed)))
    if error > 1e-6:
        raise ValueError('frozen_fusion_parity_failed')
    f = features[select[:1]]
    zeros = np.zeros_like(ids[select[:1]])
    fallback, r1 = adapter.probabilities(f, zeros, zeros)
    reference, _ = adapter.probabilities(f, zeros, zeros, 'face_only')
    if not np.array_equal(fallback, reference) or r1 != ['face_only']:
        raise ValueError('missing_hpo_fallback_failed')
    unresolved, _ = adapter.probabilities(f, ids[select[:1]], np.full_like(zeros, 2))
    if not np.array_equal(unresolved, reference):
        raise ValueError('unresolved_hpo_was_used')
    hpo_fallback, r2 = adapter.probabilities(np.zeros_like(f), ids[select[:1]], states[select[:1]])
    hpo_ref, _ = adapter.probabilities(f, ids[select[:1]], states[select[:1]], 'hpo_only')
    if not np.array_equal(hpo_fallback, hpo_ref) or r2 != ['hpo_only']:
        raise ValueError('missing_face_fallback_failed')
    absent, r3 = adapter.probabilities(np.zeros_like(f), zeros, zeros)
    if absent.any() or r3 != ['insufficient_input']:
        raise ValueError('empty_inputs_produced_a_ranking')
    bound = adapter.bind(f[0])
    if bound([]) != bound([{'kind': 'variant', 'assertion': 'unknown'}]):
        raise ValueError('unverified_variant_changed_scores')
    if adapter.bind(None)([]):
        raise ValueError('empty_callback_produced_scores')
    inverse = {i: term for term, i in adapter.terms.items()}
    sample = int(select[0])
    observations = [{'kind': 'hpo', 'concept_id': inverse[int(i)], 'assertion': 'present' if s == 0 else 'absent'}
                    for i, s in zip(ids[sample], states[sample]) if i]
    packed_ids, packed_states = adapter.pack(observations)
    if not np.array_equal(packed_ids, ids[sample]) or not np.array_equal(packed_states, states[sample]):
        raise ValueError('hpo_vocabulary_roundtrip_failed')
    callback = adapter.bind(features[sample])(observations)
    callback_p = np.exp([callback[label] for label in adapter.labels])
    one, _ = adapter.probabilities(features[sample:sample+1], ids[sample:sample+1], states[sample:sample+1])
    if np.max(np.abs(callback_p - one[0])) > 1e-6:
        raise ValueError('agent_callback_parity_failed')
    # Preserve private per-case outputs for reproducibility; never print them.
    private_output = PRIVATE / 'adapter_development_scores.npz'
    np.savez(private_output, **outputs, labels=truth)
    checks = {'direct_archived_fusion_parity_max_abs_error': error,
              'missing_hpo_visual_fallback_exact': True, 'missing_face_hpo_fallback_exact': True,
              'unresolved_hpo_excluded': True, 'missing_all_inputs_abstains': True,
              'variant_candidates_do_not_change_scores': True}
    checks.update(real_hpo_vocabulary_roundtrip=True, agent_ranker_callback_matches_frozen_model=True)
    result = {'state': 'passed', 'purpose': 'frozen_v1_adapter_integration_check',
              'patients': len(features), 'diseases': len(np.unique(truth)), 'classes': len(adapter.labels),
              'split': 'existing_GMDB_development', 'seed': 11, 'precision': 'CPU_FP32',
              'checks': checks, 'routing_counts': routing,
              'metrics': {mode: metrics(p, truth) for mode, p in outputs.items()},
              'not_a_new_training_result': True, 'not_an_independent_clinical_test': True,
              'agent_information_acquisition_evaluated': False, 'variant_evidence_used_for_scores': False,
              'expert_or_BCH_read': False, 'patient_records_in_output': False,
              'development_input_sha256': sha(source), 'private_output_sha256': sha(private_output),
              'release_manifest_sha256': sha(adapter.manifest_path),
              'adapter_code_sha256': sha(ROOT / 'frozen_v1.py'), 'updated_at_utc': now()}
    save(ROOT / 'adapter_check_receipt.json', result)
    save(ROOT / 'adapter_check_status.json', {'state': 'complete', 'updated_at_utc': now()})
    print(json.dumps(result))


if __name__ == '__main__':
    guarded_main(main)
