"""Use existing public positive HPO associations to propose research questions.

This relevance heuristic is not expected information gain or a frequency model.
An unlisted association is unknown, not negative. Answers must come from the
patient's actual verified evidence, never from the public disease profile.
"""
import math
import numpy as np
from common import *


def select(scores, excluded, associations, source_sha256, top_candidates=10):
    if not scores:
        return None
    ranked = sorted(scores, key=lambda d: (-scores[d], d))[:top_candidates]
    vals = [float(scores[d]) for d in ranked]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError('invalid_question_scores')
    weights = {d: math.exp(float(scores[d]) - max(vals)) for d in ranked}
    normalizer = sum(weights.values())
    supports = {}
    for disease in ranked:
        for term in associations.get(disease, []):
            if term not in excluded:
                supports.setdefault(term, set()).add(disease)
    if not supports:
        return None
    relevance = {term: sum(weights[d] / normalizer for d in diseases) * math.log1p(len(ranked) / len(diseases))
                 for term, diseases in supports.items()}
    term = min(relevance, key=lambda t: (-relevance[t], t))
    return {'tool': 'request_hpo_observation', 'concept_id': term,
            'selection_method': 'positive_association_relevance_heuristic',
            'planning_relevance_score': relevance[term], 'supporting_candidate_classes': sorted(supports[term]),
            'public_association_sha256': source_sha256,
            'unlisted_association_means_absent': False, 'calibrated_probability': False}


def selector():
    path = ROOT / 'assets/class_hpo_associations.json'
    receipt = load(ROOT / 'hpo_question_receipt.json')
    if sha(path) != receipt['output_sha256']:
        raise ValueError('public_profile_index_changed')
    associations = load(path)['associations']
    return lambda scores, excluded: select(scores, excluded, associations, receipt['output_sha256'])


def main():
    staged = PROJECT / 'hpo_fusion_staged_20261031'
    receipt = load(staged / 'data_receipt.json')
    source = staged / 'private/profiles.npz'
    if sha(source) != receipt['files']['profiles.npz']:
        raise ValueError('public_profiles_receipt_mismatch')
    registry = load(PROJECT / 'model_releases/CURRENT.json')
    manifest_path = Path(registry['manifest'])
    if sha(manifest_path) != registry['sha256']:
        raise ValueError('release_manifest_mismatch')
    manifest = load(manifest_path)
    def asset(name):
        item = manifest['artifacts'][name]
        path = manifest_path.parent / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('frozen_asset_changed')
        return load(path)
    labels = asset('disease_vocabulary')['labels']
    terms = asset('HPO_term_index')['term_ids']
    inverse = {i + 1: term for i, term in enumerate(terms)}
    with np.load(source) as data:
        if len(labels) != receipt['counts']['GMDB_classes'] or len(data['ids']) < len(labels):
            raise ValueError('unexpected_public_profile_layout')
        profiles = {d: sorted({inverse[int(i)] for i, s in zip(data['ids'][n], data['states'][n]) if i and s == 0})
                    for n, d in enumerate(labels)}
    output = ROOT / 'assets/class_hpo_associations.json'
    save(output, {'associations': profiles, 'source_sha256': sha(source),
                  'source': 'existing frozen-training public HPO/Orphadata profiles',
                  'profile_order': 'GMDB classes first, same sorted class vocabulary as frozen release',
                  'profile_maximum_terms': 128, 'unlisted_means_unknown': True,
                  'patient_observations_or_answers_included': False})
    summary = {'state': 'public_association_question_selector_ready', 'classes': len(labels),
               'classes_with_profile': sum(bool(x) for x in profiles.values()),
               'distinct_hpo_terms': len(set().union(*map(set, profiles.values()))),
               'source_profile_sha256': sha(source), 'output_sha256': sha(output),
               'disease_frequency_probabilities_inferred': False, 'patient_records_used': False,
               'policy_efficacy_evaluated': False, 'updated_at_utc': now()}
    save(ROOT / 'hpo_question_receipt.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    guarded_main(main)
