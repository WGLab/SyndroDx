"""HPC-local variant evidence cards. No genetic diagnosis or score adjustment.

Candidate identity, patient detection, public assertion and clinical causality
are separate fields. The existing pinned source audit remains authoritative.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from common import *
from engine import observation_reasons


def review_card(record, patient_key, cutoff, provenance):
    candidate, observation = record['candidate_knowledge'], record['observation']
    if any(x.get('patient_key') != patient_key for x in (record, candidate, observation)):
        raise ValueError('variant_review_patient_mismatch')
    if record['variant_key'] != observation['evidence_id'] or record['variant_key'] != candidate['variant_key']:
        raise ValueError('variant_review_identifier_mismatch')
    exact = sorted({str(m['VariationID']) for m in candidate['matches']
                    if m['match_type'] == 'exact_versioned_nucleotide_expression'})
    reasons = observation_reasons(observation, cutoff)
    return {'evidence_id': record['variant_key'], 'patient_key': patient_key,
            'branch': 'variant_evidence_review',
            'status': 'candidate_only_requires_source_review',
            'source_gene': candidate.get('source_gene'),
            'patient_detection_status': 'unresolved',
            'local_causality_asserted': False, 'used_for_diagnostic_ranking': False,
            'unique_exact_versioned_nucleotide_match': len(exact) == 1,
            'clinvar_candidate_ids': exact,
            'public_assertions_are_patient_findings': False,
            'public_evidence': {key: deepcopy(candidate.get(key, {} if key == 'public_variant_assertions' else []))
                                for key in ['public_variant_assertions', 'gene_disease_knowledge',
                                            'dosage_knowledge', 'orphadata_knowledge']},
            'public_evidence_interpretation': 'Reference assertions about variants or genes; not proof of patient detection, pathogenicity in this patient, or causal diagnosis.',
            'observation_gate_reasons': reasons,
            'next_action': {'tool': 'request_evidence_verification', 'evidence_id': record['variant_key'],
                            'required_checks': sorted(set(reasons + ['decode_local_assessment_and_genotype',
                                'verify_variant_condition_and_inheritance', 'verify_report_link_and_time']))},
            'provenance': deepcopy(provenance), 'rank_adjustment': 0.0}


class VariantReviewIndex:
    def __init__(self):
        audit = load(ROOT / 'source_audit_receipt.json')
        source = PRIVATE / 'variant_evidence_review.jsonl'
        matching = load(PROJECT / 'variant_knowledge_20261101/matching_receipt.json')
        if sha(source) != audit['private_output_sha256'] or audit['candidate_sha256'] != matching['private_output_sha256']:
            raise ValueError('variant_review_provenance_changed')
        self.provenance = {'review_snapshot_sha256': sha(source),
                           'source_audit_sha256': sha(ROOT / 'source_audit_receipt.json'),
                           'public_source_sha256': matching['public_source_sha256']}
        self.by_patient = defaultdict(dict)
        for record in jsonl(source):
            key, identifier = record['patient_key'], record['variant_key']
            if identifier in self.by_patient[key]:
                raise ValueError('duplicate_variant_review_record')
            self.by_patient[key][identifier] = record

    def cards(self, patient_key, evidence, cutoff):
        cards = []
        for item in evidence:
            if item.get('patient_key') != patient_key:
                raise ValueError('variant_review_patient_mismatch')
            record = self.by_patient.get(patient_key, {}).get(item['evidence_id'])
            if record is None:
                cards.append({'evidence_id': item['evidence_id'], 'patient_key': patient_key,
                              'status': 'no_pinned_variant_review_snapshot',
                              'used_for_diagnostic_ranking': False, 'rank_adjustment': 0.0})
                continue
            if item.get('source_sha256') != record['observation']['source_sha256']:
                raise ValueError('variant_observation_source_mismatch')
            cards.append(review_card(record, patient_key, cutoff, self.provenance))
        return cards


def main():
    index = VariantReviewIndex()
    results, reasons = [], Counter()
    for case in jsonl(PRIVATE / 'development_cases.jsonl'):
        if case['disposition'] != 'train_existing_split' or not case['split_audit_passed']:
            raise ValueError('review_case_not_development')
        cards = index.cards(case['patient_key'], case['evidence'], case.get('decision_time'))
        for card in cards:
            reasons.update(card.get('observation_gate_reasons', []))
        results.append({'patient_key': case['patient_key'], 'variant_evidence_cards': cards,
                        'variants_used_for_ranking': False})
    output = PRIVATE / 'variant_agent_review_cards.jsonl'
    write_jsonl(output, results)
    cards = [c for row in results for c in row['variant_evidence_cards']]
    receipt = {'state': 'variant_review_branch_connected', 'patients': len(results), 'review_cards': len(cards),
               'unique_exact_nucleotide_candidate_records': sum(c.get('unique_exact_versioned_nucleotide_match', False) for c in cards),
               'records_with_public_knowledge': sum(any(c.get('public_evidence', {}).values()) for c in cards),
               'observation_gate_reason_counts': dict(reasons), 'patient_detection_assertions_added': 0,
               'genetic_diagnosis_score_adjustments': 0, 'public_knowledge_is_patient_evidence': False,
               'output_sha256': sha(output), 'patient_records_in_output': False,
               'external_api_calls': 0, 'updated_at_utc': now()}
    save(ROOT / 'variant_agent_receipt.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    guarded_main(main)
