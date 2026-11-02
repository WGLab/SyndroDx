"""Join existing private variant candidates back to raw structured exports.

No raw MRN, patient identifier, raw report ID, image, note or mapping is exported.
Source variant IDs are HMACed using the existing linkage namespace. This pass
does not decode category numbers or claim a patient pathogenic variant.
"""
import collections
import hmac
import hashlib
import re
import pyarrow.parquet as pq
from common import *
from engine import run

RAW = Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs/Clarity_20260704/clarity_raw_exports')
KNOWLEDGE = PROJECT / 'variant_knowledge_20261101'
LINK = PROJECT / 'chop_patient_linkage_20261101'
SALT = Path('/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/direct_identity_salt.bin')


def norm(value):
    text = str(value).strip() if value is not None else ''
    if text.lower() in {'', 'none', 'nan', 'null', 'na', 'n/a'}:
        return ''
    if re.fullmatch(r'\d+\.0', text):
        text = text[:-2]
    return (text.lstrip('0') or '0') if text.isdigit() else text


def rows(path, columns):
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=8192):
        yield from batch.to_pylist()


def main():
    registry = load(PROJECT / 'model_releases/CURRENT.json')
    manifest_before = sha(registry['manifest'])
    if manifest_before != registry['sha256']:
        raise ValueError('locked_release_manifest_mismatch')
    candidate_file = KNOWLEDGE / 'private/variant_evidence_candidates.jsonl'
    receipt = load(KNOWLEDGE / 'matching_receipt.json')
    if sha(candidate_file) != receipt['private_output_sha256']:
        raise ValueError('candidate_receipt_mismatch')
    candidates = list(jsonl(candidate_file))
    release = LINK / 'private/review_release/patient_disposition.tsv'
    people = {r['patient_key']: r for r in tsv(release)}
    master = {r['patient_key']: r for r in tsv(LINK / 'private/patient_master.tsv')}
    protected_groups = {r['group_id'] for r in master.values() if r['protected'] == 'True' or r['identity_conflict'] == 'True'}
    wanted = {r['variant_key']: r for r in candidates}
    if len(wanted) != len(candidates):
        raise ValueError('duplicate_candidate_variant_key')
    for r in candidates:
        p = people[r['patient_key']]
        if p['disposition'] != 'train_existing_split' or p['split'] != 'train' or p['reasons'] or p['group_id'] != r['group_id'] or p['group_id'] in protected_groups:
            raise ValueError('candidate_split_guard_failed')
    salt = SALT.read_bytes()
    def token(kind, value):
        return hmac.new(salt, (kind + ':' + norm(value)).encode(), hashlib.sha256).hexdigest()
    selected = ['variant_id', 'hgvs_name', 'assessment_c', 'clinical_signif_c',
                'allelic_state_c', 'allelic_frequency', 'genomic_source_c', 'method_type_c',
                'variant_type_c', 'dna_var_type_c', 'variant_molec_conseq_c', 'variant_func_effect_c']
    raw_variants = {}
    for row_number, r in enumerate(rows(RAW / 'variant.parquet', selected)):
        key = token('variant', r['variant_id'])
        if key in wanted:
            if key in raw_variants:
                raise ValueError('non_unique_raw_variant_record')
            raw_variants[key] = ({k: norm(v) for k, v in r.items() if k != 'variant_id'}, row_number)
    relations = collections.defaultdict(set)
    for r in rows(RAW / 'result_variant.parquet', ['variant_id', 'result_id']):
        key = token('variant', r['variant_id'])
        if key in wanted and norm(r['result_id']):
            relations[key].add(token('result', r['result_id']))
    source_sha = {n: sha(RAW / n) for n in ['variant.parquet', 'result_variant.parquet']}
    counts, patients = collections.Counter(), collections.defaultdict(set)
    def count(name, candidate):
        counts[name] += 1
        patients[name].add(candidate['patient_key'])
    output, case_evidence = [], collections.defaultdict(list)
    for r in candidates:
        key = r['variant_key']
        source, row_number = raw_variants.get(key, ({}, None))
        if source:
            count('raw_variant_record_recovered', r)
            if norm(source['assessment_c']) != norm(r['source_assessment_code']) or norm(source['hgvs_name']) != norm(r['source_HGVS']):
                raise ValueError('source_candidate_disagreement')
        for field, value in source.items():
            if value:
                count('nonmissing_' + field, r)
        if relations[key]:
            count('result_record_link_present', r)
        nucleotide = sorted({m['VariationID'] for m in r['matches'] if m['match_type'] == 'exact_versioned_nucleotide_expression'})
        evidence = {'evidence_id': key, 'patient_key': r['patient_key'], 'kind': 'variant',
                    'concept_id': 'ClinVar:' + nucleotide[0] if len(nucleotide) == 1 else None,
                    'source_kind': 'clinical_record', 'source_ref': 'variant.parquet#row=' + str(row_number),
                    'source_sha256': source_sha['variant.parquet'],
                    'review_status': 'pending', 'reviewer_ref': None, 'experiencer': 'unverified',
                    'event_at': None, 'available_at': None, 'assertion': 'unknown',
                    'identity_verified': False, 'assay_scope_verified': False}
        case_evidence[r['patient_key']].append(evidence)
        output.append({'variant_key': key, 'patient_key': r['patient_key'], 'group_id': r['group_id'],
                       'observation': evidence, 'raw_structured_fields': source,
                       'result_keys': sorted(relations[key]), 'raw_source_row_zero_based': row_number,
                       'candidate_knowledge': r, 'genome_assembly': None,
                       'local_causality_asserted': False, 'training_admitted': False})
    write_jsonl(PRIVATE / 'variant_evidence_review.jsonl', output)
    cases, replays = [], []
    stops = collections.Counter()
    accepted_total = 0
    for key, evidence in case_evidence.items():
        case = {'patient_key': key, 'group_id': people[key]['group_id'],
                'disposition': 'train_existing_split', 'split_audit_passed': True,
                'decision_time': None, 'evidence': evidence}
        replay = run(case, max_actions=5)
        accepted_total += len(replay['accepted_observation_ids'])
        stops[replay['stop_reason']] += 1
        cases.append(case)
        replays.append({'patient_key': key, 'result': replay})
    write_jsonl(PRIVATE / 'development_cases.jsonl', cases)
    write_jsonl(PRIVATE / 'evidence_readiness_replay.jsonl', replays)
    # Empty, explicit template: a reviewer fills this locally after source review.
    save(ROOT / 'review_contract.json', {
        'state': 'required_fields_not_a_patient_record',
        'required_observation_fields': ['evidence_id', 'patient_key', 'kind', 'concept_id', 'assertion',
            'source_ref', 'source_sha256', 'source_kind', 'experiencer', 'event_at', 'available_at',
            'review_status', 'reviewer_ref', 'identity_verified', 'assay_scope_verified'],
        'needed_local_exports': ['assessment_c code-to-label dictionary with source/version',
            'clinical_signif_c, allelic_state_c, genomic_source_c, method_type_c dictionaries',
            'RESULT master / documented variant-result-report relationship with availability time',
            'variant genome assembly / reference / coordinates for allele-frequency use'],
        'must_not_join': 'RESULT_VARIANT.result_id to ORDER_RESULTS.order_proc_id without a documented bridge',
        'public_schema_references': ['https://open.epic.com/EHITables/GetTable/VARIANT.htm',
                                     'https://open.epic.com/EHITables/GetTable/RESULT_VARIANT.htm'],
        'numeric_category_codes_inferred_from_public_label_order': False})
    if sha(registry['manifest']) != manifest_before:
        raise ValueError('locked_release_changed_during_audit')
    result = {'state': 'private_source_audit_and_readiness_replay_complete',
              'patients': len(cases), 'variant_records': len(candidates),
              'record_counts': dict(counts), 'patient_counts': {k: len(v) for k, v in patients.items()},
              'replay_stop_counts': dict(stops), 'accepted_patient_observations': accepted_total,
              'training_admissions': 0, 'diagnostic_rankings_generated': 0,
              'clinvar_matches_remain_candidates': True, 'numeric_categories_decoded': False,
              'protected_group_overlap': 0, 'bch_identity_overlap_certified': False,
              'images_or_notes_read': False, 'mrn_mapping_read': False,
              'patient_data_sent_to_external_apis': False, 'locked_v1_modified': False,
              'source_hashes': source_sha, 'candidate_sha256': sha(candidate_file),
              'split_disposition_sha256': sha(release), 'master_sha256': sha(LINK / 'private/patient_master.tsv'),
              'private_output_sha256': sha(PRIVATE / 'variant_evidence_review.jsonl'),
              'replay_sha256': sha(PRIVATE / 'evidence_readiness_replay.jsonl'),
              'updated_at_utc': now()}
    save(ROOT / 'source_audit_receipt.json', result)
    print(json.dumps({k: result[k] for k in ['state', 'patients', 'variant_records', 'record_counts',
                                           'replay_stop_counts', 'accepted_patient_observations']}))


if __name__ == '__main__':
    guarded_main(main)
