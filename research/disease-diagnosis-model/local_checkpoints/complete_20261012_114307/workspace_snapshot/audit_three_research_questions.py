"""Read-only feasibility audit: dated photos/notes, non-22q evidence, evaluation.

No note text or patient identifiers are exported. All eligibility stays pending.
"""
import bisect
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from audit_metadata_sources import mrnkey, clean

ROOT = Path(__file__).resolve().parent
CF = Path('/controlled-data/craniofacial_data')
MED = Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs')
OUT = ROOT / 'research_feasibility_audit'
ASOF = date(2026, 10, 11)
Q22 = {'MONDO:0018923', 'MONDO:0012020', 'MONDO:0008564'}


def rows(path):
    with Path(path).open() as f:
        yield from csv.DictReader(f, delimiter='\t')


def mids(s):
    return set(re.findall(r'MONDO:\d{7}', s or ''))


@lru_cache(None)
def parse_date(s):
    s = s.strip()
    fmt = '%Y%m%d' if re.fullmatch(r'\d{8}', s) else '%Y-%m-%d' if re.fullmatch(r'\d{4}-\d{2}-\d{2}', s) else None
    if not fmt:
        return None
    try:
        d = datetime.strptime(s, fmt).date()
    except ValueError:
        return None
    return d.toordinal() if date(1900, 1, 1) <= d <= ASOF else None


def date_options(s):
    """Ambiguous dotted dates retain both interpretations, never silently choose."""
    parsed = parse_date(s)
    if parsed:
        return {parsed}
    opts = set()
    for fmt in ['%m.%d.%y', '%d.%m.%y', '%m.%d.%Y', '%d.%m.%Y']:
        try:
            d = datetime.strptime(s.strip(), fmt).date()
            if date(1900, 1, 1) <= d <= ASOF:
                opts.add(d.toordinal())
        except ValueError:
            pass
    return opts


def main():
    OUT.mkdir(exist_ok=True)
    csv.field_size_limit(10**8)
    strict_path = ROOT / 'data_release_audit/chop_patient_consistent_candidates.tsv'
    strict = list(rows(strict_path))
    all_ids = {r['mrn'] for r in strict}
    cf_rows = [r for r in strict if r['source'] == 'CHOP_craniofacial']
    cf_ids = {r['mrn'] for r in cf_rows}
    bypath = {r['image_path']: r for r in cf_rows}
    evidence = {r['mrn']: r for r in rows(ROOT / 'data_release_audit/patient_manifest.tsv') if r['mrn'] in all_ids}
    expert = {r['mrn']: r for r in rows(ROOT / 'expert_holdout_audit/label_audit.tsv')}
    plan = {r['mrn']: r for r in rows(ROOT / 'split_plan_v0/patient_split_proposal.tsv')}
    reserved = {r['mrn'] for r in rows(ROOT / 'four_subset_inventory/patient_subsets.tsv') if r['subset'] == '1'}
    note_available = {p for p, r in evidence.items() if int(r['raw_note_text_rows'] or 0)}
    hpo_available = {p for p, r in evidence.items() if int(r['ctakes_hpo_candidate_rows'] or 0) or int(r['phenominal_hpo_candidate_rows'] or 0)}
    variants = {p for p, r in evidence.items() if int(r['variant_records'] or 0)}
    variant_significance = {p for p, r in evidence.items() if int(r['variant_significance_records'] or 0)}

    old_path = CF / 'craniofacial_metadata_frontal_updated_123025_deduplicated.csv'
    new_path = CF / 'complete_full_craniofacial_metadata_updated_03152026.tsv'
    photo_dates, old_by_path = defaultdict(set), {}
    photo_counts = Counter()
    for r in rows(old_path):
        if r['image_path'] not in bypath:
            continue
        d = parse_date(r['date_taken'])
        photo_counts['matched_paths'] += 1
        if d:
            p = bypath[r['image_path']]['mrn']
            photo_dates[p].add(d)
            old_by_path[r['image_path']] = d
            photo_counts['valid_date_paths'] += 1
        elif r['date_taken'].strip().lower() not in ['', 'nan', 'none', 'null']:
            photo_counts['nonempty_unparseable_or_out_of_range_paths'] += 1
    agreement = Counter()
    concordant_dates = defaultdict(set)
    for r in rows(new_path):
        path = r['image_path']
        if path not in old_by_path:
            continue
        options = date_options(r['date_taken'])
        old = old_by_path[path]
        if options == {old}:
            agreement['unambiguous_agreement_paths'] += 1
            concordant_dates[bypath[path]['mrn']].add(old)
        elif old in options:
            agreement['ambiguous_format_compatible_paths'] += 1
        elif options:
            agreement['date_disagreement_paths'] += 1
        else:
            agreement['new_date_missing_or_unparseable_paths'] += 1
    print('Photo dates indexed; starting full clinical-note date scan', flush=True)

    # Scan the complete source, retain only distinct dates for candidate patients.
    # No note contents are kept, copied, exported or printed.
    notes_path = MED / 'CHOP_Images/all_clarity_notes_merged.csv'
    note_dates = defaultdict(set)
    note_scan = Counter()
    for r in rows(notes_path):
        note_scan['rows_scanned'] += 1
        p = mrnkey(r.get('MRN'))
        if p in all_ids and clean(r.get('TEXT')):
            note_scan['candidate_text_rows'] += 1
            d = parse_date(r.get('DATE', ''))
            if d:
                note_dates[p].add(d)
                note_scan['candidate_text_rows_with_valid_date'] += 1
            else:
                note_scan['candidate_text_rows_invalid_date'] += 1
        if note_scan['rows_scanned'] % 200000 == 0:
            print('Note rows scanned:', note_scan['rows_scanned'], flush=True)
    print('Clinical-note date scan complete', flush=True)

    def time_metrics(ps, dates):
        result = {'patients': len(ps), 'patients_with_valid_photo_date': 0, 'patients_with_at_least_two_photo_dates': 0, 'multi_date_and_any_note_text': 0, 'multi_date_and_HPO': 0}
        for gap in [30, 90, 180, 365]:
            result[f'photo_span_ge_{gap}_days'] = 0
        for window in [30, 90]:
            for direction in ['either_side', 'on_or_before']:
                result[f'photo_dates_ge90days_apart_each_with_note_{direction}_{window}days'] = 0
        qualified = set()
        for p in ps:
            dd = sorted(dates.get(p, set()))
            if dd:
                result['patients_with_valid_photo_date'] += 1
            if len(dd) < 2:
                continue
            result['patients_with_at_least_two_photo_dates'] += 1
            result['multi_date_and_any_note_text'] += p in note_available
            result['multi_date_and_HPO'] += p in hpo_available
            span = dd[-1] - dd[0]
            for gap in [30, 90, 180, 365]:
                result[f'photo_span_ge_{gap}_days'] += span >= gap
            nd = sorted(note_dates.get(p, set()))
            for window in [30, 90]:
                for direction in ['either_side', 'on_or_before']:
                    matched = []
                    for d in dd:
                        left = bisect.bisect_left(nd, d - window)
                        right = bisect.bisect_right(nd, d + window if direction == 'either_side' else d)
                        if left < right:
                            matched.append(d)
                    # > window ensures one dated note cannot service both endpoints in prior-only mode.
                    if len(matched) >= 2 and matched[-1] - matched[0] >= 90:
                        result[f'photo_dates_ge90days_apart_each_with_note_{direction}_{window}days'] += 1
                        if window == 30 and direction == 'on_or_before':
                            qualified.add(p)
        return result, qualified

    timeline, qualified = time_metrics(cf_ids, photo_dates)
    timeline_concordant, _ = time_metrics(cf_ids, concordant_dates)
    timeline_nonreserved, _ = time_metrics(cf_ids - reserved, photo_dates)

    def has_22q(r):
        return bool(mids(r.get('mondo_ids')) & Q22) or bool(re.search(r'22\s*q|digeorge|velo.?cardio.?facial', r.get('diagnosis_raw', ''), re.I))

    def evidence_counts(ps):
        return {'patients': len(ps), 'with_note_text': len(ps & note_available), 'with_HPO_candidates': len(ps & hpo_available), 'with_notes_and_HPO': len(ps & note_available & hpo_available), 'with_variant_records': len(ps & variants), 'with_variant_clinical_significance_field': len(ps & variant_significance), 'with_ge90day_photo_timepoints_and_prior30day_notes': len(ps & qualified)}

    expert_present = set(expert) & all_ids
    expert_22q = {p for p in expert_present if has_22q(expert[p])}
    non22 = {p for p in expert_present - expert_22q if mids(expert[p]['mondo_ids'])}
    unresolved = expert_present - expert_22q - non22
    non22_all_source = {p for p, r in expert.items() if not has_22q(r) and mids(r['mondo_ids'])}
    recommended = {p for p in non22 if 'genetest' in expert[p]['source_tags'].split(',')}
    severe = {'missing_mondo', 'invalid_mondo_cell', 'ragged_source_row', 'missing_diagnosis_text', 'multiple_mondo_requires_adjudication', 'manual_fix_mondo_disagreement', 'obsolete_mondo', 'id_missing_from_local_ontology', 'diagnosis_assertion_review'}
    cleaner = {p for p in non22 - recommended if not set(expert[p]['issues'].split('|')) & severe and expert[p]['diagnosis_raw'].strip()}
    single = {p for p in cleaner if len(mids(expert[p]['mondo_ids'])) == 1}
    disease_patients = defaultdict(set)
    for p in non22:
        for d in mids(expert[p]['mondo_ids']):
            disease_patients[d].add(p)
    names = {}
    for block in (ROOT / 'ontology/mondo.obo').read_text().split('[Term]')[1:]:
        mid = re.search(r'^id: (MONDO:\d{7})$', block, re.M)
        name = re.search(r'^name: (.+)$', block, re.M)
        if mid and name:
            names[mid[1]] = name[1]
    disease_rows = [{'mondo_id': d, 'diagnosis_name': names.get(d, ''), 'expert_source_candidate_patients': len(ps), 'patients_with_notes': len(ps & note_available), 'patients_with_HPO': len(ps & hpo_available), 'patients_with_genetest_source': len(ps & recommended)} for d, ps in sorted(disease_patients.items(), key=lambda x: (-len(x[1]), x[0]))]

    # Image-label syndromes are a separate lexical tier, never pooled with expert truth.
    image_syndromes = set()
    for r in rows(ROOT / 'diagnosis_strata/diagnosis_image_manifest.tsv'):
        if any('syndrome' in names.get(d, '').casefold() for d in mids(r['mondo_ids'])) and not mids(r['mondo_ids']) & Q22:
            image_syndromes.add(r['mrn'])
    ian = {p for p, r in plan.items() if r['ian_record'] == 'True'} & all_ids
    old_label_patients = set()
    old_train = set()
    old_by_split = defaultdict(set)
    for r in rows(ROOT / 'labels.tsv'):
        p = mrnkey(r['mrn'])
        old_label_patients.add(p)
        old_by_split[r['split']].add(p)
        if r['split'] in ['train', 'ssl']:
            old_train.add(p)
    def evaluation(ps):
        return {**evidence_counts(ps), 'in_historical_labels_any_split': len(ps & old_label_patients), 'in_historical_train_or_ssl_split': len(ps & old_train), 'absent_from_historical_labels': len(ps - old_label_patients), 'non22q_expert_source_candidates': len(ps & non22)}
    summary = {
        'as_of': ASOF.isoformat(),
        'scope': {'craniofacial_patients': len(cf_ids), 'craniofacial_images': len(cf_rows), 'combined_CHOP_patients': len(all_ids), 'combined_CHOP_images': len(strict)},
        'Q1_longitudinal': {'photo_date_parse': dict(photo_counts), 'cross_version_date_comparison': dict(agreement), 'old_date_field_cohort': timeline, 'unambiguous_cross_version_concordant_date_subset': timeline_concordant, 'nonreserved_old_date_field_cohort': timeline_nonreserved, 'note_scan': dict(note_scan), 'patients_with_valid_dated_notes_combined': len(note_dates), 'verified_acquisition_time_and_contemporaneous_clinical_evidence_patients': None},
        'Q2_non22q': {'definition': 'Exclude expert labels with 22q11.2 deletion, duplication, DiGeorge IDs or explicit 22q/DiGeorge/velocardiofacial diagnosis text. Missing MONDO is unresolved, not non-22q.', 'expert_source_with_images': len(expert_present), 'expert_source_22q_patients': len(expert_22q), 'expert_source_unresolved_patients': len(unresolved), 'non22q_expert_source_combined': evidence_counts(non22), 'non22q_expert_source_craniofacial': evidence_counts(non22 & cf_ids), 'non22q_expert_source_all_patients_including_no_images': len(non22_all_source), 'non22q_expert_source_without_current_images': len(non22_all_source - all_ids), 'non22q_expert_mondo_concepts': len(disease_patients), 'disease_count_thresholds': {str(n): sum(len(ps) >= n for ps in disease_patients.values()) for n in [2, 5, 10, 20]}, 'genetest_source_requires_result_verification': evidence_counts(recommended), 'without_genetest_or_major_label_flags': evidence_counts(single), 'source_tags_in_non22q_patients': dict(Counter(t for p in non22 for t in expert[p]['source_tags'].split(',') if t)), 'image_label_syndrome_name_heuristic_combined': evidence_counts(image_syndromes), 'clinically_adjudicated_etiologic_non22q_patients': None},
        'Q3_evaluation': {'all_expert_Ian_reserved': evaluation(reserved), 'Ian_direct_records_with_images': evaluation(ian), 'non22q_expert_source_candidates': evaluation(non22), 'Ian_not_in_expert_source_with_images': len(ian - expert_present), 'ordinary_proposed_test': evaluation({p for p, r in plan.items() if r['proposed_role'] == 'test_candidate'} & all_ids), 'final_independent_test_patients': None, 'family_isolation': 'unverified', 'near_duplicate_cross_dataset_check': 'not_completed', 'external_center_clinical_cohort': 'not_established_in_audited_inventory', 'GMDB_exact_craniofacial_overlap': json.loads((ROOT / 'public_resource_coverage/summary.json').read_text())['coverage']['gmdb_patients']},
        'limitations': ['Dates are metadata fields, not verified acquisition times. Two versions may share upstream extraction; agreement is not independent validation.', 'Note DATE is parsed as calendar date, not verified encounter/acquisition time. Matching windows measure temporal proximity only, not clinical relevance or prediagnostic availability.', 'No diagnosis, surgery or treatment event timeline has been adjudicated. No negative clinical facts inferred from absent records.', 'Expert-source labels are all pending clinical adjudication. Genetest recommendations and gene fields do not establish a positive molecular result.', 'Historical split overlap is exposure risk, not proof a particular image was loaded. Absence from labels.tsv does not prove absence from every prior training source.', 'CHOP craniofacial and genetics are the same institution. Public GMDB resource counts are not independent test eligibility.', 'No emails, external communications or recruitment were performed. External collaboration cannot be inferred from the data inventory.']
    }
    assert len(expert_present) == len(expert_22q) + len(non22) + len(unresolved)
    assert single <= non22 and reserved <= all_ids
    assert timeline['patients_with_at_least_two_photo_dates'] <= timeline['patients_with_valid_photo_date'] <= len(cf_ids)
    assert timeline['photo_dates_ge90days_apart_each_with_note_on_or_before_30days'] <= timeline['photo_dates_ge90days_apart_each_with_note_either_side_30days']
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (OUT / 'non22q_expert_disease_counts.tsv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(disease_rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(disease_rows)
    inputs = [strict_path, old_path, new_path, ROOT / 'data_release_audit/patient_manifest.tsv', ROOT / 'expert_holdout_audit/label_audit.tsv', ROOT / 'split_plan_v0/patient_split_proposal.tsv', ROOT / 'four_subset_inventory/patient_subsets.tsv', ROOT / 'labels.tsv', Path(__file__)]
    manifest = {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs}
    manifest[str(notes_path)] = {'bytes': notes_path.stat().st_size, 'mtime_ns': notes_path.stat().st_mtime_ns, 'scan': 'complete CSV scan; no note text retained; no extra hash pass over 9GB file'}
    (OUT / 'input_versions.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    enrich_and_report(summary)


def enrich_and_report(summary):
    """Small follow-up joins and report generation; avoids repeating the note scan."""
    expert = {r['mrn']: r for r in rows(ROOT / 'expert_holdout_audit/label_audit.tsv')}
    present = {r['mrn'] for r in rows(ROOT / 'data_release_audit/chop_patient_consistent_candidates.tsv')}
    def is_non22(r):
        return bool(mids(r['mondo_ids'])) and not mids(r['mondo_ids']) & Q22 and not re.search(r'22\s*q|digeorge|velo.?cardio.?facial', r['diagnosis_raw'], re.I)
    no_image = {p for p, r in expert.items() if is_non22(r)} - present
    genetics_index = MED / 'CHOP_Images/final_results/arcus_genetics_photos_faith.csv'
    genetics_ids = {mrnkey(r['MRN']) for r in rows(genetics_index) if mrnkey(r['MRN'])}
    summary['Q2_non22q']['non22q_expert_without_images_but_in_full_genetics_index'] = len(no_image & genetics_ids)
    summary['Q1_longitudinal']['date_provenance'] = {'upstream_code': str(MED / 'CHOP_Images/qwen3_extraction.py'), 'finding': 'build_prompt explicitly extracts date_taken from folder/file paths using an LLM; not an EXIF acquisition-time certificate. Complete row-level lineage for current metadata remains unverified.', 'cross_version_parser': 'ISO/compact dates and dotted month-day-year/day-month-year interpretations; other formats are not resolved. These counts do not validate acquisition times.'}
    expected_text_rows = sum(int(r['raw_note_text_rows'] or 0) for r in rows(ROOT / 'data_release_audit/patient_manifest.tsv') if r['mrn'] in present)
    q2 = summary['Q2_non22q']
    checks = {'fresh_note_scan_matches_previous_candidate_text_count': summary['Q1_longitudinal']['note_scan']['candidate_text_rows'] == expected_text_rows,
              'expert_partition_complete': q2['expert_source_with_images'] == q2['expert_source_22q_patients'] + q2['expert_source_unresolved_patients'] + q2['non22q_expert_source_combined']['patients'],
              'non22q_image_partition_complete': q2['non22q_expert_source_all_patients_including_no_images'] == q2['non22q_expert_source_without_current_images'] + q2['non22q_expert_source_combined']['patients'],
              'recovery_target_is_subset': q2['non22q_expert_without_images_but_in_full_genetics_index'] <= q2['non22q_expert_source_without_current_images']}
    (OUT / 'validation_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    assert all(checks.values()), checks
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest_path = OUT / 'input_versions.json'
    manifest = json.loads(manifest_path.read_text())
    for p in [genetics_index, MED / 'CHOP_Images/qwen3_extraction.py', ROOT / 'ontology/mondo.obo', ROOT / 'diagnosis_strata/diagnosis_image_manifest.tsv', ROOT / 'public_resource_coverage/summary.json', Path(__file__)]:
        manifest[str(p)] = {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    q1, q2, q3 = (summary[k] for k in ['Q1_longitudinal', 'Q2_non22q', 'Q3_evaluation'])
    t = q1['old_date_field_cohort']
    n = q1['nonreserved_old_date_field_cohort']
    lines = ['# Data Feasibility: Three Research Questions', '', 'Audit date: October 11, 2026.', '',
             '## Findings', '',
             'The current data support substantial candidate cohorts, but verified longitudinal acquisition times, adjudicated non-22q etiologic labels, and a final independent evaluation cohort remain unestablished. Null counts in the summary mean undetermined, not zero.', '',
             '## 1. Multi-timepoint photos with contemporaneous clinical records', '',
             'Denominator: 18,715 craniofacial patients / 121,308 unique candidate crops. Use the old frontal metadata date_taken field because its dates are standardized; this does not certify its provenance. Original representative paths from the conservative pool are used, not every duplicate path.', '',
             '| Availability criterion | Patients |', '|---|---:|']
    for label, key in [('At least one parseable, in-range photo date', 'patients_with_valid_photo_date'), ('At least two distinct photo dates', 'patients_with_at_least_two_photo_dates'), ('Two or more photo dates plus any linked note text', 'multi_date_and_any_note_text'), ('Photo dates spanning at least 90 days', 'photo_span_ge_90_days'), ('Photo dates spanning at least one year', 'photo_span_ge_365_days'), ('Photo dates at least 90 days apart, each with a note within ±30 days', 'photo_dates_ge90days_apart_each_with_note_either_side_30days'), ('Photo dates at least 90 days apart, each with a note on/before the photo date within 30 days', 'photo_dates_ge90days_apart_each_with_note_on_or_before_30days')]:
        lines.append(f'| {label} | {t[key]:,} |')
    lines += ['', f"After excluding Expert / Ian reserved patients, the final prior-30-day criterion retains **{n['photo_dates_ge90days_apart_each_with_note_on_or_before_30days']:,} candidate patients**.", '',
              f"The full note scan examined **{q1['note_scan']['rows_scanned']:,} rows**. Only calendar dates were retained in memory; no note text or patient identifiers were exported. The 30/90-day windows are exploratory availability definitions, not clinical standards.", '',
              '**Reliability is still unverified.** Upstream code explicitly asks an LLM to extract date_taken from file/folder paths. Agreement between metadata versions is not independent evidence. Cross-version parsing retains month/day ambiguity and does not resolve all other date conventions. Acquisition dates must be checked against source files or clinical documentation. We have not established surgery/treatment timing, note relevance, diagnostic cutoff dates, or that same-day information predates imaging. Therefore this is a candidate timeline, not validated longitudinal follow-up.', '',
              'The generation and patient-selection provenance of the note export remains pending as documented in DATA_SCOPE.md. Here it is joined to the independently defined image cohort only to measure resource availability; it is not approved as a predictive input.', '',
              '## 2. Non-22q cases with diagnostic evidence', '',
              'Use the expert-source patient labels, independently of the 165 image-label concept inventory. Exclude 22q11.2 deletion/duplication and DiGeorge MONDO IDs, plus explicit 22q/DiGeorge/velocardiofacial diagnosis text. Missing MONDO remains unresolved. Non-22q does not automatically mean etiologically or molecularly confirmed.', '',
              '| Expert-source criterion | Patients |', '|---|---:|',
              f"| Expert-source patients with conservative images (combined CHOP) | {q2['expert_source_with_images']:,} |",
              f"| 22q-related under this definition | {q2['expert_source_22q_patients']:,} |",
              f"| Non-22q, with a MONDO mapping | {q2['non22q_expert_source_combined']['patients']:,} |",
              f"| Unresolved | {q2['expert_source_unresolved_patients']:,} |",
              f"| Non-22q candidates with craniofacial images | {q2['non22q_expert_source_craniofacial']['patients']:,} |",
              f"| Non-22q candidates with note text | {q2['non22q_expert_source_combined']['with_note_text']:,} |",
              f"| Non-22q candidates with HPO | {q2['non22q_expert_source_combined']['with_HPO_candidates']:,} |",
              f"| Non-22q candidates with both notes and HPO | {q2['non22q_expert_source_combined']['with_notes_and_HPO']:,} |",
              f"| Non-22q candidates carrying genetest source | {q2['genetest_source_requires_result_verification']['patients']:,} |",
              f"| Single-label candidates without genetest or major label flags | {q2['without_genetest_or_major_label_flags']['patients']:,} |", '',
              f"Non-22q candidates cover **{q2['non22q_expert_mondo_concepts']} expert-source MONDO concepts**. Disease counts are not additive for multi-label patients. Counts of concepts with at least 2/5/10/20 candidate patients are {q2['disease_count_thresholds']}.", '',
              'All expert-source clinical adjudication remains pending. The cleaner subset is an audit priority, not an approved gold standard. Genetest may represent a recommended test rather than a positive result; variant availability does not establish pathogenicity.', '',
              f"There are **{q2['non22q_expert_source_without_current_images']:,} additional non-22q expert-source patients without current conservative images**. Of these, **{q2['non22q_expert_without_images_but_in_full_genetics_index']:,} occur in the full genetics photo index**. This is a concrete recovery target, not a count of available or QC-approved images. The prior genetics-path audit could not resolve the indexed originals; this audit does not re-certify their current file availability.", '',
              '## 3. Independent evaluation and external validation', '',
              '| Candidate pool | Patients | In historical train/SSL split | Absent from historical labels.tsv |', '|---|---:|---:|---:|']
    for label, key in [('Expert / Ian reserved union', 'all_expert_Ian_reserved'), ('Ian direct records with images', 'Ian_direct_records_with_images'), ('Non-22q expert-source candidates', 'non22q_expert_source_candidates'), ('Ordinary proposed test', 'ordinary_proposed_test')]:
        r = q3[key]
        lines.append(f"| {label} | {r['patients']:,} | {r['in_historical_train_or_ssl_split']:,} | {r['absent_from_historical_labels']:,} |")
    lines += ['', 'Rows overlap; do not add them. Historical split membership indicates exposure risk, not proof of individual image use. Absence from labels.tsv does not prove a patient was unseen by all previous models. The prior all-image SSL setting also makes other historical split members relevant to exposure review.', '',
              f"Ian has **{q3['Ian_not_in_expert_source_with_images']} image-bearing patients outside the expert-source TSV**, whose original expert diagnoses require separate mapping and review. The ordinary proposed test covers only two 22q categories. Existing proposals and reservation rules were not changed.", '',
              'Family isolation and cross-dataset near-duplicate review are incomplete. CHOP craniofacial and genetics are one institution, so department separation is not external-center validation. Final independent test eligibility remains undetermined.', '',
              'GMDB provides 594 patient IDs across 18 exact craniofacial MONDO overlaps in the local resource audit, potentially supporting a scoped external image evaluation. These are resource counts, not approved test cases; subject type, official splits, training/retrieval overlap, CHOP case overlap and intended-use permissions must be resolved. It does not provide validation of all 165 concepts or of the complete CHOP multimodal workflow.', '',
              'No separate external-center clinical cohort or confirmed collaboration was established from the audited files. This audit did not contact collaborators; willingness to recruit or share data cannot be inferred from filesystem inventories.', '',
              '## Priorities', '',
              '1. Validate date provenance and treatment timing on the temporally linked candidate cohort before claiming longitudinal learning.',
              '2. Adjudicate the image-bearing non-22q expert-source cases and map the remaining Ian diagnoses without releasing reservations.',
              '3. Investigate full-genetics image recovery, starting with non-22q expert-source patients already present in its index.',
              '4. Lock independent internal patients/families and separately establish an external clinical cohort; public image validation is complementary.', '',
              '## Outputs', '',
              '- [Full aggregate statistics](summary.json)',
              '- [Non-22q expert-source disease counts](non22q_expert_disease_counts.tsv)',
              '- [Input provenance](input_versions.json)',
              '- [Count validation](validation_checks.json)',
              '- Reproduce with `python audit_three_research_questions.py` from the repository root. This streams the approximately 9.2 GB note source.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
