"""Build the consolidated, aggregate-only research data inventory.

Recounts local manifests and reconciles audit snapshots. Does not modify source
data, labels, splits, QC decisions or training eligibility. No patient IDs/text
are exported. File availability audits retain their original audit dates.
"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'complete_data_inventory'
USED = set()


def table(rel):
    path = ROOT / rel
    USED.add(path)
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def js(rel):
    path = ROOT / rel
    USED.add(path)
    return json.loads(path.read_text())


def ids(value):
    return set(re.findall(r'MONDO:\d{7}', value or ''))


def write_table(name, data):
    if not data:
        return
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), delimiter='\t')
        w.writeheader()
        w.writerows(data)


def main():
    OUT.mkdir(exist_ok=True)
    available = js('data_release_audit/summary.json')
    sources = js('metadata_audit/source_audit.json')
    feasibility = js('research_feasibility_audit/summary.json')
    public = js('public_resource_coverage/summary.json')
    previous_source_counts = js('diagnosis_strata/craniofacial_statistics.json')
    strata = js('diagnosis_strata/summary.json')
    subsets_expected = js('four_subset_inventory/summary.json')
    concepts_summary = js('candidate_concept_audit/summary.json')
    expert_summary = js('expert_holdout_audit/summary.json')
    split_summary = js('split_plan_v0/summary.json')
    auxiliary = js('metadata_audit/medicaldbs_auxiliary.json')
    medical = js('metadata_audit/medicaldbs_audit.json')
    genetics_full = js('data_release_audit/full_genetics_path_audit.json')
    images = table('four_subset_inventory/image_subsets.tsv')
    diagnosis = table('diagnosis_strata/diagnosis_image_manifest.tsv')
    patients = {r['mrn']: r for r in table('data_release_audit/patient_manifest.tsv')}
    expert = {r['mrn']: r for r in table('expert_holdout_audit/label_audit.tsv')}
    plan = {r['mrn']: r for r in table('split_plan_v0/patient_split_proposal.tsv')}
    concept_rows = table('candidate_concept_audit/raw_label_concepts.tsv')
    qc = table('face_qc/subset4_screen_v1/results.tsv')
    issues = table('data_release_audit/image_issue_queue.tsv')
    public_rows = {r['mondo_id']: r for r in table('public_resource_coverage/disease_resource_matrix.tsv')}
    registry = table('metadata_audit/medicaldbs_source_registry.tsv')
    bypatient = defaultdict(list)
    byhash = {}
    for r in images:
        bypatient[r['mrn']].append(r)
        byhash[r['crop_sha256']] = r
    all_patients = set(bypatient)
    cf = {p for p, rr in bypatient.items() if any(r['source'] == 'CHOP_craniofacial' for r in rr)}
    genetics = {p for p, rr in bypatient.items() if any(r['source'] == 'CHOP_genetics' for r in rr)}
    reserved = {p for p, rr in bypatient.items() if rr[0]['subset'] == '1'}
    im_ids_by_patient = defaultdict(set)
    disease_images, disease_patients = defaultdict(set), defaultdict(set)
    for r in diagnosis:
        for mid in ids(r['mondo_ids']):
            im_ids_by_patient[r['mrn']].add(mid)
            disease_images[mid].add(r['crop_sha256'])
            disease_patients[mid].add(r['mrn'])
    ex_patients = defaultdict(set)
    for p in all_patients & expert.keys():
        for mid in ids(expert[p]['mondo_ids']):
            ex_patients[mid].add(p)
    qc_byhash = {r['upstream_crop_sha256']: r for r in qc}

    feature_sets = {}
    for name, columns in {'clinical_dx': ['clinical_dx_rows'], 'hpo': ['ctakes_hpo_candidate_rows', 'phenominal_hpo_candidate_rows'], 'notes': ['raw_note_text_rows'], 'variants': ['variant_records'], 'hgvs': ['variant_hgvs_records'], 'variant_significance': ['variant_significance_records']}.items():
        feature_sets[name] = {p for p in all_patients if any(int(patients[p].get(c) or 0) > 0 for c in columns)}
    feature_sets['image_mondo'] = set(im_ids_by_patient)
    feature_sets['expert_source'] = all_patients & expert.keys()
    feature_sets['Ian_direct'] = {p for p in all_patients if plan.get(p, {}).get('ian_record') == 'True'}
    for col in ['birth_date_present', 'sex_code_present', 'ethnicity_code_present', 'patient_bridge']:
        feature_sets[col] = {p for p in all_patients if patients[p].get(col) == 'True'}

    def cohort(label, rr):
        ps = {r['mrn'] for r in rr}
        hashes = {r['crop_sha256'] for r in rr}
        counts = Counter(r['mrn'] for r in rr)
        mm = set().union(*(ids(r['image_mondo_ids']) for r in rr)) if rr else set()
        p_mondo = {r['mrn'] for r in rr if r['image_has_mondo'] == 'True'}
        record = {'cohort': label, 'patients': len(ps), 'unique_crops': len(hashes), 'image_mondo_patients': len(p_mondo), 'image_mondo_crops': sum(r['image_has_mondo'] == 'True' for r in rr), 'image_mondo_ids': len(mm), 'expert_source_patients': len(ps & feature_sets['expert_source']), 'expert_source_mondo_ids': len(set().union(*(ids(expert[p]['mondo_ids']) for p in ps & expert.keys()))), 'Ian_direct_patients': len(ps & feature_sets['Ian_direct']), 'reserved_patients': len(ps & reserved), 'patients_ge2_crops': sum(n >= 2 for n in counts.values()), 'patients_ge5_crops': sum(n >= 5 for n in counts.values()), 'patients_ge10_crops': sum(n >= 10 for n in counts.values()), 'median_crops_per_patient': median(counts.values()) if counts else 0}
        for key in ['clinical_dx', 'hpo', 'notes', 'variants', 'hgvs', 'variant_significance', 'birth_date_present', 'sex_code_present', 'ethnicity_code_present', 'patient_bridge']:
            record[key + '_patients'] = len(ps & feature_sets[key])
        record['hpo_notes_patients'] = len(ps & feature_sets['hpo'] & feature_sets['notes'])
        record['hpo_notes_image_mondo_patients'] = len(p_mondo & feature_sets['hpo'] & feature_sets['notes'])
        decisions = Counter(qc_byhash[h]['decision'] if h in qc_byhash else 'not_screened_in_subset4_run' for h in hashes)
        for k in ['frontal_candidate', 'nonfrontal_candidate', 'review', 'not_screened_in_subset4_run']:
            record['qc_' + k + '_images'] = decisions[k]
        return record

    cohorts = [cohort('craniofacial', [r for r in images if r['source'] == 'CHOP_craniofacial']), cohort('genetics', [r for r in images if r['source'] == 'CHOP_genetics']), cohort('combined_CHOP', images), cohort('reserved_combined', [r for r in images if r['mrn'] in reserved]), cohort('nonreserved_combined', [r for r in images if r['mrn'] not in reserved])]
    subset_rows = [cohort('subset_' + s, [r for r in images if r['subset'] == s]) for s in ['1', '2', '3', '4']]
    combinations = []
    for name, ps in [('craniofacial', cf), ('genetics', genetics), ('combined_CHOP', all_patients)]:
        patterns = Counter(tuple(p in feature_sets[k] for k in ['hpo', 'notes', 'image_mondo', 'expert_source']) for p in ps)
        for flags, n in sorted(patterns.items()):
            combinations.append(dict(cohort=name, hpo=flags[0], note_text=flags[1], any_source_image_mondo=flags[2], expert_source=flags[3], patients=n))

    names = {}
    ontology_path = ROOT / 'ontology/mondo.obo'
    USED.add(ontology_path)
    for block in ontology_path.read_text().split('[Term]')[1:]:
        mid = re.search(r'^id: (MONDO:\d{7})$', block, re.M)
        name = re.search(r'^name: (.+)$', block, re.M)
        if mid and name:
            names[mid[1]] = name[1]
    concept_types = defaultdict(set)
    for r in concept_rows:
        for mid in ids(r['mondo_ids']):
            concept_types[mid].add(r['concept_type'])
    disease_rows = []
    for mid in sorted(set(disease_images) | set(ex_patients)):
        ips, eps = disease_patients.get(mid, set()), ex_patients.get(mid, set())
        ih = disease_images.get(mid, set())
        cf_h = {h for h in ih if byhash[h]['source'] == 'CHOP_craniofacial'}
        gen_h = {h for h in ih if byhash[h]['source'] == 'CHOP_genetics'}
        row = {'mondo_id': mid, 'diagnosis_name': names.get(mid, ''), 'raw_image_label_concept_types': ';'.join(sorted(concept_types[mid])), 'image_label_patients': len(ips), 'image_label_crops': len(ih), 'craniofacial_image_label_patients': len({byhash[h]['mrn'] for h in cf_h}), 'craniofacial_image_label_crops': len(cf_h), 'genetics_image_label_patients': len({byhash[h]['mrn'] for h in gen_h}), 'genetics_image_label_crops': len(gen_h), 'expert_label_patients_with_any_candidate_image': len(eps), 'patients_with_both_same_ID_sources': len(ips & eps), 'unique_patients_either_label_source': len(ips | eps), 'image_label_patients_with_notes': len(ips & feature_sets['notes']), 'image_label_patients_with_HPO': len(ips & feature_sets['hpo']), 'image_label_patients_with_notes_HPO': len(ips & feature_sets['notes'] & feature_sets['hpo']), 'expert_label_patients_with_notes': len(eps & feature_sets['notes']), 'image_label_nonreserved_patients': len(ips - reserved), 'image_label_reserved_patients': len(ips & reserved), 'image_label_subset4_frontal_candidate_crops': sum(h in qc_byhash and qc_byhash[h]['decision'] == 'frontal_candidate' for h in ih), 'clinical_adjudication': 'pending', 'public_resource_coverage_scope': 'audited' if mid in public_rows else 'not_audited_outside_165_craniofacial_concepts'}
        for col in ['hpo_positive_terms_exact', 'orphadata_positive_hpo_terms_exact', 'phenopacket_case_records_exact', 'gmdb_patients_exact', 'genereviews_chapters_exact']:
            row[col] = public_rows[mid][col] if mid in public_rows else ''
        disease_rows.append(row)

    paired = feature_sets['expert_source'] & feature_sets['image_mondo']
    label_compare = Counter()
    for p in paired:
        a, b = im_ids_by_patient[p], ids(expert[p]['mondo_ids'])
        key = 'expert_id_missing' if not b else 'identical_ID_sets' if a == b else 'overlapping_ID_sets' if a & b else 'disjoint_ID_sets_review'
        label_compare[key] += 1
    issue_counts = Counter()
    for r in issues:
        for reason in filter(None, r['review_reasons'].split('|')):
            issue_counts[reason] += 1
    error_rows = [r for r in qc if 'processing_error_FileNotFoundError' in r['reasons']]
    error_exist_now = sum(Path(r['image_path']).exists() for r in error_rows)

    # Reconcile all-image metadata indexing; do not decode/re-admit nonfrontal images.
    full_path = Path(sources['sources']['full_mar15']['file'])
    frontal_path = Path(sources['sources']['frontal']['file'])
    USED.update([full_path, frontal_path])
    full_paths, full_mrns, view_counts = set(), set(), Counter()
    from audit_metadata_sources import mrnkey
    with full_path.open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            full_paths.add(r['image_path'])
            if mrnkey(r['norm_mrn']):
                full_mrns.add(mrnkey(r['norm_mrn']))
            view_counts[r['EVA02_Organ']] += 1
    with frontal_path.open() as f:
        frontal_paths = {r['image_path'] for r in csv.DictReader(f, delimiter='\t')}
    raw_index = {'latest_full_metadata_rows': sum(view_counts.values()), 'latest_full_unique_paths': len(full_paths), 'latest_full_unique_nonempty_mrn_values': len(full_mrns), 'frontal_table_paths': len(frontal_paths), 'frontal_paths_missing_from_latest_full': len(frontal_paths - full_paths), 'full_plus_frontal_unique_paths': len(full_paths | frontal_paths), 'view_counts': dict(view_counts), 'meaning': 'Metadata index only, mixed views, provisional identities; not an exhaustive physical archive or approved image cohort'}

    image_mondo_ids = set(disease_images)
    expert_mondo_ids = set(ex_patients)
    summary = {'compiled_date': '2026-10-11', 'scope': 'Consolidated available data inventory; fresh manifest aggregation plus dated technical/provenance audits; not a new full archive decode or clinical adjudication', 'raw_craniofacial_index': raw_index, 'cohorts': cohorts, 'subsets': subset_rows, 'cross_source_patient_overlap': len(cf & genetics), 'cross_source_unique_image_hash_overlap': 0, 'image_label_mondo_ids': len(image_mondo_ids), 'expert_source_mondo_ids_with_images': len(expert_mondo_ids), 'label_source_mondo_ID_intersection': len(image_mondo_ids & expert_mondo_ids), 'label_source_mondo_ID_union_NOT_confirmed_diseases': len(image_mondo_ids | expert_mondo_ids), 'image_vs_expert_patient_ID_sets': dict(label_compare), 'candidate_disease_sample_size': {scope: {str(n): sum(int(r[column]) >= n for r in disease_rows) for n in [1, 2, 5, 10, 20, 50, 100]} for scope, column in [('craniofacial_image_labels', 'craniofacial_image_label_patients'), ('combined_image_labels', 'image_label_patients'), ('expert_labels_with_images', 'expert_label_patients_with_any_candidate_image')]}, 'issue_queue_rows': len(issues), 'issue_reason_counts_nonexclusive': dict(issue_counts), 'subset4_file_errors': len(error_rows), 'error_input_paths_existing_at_compilation': error_exist_now, 'historical_audit': {'availability': available, 'concept_audit': concepts_summary, 'expert_audit': expert_summary, 'split_proposal': split_summary, 'full_genetics_path_audit': genetics_full}, 'research_feasibility': feasibility, 'public_resource_coverage': public, 'auxiliary_text_resources': auxiliary, 'verified_final_counts': {'frontal_QC_approved_images': None, 'clinically_adjudicated_disease_types': None, 'molecularly_confirmed_patients': None, 'independent_test_patients': None, 'verified_longitudinal_patients': None}, 'new_training_admissions': 0}
    checks = {'unique_image_hashes': len(byhash) == len(images), 'conservative_population': len(all_patients) == 19448 and len(images) == 123982, 'cohort_source_reconciliation': all(next(r for r in cohorts if r['cohort'] == key)['patients'] == value['patients'] and next(r for r in cohorts if r['cohort'] == key)['unique_crops'] == value['images'] for key, value in previous_source_counts.items()), 'subsets_partition_patients': sum(r['patients'] for r in subset_rows) == len(all_patients), 'subsets_partition_images': sum(r['unique_crops'] for r in subset_rows) == len(images), 'one_subset_per_patient': all(len({r['subset'] for r in rr}) == 1 for rr in bypatient.values()), 'subset4_exact_QC_coverage': set(qc_byhash) == {r['crop_sha256'] for r in images if r['subset'] == '4'} and len(qc_byhash) == len(qc), 'image_diagnosis_counts': len(diagnosis) == strata['all_conservative']['all_diagnosis']['images'] and len(image_mondo_ids) == 178, 'image_manifest_labels_consistent': all(ids(r['mondo_ids']) == ids(byhash[r['crop_sha256']]['image_mondo_ids']) for r in diagnosis), 'public_matrix_165_unique_concepts': len(public_rows) == 165, 'fresh_full_metadata_counts_match_snapshot': len(full_paths) == sources['sources']['full_mar15']['distinct_image_paths'] and dict(view_counts) == sources['sources']['full_mar15']['organ_counts'], 'no_training_admission': all(r['training_admitted'] == 'False' for r in images), 'modality_patterns_complete': all(sum(r['patients'] for r in combinations if r['cohort'] == name) == len(ps) for name, ps in [('craniofacial', cf), ('genetics', genetics), ('combined_CHOP', all_patients)]), 'fresh_feasibility_audit_checks': all(js('research_feasibility_audit/validation_checks.json').values())}
    for r in subset_rows:
        expected = subsets_expected['subsets'][r['cohort'][-1]]
        checks[r['cohort'] + '_counts'] = r['patients'] == expected['patients'] and r['unique_crops'] == expected['images']
    (OUT / 'validation_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    assert all(checks.values()), checks
    write_table('cohort_summary.tsv', cohorts)
    write_table('subset_summary.tsv', subset_rows)
    write_table('modality_combinations.tsv', combinations)
    write_table('disease_inventory.tsv', disease_rows)
    write_table('raw_image_label_categories.tsv', concepts_summary['categories'])
    write_table('source_registry.tsv', registry)
    write_table('craniofacial_all_view_counts.tsv', [{'metadata_view_label': k, 'rows': v, 'status': 'metadata_only_not_QC_approved'} for k, v in view_counts.most_common()])
    write_table('issue_reason_counts.tsv', [{'reason': k, 'rows': v} for k, v in issue_counts.most_common()])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    used = {str(p): {'bytes': p.stat().st_size, 'mtime_utc': datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(USED | {Path(__file__)})}
    (OUT / 'input_versions.json').write_text(json.dumps(used, indent=2) + '\n')
    write_report(summary, disease_rows, registry, medical, checks)
    print(json.dumps({'cohorts': cohorts, 'label_comparison': label_compare, 'image_mondo_IDs': len(image_mondo_ids), 'expert_mondo_IDs': len(expert_mondo_ids), 'label_ID_union': len(image_mondo_ids | expert_mondo_ids), 'source_patient_overlap': len(cf & genetics), 'checks_passed': all(checks.values())}, indent=2))


def write_report(s, diseases, registry, medical, checks):
    def md(rows, columns):
        labels = [label for _, label in columns]
        lines = ['| ' + ' | '.join(labels) + ' |', '| ' + ' | '.join(['---'] * len(labels)) + ' |']
        for row in rows:
            values = []
            for key, _ in columns:
                value = row[key]
                values.append(f'{value:,}' if isinstance(value, int) else str(value).replace('|', '/'))
            lines.append('| ' + ' | '.join(values) + ' |')
        return '\n'.join(lines)

    gaps = [
        ('P0', 'Define clinical targets and verify labels', '178 image-label concepts; 122 expert-source concepts', 'Adjudicated diagnosis, evidence source, certainty and diagnosis date per patient; separate structural diagnoses, phenotypes and etiologic diseases', 'Clinically adjudicated disease and patient counts are unknown'),
        ('P0', 'Establish independent evaluation', '1,036 reserved patients; 457 listed in historical train/SSL splits', 'Freeze patient/family partitions, check near duplicates and all prior model exposure; document external validation access', 'No verified independent test denominator'),
        ('P0', 'Complete frontal image QC', '52,632 subset-4 crops screened; 71,350 other crops outside this run', 'Calibrate on human-reviewed samples, resolve errors, review uncertain images, publish approval counts by patient and disease', 'No final approved frontal image count'),
        ('P1', 'Verify temporal provenance', '7,073 multi-date CF candidates; 2,646 with two prior-note-linked timepoints', 'Verify acquisition dates and contemporaneous notes; adjudicate diagnosis/treatment events and input availability', 'No verified longitudinal clinical cohort'),
        ('P1', 'Expand non-22q evidence', '144 expert-source candidates across 117 concepts; 95 additional candidates indexed in full genetics metadata', 'Verify diagnosis/results and recover accessible original images; report disease-specific sample depth', 'Only one non-22q concept currently has at least five expert-source patients'),
        ('P1', 'Reconcile full genetics archive', '37,079 metadata rows / 5,664 MRN values', 'Resolve original/modified path roots and rerun availability, identity and frontal QC; retain copied-subset distinction', 'Prior full-archive paths were unavailable; recoverability unknown'),
        ('P1', 'Audit demographic and subgroup coverage', 'Presence flags available; distributions not yet validated', 'Validate sex/ethnicity codes, age at verified acquisition, age bands, missingness and disease-by-subgroup coverage', 'No validated age or demographic fairness denominator'),
        ('P2', 'Prepare agent retrieval and evidence tasks', '165 CF concepts audited against five public resources', 'Version evidence, verify permitted use, separate retrieval sources from evaluation cases, evaluate grounded outputs and abstention', 'Resource coverage is not task performance or independent evaluation'),
    ]
    gap_rows = [dict(zip(['priority', 'workstream', 'available_starting_point', 'completion_criterion', 'current_gap'], r)) for r in gaps]
    write_table('prioritized_gaps.tsv', gap_rows)
    cohort_table = md(s['cohorts'], [('cohort','Scope'), ('patients','Patients'), ('unique_crops','Unique crops'), ('image_mondo_patients','Patients with image MONDO'), ('image_mondo_ids','Image MONDO IDs')])
    modality_table = md(s['cohorts'][:3], [('cohort','Scope'), ('clinical_dx_patients','Clinical diagnosis rows'), ('hpo_patients','HPO candidates'), ('notes_patients','Note text'), ('variants_patients','Variant records'), ('hpo_notes_patients','HPO + notes'), ('hpo_notes_image_mondo_patients','HPO + notes + image MONDO')])
    depth = [dict(scope=k, **{'n'+n: v for n, v in vals.items()}) for k, vals in s['candidate_disease_sample_size'].items()]
    public = [dict(resource=k, **v) for k,v in s['public_resource_coverage']['coverage'].items()]
    q = s['research_feasibility']
    views = [{'view': k, 'rows': v} for k,v in sorted(s['raw_craniofacial_index']['view_counts'].items(), key=lambda x:-x[1])]
    text = f'''# Complete research data inventory

Compiled **{s['compiled_date']}**. This is the consolidated entrypoint for the currently audited resources. It combines fresh counts from local manifests with explicitly retained earlier audit findings. It is not a new exhaustive physical archive scan, clinical adjudication or training release.

See the [data-use and evaluation plan](DATA_USE_PLAN.md) for proposed roles, admission conditions and cross-modal isolation, and the [26-source use matrix](data_use_matrix.tsv) for source-by-source assignments.

## 1. Decision summary

The conservative CHOP candidate pool contains **19,448 patients and 123,982 byte-unique image crops**. Craniofacial contributes **18,715 patients / 121,308 crops**; the copied genetics subset contributes **1,034 patients / 2,674 crops**, with **301 shared patients**. The image-label space contains **178 MONDO IDs**, including **165 in craniofacial**. These are mapped label concepts, not 178 clinically confirmed etiologic diseases.

There is substantial material for image representation, structural diagnosis and multimodal evidence work. The immediate constraints are clinical label verification, image quality, temporal provenance and independent evaluation. An agentic architecture can organize evidence retrieval and review, but does not resolve those evidence gaps by itself. Journal suitability cannot be established from volume alone.

**Unknown final counts:** approved frontal images, clinically adjudicated diseases, molecularly confirmed patients, verified longitudinal patients and independent test patients. Unknown is not zero. All current candidate images remain `training_admitted=False`; this inventory changes no eligibility or reservation.

## 2. Counting rules

- A current patient is a normalized MRN in the conservative manifest, not a newly validated person identity. Family relationships are not established.
- A current image is a unique crop byte hash. Re-encoded near duplicates and related originals/crops require further checks.
- A raw archive row, unique path, crop, patient, diagnosis string and MONDO concept are different units.
- A MONDO ID can describe a broad structural condition or disease grouping. Multiple IDs may apply to one patient; disease-row patient counts are not additive.
- Clinical diagnosis rows, NLP-derived HPO candidates, note availability and variant records indicate evidence availability, not confirmed diagnoses or usable prediagnostic inputs.
- Expert-source labels are tracked separately from image-label mappings and remain pending clinical adjudication. A genetic-test recommendation is not a positive genetic result.
- Reservation applies at patient level across sources. Craniofacial and genetics are from the same institution.

## 3. Full metadata index and candidate-pool derivation

The latest full craniofacial metadata contains **378,471 rows / distinct paths** and **26,897 nonempty normalized MRN values**. These are provisional metadata identities. The frontal table has **154,825 paths**, including **334 absent from the latest full table**; their path union is **378,805**. The latest full table includes:

{md(views, [('view','Metadata view label'), ('rows','Rows')])}

This index includes nonfrontal views and clinical document images; it is not the approved face cohort. The older cropping table has 405,522 rows and is a recovery source, not an additional cohort to add.

| Stage | Craniofacial | Copied genetics | Combined |
| --- | ---: | ---: | ---: |
| Frontal source rows in technical audit | 154,825 | 2,749 | 157,574 |
| Byte-unique crops before record-level screening | 139,636 | 2,741 | 142,377 |
| Record-level screened crops | 132,906 | 2,735 | 135,641 |
| Conservative crops after patient-level conflict exclusion | 121,308 | 2,674 | 123,982 |

The screened combined cohort had 20,500 patients; excluding 1,052 patients with other known image identity conflicts leaves 19,448. These are successive stages and must not be added. Technical availability and decoding come from the earlier availability audit, not a full rerun today.

## 4. Current cohort and reservations

{cohort_table}

Source patient counts overlap; combined counts deduplicate patients. Image-label coverage uses labels on that source's own images. Expert labels use patient linkage and do not establish that a specific image depicts the labelled diagnosis.

{md(s['subsets'], [('cohort','Patient-disjoint subset'), ('patients','Patients'), ('unique_crops','Crops'), ('image_mondo_crops','Crops with image MONDO')])}

Subset 1 is expert/Ian reserved; subset 2 is A/B with MONDO; subset 3 is A/B without MONDO; subset 4 is the remaining pool. A/B denotes historical directory grouping, not an independently adjudicated diagnosis grade. The four subsets partition the combined cohort exactly.

{md(s['cohorts'][:3], [('cohort','Scope'), ('patients_ge2_crops','Patients ≥2 crops'), ('patients_ge5_crops','≥5'), ('patients_ge10_crops','≥10'), ('median_crops_per_patient','Median crops')])}

Multiple crops alone do not establish multiple visits or longitudinal clinical change.

## 5. Clinical evidence and multimodal completeness

Counts below are patients with at least one matching record/field:

{modality_table}

The detailed cohort table additionally records DOB, sex and ethnicity field presence, patient bridges and HGVS/significance fields. Presence flags do not validate demographic values, age at image acquisition, pathogenicity, note relevance or phenotype assertion polarity. The audited variant significance fields are empty; that does not establish absence of molecular diagnoses in unreviewed clinical records.

The full note source contains 1,577,548 rows; the recent feasibility scan found 835,532 text rows linked to current candidates. These are note rows, not independent visits or patients. No raw clinical text or patient identifiers are exported in this consolidated inventory.

[Modality combinations](modality_combinations.tsv) provides mutually exclusive HPO × note × image-MONDO × expert-source patterns. Its image-MONDO flag uses **any source for that patient**, whereas source-specific cohort coverage uses that source's own images. This distinction matters for the 301 shared patients.

## 6. Complete disease/concept inventory and sample depth

[All MONDO IDs and diagnosis names](disease_inventory.tsv) lists every concept found in either current image labels or linked expert-source labels, with source-specific patient/crop counts, evidence coverage, reservation and public-resource coverage.

| Label space | Distinct IDs |
| --- | ---: |
| Craniofacial image labels | 165 |
| Copied genetics image labels | 21 |
| Image-label union | {s['image_label_mondo_ids']} |
| Expert-source labels with candidate images | {s['expert_source_mondo_ids_with_images']} |
| Intersection of image and expert ID spaces | {s['label_source_mondo_ID_intersection']} |
| Union of the two evidence spaces | {s['label_source_mondo_ID_union_NOT_confirmed_diseases']} |

The last row is an **inventory union**, not a new confirmed disease count. Legacy clinical MONDO fields are a separate evidence source and are not pooled into this table. Label agreement at the patient level is also not clinical validation:

{md([{'comparison': k, 'patients': v} for k,v in s['image_vs_expert_patient_ID_sets'].items()], [('comparison','Patients with both label sources: comparison'), ('patients','Patients')])}

Disjoint ID sets require review, but can reflect broad-versus-specific labels or multiple conditions rather than a definite error.

{md(depth, [('scope','Label scope'), ('n1','IDs ≥1 patient'), ('n2','≥2'), ('n5','≥5'), ('n10','≥10'), ('n20','≥20'), ('n50','≥50'), ('n100','≥100')])}

The raw-label audit has 1,209 nonempty strings: 362 with lexical MONDO/HPO mappings and 847 requiring review. Concept categories include structural diagnoses, disease/disorder, phenotype, ambiguous abbreviations, procedures and workflow labels. Their patient counts overlap. Structural diagnoses can support a well-defined study endpoint, but must remain distinct from etiologic disease diagnosis. See [raw label categories](raw_image_label_categories.tsv).

## 7. Expert labels and non-22q depth

There are **946 expert-source patients with candidate images**, spanning **122 MONDO concepts**. The broader expert/Ian reservation includes 1,036 patients / 5,239 crops. Among the 946, the current classification gives 799 with 22q-related labels, 144 non-22q candidates and 3 unresolved patients.

The **144 non-22q candidates** span **117 concepts**; 68 have notes, all 144 have HPO candidates and 15 have variant records. Only 19 concepts have at least two patients; only one has at least five (Beckwith–Wiedemann, 11 patients, all with a genetic-test-source tag requiring result verification). None has at least 20. Thus broad rare-disease coverage currently has limited expert-labelled per-disease depth.

There are 1,777 non-22q expert-source patients without current conservative images, of whom **95 appear in the full genetics photo index**. This is a targeted recovery opportunity, not 95 newly available image patients. Expert-source provenance and final diagnosis must still be checked.

## 8. Longitudinal candidates

{md([{'metric': k, 'patients': v} for k,v in q['Q1_longitudinal']['old_date_field_cohort'].items()], [('metric','Craniofacial temporal candidate criterion'), ('patients','Patients')])}

Of 121,308 CF crops, 104,262 have a valid parsed date in the audited metadata; 21 nonempty values were invalid/unparseable or out of range. There are **7,073 patients with at least two distinct dates** and **2,646** with two dates at least 90 days apart, each with a note on or before it within 30 days (2,503 outside reservation).

These are temporal linkage candidates. Upstream extraction code infers dates from file/folder paths; acquisition-time lineage remains unverified. Metadata-version agreement is not independent validation. Notes near a date need review for relevance and information available at that time. Diagnosis, surgery and treatment event timelines have not been adjudicated. Verified longitudinal patient count remains unknown.

## 9. Image quality and identity issues

{md(s['cohorts'][:3], [('cohort','Scope'), ('qc_frontal_candidate_images','Frontal candidates'), ('qc_nonfrontal_candidate_images','Nonfrontal candidates'), ('qc_review_images','Review'), ('qc_not_screened_in_subset4_run_images','Outside this QC run')])}

The subset-4 run covers all 52,632 subset-4 crops exactly once. Its output is an uncalibrated triage screen, not final approval or a validated frontal prevalence estimate. Other subsets contain 71,350 crops outside this run; prior pilots do not constitute full cohort QC.

There were **{s['subset4_file_errors']} file-not-found processing errors** within the review bucket. **{s['error_input_paths_existing_at_compilation']} of those input paths exist at compilation**, supporting a targeted retry; existence does not prove successful processing. The image issue queue contains **{s['issue_queue_rows']:,} rows**, with overlapping reasons in [issue counts](issue_reason_counts.tsv). The earlier byte-copy audit found 13,603 exact-copy groups, including 1,311 with multiple nonempty IDs. Family linkage and re-encoded near-duplicate detection remain incomplete.

## 10. Evaluation readiness

The 1,036 reserved patients include 833 with notes and 1,014 with HPO candidates. Of these, 511 appear in historical labels and 457 in historical train/SSL splits. Direct Ian records cover 117 current image patients, including 100 with notes; 41 appear in historical train/SSL splits.

The current ordinary test proposal has 117 patients but no expert-source non-22q candidates. It cannot support broad rare-disease claims as presently constituted. The split plan is a proposal, not an admitted training release or validated benchmark.

Historical label overlap is exposure risk, not proof a particular model loaded every image. Absence from that file does not prove independence from every earlier training source. A new study can establish prospective isolation with appropriate model provenance, patient/family separation and duplicate checks. No independent external-center clinical cohort is established in the audited inventory. GMDB overlap is not by itself independent test eligibility.

## 11. Public evidence for an agent system

Coverage below is restricted to the **165 CF image-label concepts**:

{md(public, [('resource','Resource'), ('exact_diseases','Exact concepts'), ('exact_or_descendant_diseases','Exact or strict descendant'), ('exact_unique_resource_items','Exact-linked resource items')])}

The descendant column includes the exact column and is not additive. Descendant coverage can provide related evidence but is not equivalent to the same diagnosis. Resource-item units differ: HPO terms, case records, GMDB patient IDs and GeneReviews chapters. The 470 exact-linked Phenopacket cases include 419 NF1 cases; resource breadth and case balance must be considered separately. Four concepts have exact coverage in all five resources: Apert, Cohen, Muenke and NF1.

Local snapshots include HPO 2025-01-16, MONDO 2026-09-01 and Orphadata export 2025-12-09. All 8,207 Phenopacket JSON cases and 987 GeneReviews XML files were read in the resource audit. GeneReviews matching uses strict title/synonym or chapter OMIM-entry evidence; zero is not proof no relevant overview exists. PMC coverage was not audited. Public fields are blank for concepts outside the audited 165, rather than falsely marked absent.

These resources support phenotype normalization, evidence retrieval and candidate ranking. They do not replace patient-specific clinical confirmation. Resource/case overlap, permitted use and retrieval leakage need evaluation before benchmark admission. See the [public-resource report](../public_resource_coverage/REPORT.md).

## 12. Wider source registry and recovery inventory

[Source registry](source_registry.tsv) retains all {len(registry)} audited entries and their recorded status. Important scope distinctions:

| Resource | Audited scale | Interpretation |
| --- | --- | --- |
| Full genetics photo metadata | 37,079 rows; 5,664 MRN values; 35,537 original relative paths | Metadata exists. Prior path audit could not access these under the stated root; full availability has not been reverified. |
| Copied genetics all-view subset | 8,256 rows; 1,165 patients | Includes the smaller frontal source; do not add both. |
| GMDB | 10,980 metadata rows; 8,346 patient IDs; 581 syndrome IDs | External disease-face resource; access and test eligibility are separate. |
| CFAD | 4,341 rows; 2,761 frontal rows; 2,695 byte-unique frontal crops | Contains processed variants; 301 original/original-crop frontal rows; patient count unverified. |
| RareBench | LIRICAL 370; RAMEDIS 624; MME 40; HMS 88 records | Phenotype/reasoning records; no verified face pairing; do not assume cross-source independence. |
| Clinical Genetics physical-exam annotations | Train 2,767 rows / 1,716 observation IDs; dev 734 / 454 | Text phenotype/negation support; observation IDs are not verified patient counts. |
| FairFace / Caltech | 97,698 / 7,092 JPG files | Auxiliary image inventories; not established healthy clinical controls. |
| Raw notes and structured clinical tables | Available in the broader MedicalDBs audit | Full-source rows are not the current image-linked patient denominator. |
| PubMind and derivatives | Excluded | No inclusion in cohort, training or evaluation counts. |

Therefore, genetics photo metadata is **present**. The unresolved issue is availability and linkage of the full archive, not absence of metadata.

## 13. Prioritized work and completion criteria

{md(gap_rows, [('priority','Priority'), ('workstream','Workstream'), ('available_starting_point','Starting point'), ('completion_criterion','Completion criterion')])}

Start with a clinically defined endpoint and an isolated evaluation protocol, alongside calibrated image QC. Then measure how much verified non-22q and longitudinal evidence survives those requirements. Use those verified denominators to choose the main scientific claim and the agent tasks. Extensive GPU capacity can accelerate representation learning and experiments once admission is justified; it cannot supply missing clinical truth.

## 14. Reproducibility, freshness and deliverables

Run `python build_complete_data_inventory.py` from the repository root. The build checks unique image hashes, source reconciliation, subset partitions, label consistency, QC coverage, public-resource scope, full metadata consistency and unchanged training admission. **{sum(checks.values())}/{len(checks)} checks passed.**

Freshly recomputed here: cohort/subset/modalities/concept counts from current manifests, image/expert label comparisons, full CF metadata view/path counts and existence of the previously failed QC inputs. Earlier technical availability/decoding and full-genetics path results retain their original audit scope (September 28); subset-4 QC is the September 29 run. Temporal note-linkage and public-resource findings use their linked audit outputs. This compilation does not rerun image decoding, clinical annotation or all source downloads.

- [Cohort statistics](cohort_summary.tsv), [subset statistics](subset_summary.tsv), [modality patterns](modality_combinations.tsv).
- [Complete MONDO/diagnosis inventory](disease_inventory.tsv), [raw-label categories](raw_image_label_categories.tsv).
- [All-view metadata counts](craniofacial_all_view_counts.tsv), [issue reasons](issue_reason_counts.tsv), [source registry](source_registry.tsv).
- [Prioritized gaps](prioritized_gaps.tsv), [machine-readable summary](summary.json), [input hashes and modification times](input_versions.json), [validation checks](validation_checks.json).
- Supporting audits: [availability](../data_release_audit/AVAILABILITY_REPORT.md), [research feasibility](../research_feasibility_audit/REPORT.md), [public coverage](../public_resource_coverage/REPORT.md).

This inventory is comprehensive across the audited source registry, while explicitly retaining unmeasured dimensions. Demographic distributions, independently confirmed diagnoses, surgical events, family structure, full archive accessibility and independent evaluation eligibility remain gaps, not silently assumed negatives.
'''
    (OUT / 'REPORT.md').write_text(text)

if __name__ == '__main__':
    main()
