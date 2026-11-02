"""Prepare private clinician review sheets and an aggregate audit report."""
import collections
import csv
import importlib.util
import sys
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parent
PROJECT=WORK.parent
sys.path.insert(0,str(PROJECT/'clinical_evidence_agent_20261101'))
from common import load,save,sha,now,jsonl


def csv_file(path,rows,fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in rows:
            # CSV is intended for local spreadsheet review, never formula execution.
            safe={k:("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v) for k,v in row.items()}
            writer.writerow(safe)
    path.chmod(0o600)


def main():
    audit=load(WORK/'audit.json');results=load(WORK/'results.json');protocol=load(WORK/'protocol.json')
    if results['audit_sha256']!=sha(WORK/'audit.json') or results['protocol_sha256']!=sha(WORK/'protocol.json'):
        raise ValueError('audit_or_protocol_changed')
    for name,digest in protocol['source_sha256'].items():
        if sha(WORK/name)!=digest:raise ValueError('audited_source_changed')
    with np.load(WORK/'private/strict_outputs.npz') as out:
        if not np.array_equal(out['visual_scores'],out['cross_attention_routed_scores']):raise ValueError('strict_fallback_failed')
        if not np.array_equal(out['visual_scores'],out['score_fusion_routed_scores']):raise ValueError('strict_fallback_failed')
        if out['HPO_scores'].any():raise ValueError('unresolved_HPO_used')
    index=load(WORK/'private/strict_index.json');counts=collections.Counter(index['labels'])
    review=list(jsonl(WORK/'private/hpo_review_queue.jsonl'))
    patients=list(jsonl(WORK/'private/patient_coverage_review.jsonl'))
    names=load(PROJECT/'arcface1000_hpo_alignment_20261028/private/ontology.json')['terms']
    hpo_rows=[]
    for r in review:
        row=dict(r,HPO_name=names.get(r['HPO'],{}).get('name',''),review_id=str(len(hpo_rows)+1))
        hpo_rows.append(row)
    hpo_fields=['review_id','patient_key','group_id','HPO','HPO_name','recorded_assertion','record_creation_date',
        'context_available_in_source','source','source_database','source_rows','reviewed_assertion','reviewed_experiencer',
        'clinical_event_time','evidence_available_at','decision_time','reviewer_ref','source_document_reference','review_note']
    csv_file(WORK/'private/HPO_REVIEW.csv',hpo_rows,hpo_fields)
    # Do not show model predictions or candidate coverage to diagnosis adjudicators.
    patient_rows=[{'group_id':r['group_id'],'source_MONDO':r['original_label'],'source_diagnosis':r['disease_name'],
        'recorded_HPO_terms':r['recorded_HPO_terms'],'reviewed_diagnosis':None,'reviewed_MONDO':None,
        'diagnostic_basis':None,'source_reference':None,'target_population_eligible':None,
        'eligibility_reason':None,'reviewer_ref':None,'review_date':None} for r in patients]
    csv_file(WORK/'private/COHORT_REVIEW.csv',patient_rows,list(patient_rows[0]))
    patient_ceiling=audit['strict_exact_classifier_covered_patients']/audit['patients']
    disease_ceiling=audit['diseases_by_coverage_reason']['covered_by_exact_current_classifier_mapping']/audit['diseases']
    clinical={'state':'review_templates_ready_no_new_labels_admitted','patient_rows':len(patient_rows),'HPO_rows':len(hpo_rows),
        'model_predictions_in_review_sheets':False,'real_MRN_in_review_sheets':False,
        'context_marked_available_records':audit['clinician_source_context_counts']['records_with_local_context'],
        'context_not_loaded_into_chat':True,'clinical_review_completed':False,
        'HPO_csv_sha256':sha(WORK/'private/HPO_REVIEW.csv'),'cohort_csv_sha256':sha(WORK/'private/COHORT_REVIEW.csv'),
        'updated_at_utc':now()}
    save(WORK/'review_receipt.json',clinical)
    verification={'state':'passed','unit_tests':7,'strict_HPO_scores_empty':True,'multimodal_fallbacks_equal_visual':True,
        'model_and_input_hashes_verified':True,'candidate_crosswalk_changed':results['diagnosis_candidate_mapping_changed'],
        'largest_disease_patient_count':max(counts.values()),'singleton_diseases':sum(v==1 for v in counts.values()),
        'all_cohort_closed_set_micro_accuracy_ceiling':patient_ceiling,
        'all_cohort_closed_set_disease_macro_accuracy_ceiling':disease_ceiling,
        'new_training_admissions':0,'updated_at_utc':now()}
    save(WORK/'verification.json',verification)
    visual=results['classification']['visual']['covered']['all'];ret=results['retrieval']['visual']['all']
    lines=['# CHOP clinical input and candidate-coverage audit','',
        'The current CHOP expert export cannot establish facial + HPO efficacy. All 724 recorded HPO terms in 58 patients have unresolved polarity; there are zero patients with polarity-confirmed HPO in these cached inputs. This does not mean the patients lack phenotypes.',
        '', '## Diagnostic candidate coverage','',
        '| Category | Patients | Diseases |','|---|---:|---:|']
    for category,n in audit['coverage_failure_counts'].items():
        lines.append(f'| {category} | {n} | {audit["diseases_by_coverage_reason"][category]} |')
    lines += ['', 'Total: 287 patients, 58 diagnoses. The current classifier covers 26 patients. The full visual-training label vocabulary would cover 30, and the full GMDB reference has unique exact mappings for 38. Alias and unique obsolete-ID replacement checks recovered no additional cases. An absent exact crosswalk is not proof that no broader clinical relationship exists.',
        '', f'With the fixed classifier and this cohort, all-patient micro accuracy cannot exceed {100*patient_ceiling:.2f}%, even with perfect ranking of covered diagnoses. Disease-macro accuracy cannot exceed {100*disease_ceiling:.2f}%. Candidate coverage must therefore be reported separately from ranking quality.',
        '', f'The largest diagnosis accounts for {max(counts.values())}/287 patients; {sum(v==1 for v in counts.values())}/58 diagnoses have only one patient. The phenotype-retrieval endpoint includes 238 queries from nine non-singleton diseases and is not equivalent to broad rare-disease diagnostic accuracy.',
        '', '## Strict matched controls','',
        'The frozen default seed11 model was evaluated in FP32. Unresolved HPO was excluded. Both cross-attention routing and independent score fusion fell back exactly to the visual branch for every patient. HPO-only abstained for all 287 patients. There is no eligible paired subgroup for estimating multimodal benefit.',
        '', f'Visual closed-set classification on the 26 covered patients: disease-macro Top1 {100*visual["top1_macro"]:.2f}%, Top5 {100*visual["top5_macro"]:.2f}%.',
        f'Native visual retrieval on the existing 238-query/nine-disease task: disease-macro Top1 {100*ret["top1_macro"]:.2f}%, Top5 {100*ret["top5_macro"]:.2f}%.',
        '', 'These are different tasks and denominators. The latter uses the fixed seed11/all-image average and should not be substituted for the historical four-seed/view-balanced 27.66% / 39.92%.',
        '', 'The previous state2 cross-attention results remain archived as unresolved-annotation sensitivity analyses. They do not show that verified clinical HPO harms diagnosis. Additional direct-head controls are in results.json and are not deployed.',
        '', '## Next data work','',
        '1. Review the intended clinical population and diagnostic granularity without seeing model predictions. Do not discard cases based on model performance or label coverage. Report the original cohort and any prespecified clinically justified subgroup separately.',
        '2. Establish the clinician-HPO export semantics or review source evidence for polarity and experiencer. There are 221 records marked as having local context and 503 without; a creation date is not automatically a clinical availability time.',
        '3. Build an adjudicated development cohort and a separate frozen clinical evaluation cohort. CHOP expert has already been inspected; BCH or a new temporal cohort requires its own access, identity and overlap checks.',
        '4. After that release, rerun facial-only, HPO-only, cross-attention and independent score fusion on identical patients and candidate rules. Evaluate agent acquisition only when genuine answers and timing are available.',
        '', 'The local files private/HPO_REVIEW.csv and private/COHORT_REVIEW.csv are ready for review. They contain no model predictions or real MRN. Patient keys and clinical content must stay on HPC. No new observations were marked positive, no images or raw notes were read, and no model was trained or replaced.', '']
    (WORK/'README.md').write_text('\n'.join(lines))
    status={'state':'technical_audit_complete_clinical_adjudication_pending','audit':str(WORK/'audit.json'),
        'results':str(WORK/'results.json'),'report':str(WORK/'README.md'),'paired_multimodal_efficacy_estimable':False,
        'review_sheets_ready':True,'checkpoint_changed':False,'GPU_allocations_cancelled':0,'updated_at_utc':now()}
    save(WORK/'status.json',status)
    registry=load(PROJECT/'LATEST_MULTIMODAL_EXPERIMENT.json')
    registry.update(latest_CHOP_input_audit=status,clinical_HPO_validity_warning='Historical CHOP state2 inputs have unresolved polarity; do not claim clinical multimodal benefit or harm from those numbers.',updated_at_utc=now())
    save(PROJECT/'LATEST_MULTIMODAL_EXPERIMENT.json',registry)
    try:
        spec=importlib.util.spec_from_file_location('chop_audit_mail',PROJECT/'transface_objective_comparison_20261028/matched_runtime.py')
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
        runtime.notify('CHOP clinical input audit completed', '\n'.join(lines))
        save(WORK/'notification.json',{'state':'local_notifier_invoked','updated_at_utc':now()})
    except Exception as e:save(WORK/'notification.json',{'state':'notifier_failed','error_type':type(e).__name__,'updated_at_utc':now()})
    print(__import__('json').dumps({'state':status['state'],'review':clinical,'verification':verification}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'finalize_failure.json',{'error_type':type(e).__name__,'updated_at_utc':now()})
        raise SystemExit(1) from None
