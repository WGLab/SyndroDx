"""Check inventory invariants without rescanning medical images or notes."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'data_release_audit'
def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))

summary = json.loads((OUT / 'summary.json').read_text())
broad = read('chop_unique_candidates.tsv')
strict = read('chop_patient_consistent_candidates.tsv')
patients = read('patient_manifest.tsv')
by_id = {r['mrn']: r for r in patients}
checks = {}
def check(name, condition):
    checks[name] = bool(condition)

e = summary['evidence_in_identity_screened_CHOP']
q = summary['patient_consistent_subset']
check('broad_counts', len(broad) == e['unique_crops'] and len(by_id) == len(patients) == e['patients'])
check('patient_membership', set(by_id) == {r['mrn'] for r in broad})
check('unique_crop_hashes', len({r['crop_sha256'] for r in broad}) == len(broad))
check('strict_counts', len(strict) == q['unique_crops'] and len({r['mrn'] for r in strict}) == q['patients'])
check('strict_subset', {r['crop_sha256'] for r in strict} <= {r['crop_sha256'] for r in broad})
check('strict_patient_consistency', all(by_id[r['mrn']]['any_known_identity_conflict'] == 'False' for r in strict))
check('technical_and_identity_eligibility', all(r['original_file_status'] == 'exists' and r['crop_file_status'] == 'decoded' and r['eligible_identity_candidate'] == 'True' and r['identity_status'] != 'conflict' and r['duplicate_identity_conflict'] != 'True' for r in broad))
check('visual_rejections_absent', all(r['visual_qc'] in {'pending', 'assistant_visual_frontal_candidate'} for r in broad))
check('nonnumeric_ids_require_bridge', all(r['mrn'].isdigit() or by_id[r['mrn']]['patient_bridge'] == 'True' for r in broad))
check('expert_counts', sum(r['expert_record'] == 'True' for r in strict) == q['expert_record_unique_crops'] and len({r['mrn'] for r in strict if r['expert_record'] == 'True'}) == q['expert_record_patients'])
versions = json.loads((OUT / 'input_versions.json').read_text())
check('candidate_input_versions_unchanged', all(Path(p).stat().st_size == v['bytes'] and Path(p).stat().st_mtime_ns == v['mtime_ns'] for p, v in versions.items()))
check('candidate_sources_allowlisted', all(r['source'] in {'CHOP_craniofacial', 'CHOP_genetics'} for r in broad))
result = {'all_passed': all(checks.values()), 'checks': checks, 'broad_crops': len(broad), 'strict_crops': len(strict), 'limitations': 'Structural consistency only; does not certify frontal view, clinical diagnoses, identity truth, or independence.'}
(OUT / 'validation_checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if not result['all_passed']:
    raise SystemExit(1)
