"""Add exact-copy groups and a conservative patient-level subset."""
import csv,json,collections
from pathlib import Path
OUT=Path(__file__).resolve().parent/'data_release_audit'
def read(p):
 with Path(p).open() as f:yield from csv.DictReader(f,delimiter='\t')
def write(p,rr,fields=None):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rr[0]),delimiter='\t');w.writeheader();w.writerows(rr)
images=list(read(OUT/'image_manifest.tsv'));groups=collections.defaultdict(list);flagged=set()
for r in images:
 if r['crop_file_status']=='decoded':groups[r['crop_sha256']].append(r)
 if not r['source'].startswith('CHOP'):continue
 if r['identity_status']=='conflict':flagged.update(x for x in r['identity_candidates'].split('|') if x)
 if r['duplicate_identity_conflict']=='True' and r['mrn']:flagged.add(r['mrn'])
rr=[]
for h,gg in groups.items():
 if len(gg)<2:continue
 ids={r['mrn'] for r in gg if r['mrn']};rr.append({'crop_sha256':h,'rows':len(gg),'sources':'|'.join(sorted({r['source'] for r in gg})),'distinct_nonempty_ids':len(ids),'ids':'|'.join(sorted(ids)),'identity_review':len(ids)>1})
write(OUT/'exact_duplicate_groups.tsv',rr,['crop_sha256','rows','sources','distinct_nonempty_ids','ids','identity_review'])
patients=list(read(OUT/'patient_manifest.tsv'))
for r in patients:r['any_known_identity_conflict']=r['mrn'] in flagged
write(OUT/'patient_manifest.tsv',patients)
strict=[r for r in read(OUT/'chop_unique_candidates.tsv') if r['mrn'] not in flagged]
write(OUT/'chop_patient_consistent_candidates.tsv',strict)
s=json.loads((OUT/'summary.json').read_text());s['exact_copy_groups']={'groups':len(rr),'groups_with_multiple_nonempty_ids':sum(r['identity_review'] for r in rr)}
s['patient_consistent_subset']={'patients':len({r['mrn'] for r in strict}),'unique_crops':len(strict),'patients_flagged_in_other_images':sum(r['any_known_identity_conflict'] for r in patients),'expert_record_patients':len({r['mrn'] for r in strict if r['expert_record']=='True'}),'expert_record_unique_crops':sum(r['expert_record']=='True' for r in strict),'scope':'Exclude any candidate patient appearing in known metadata-ID conflicts or cross-ID exact-copy groups. Still pending frontal QC, near-duplicate checks, identity adjudication and diagnosis confirmation.'}
consistent=[r for r in patients if not r['any_known_identity_conflict']]
def hpo(r):return int(r['ctakes_hpo_candidate_rows'])>0 or int(r['phenominal_hpo_candidate_rows'])>0
s['patient_consistent_subset'].update(
 clinical_dx_patients=sum(int(r['clinical_dx_rows'])>0 for r in consistent),
 hpo_candidate_patients=sum(hpo(r) for r in consistent),
 hpo_birth_date_sex_fields_patients=sum(hpo(r) and r['birth_date_present']=='True' and r['sex_code_present']=='True' for r in consistent),
 raw_note_text_patients=sum(int(r['raw_note_text_rows'])>0 for r in consistent),
 existing_mondo_patients=sum(bool(r['existing_mondo_ids']) for r in consistent),
 existing_mondo_ids=len({x for r in consistent for x in r['existing_mondo_ids'].split('|') if x}),
 variant_patients=sum(int(r['variant_records'])>0 for r in consistent))
(OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(s['patient_consistent_subset']);print(s['exact_copy_groups'])
