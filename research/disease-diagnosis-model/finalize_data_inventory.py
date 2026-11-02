"""Join completed file QC and evidence; make review queues and aggregate report."""
import csv,json,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'data_release_audit';csv.field_size_limit(10**8)
def read(p):
 with Path(p).open() as f:yield from csv.DictReader(f,delimiter='\t')
def write(p,rows,fields=None):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
def yes(x):return x is True or x=='True'
def main():
 images=list(read(OUT/'image_manifest.tsv'));ev={r['mrn']:r for r in read(OUT/'patient_evidence.tsv')}
 # Apply explicit sample exclusions and nonstandard identity quarantine.
 sample={r['cropping_path']:r for r in read(OUT/'view_qc_sample.tsv')}
 sample_by_hash={}
 severity={'assistant_visual_frontal_candidate':0,'assistant_visual_review_needed':1,'assistant_visual_reject_candidate':2}
 for r in images:
  q=sample.get(r['cropping_path']);h=r.get('crop_sha256')
  if q and h:
   old=sample_by_hash.get(h)
   if old is None or severity.get(q['status'],0)>severity.get(old['status'],0):sample_by_hash[h]=q
 newly_flagged=0;nonstandard_id_rows=0
 for r in images:
  if r['source'].startswith('CHOP') and r['mrn'] and not r['mrn'].isdigit() and not yes(ev.get(r['mrn'],{}).get('patient_bridge')):
   nonstandard_id_rows+=1;r['identity_status']='non_numeric_unverified';r['eligible_identity_candidate']='False'
   if 'non_numeric_id_without_patient_bridge' not in r['review_reasons']:r['review_reasons']+='|non_numeric_id_without_patient_bridge'
  q=sample_by_hash.get(r.get('crop_sha256')) or sample.get(r['cropping_path'])
  if q:
   r['visual_qc']=q['status']
   if q['status'] in ['assistant_visual_reject_candidate','assistant_visual_review_needed']:
    newly_flagged+=yes(r['eligible_identity_candidate']);r['eligible_identity_candidate']='False'
    reason='sample_visual_'+q['reason']
    if reason not in r['review_reasons']:r['review_reasons']+='|'+reason
 write(OUT/'image_manifest.tsv',images)
 # Remove placeholder columns from a previous running version; not-measured is not zero.
 for r in ev.values():r.pop('clinician_hpo_rows',None)
 write(OUT/'patient_evidence.tsv',list(ev.values()))
 valid=[r for r in images if yes(r['eligible_identity_candidate'])]
 byhash=collections.defaultdict(list)
 for r in valid:byhash[r['crop_sha256']].append(r)
 unique=[]
 for h,rs in sorted(byhash.items()):
  # Choose a reproducible representative; keep every original row in image_manifest.
  row=dict(min(rs,key=lambda r:(r['source'],r['image_path'])));e=ev.get(row['mrn'],{})
  row.update(exact_copy_rows=len(rs),has_clinical_dx=int(e.get('clinical_dx_rows') or 0)>0,existing_mondo_ids=e.get('existing_mondo_ids',''),existing_mondo_source_disagreement=e.get('existing_mondo_source_disagreement',''),has_variant_record=int(e.get('variant_records') or 0)>0,has_other_identity_conflict=e.get('has_other_identity_conflict',''),diagnosis_adjudicated=False)
  unique.append(row)
 write(OUT/'chop_unique_candidates.tsv',unique,list(images[0])+['exact_copy_rows','has_clinical_dx','existing_mondo_ids','existing_mondo_source_disagreement','has_variant_record','has_other_identity_conflict','diagnosis_adjudicated'])
 issues=[r for r in images if r['source'].startswith('CHOP') and not yes(r['eligible_identity_candidate'])]
 write(OUT/'image_issue_queue.tsv',issues,list(images[0]))
 patients=collections.defaultdict(list)
 for r in unique:patients[r['mrn']].append(r)
 pr=[];mondo=collections.defaultdict(set);folder=collections.defaultdict(set)
 for m,rr in sorted(patients.items()):
  e=ev.get(m,{});r=dict(e)
  r.update(unique_candidate_crops=len(rr),image_sources='|'.join(sorted({x['source'] for x in rr})),expert_review_status='record_linked_criteria_pending' if yes(e.get('expert_review_record')) else 'no_record_linked',view_review_status='pending',final_test_eligible='not_yet_adjudicated')
  pr.append(r)
  for mid in e.get('existing_mondo_ids','').split('|'):
   if mid:mondo[mid].add(m)
  for t in e.get('folder_tiers','').split('|'):
   if t:folder[t].add(m)
 write(OUT/'patient_manifest.tsv',pr)
 # Source MONDO IDs remain separate from folder-term normalization.
 names={};cur=None
 with (HERE/'ontology/mondo.obo').open() as f:
  for l in f:
   if l.startswith('['):cur=None
   elif l.startswith('id: MONDO:'):cur=l.strip()[4:]
   elif cur and l.startswith('name: '):names[cur]=l.strip()[6:]
 mr=[dict(mondo_id=k,mondo_name=names.get(k,''),identity_screened_patients=len(v),status='existing_source_mapping_unadjudicated') for k,v in sorted(mondo.items(),key=lambda x:(-len(x[1]),x[0]))]
 write(OUT/'existing_mondo_candidate_counts.tsv',mr,['mondo_id','mondo_name','identity_screened_patients','status'])
 rawmap={r['raw_label']:r for r in read(HERE/'mondo_normalization/raw_label_mondo_map.tsv')}
 rawcounts=collections.defaultdict(set);mapstats=collections.Counter()
 for r in unique:
  if not r['source'].startswith('CHOP_craniofacial'):continue
  mapping=rawmap.get(r['raw_label'],{});mapstats[mapping.get('status','not_in_previous_map')]+=1
  if mapping.get('status')=='mapped' and mapping.get('mondo_id'):rawcounts[mapping['mondo_id']].add(r['mrn'])
 rawrows=[dict(mondo_id=k,patients=len(v),status='folder_term_mapping_only_not_clinical_confirmation') for k,v in sorted(rawcounts.items(),key=lambda x:(-len(x[1]),x[0]))]
 write(OUT/'folder_mondo_candidate_counts.tsv',rawrows,['mondo_id','patients','status'])
 def count(test):return sum(bool(test(r)) for r in pr)
 s=json.loads((OUT/'summary.json').read_text())
 s.setdefault('CHOP_before_sample_visual_screen',s['CHOP_union'])
 s['CHOP_union']={'identity_screened_rows':len(valid),'exact_unique_crops':len(unique),'patients':len(patients),'expert_record_patients':count(lambda r:yes(r.get('expert_review_record')))}
 s['nonstandard_id_rows_quarantined']=nonstandard_id_rows
 s.setdefault('sample_visual_screen_newly_quarantined_rows',newly_flagged)
 s['view_qc_sample']=json.loads((OUT/'view_qc_sample_summary.json').read_text())
 for src,rec in s['by_source'].items():
  sr=[r for r in valid if r['source']==src]
  rec['identity_status_rows']=dict(collections.Counter(r['identity_status'] for r in images if r['source']==src))
  rec.update(identity_screened_candidate_rows=len(sr),identity_screened_unique_crops=len({r['crop_sha256'] for r in sr}),identity_screened_patients=len({r['mrn'] for r in sr}),expert_linked_screened_patients=len({r['mrn'] for r in sr if yes(r['expert_record'])}))
 s['evidence_in_identity_screened_CHOP']={'patients':len(pr),'unique_crops':len(unique),'patient_bridge':count(lambda r:yes(r.get('patient_bridge'))),'ctakes_hpo_candidate_patients':count(lambda r:int(r.get('ctakes_hpo_candidate_rows') or 0)>0),'phenominal_hpo_candidate_patients':count(lambda r:int(r.get('phenominal_hpo_candidate_rows') or 0)>0),'raw_note_text_patients':count(lambda r:int(r.get('raw_note_text_rows') or 0)>0),'image_hpo_birth_date_sex_field_patients':count(lambda r:(int(r.get('ctakes_hpo_candidate_rows') or 0)>0 or int(r.get('phenominal_hpo_candidate_rows') or 0)>0) and yes(r.get('birth_date_present')) and yes(r.get('sex_code_present'))),'clinical_dx_records':count(lambda r:int(r.get('clinical_dx_rows') or 0)>0),'existing_mondo_patients':count(lambda r:bool(r.get('existing_mondo_ids'))),'existing_mondo_ids':len(mondo),'existing_mondo_source_disagreement_patients':count(lambda r:yes(r.get('existing_mondo_source_disagreement'))),'expert_review_record_patients':count(lambda r:yes(r.get('expert_review_record'))),'expert_record_unique_crops':sum(yes(r['expert_record']) for r in unique),'variant_record_patients':count(lambda r:int(r.get('variant_records') or 0)>0),'hgvs_record_patients':count(lambda r:int(r.get('variant_hgvs_records') or 0)>0),'variant_significance_record_patients':count(lambda r:int(r.get('variant_significance_records') or 0)>0),'birth_date_field_present':count(lambda r:yes(r.get('birth_date_present'))),'sex_code_field_present':count(lambda r:yes(r.get('sex_code_present'))),'ethnicity_code_field_present':count(lambda r:yes(r.get('ethnicity_code_present'))),'other_image_identity_conflict_patients':count(lambda r:yes(r.get('has_other_identity_conflict'))),'crossfolder_A_or_B_patients':len(folder['A_corroborated']|folder['B_multi_compatible']),'existing_mondo_ids_with_at_least_5_patients':sum(len(v)>=5 for v in mondo.values()),'clinically_adjudicated_disease_count':None,'manual_frontal_qc_pass_count':None,'final_independent_test_count':None}
 s['folder_normalization']={'unique_crop_status_counts':dict(mapstats),'candidate_mondo_ids':len(rawcounts),'note':'Whole-label lexical mappings; broad mappings allowed by prior policy. Not combined with expert-source IDs as disease truth.'}
 cfad=[r for r in images if r['source']=='CFAD' and yes(r['technical_candidate'])]
 s['CFAD_preprocessing_counts']=dict(collections.Counter(r['official_group'] for r in cfad))
 s['CFAD_original_or_original_crop_rows']=sum(r['official_group'] in ['1. Original','2. Original Cropped'] for r in cfad)
 s['review_queue_rows']=len(issues)
 (OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
 assert len(unique)==s['CHOP_union']['exact_unique_crops']
 assert len(patients)==s['CHOP_union']['patients']
 assert all(r['crop_file_status']=='decoded' and r['mrn'] for r in unique)
 assert len({r['crop_sha256'] for r in unique})==len(unique)
 print(json.dumps(s['evidence_in_identity_screened_CHOP'],indent=2))
if __name__=='__main__':main()
