"""Four mutually exclusive patient subsets; image labels remain image-specific."""
import csv,json,hashlib,collections
from pathlib import Path
from audit_metadata_sources import mrnkey
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'four_subset_inventory'
def read(name):
 with (ROOT/name).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(name,rows):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(exist_ok=True)
 imagefile='data_release_audit/chop_patient_consistent_candidates.tsv'
 images=read(imagefile)
 protected={mrnkey(r['mrn']) for r in read('expert_holdout_audit/protected_mrns.tsv')}|{mrnkey(r['mrn']) for r in read('split_plan_v0/ian_protected_mrns.tsv')}
 proposal={r['mrn']:r for r in read('split_plan_v0/patient_split_proposal.tsv')}
 pats={mrnkey(r['mrn']):r for r in read('patients.tsv')}
 mapped=read('diagnosis_strata/diagnosis_image_manifest.tsv')
 image_mondo={(r['mrn'],r['crop_sha256']):set(r['mondo_ids'].split('|')) for r in mapped}
 patient_mondo=collections.defaultdict(set)
 for (m,h),ids in image_mondo.items():patient_mondo[m].update(ids)
 bypatient=collections.defaultdict(list)
 for r in images:bypatient[r['mrn']].append(r)
 names={1:'expert_source_reserved',2:'crossfolder_AB_with_MONDO',3:'crossfolder_AB_without_MONDO',4:'not_crossfolder_AB'}
 outpatients=[];imageout=[];stats={}
 for m,rr in sorted(bypatient.items()):
  tier=pats.get(m,{}).get('tier','not_available');ab=tier in {'A_corroborated','B_multi_compatible'}
  idx=1 if m in protected else (2 if ab and patient_mondo[m] else (3 if ab else 4))
  rec=dict(mrn=m,subset=idx,subset_name=names[idx],candidate_images=len(rr),images_with_mondo=sum((m,r['crop_sha256']) in image_mondo for r in rr),has_mondo_image=bool(patient_mondo[m]),mondo_ids='|'.join(sorted(patient_mondo[m])),legacy_crossfolder_tier=tier,expert_source_member=proposal.get(m,{}).get('source_label_member','False'),ian_record=proposal.get(m,{}).get('ian_record','False'),expert_status='source_record_pending_adjudication' if idx==1 else 'not_in_reserved_expert_sources',frontal_qc='pending',crossfolder_mondo_revalidation='pending',training_admitted=False)
  outpatients.append(rec)
  for r in rr:imageout.append(dict(mrn=m,subset=idx,subset_name=names[idx],crop_sha256=r['crop_sha256'],cropping_path=r['cropping_path'],source=r['source'],image_has_mondo=(m,r['crop_sha256']) in image_mondo,image_mondo_ids='|'.join(sorted(image_mondo.get((m,r['crop_sha256']),set()))),training_admitted=False))
 for idx in names:
  pp=[r for r in outpatients if r['subset']==idx]
  stats[str(idx)]=dict(name=names[idx],patients=len(pp),images=sum(r['candidate_images'] for r in pp),images_with_mondo=sum(r['images_with_mondo'] for r in pp),patients_with_mondo_images=sum(r['has_mondo_image'] for r in pp),mondo_ids=len({v for r in pp for v in r['mondo_ids'].split('|') if v}),legacy_tier_patient_counts=dict(collections.Counter(r['legacy_crossfolder_tier'] for r in pp)))
 checks=dict(patient_partition=sum(x['patients'] for x in stats.values())==len(bypatient),image_partition=sum(x['images'] for x in stats.values())==len(images),unique_images=len({r['crop_sha256'] for r in imageout})==len(imageout),no_patient_crosses_subsets=len({r['mrn'] for r in outpatients})==len(outpatients),expert_precedence=all(r['subset']==1 for r in outpatients if r['mrn'] in protected),without_mondo_has_no_image_labels=stats['3']['images_with_mondo']==0)
 assert all(checks.values())
 write('patient_subsets.tsv',outpatients);write('image_subsets.tsv',imageout)
 summary=dict(denominator='Conservative CHOP candidate pool; not final frontal-QC-approved data',total_patients=len(bypatient),total_images=len(images),assignment='patient-level: expert reservation > A/B with any mapped image > A/B without mapped images > rest',subset1_definition='Expert TSV + complete Ian reservation and known identity/copy links. Source membership, NOT adjudicated golden truth.',mondo_definition='Explicit lexical MONDO mapping of image raw labels, including structural diagnoses; not inferred onto other photos.',subsets=stats,checks=checks)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 files=[imagefile,'expert_holdout_audit/protected_mrns.tsv','split_plan_v0/ian_protected_mrns.tsv','split_plan_v0/patient_split_proposal.tsv','patients.tsv','diagnosis_strata/diagnosis_image_manifest.tsv','count_four_subsets.py']
 (OUT/'input_versions.json').write_text(json.dumps({f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
