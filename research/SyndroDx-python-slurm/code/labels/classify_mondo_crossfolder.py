"""Stratify mapped MONDO diagnosis samples by existing patient A/B evidence."""
import csv,json,collections,hashlib
from pathlib import Path
from audit_metadata_sources import mrnkey
from expert_reservation import filter_reserved
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'diagnosis_strata'
def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(name,rows):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(exist_ok=True)
 concepts={r['raw_label']:r for r in read(ROOT/'candidate_concept_audit/raw_label_concepts.tsv')}
 patients={mrnkey(r['mrn']):r for r in read(ROOT/'patients.tsv')}
 strict=read(ROOT/'data_release_audit/chop_patient_consistent_candidates.tsv');remaining,_=filter_reserved(strict)
 keys={(r['mrn'],r['crop_sha256']) for r in strict};remkeys={(r['mrn'],r['crop_sha256']) for r in remaining}
 matches=collections.defaultdict(lambda:dict(mondo=set(),labels=set(),sources=set()))
 for r in read(ROOT/'data_release_audit/image_manifest.tsv'):
  pair=(r['mrn'],r['crop_sha256'])
  if not r['source'].startswith('CHOP') or r['eligible_identity_candidate']!='True' or pair not in keys:continue
  mids=concepts[r['raw_label']]['mondo_ids']
  if not mids:continue
  x=matches[pair];x['mondo'].update(mids.split('|'));x['labels'].add(concepts[r['raw_label']]['raw_label_id']);x['sources'].add(r['source'])
 rows=[];links=collections.defaultdict(lambda:dict(images=set(),labels=set()))
 for r in strict:
  pair=(r['mrn'],r['crop_sha256'])
  if pair not in matches:continue
  x=matches[pair];pat=patients.get(mrnkey(r['mrn']),{});tier=pat.get('tier','not_available')
  ab=tier in {'A_corroborated','B_multi_compatible'}
  rec=dict(mrn=r['mrn'],crop_sha256=r['crop_sha256'],cropping_path=r['cropping_path'],mondo_ids='|'.join(sorted(x['mondo'])),is_diagnosis=True,crossfolder_validated_legacy=ab,crossfolder_tier=tier,n_folders_legacy=pat.get('n_folders',''),diagnosis_stratum='diagnosis_crossfolder_AB' if ab else 'diagnosis_other',crossfolder_mondo_revalidation='pending',label_evidence='raw_label_lexical_mapping_not_clinical_confirmation',raw_label_ids='|'.join(sorted(x['labels'])),sources='|'.join(sorted(x['sources'])),remaining_after_reservation=pair in remkeys,training_admitted=False)
  rows.append(rec)
  for mid in x['mondo']:
   z=links[(r['mrn'],mid)];z['images'].add(r['crop_sha256']);z['labels'].update(x['labels'])
 write('diagnosis_image_manifest.tsv',rows)
 remain=[r for r in rows if r['remaining_after_reservation']]
 for name,rr in [('remaining_diagnosis_crossfolder_AB.tsv',[r for r in remain if r['crossfolder_validated_legacy']]),('remaining_diagnosis_other.tsv',[r for r in remain if not r['crossfolder_validated_legacy']])]:write(name,rr)
 def count(rr):return dict(images=len(rr),patients=len({r['mrn'] for r in rr}),mondo_ids=len({m for r in rr for m in r['mondo_ids'].split('|')}))
 summary={}
 for name,rr in [('all_conservative',rows),('after_reservation',remain)]:
  summary[name]={'all_diagnosis':count(rr),'crossfolder_AB':count([r for r in rr if r['crossfolder_validated_legacy']]),'other':count([r for r in rr if not r['crossfolder_validated_legacy']]),'tiers':{t:count([r for r in rr if r['crossfolder_tier']==t]) for t in sorted({r['crossfolder_tier'] for r in rr})}}
 classes=[]
 for mid in sorted({m for r in remain for m in r['mondo_ids'].split('|')}):
  rr=[r for r in remain if mid in r['mondo_ids'].split('|')]
  for label,subset in [('crossfolder_AB',[r for r in rr if r['crossfolder_validated_legacy']]),('other',[r for r in rr if not r['crossfolder_validated_legacy']])]:
   classes.append(dict(mondo_id=mid,stratum=label,images=len(subset),patients=len({r['mrn'] for r in subset})))
 write('remaining_disease_strata.tsv',classes)
 for rr in [rows,remain]:
  a=[r for r in rr if r['crossfolder_validated_legacy']];b=[r for r in rr if not r['crossfolder_validated_legacy']]
  assert len(a)+len(b)==len(rr)
  assert not ({r['mrn'] for r in a}&{r['mrn'] for r in b})
  assert len({r['crop_sha256'] for r in rr})==len(rr)
 summary['checks_passed']=True
 summary['meaning']='Existing patient-level A/B grades only; specific MONDO evidence across folders has not been revalidated.'
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 sources=[ROOT/'patients.tsv',ROOT/'labelmap.py',ROOT/'build_labels.py',ROOT/'candidate_concept_audit/raw_label_concepts.tsv',ROOT/'data_release_audit/image_manifest.tsv',ROOT/'data_release_audit/chop_patient_consistent_candidates.tsv',Path(__file__),ROOT/'expert_reservation.py']+list((ROOT/'expert_holdout_audit').glob('protected_mrns.tsv'))+list((ROOT/'expert_holdout_audit').glob('blocked*.tsv'))+list((ROOT/'split_plan_v0').glob('ian_*.tsv'))
 (OUT/'input_versions.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
