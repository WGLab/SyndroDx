"""Build patient-level source evidence independently of file QC; never output notes."""
import csv,json,re,collections
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from audit_metadata_sources import mrnkey,clean
from audit_validation_evidence import xlsx
HERE=Path(__file__).resolve().parent;OUT=HERE/'data_release_audit';MED=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs');csv.field_size_limit(10**8)
def read(p):
 with Path(p).open() as f:
  first=f.readline();f.seek(0);yield from csv.DictReader(f,delimiter='\t' if first.count('\t')>first.count(',') else ',')
def key(x):return mrnkey(str(x)) if x is not None else ''
def main():
 ids=set();conflicted=set()
 for r in read(OUT/'candidate_inventory.tsv'):
  if not r['source'].startswith('CHOP'):continue
  if r['mrn']:ids.add(r['mrn'])
  if r['identity_status']=='conflict':conflicted.update(r['identity_candidates'].split('|'))
 data={m:dict(mrn=m,has_other_identity_conflict=m in conflicted,patient_bridge=False,clinical_dx_rows=0,clinical_dx_ids=set(),raw_icd_strings=set(),variant_records=0,variant_hgvs_records=0,variant_significance_records=0,expert_review_record=False,expert_diagnosis_raw='',existing_mondo_ids=set(),mondo_by_source={},historical_splits=set(),folder_tiers=set()) for m in ids}
 for r in read(HERE/'labels.tsv'):
  m=key(r['mrn'])
  if m in data:
   data[m]['historical_splits'].add(r['split'])
   if r['tier']:data[m]['folder_tiers'].add(r['tier'])
 for r in xlsx(MED/'CHOP_Images/examples/Ian_verified.xlsx'):
  m=key(r['pat_mrn_id'])
  if m in data:data[m]['expert_review_record']=True;data[m]['expert_diagnosis_raw']=r.get('Diagnosis','')
 names=['CHOP_Diagnosis/expert_chop_disease_labels_combined_20260511.tsv','CHOP_Diagnosis/expert_labels_manual_fix_20260511.tsv','Arcus_old/complete_processing_analysis.csv']
 for name in names:
  for r in read(MED/name):
   m=key(r['MRN'])
   if m in data:
    mids=set(re.findall(r'MONDO:\d+',r.get('mondo') or ''));data[m]['existing_mondo_ids'].update(mids)
    if mids:data[m]['mondo_by_source'][name]=sorted(mids)
 print('Expert/MONDO sources linked',flush=True)
 for r in read(MED/'Clarity/all_mrns_chop_images_clarity_edg_dx.csv'):
  m=key(r['MRN'])
  if m in data:
   x=data[m];x['clinical_dx_rows']+=1
   if clean(r.get('DX_ID')):x['clinical_dx_ids'].add(r['DX_ID'])
   if clean(r.get('CURRENT_ICD10_LIST')):x['raw_icd_strings'].add(r['CURRENT_ICD10_LIST'])
 print('Clinical diagnosis source linked',flush=True)
 raw=MED/'Clarity_20260704/clarity_raw_exports';bridge=collections.defaultdict(set);reverse=collections.defaultdict(set)
 for b in pq.ParquetFile(raw/'patient.parquet').iter_batches(columns=['pat_id','pat_mrn_id','birth_date','sex_c','ethnic_group_c']):
  for r in b.to_pylist():
   m=key(r['pat_mrn_id']);p=str(r['pat_id'])
   if m not in data:continue
   bridge[p].add(m);reverse[m].add(p)
   data[m].update(birth_date_present=r['birth_date'] is not None,sex_code_present=r['sex_c'] is not None,ethnicity_code_present=r['ethnic_group_c'] is not None)
 good={p:next(iter(ms)) for p,ms in bridge.items() if len(ms)==1 and len(reverse[next(iter(ms))])==1}
 for m in good.values():data[m]['patient_bridge']=True
 for b in pq.ParquetFile(raw/'variant.parquet').iter_batches(columns=['pat_id','hgvs_name','clinical_signif_c']):
  for r in b.to_pylist():
   m=good.get(str(r['pat_id']))
   if m:
    data[m]['variant_records']+=1;data[m]['variant_hgvs_records']+=bool(r['hgvs_name']);data[m]['variant_significance_records']+=r['clinical_signif_c'] is not None
 print('Structured patient and variant sources linked',flush=True)
 # This bridge is specific to Clarity. Do not join Arcus HPO via unrelated pat_id namespaces.
 rr=[]
 for m,x in sorted(data.items()):
  vals=list(x['mondo_by_source'].values());x['existing_mondo_source_disagreement']=len({tuple(v) for v in vals})>1
  x['mondo_by_source']=json.dumps(x['mondo_by_source'],sort_keys=True)
  x['clinical_dx_id_count']=len(x.pop('clinical_dx_ids'));x['raw_icd_string_count']=len(x.pop('raw_icd_strings'))
  for k,v in list(x.items()):
   if isinstance(v,set):x[k]='|'.join(sorted(v))
  x['diagnosis_adjudicated']=False;x['review_status']='pending';rr.append(x)
 fields=sorted(set().union(*(r.keys() for r in rr)))
 with (OUT/'patient_evidence.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rr)
 print('Patient evidence rows',len(rr),flush=True)
if __name__=='__main__':main()
