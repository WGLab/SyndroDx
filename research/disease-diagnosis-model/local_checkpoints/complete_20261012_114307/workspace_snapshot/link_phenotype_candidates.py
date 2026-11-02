"""Add raw HPO candidate counts; clinical assertions remain unverified."""
import csv,json,collections
from pathlib import Path
from audit_metadata_sources import mrnkey,clean
HERE=Path(__file__).resolve().parent;OUT=HERE/'data_release_audit';MED=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs');csv.field_size_limit(10**8)
def read(p):
 with Path(p).open() as f:
  first=f.readline();f.seek(0);yield from csv.DictReader(f,delimiter='\t' if first.count('\t')>first.count(',') else ',')
rr=list(read(OUT/'patient_evidence.tsv'));by={r['mrn']:r for r in rr}
versions={}
for filename,column in [('ctakes_hpo.csv','ctakes_hpo_candidate_rows'),('phenominal_hpo.csv','phenominal_hpo_candidate_rows')]:
 for r in rr:r[column]=0
 p=MED/'CHOP_Images/final_results'/filename;s=p.stat();versions[str(p)]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
 for r in read(p):
  m=mrnkey(r.get('pat_mrn_id'))
  if m in by and clean(r.get('HPOId')):by[m][column]+=1
 print(filename,'linked patients',sum(r[column]>0 for r in rr),flush=True)
for r in rr:
 r.pop('clinician_hpo_rows',None);r['phenotype_assertions_adjudicated']=False
with (OUT/'patient_evidence.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rr[0]),delimiter='\t');w.writeheader();w.writerows(rr)
(OUT/'phenotype_input_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
