"""Stream note coverage into the candidate ledger without retaining note text."""
import csv,json
from pathlib import Path
from audit_metadata_sources import mrnkey,clean
HERE=Path(__file__).resolve().parent;OUT=HERE/'data_release_audit';csv.field_size_limit(10**8)
with (OUT/'patient_evidence.tsv').open() as f:rr=list(csv.DictReader(f,delimiter='\t'))
by={r['mrn']:r for r in rr}
for r in rr:r.update(raw_note_rows=0,raw_note_text_rows=0,raw_note_dated_text_rows=0,raw_note_selection_provenance='pending')
p=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs/CHOP_Images/all_clarity_notes_merged.csv');st=p.stat();n=0
with p.open() as f:
 for r in csv.DictReader(f,delimiter='\t'):
  n+=1;m=mrnkey(r.get('MRN'))
  if m in by:
   x=by[m];x['raw_note_rows']+=1
   if clean(r.get('TEXT')):
    x['raw_note_text_rows']+=1;x['raw_note_dated_text_rows']+=bool(clean(r.get('DATE')))
  if n%250000==0:print('Note rows scanned',n,flush=True)
with (OUT/'patient_evidence.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rr[0]),delimiter='\t');w.writeheader();w.writerows(rr)
(OUT/'note_input_versions.json').write_text(json.dumps({str(p):{'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'rows':n,'text_patients':sum(r['raw_note_text_rows']>0 for r in rr),'provenance':'selection lineage pending; no PubMind outputs admitted'}},indent=2)+'\n')
print('Note text linked patients',sum(r['raw_note_text_rows']>0 for r in rr),flush=True)
