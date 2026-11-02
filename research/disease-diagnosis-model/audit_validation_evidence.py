"""Read-only audit of cross-folder tiers and existing reviewed cohorts; aggregates only."""
import csv,json,collections,random,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
from audit_metadata_sources import mrnkey,FRONT
HERE=Path(__file__).resolve().parent
ROOT=Path('/controlled-data/craniofacial_data')
EX=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs/CHOP_Images/examples')
csv.field_size_limit(10**8)
def rows(p):
 with p.open() as f:
  first=f.readline();f.seek(0)
  return list(csv.DictReader(f,delimiter='\t' if first.count('\t')>first.count(',') else ','))
def xlsx(p):
 ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(p) as z:
  ss=[''.join(t.text or '' for t in e.findall('.//m:t',ns)) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
  rr=[]
  for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:sheetData/m:row',ns):
   rec={}
   for c in row:
    col=''.join(a for a in c.attrib['r'] if a.isalpha());v=c.find('m:v',ns);value=v.text if v is not None else ''
    if c.get('t')=='s':value=ss[int(value)] if value else ''
    elif c.get('t')=='inlineStr':value=''.join(t.text or '' for t in c.findall('.//m:t',ns))
    rec[col]=value
   rr.append(rec)
  header=rr[0];return [{header[k]:r.get(k,'') for k in header} for r in rr[1:] if any(r.values())]
def main():
 labels=rows(HERE/'labels.tsv');pats=rows(HERE/'patients.tsv')
 cf=collections.defaultdict(set);gf=collections.defaultdict(set);gall=collections.defaultdict(set);split=collections.defaultdict(set);tier={};imrn=collections.defaultdict(set)
 for r in labels:
  m=mrnkey(r['mrn'])
  if not m:continue
  cf[m].add(r['image_path']);split[m].add(r['split'])
  for k in ['image_path','cropping_path']:imrn[r[k]].add(m)
 for r in rows(ROOT/'genetics_photos_metdata_top10dx.csv'):
  m=mrnkey(r['pat_mrn_id']);gall[m].add(r['image_path']);imrn[r['image_path']].add(m)
  if r['EVA02_Organ']==FRONT:gf[m].add(r['image_path'])
 for r in pats:tier[mrnkey(r['mrn'])]=r['tier']
 # Add explicit crop mapping from the April metadata, without using its diagnoses.
 for r in rows(Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs/CHOP_Images/final_results/genetics_photos_top10_metadata_04082026.tsv')):
  if r.get('cropping_path'):imrn[r['cropping_path']].add(mrnkey(r['pat_mrn_id']))
 def describe(ids):
  return {'patients':len(ids),'union_frontal_patients':len(ids & (set(cf)|set(gf))),'union_frontal_exact_paths':len(set().union(*(cf.get(m,set())|gf.get(m,set()) for m in ids))) if ids else 0,'frontal_patients_absent_from_craniofacial_archive':len((ids&set(gf))-set(cf)),'craniofacial_frontal_patients':len(ids&set(cf)),'craniofacial_frontal_images':len(set().union(*(cf.get(m,set()) for m in ids))) if ids else 0,'genetics_frontal_patients':len(ids&set(gf)),'genetics_frontal_images':len(set().union(*(gf.get(m,set()) for m in ids))) if ids else 0,'genetics_allview_images':len(set().union(*(gall.get(m,set()) for m in ids))) if ids else 0,'patients_by_existing_label_split_nonexclusive':dict(collections.Counter(s for m in ids for s in split.get(m,set()))),'patients_by_crossfolder_tier':dict(collections.Counter(tier.get(m,'not_in_patient_table') for m in ids))}
 tiers={t:describe({m for m,v in tier.items() if v==t}) for t in set(tier.values())}
 expert=xlsx(EX/'Ian_verified.xlsx');eids={mrnkey(r['pat_mrn_id']) for r in expert if r.get('pat_mrn_id')};result={'crossfolder_tiers':tiers,'Ian_verified':dict(rows=len(expert),**describe(eids),diagnosis_nonempty_rows=sum(bool(r.get('Diagnosis')) for r in expert),notes_nonempty_rows=sum(bool(r.get('Notes')) for r in expert),distinct_diagnosis_strings=len({r.get('Diagnosis') for r in expert if r.get('Diagnosis')}))}
 # Only aggregate diagnosis names, never notes or identifiers.
 result['Ian_verified']['diagnosis_row_counts']=dict(collections.Counter(r.get('Diagnosis','') for r in expert))
 result['confirmation_tables']={}
 allids={}
 for p in sorted(EX.glob('*withConfirmation.csv')):
  rr=rows(p);ids={mrnkey(r['pat_mrn_id']) for r in rr if r.get('pat_mrn_id')};allids[p.name]=ids
  result['confirmation_tables'][p.name]=dict(rows=len(rr),unique_listed_image_paths=len({r['image_path'] for r in rr}),**describe(ids),overlap_Ian_patients=len(ids&eids),confirmed_values=dict(collections.Counter(r.get('confirmed') for r in rr)),distinct_extracted_labels=len({r.get('extracted_syndrome') for r in rr}))
 d=json.loads((ROOT/'MAI_CHOP_testing_set.json').read_text());images=set();shapes=collections.Counter()
 for rec in d:
  im=rec.get('image');shapes[type(im).__name__]+=1
  if isinstance(im,str):images.add(im)
  elif isinstance(im,list):images.update(x for x in im if isinstance(x,str))
 linked=set();ambiguous=0
 for im in images:
  if len(imrn.get(im,set()))==1:linked.update(imrn[im])
  elif len(imrn.get(im,set()))>1:ambiguous+=1
 result['MAI_testing_json']={'conversation_records':len(d),'image_field_types':dict(shapes),'unique_image_strings':len(images),'exact_image_strings_matched_to_metadata':sum(bool(imrn.get(i)) for i in images),'ambiguous_image_strings':ambiguous,'unique_linked_patients':len(linked),'linked_Ian_patients':len(linked&eids),'linked_121225_patients':len(linked&allids.get('testing_data_for_MAI_121225_withConfirmation.csv',set())),'file_availability_sample_size':min(50,len(images)),'file_availability_sample_existing':sum(Path(i).is_file() for i in random.Random(20260928).sample(sorted(images),min(50,len(images)))),'note':'Repeated conversation records are not independent images. Exact path mapping only; no content-based duplicate checks or clinical text inspection.'}
 result['limitations']=['Crossfolder tiers are folder agreement, not independent clinical verification.','Ian review criteria and completeness await user/source confirmation. Diagnosis names are raw, not adjudicated MONDO.','Existing split membership does not prove actual training exposure; all-image SSL and prior MAI use need provenance checks.','PubMind and derivatives excluded; unknown MAI generation provenance quarantined.','Raw patient metadata and clinical notes are never emitted.']
 (HERE/'metadata_audit/validation_evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
