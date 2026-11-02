"""Expand expert classification eligibility; keep the prior exposure quarantine."""
import csv,gzip,json,os,re,collections,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;PRIOR=PROJECT/'chop_expert_comparison_20261027';QWEN=PROJECT/'cranioface_qwen_alignment_20261027'
def load(p):return json.loads(Path(p).read_text())
def save(p,d):p=Path(p);p.write_text(json.dumps(d,indent=2)+'\n');os.chmod(p,0o600)
def norm(s):
 s=re.sub(r'\.0$','',str(s or '').strip());return s.lstrip('0') or '0' if s.isdigit() else s
def rows(p):
 p=Path(p)
 with p.open('rb') as f:gz=f.read(2)==b'\x1f\x8b'
 with (gzip.open(p,'rt') if gz else p.open()) as f:
  first=f.readline();f.seek(0);yield from csv.DictReader(f,delimiter='\t' if first.count('\t')>first.count(',') else ',')
def main():
 os.umask(0o077);csv.field_size_limit(10**8)
 clean=load(PRIOR/'private/released.json')['rows'];eligible=load(PRIOR/'private/eligible.json')['rows'];groups=collections.defaultdict(list)
 for r in clean:groups[norm(r['patient'])].append(r)
 bridge=collections.defaultdict(set);reverse=collections.defaultdict(set);cf=Path('/controlled-data/craniofacial_data/arcus_export')
 for path in sorted((cf/'arcus_2026_06_29_full_ctakes').glob('patient-*.csv')):
  for r in rows(path):
   m=norm(r['pat_mrn_id']);pid=norm(r['pat_id'])
   if m in groups:bridge[pid].add(m);reverse[m].add(pid)
 good={p:next(iter(v)) for p,v in bridge.items() if len(v)==1 and len(reverse[next(iter(v))])==1}
 terms=load(QWEN/'private/ontology.json')['terms'] if (QWEN/'private/ontology.json').exists() else load(PROJECT/'transface_biomedbert_alignment_20261027/private/ontology.json')['terms']
 hp=collections.defaultdict(set);uncertain=collections.Counter()
 for path in sorted((cf/'arcus_2026_06_29').glob('clinician_hpo_element*')):
  for r in rows(path):
   m=good.get(norm(r['pat_id']))
   if m is None:continue
   ids=set(re.findall(r'HP:\d{7}',r.get('hpo_code','')))&terms.keys()
   try:d=json.loads(r.get('code_json') or '{}')
   except Exception:d={}
   note=d.get('Notes','') if isinstance(d,dict) else ''
   if not isinstance(note,str):note=''
   if re.search(r'\b(no|not|denies|absent|negative|possible|suspect|rule out|mother|father|family history)\b',note,re.I):uncertain[m]+=1;continue
   hp[m].update(ids)
 xrefs=collections.defaultdict(set)
 for block in (PROJECT/'ontology/mondo.obo').read_text().split('[Term]')[1:]:
  mid=re.search(r'^id: (MONDO:\d{7})$',block,re.M)
  if not mid or 'is_obsolete: true' in block:continue
  for line in block.splitlines():
   match=re.match(r'xref: (OMIM:\d+)',line)
   if match and 'MONDO:equivalentTo' in line:xrefs[match[1]].add(mid[1])
   match=re.match(r'property_value: skos:exactMatch (OMIM:\d+)',line)
   if match:xrefs[match[1]].add(mid[1])
 meta=Path('/controlled-data/GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata');mapping={}
 for r in rows(meta/'gmdb_syndromes_v1.1.0.tsv'):
  mids=xrefs.get('OMIM:'+r['OMIM'].removesuffix('.0'),set())
  if len(mids)==1:mapping[r['syndrome_id']]=next(iter(mids))
 vocab=load(QWEN/'private/sft_patients.json')['vocabulary'];covered={mapping[v['label']] for v in vocab if v['label'] in mapping}
 patients=[]
 for m,images in sorted(groups.items()):
  labels={r['label'] for r in images}
  if len(labels)!=1:raise RuntimeError('expert_label_conflict')
  h=sorted(hp[m]);text='Present findings: '+('; '.join(terms[x]['name'] for x in h) if h else 'none recorded')+'. Explicitly absent findings: none recorded. Unrecorded findings are unknown.'
  patients.append(dict(patient=m,label=next(iter(labels)),images=images,hpo=h,clinical_input=text,has_HPO=bool(h),covered=next(iter(labels)) in covered))
 save(ROOT/'private/patients.json',dict(patients=patients,vocabulary=vocab,class_to_mondo=mapping))
 save(ROOT/'audit.json',dict(state='prepared',primary_patients=len(patients),primary_images=len(clean),primary_diseases=len({p['label'] for p in patients}),
  HPO_paired_patients=sum(p['has_HPO'] for p in patients),classifier_covered_patients=sum(p['covered'] for p in patients),paired_and_covered=sum(p['covered'] and p['has_HPO'] for p in patients),
  expanded_from_previous_retrieval_queries=load(PRIOR/'release.json')['retrieval_evaluable_patients'],
  singleton_diseases_included_for_classification=True,quarantined_patients=len({r['patient'] for r in eligible})-len(patients),
  near_duplicate_quarantine_preserved=True,expert_used_for_training=False,HPO_timing_and_assertions_not_clinically_adjudicated=True,
  source_labels_pending_new_clinical_adjudication=True,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 print('Expanded expert cohort prepared locally; no records emitted.')
if __name__=='__main__':main()
