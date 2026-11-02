"""Candidate-wide terminology audit. Lexical mapping is never clinical confirmation."""
import collections,csv,hashlib,json,re
from pathlib import Path
from ontology import parse_obo,normstr
from normalize_mondo import exact_index,ALIASES,BLOCKED,SUTURE_LOOKUP,BROAD_ALIASES
from expert_reservation import filter_reserved
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'candidate_concept_audit'

def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(name,rr,fields):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rr)
def key(raw):
 # Remove only a dotted administrative prefix, never chromosome identifiers.
 return normstr(re.sub(r'^\s*\d+(?:\.\d+)+\s+(?=[A-Za-z])','',raw))

class Normalizer:
 def __init__(self):
  self.m=parse_obo(ROOT/'ontology/mondo.obo');self.h=parse_obo(ROOT/'ontology/hp.obo')
  self.mi=exact_index(self.m,'MONDO:');self.hi=exact_index(self.h,'HP:')
  # Do not inherit historical diagnosis groups or clinically lossy aliases.
  self.alias={normstr(a):mid for mid,aa in ALIASES.items() for a in aa if normstr(a) not in {'hfm','bifac microsomia','nf','cp','cl','dermoid'}}
  self.hp_alias={'pierre robin':'HP:0000201','prs':'HP:0000201','hemihyperplasia':'HP:0001528','mand hypoplasia':'HP:0000347'}
  self.hp_alias.update({'deviated septum':'HP:0004411','trigon':'HP:0000243',
   'fac asymmetry':'HP:0000324','facial asym':'HP:0000324','facial asymetry':'HP:0000324',
   'hypernasal':'HP:0001611','syndactaly':'HP:0001159','orbital hypertelorism':'HP:0000316',
   'dolicocephaly':'HP:0000268','scar':'HP:0100699',
   'acquired facial paralysis':'HP:0007209','aquired facial paralysis':'HP:0007209'})
  for v in self.hp_alias.values():assert v in self.h and not self.h[v].obsolete
  self.structural={'MONDO:0004747','MONDO:0016064','MONDO:0016044','MONDO:0015469','MONDO:0010920','MONDO:0015397'}
  self.morphology={'hemihypertrophy','hemihyperplasia','pierre robin','pierre robin sequence','prs','trigonocephaly','scaphocephaly','plagiocephaly','metopic ridge','nevus','port wine stain'}
  self.abbrev={'cp':['MONDO:0016064'],'cl':['MONDO:0004747'],'nf':['MONDO:0021061'],'hfm':['MONDO:0015397'],'22q 11 2':['MONDO:0018923','MONDO:0012020'],'22q11 2':['MONDO:0018923','MONDO:0012020']}
  self.proc=re.compile(r'\b(rhinoplasty|otoplasty|blepharoplasty|facelift|browlift|cranioplasty|genioplasty|reconstruction|recon|grafts?|grafting|flaps?|rhino|bleph|cpt|proc)\b|distraction osteogenesis|cranial vault reshaping|scar revision|ear molding|combo facial|free muscle transfer|breast reduction|\bcp team\b')
  self.workflow={'oj','not done','follow up','preop','postop','post op','pre op','consult','unknown','unclassified','undetermined'}
 def atomic(self,raw):
  k=key(raw);ms=set(self.mi.get(k,()));hs=set(self.hi.get(k,()))
  b=dict(component=raw,normalized_text=k,concept_type='unresolved',mondo_ids='',candidate_mondo_ids='',hpo_ids='',candidate_hpo_ids='',mapping_status='needs_review',mapping_relation='',evidence='',assertion='not_established_from_folder',notes='')
  if not k:return dict(b,concept_type='missing',mapping_status='missing')
  assertion=re.search(r'\b(suspected|suspect|possible|probable|rule out|family history|history of|no evidence of)\b|\?',raw,re.I)
  if assertion:return dict(b,concept_type='assertion_requires_review',notes='Do not turn suspicion, history or negation into a positive diagnosis.')
  if k in self.abbrev:return dict(b,concept_type='ambiguous_abbreviation',candidate_mondo_ids='|'.join(self.abbrev[k]),notes='Contextual interpretation required; no disease selected.')
  if k in self.hp_alias:
   hid=self.hp_alias[k]
   return dict(b,concept_type='phenotype',hpo_ids=hid,candidate_mondo_ids='|'.join(sorted(ms)),mapping_status='mapped_lexically',mapping_relation='phenotype_alias_or_broader',evidence='explicit_alias_pending_domain_review',notes='Original qualifier retained in raw text; phenotype does not establish etiologic disease. Short scar is not expanded to the SCAR drug-reaction acronym.')
  if k in self.morphology:
   return dict(b,concept_type='phenotype',candidate_mondo_ids='|'.join(sorted(ms)),hpo_ids='|'.join(sorted(hs)) if len(hs)==1 else '',candidate_hpo_ids='|'.join(sorted(hs)) if len(hs)>1 else '',mapping_status='mapped_lexically' if len(hs)==1 else 'needs_review',mapping_relation='exact_phenotype' if len(hs)==1 else '',evidence='local_HPO_or_explicit_alias',notes='No etiologic syndrome or suture fusion inferred from morphology; MONDO hits remain candidates.')
  if k in SUTURE_LOOKUP or k=='multiple suture synostosis':
   return dict(b,concept_type='structural_diagnosis',mondo_ids='MONDO:0015469',hpo_ids=SUTURE_LOOKUP.get(k,''),mapping_status='mapped_lexically',mapping_relation='broader_disease',evidence='explicit_synostosis_whole_label',notes='Suture subtype retained; does not establish a syndromic cause.')
  alias=self.alias.get(k)
  if alias:ms.add(alias)
  # Short ontology acronyms are not automatically equivalent to ordinary folder words.
  if not alias and len(k.replace(' ',''))<=6 and ms:
   acronym_only=all(normstr(self.m[d].name or '')!=k and not any(normstr(s)==k and not (s.isupper() and len(s)<=6) for s,scope in self.m[d].synonyms if scope=='EXACT') for d in ms)
   if acronym_only:return dict(b,concept_type='ambiguous_abbreviation',candidate_mondo_ids='|'.join(sorted(ms)),notes='Only short uppercase ontology synonyms match; contextual review required.')
  # Exact ontology concepts take precedence over procedural keyword heuristics.
  if len(ms)==1:
   mid=next(iter(ms))
   if mid in BLOCKED:return dict(b,concept_type='specificity_requires_review',candidate_mondo_ids=mid,hpo_ids='|'.join(hs) if len(hs)==1 else '',notes=BLOCKED[mid])
   typ='structural_diagnosis' if mid in self.structural else ('disease_phenotype_overlap' if hs else 'disease_or_disorder')
   if re.search(r'\b(trauma|injury|burns?|fractures?|lacerations?)\b',k):typ='injury_description'
   return dict(b,concept_type=typ,mondo_ids=mid,hpo_ids='|'.join(sorted(hs)) if len(hs)==1 else '',candidate_hpo_ids='|'.join(sorted(hs)) if len(hs)>1 else '',mapping_status='mapped_lexically',mapping_relation='broader_concept' if k in BROAD_ALIASES else ('alias_candidate' if alias else 'ontology_exact'),evidence='explicit_alias_pending_domain_review' if alias else 'local_MONDO_label_or_EXACT_synonym',notes='Ontology match is terminology evidence only. Disease/HPO overlap requires target-policy review.' if hs else 'Clinical assertion and source still pending.')
  if ms:return dict(b,concept_type='ambiguous_concept',candidate_mondo_ids='|'.join(sorted(ms)),candidate_hpo_ids='|'.join(sorted(hs)),notes='Multiple exact candidates; none selected.')
  if hs:return dict(b,concept_type='phenotype',hpo_ids='|'.join(sorted(hs)) if len(hs)==1 else '',candidate_hpo_ids='|'.join(sorted(hs)) if len(hs)>1 else '',mapping_status='mapped_lexically' if len(hs)==1 else 'needs_review',mapping_relation='ontology_exact',evidence='local_HPO_label_or_EXACT_synonym')
  if re.fullmatch(r'(?:fgfr[1234]|[A-Z0-9]{2,12})',raw.strip()) and raw.strip().upper() in {'FGFR1','FGFR2','FGFR3','FGFR4','PTPN11','NIPBL','ANKRD11','CHD7'}:
   return dict(b,concept_type='gene_only',notes='Gene name alone is not a disease diagnosis.')
  if k in self.workflow:return dict(b,concept_type='workflow',mapping_status='not_a_disease_label')
  if self.proc.search(k):return dict(b,concept_type='procedure_or_workflow',mapping_status='needs_review',notes='Procedure keyword; mixed diagnosis may be present. Do not infer disease from intervention.')
  if re.search(r'\b(fx|fractures?|trauma|injury|bite|burn|burns|laceration)\b',k):return dict(b,concept_type='injury_description',notes='Retain clinical condition, separate from syndrome targets; ontology mapping unresolved.')
  if re.search(r'\b(lesion|mass|cyst|tumor|scar|defect|defects|deformity|deformities|head shape)\b',k):return dict(b,concept_type='nonspecific_finding',notes='Unspecified finding; no subtype or etiology inferred.')
  return b
 def map(self,raw):
  whole=self.atomic(raw)
  # Keep exact ontology names intact (e.g. cleft lip/palate).
  if whole['mapping_status']=='mapped_lexically' or whole['concept_type']=='assertion_requires_review':return [whole],'single'
  if '/' in raw:
   parts=[x.strip() for x in raw.split('/') if x.strip()]
   if len(parts)>1:return [self.atomic(x) for x in parts],'slash_relation_unknown'
  if ';' in raw or '+' in raw:
   parts=[x.strip() for x in re.split(r'[;+]',raw) if x.strip()]
   if len(parts)>1:return [self.atomic(x) for x in parts],'multiple_relation_unknown'
  if key(raw) in {'aperts pfeiffer','apert pfeiffer','aperts phieffer','aperts pfeiffers'}:
   return [self.atomic('aperts'),self.atomic('pfeiffer')],'differential_or_combined_folder'
  if key(raw) in {'cp pr','cp prs'}:
   return [self.atomic('cp'),self.atomic('pierre robin sequence')],'abbreviation_relation_unknown'
  if key(raw)=='hfm microtia':
   return [self.atomic('hfm'),self.atomic('microtia')],'multiple_relation_unknown'
  return [whole],'single'

def main():
 OUT.mkdir(exist_ok=True);n=Normalizer()
 images=[r for r in read(ROOT/'data_release_audit/image_manifest.tsv') if r['source'].startswith('CHOP')]
 strict=read(ROOT/'data_release_audit/chop_patient_consistent_candidates.tsv');remaining,_=filter_reserved(strict)
 strictkeys={(r['mrn'],r['crop_sha256']) for r in strict};trainkeys={(r['mrn'],r['crop_sha256']) for r in remaining}
 # Retain all eligible raw-label provenance even when one duplicate crop is selected as representative.
 stats={};patientraw=collections.defaultdict(lambda:dict(hashes=set(),sources=set(),tiers=set(),rows=0,strict=False,remaining=False))
 for r in images:
  raw=r['raw_label'];s=stats.setdefault(raw,dict(rows=0,patients=set(),strict_hashes=set(),strict_patients=set(),remaining_hashes=set(),remaining_patients=set(),sources=set()))
  s['rows']+=1;s['patients'].add(r['mrn']) if r['mrn'] else None;s['sources'].add(r['source'])
  eligible=r['eligible_identity_candidate']=='True';pair=(r['mrn'],r['crop_sha256'])
  for tag,keys in [('strict',strictkeys),('remaining',trainkeys)]:
   if eligible and pair in keys:s[tag+'_hashes'].add(r['crop_sha256']);s[tag+'_patients'].add(r['mrn'])
  if eligible and pair in strictkeys:
   p=patientraw[(r['mrn'],raw)];p['rows']+=1;p['hashes'].add(r['crop_sha256']);p['sources'].add(r['source']);p['tiers'].add(r['crossfolder_tier']);p['strict']=True;p['remaining']=pair in trainkeys
 rawrows=[];components=[];maps={}
 old={r['raw_label']:r for r in read(ROOT/'mondo_normalization/raw_label_mondo_map.tsv')}
 for raw,s in sorted(stats.items(),key=lambda x:(-len(x[1]['remaining_patients']),-x[1]['rows'],x[0])):
  rr,relation=n.map(raw);rid=hashlib.sha256(raw.encode()).hexdigest()[:16]
  for i,r in enumerate(rr):components.append(dict(raw_label_id=rid,component_index=i+1,**r))
  ms={v for r in rr for v in r['mondo_ids'].split('|') if v};hs={v for r in rr for v in r['hpo_ids'].split('|') if v}
  cats={r['concept_type'] for r in rr};category=next(iter(cats)) if relation=='single' else 'mixed_or_alternative'
  status=rr[0]['mapping_status'] if relation=='single' else 'needs_relation_review'
  legacy=old.get(raw,{})
  rec=dict(raw_label_id=rid,raw_label=raw,normalized_text=key(raw),concept_type=category,mapping_status=status,component_relation=relation,mondo_ids='|'.join(sorted(ms)) if relation=='single' else '',component_mondo_candidates='|'.join(sorted(ms)) if relation!='single' else '',candidate_mondo_ids='|'.join(sorted({v for r in rr for v in r['candidate_mondo_ids'].split('|') if v})),mondo_names='|'.join(n.m[v].name for v in sorted(ms)),hpo_ids='|'.join(sorted(hs)),hpo_names='|'.join(n.h[v].name for v in sorted(hs)),source_image_rows=s['rows'],source_patient_ids=len(s['patients']),strict_unique_crops=len(s['strict_hashes']),strict_patients=len(s['strict_patients']),remaining_unique_crops=len(s['remaining_hashes']),remaining_patients=len(s['remaining_patients']),sources='|'.join(sorted(s['sources'])),legacy_mondo=legacy.get('mondo_id',''),legacy_status=legacy.get('status',''),clinical_assertion='unadjudicated',weak_supervision_admitted=False,notes=' | '.join(r['notes'] for r in rr))
  rec['legacy_mapping_changed']=bool(rec['legacy_mondo'] and set(rec['legacy_mondo'].split(';'))!=set(rec['mondo_ids'].split('|')))
  rec['mapping_relations']='|'.join(sorted({r['mapping_relation'] for r in rr if r['mapping_relation']}))
  rec['mapping_evidence']='|'.join(sorted({r['evidence'] for r in rr if r['evidence']}))
  rec['candidate_hpo_ids']='|'.join(sorted({v for r in rr for v in r['candidate_hpo_ids'].split('|') if v}))
  maps[raw]=rec;rawrows.append(rec)
 write('raw_label_concepts.tsv',rawrows,list(rawrows[0]));write('concept_components.tsv',components,list(components[0]))
 review=[r for r in rawrows if r['mapping_status'] not in {'mapped_lexically','missing','not_a_disease_label'} or r['legacy_mapping_changed']]
 write('terminology_review_queue.tsv',review,list(rawrows[0]))
 pr=[];concepts=collections.defaultdict(lambda:dict(patients=set(),hashes=set(),labels=set()))
 for (m,raw),x in sorted(patientraw.items()):
  rec=maps[raw];ms=rec['mondo_ids'];hs=rec['hpo_ids']
  pr.append(dict(mrn=m,raw_label_id=rec['raw_label_id'],raw_label=raw,concept_type=rec['concept_type'],mondo_ids=ms,hpo_ids=hs,component_relation=rec['component_relation'],candidate_crop_count=len(x['hashes']),sources='|'.join(sorted(x['sources'])),historical_crossfolder_tiers='|'.join(sorted(x['tiers'])),remaining_after_reservation=x['remaining'],clinical_assertion='unadjudicated',weak_supervision_admitted=False))
  if x['remaining']:
   for mid in filter(None,ms.split('|')):
    z=concepts[(mid,rec['concept_type'])];z['patients'].add(m);z['hashes'].update(x['hashes']);z['labels'].add(raw)
 write('patient_label_assertions.tsv',pr,list(pr[0]))
 cc=[dict(mondo_id=mid,mondo_name=n.m[mid].name,target_category=typ,remaining_patients=len(x['patients']),remaining_unique_crops=len(x['hashes']),raw_labels='|'.join(sorted(x['labels'])),clinical_confirmation='pending') for (mid,typ),x in sorted(concepts.items(),key=lambda x:-len(x[1]['patients']))]
 write('remaining_mondo_concept_counts.tsv',cc,['mondo_id','mondo_name','target_category','remaining_patients','remaining_unique_crops','raw_labels','clinical_confirmation'])
 categories=[]
 for typ in sorted({r['concept_type'] for r in rawrows}):
  raws=[r['raw_label'] for r in rawrows if r['concept_type']==typ]
  categories.append(dict(category=typ,raw_labels=len(raws),source_rows=sum(stats[r]['rows'] for r in raws),remaining_patients=len(set().union(*(stats[r]['remaining_patients'] for r in raws))),remaining_unique_crops=len(set().union(*(stats[r]['remaining_hashes'] for r in raws)))))
 write('category_counts.tsv',categories,list(categories[0]))
 checks={'all_raw_labels_covered':len(rawrows)==len(stats),'all_remaining_crops_have_label_provenance':set().union(*(s['remaining_hashes'] for s in stats.values()))=={r['crop_sha256'] for r in remaining},'no_automatic_supervision':all(r['weak_supervision_admitted'] is False for r in rawrows),'remaining_patient_assertions_match_candidate_pool':{r['mrn'] for r in pr if r['remaining_after_reservation']}=={r['mrn'] for r in remaining},'all_mapped_mondo_valid':all(mid in n.m and not n.m[mid].obsolete for r in rawrows for mid in r['mondo_ids'].split('|') if mid),'all_hpo_valid':all(hid in n.h and not n.h[hid].obsolete for r in rawrows for hid in r['hpo_ids'].split('|') if hid)}
 assert all(checks.values())
 summary=dict(scope='All CHOP image candidates in audited image manifest, not every patient diagnosis in MedicalDBs',image_rows=len(images),distinct_raw_strings=len(rawrows),nonempty_raw_strings=sum(bool(r['raw_label']) for r in rawrows),remaining_images=len(remaining),remaining_patients=len({r['mrn'] for r in remaining}),categories=categories,lexically_mapped_raw_labels=sum(r['mapping_status']=='mapped_lexically' for r in rawrows),review_labels=len(review),remaining_mondo_ids=len({r['mondo_id'] for r in cc}),remaining_disease_or_disorder_ids=len({r['mondo_id'] for r in cc if r['target_category']=='disease_or_disorder'}),legacy_mapped_labels_changed=sum(r['legacy_mapping_changed'] for r in rawrows),checks=checks,confirmed_disease_count=None,approved_weak_supervision_patients=0)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 inputs=[ROOT/'data_release_audit/image_manifest.tsv',ROOT/'data_release_audit/chop_patient_consistent_candidates.tsv',ROOT/'mondo_normalization/raw_label_mondo_map.tsv',ROOT/'ontology/mondo.obo',ROOT/'ontology/hp.obo',Path(__file__),ROOT/'normalize_mondo.py',ROOT/'expert_reservation.py']+list((ROOT/'expert_holdout_audit').glob('protected_mrns.tsv'))+list((ROOT/'expert_holdout_audit').glob('blocked*.tsv'))+list((ROOT/'split_plan_v0').glob('ian_*.tsv'))
 (OUT/'input_versions.json').write_text(json.dumps({str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size) for p in inputs},indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
