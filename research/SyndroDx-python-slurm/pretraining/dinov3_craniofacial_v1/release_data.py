"""Patient-isolated SSL release; ignore all clinical/disease/HPO label fields.

Private identity columns are joined locally to carry forward prior exclusions.
Only pseudonymous groups and image provenance are written to the release files.
"""
import collections,csv,hashlib,hmac,json,re,sqlite3,sys,time
from pathlib import Path
from runtime import ROOT,RUN,digest,load,save,now
OLD=Path('/opt/craniofacial/disease-diagnosis-model')
METADATA=Path('/controlled-data/chop_data_metadata/metadata_internal.tsv')
LEGACY=Path('/controlled-data/craniofacial_data/complete_full_craniofacial_metadata_updated_03152026.tsv')
OUT=ROOT/'dataset_release'

class Union:
 def __init__(self):self.parent={}
 def find(self,x):
  p=self.parent.setdefault(x,x)
  if p!=x:self.parent[x]=self.find(p)
  return self.parent[x]
 def join(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.parent[max(a,b)]=min(a,b)

def rows(path):
 with Path(path).open(newline='') as f:yield from csv.DictReader(f,delimiter='\t')
def normalize(x):
 x=(x or '').strip()
 if x.lower() in {'','nan','none','null','<na>'}:return ''
 if re.fullmatch(r'\d+\.0',x):x=x[:-2]
 return (x.lstrip('0') or '0') if x.isdigit() else x

def identity_plan():
 OUT.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(RUN))
 from annotation_pipeline import references,canonical_path
 settings=load(RUN/'config.json');salt=(RUN/'direct_identity_salt.bin').read_bytes()
 def token(s):return hmac.new(salt,s.encode(),hashlib.sha256).hexdigest()
 def paths(value):return [str(canonical_path(v,settings)) for v in references(value)]
 start=METADATA.stat();g=Union();by_path=collections.defaultdict(set);patients=set()
 for row in rows(METADATA):
  pseudo=(row.get('pseudo_mrn') or '').strip()
  if not normalize(pseudo):continue
  patient=token('patient:'+pseudo);patients.add(patient);g.find('p:'+patient)
  for path in paths(row.get('image_path')):by_path[path].add(patient)
 ambiguous={p for path,ps in by_path.items() if len(ps)>1 for p in ps}
 linked=set();protected_ids=set();blocked_paths=set();blocked_hashes=set();held=set(ambiguous);external=set();historic_paths=set()
 controls=[]
 for directory,prefix in [(OLD/'expert_holdout_audit',''),(OLD/'split_plan_v0','ian_')]:
  for filename,column,dest in [(prefix+'protected_mrns.tsv','mrn',protected_ids),(prefix+'blocked_paths.tsv','path',blocked_paths),(prefix+'blocked_crop_hashes.tsv','sha256',blocked_hashes)]:
   p=directory/filename
   if not p.is_file():raise RuntimeError('protected_patient_controls_missing')
   controls.append(p)
   for row in rows(p):
    value=row.get(column,'')
    if value:dest.add(normalize(value) if column=='mrn' else str(Path(value)) if column=='path' else value)
 if not protected_ids:raise RuntimeError('empty_protected_patient_controls')
 # All aliases recorded by the legacy metadata remain in the same component.
 for row in rows(LEGACY):
  ids={normalize(row.get(k)) for k in ['norm_mrn','MRN','chart_mrn']} - {''}
  for path in paths(row.get('image_path')):
   ps=by_path.get(path,set())
   if ps and ids:
    nodes=['p:'+p for p in ps]+['i:'+v for v in ids]
    for n in nodes[1:]:g.join(nodes[0],n)
    linked.update(ps)
 # Audited image provenance carries historical split and external-source protection.
 audit=OLD/'data_release_audit/image_manifest.tsv';controls.append(audit)
 for row in rows(audit):
  oldpaths=[str(Path(row[k])) for k in ['image_path','cropping_path'] if row.get(k)]
  ps=set().union(*(by_path.get(p,set()) for p in oldpaths))
  if not ps:continue
  ids={normalize(v) for k in ['mrn','original_mrn','latest_norm_mrn','identity_candidates'] for v in (row.get(k) or '').split('|')} - {''}
  nodes=['p:'+p for p in ps]+['i:'+v for v in ids]
  for n in nodes[1:]:g.join(nodes[0],n)
  if ids:linked.update(ps)
  source=(row.get('source') or '').lower()
  if source and not source.startswith('chop'):external.update(ps)
  if any(v in (row.get('original_split') or '').lower() for v in ['test','val','dev']):historic_paths.update(ps)
  if str(row.get('duplicate_identity_conflict','')).lower()=='true':held.update(ps)
 proposal=OLD/'split_plan_v0/patient_split_proposal.tsv';controls.append(proposal)
 oldgroups={};old_hold_ids=set()
 for row in rows(proposal):
  mrn=normalize(row.get('mrn'));group=row.get('group_id')
  if mrn and group:
   if group in oldgroups:g.join('i:'+mrn,'i:'+oldgroups[group])
   else:oldgroups[group]=mrn
  if mrn and any(v in (row.get('proposed_role') or '').lower() for v in ['test','dev','hold','review','expert']):old_hold_ids.add(mrn)
  if mrn and any(v in (row.get('historical_splits') or '').lower() for v in ['test','val','dev']):old_hold_ids.add(mrn)
 protected_roots={g.find('i:'+i) for i in protected_ids|old_hold_ids}
 for path in blocked_paths:
  held.update(by_path.get(path,set()))
 held_roots={g.find('p:'+p) for p in held|external|historic_paths}
 components=collections.defaultdict(list)
 for p in patients:components[g.find('p:'+p)].append(p)
 plan={}
 for root,members in components.items():
  group=token('ssl-group:'+','.join(sorted(members)))
  reason=''
  if root in protected_roots:reason='expert_ian_or_historical_reserved'
  elif root in held_roots:reason='identity_source_or_historical_review'
  elif any(p not in linked for p in members):reason='legacy_identity_bridge_unresolved'
  fraction=int(group[:16],16)/2**64
  split='hold' if reason else 'train' if fraction<.9 else 'validation' if fraction<.95 else 'test'
  for p in members:plan[p]={'group_id':group,'split':split,'reason':reason,'source':'approved_internal','identity_bridge_verified':not bool(reason)}
 if (METADATA.stat().st_size,METADATA.stat().st_mtime_ns)!=(start.st_size,start.st_mtime_ns):raise RuntimeError('approved_metadata_changed_during_split')
 payload={'state':'patient_identity_plan_ready','metadata_sha256':digest(METADATA),'legacy_metadata_sha256':digest(LEGACY),'control_sha256':{str(p):digest(p) for p in controls},'patients':plan,'blocked_hashes':sorted(blocked_hashes),'known_aliases_grouped':True,'unknown_family_relationships':'not_available_not_claimed_verified','clinical_labels_used':False,'pubmind_used':False,'created_at_utc':now()}
 save(OUT/'patient_identity_plan.private.json',payload)
 save(OUT/'identity_plan_status.json',{'state':'patient_identity_plan_ready','created_at_utc':now(),'individual_records_remain_local':True})
 return payload

def final_release():
 if load(RUN/'crop_completion.json').get('state')!='crop_complete':return 'waiting_for_crop_completion'
 from independent_qc import COUNT
 if any(load(ROOT/'independent_qc'/f'slot_{i:02d}.json').get('state')!='finished' for i in range(COUNT)):return 'waiting_for_independent_automated_review'
 reference=load(ROOT/'assets/fairface/status.json')
 if reference.get('state')!='public_reference_ready':return 'waiting_for_reference_faces'
 identity=load(OUT/'patient_identity_plan.private.json')
 if identity.get('state')!='patient_identity_plan_ready':identity=identity_plan()
 if digest(METADATA)!=identity['metadata_sha256']:raise RuntimeError('approved_metadata_changed_since_identity_plan')
 for name,sha in identity['control_sha256'].items():
  if digest(name)!=sha:raise RuntimeError('protected_patient_controls_changed')
 candidates=list(rows(RUN/'cropped_candidates.tsv'));reviews={}
 for i in range(COUNT):
  db=sqlite3.connect((ROOT/'independent_qc'/f'slot_{i:02d}.sqlite').resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
  try:
   if db.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise RuntimeError('qc_database_integrity_failed')
   for encoded, in db.execute('SELECT record FROM reviews'):
    v=json.loads(encoded)
    if v['photo_id'] in reviews:raise RuntimeError('duplicate_independent_review_ownership')
    reviews[v['photo_id']]=v
  finally:db.close()
 # Exact source/crop copies with conflicting pseudopatient owners remain excluded.
 owners=collections.defaultdict(set)
 for r in reviews.values():
  if r['state']=='pass':
   for k in ['source_sha256','crop_sha256','pixel_sha256']:
    if r.get(k):owners[(k,r[k])].add(identity['patients'].get(r['patient_key'],{}).get('group_id',r['patient_key']))
 conflict_groups=set().union(*(v for v in owners.values() if len(v)>1)) if owners else set()
 blocked=set(identity['blocked_hashes']);kept=[];holds=[];seen=set();pixels=set()
 for row in candidates:
  photo=row['photo_id'];qc=reviews.get(photo);patient=identity['patients'].get(row['patient_key']);reason=''
  if qc is None:raise RuntimeError('final_crop_missing_independent_review')
  if qc.get('state')!='pass':reason='independent_qc_hold'
  elif not patient or patient['split']=='hold':reason='protected_or_unresolved_patient'
  elif patient['group_id'] in conflict_groups:reason='cross_patient_copy_conflict'
  elif qc.get('crop_sha256') in blocked or qc.get('source_sha256') in blocked:reason='protected_copy_hash'
  elif qc['crop_sha256']!=row['crop_sha256'] or qc['patient_key']!=row['patient_key'] or qc['crop_path']!=row['crop_path'] or qc['view']!=row['view']:raise RuntimeError('qc_and_final_crop_manifest_conflict')
  elif digest(row['crop_path'])!=qc['crop_sha256']:reason='crop_changed_after_independent_qc'
  elif qc['pixel_sha256'] in seen:reason='redundant_identical_crop'
  if reason:holds.append({'photo_id':photo,'patient_key':row['patient_key'],'reason':reason});continue
  seen.add(qc['pixel_sha256']);pixels.add(qc['pixel_sha256'])
  kept.append(dict(row,group_id=patient['group_id'],source=patient['source'],split=patient['split'],training_admitted=str(patient['split']=='train').lower(),pixel_sha256=qc['pixel_sha256']))
 if not kept or not any(v['split']=='train' for v in kept):raise RuntimeError('no_admissible_training_faces')
 fields=['photo_id','patient_key','group_id','crop_path','source','view','crop_sha256','pixel_sha256','training_admitted','split']
 manifests={}
 for split in ['train','validation','test']:
  p=OUT/(split+'.tsv');tmp=p.with_suffix('.tmp')
  with tmp.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(v for v in kept if v['split']==split)
  tmp.replace(p);manifests[split]={'path':str(p),'sha256':digest(p)}
 with (OUT/'held_records.private.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['photo_id','patient_key','reason'],delimiter='\t');w.writeheader();w.writerows(holds)
 groups={s:{r['group_id'] for r in kept if r['split']==s} for s in manifests}
 if groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test']:raise RuntimeError('patient_split_overlap')
 # Reference images are public, unlabelled for clinical health. Cross-source exact pixel copies are removed.
 from PIL import Image
 source=Path(reference['manifest'])
 if digest(source)!=reference['manifest_sha256']:raise RuntimeError('reference_manifest_changed')
 reference_out=OUT/'reference_train.tsv'
 with reference_out.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore');writer.writeheader()
  for row in rows(source):
   with Image.open(row['crop_path']) as im:
    im=im.convert('RGB');pixel=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
   if pixel in pixels:continue
   row.update(group_id=row['patient_key'],pixel_sha256=pixel);writer.writerow(row)
 manifests['reference_train']={'path':str(reference_out),'sha256':digest(reference_out)}
 validation={'state':'crop_validation_complete','method':'separate_offline_crop_only_Qwen_pass_and_exact_native_pixel_integrity','human_review_complete':False,'independent_model':False,'created_at_utc':now()}
 save(RUN/'crop_validation_completion.json',validation)
 save(RUN/'dataset_split_completion.json',{'state':'patient_isolated_split_complete','manifest_sha256':{k:v['sha256'] for k,v in manifests.items()},'known_aliases_and_protected_pools_excluded':True,'unknown_family_relationships':'not_available_not_claimed_verified','created_at_utc':now()})
 release={'state':'ssl_data_ready','manifests':manifests,'clinical_train_images':sum(v['split']=='train' for v in kept),'sampling_weights':{'approved_internal':.8,'fairface_reference':.2},'clinically_verified_healthy_controls':False,'patient_split_disjoint':True,'protected_pools_excluded':True,'pubmind_used':False,'labels_used':False,'automated_qc_only':True,'created_at_utc':now()}
 save(OUT/'release.json',release);save(RUN/'ssl_data_ready.json',release)
 return 'ssl_data_ready'

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--identity-only',action='store_true');a=p.parse_args()
 if a.identity_only:identity_plan()
 else:final_release()
