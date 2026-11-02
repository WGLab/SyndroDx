"""Deterministic patient-level proposal, never a training admission manifest."""
import collections,csv,hashlib,json
from pathlib import Path
from audit_metadata_sources import mrnkey
from audit_validation_evidence import xlsx
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'split_plan_v0'
SEED='20260928-v0'
IAN=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs/CHOP_Images/examples/Ian_verified.xlsx')

def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(name,rows,fields):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
def rank(x):return hashlib.sha256((SEED+'|'+x).encode()).hexdigest()
class Groups:
 def __init__(self):self.parent={}
 def find(self,x):
  self.parent.setdefault(x,x)
  if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
  return self.parent[x]
 def join(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.parent[max(a,b)]=min(a,b)

def source_order(rows):
 # Interleave source strata proportionally; hash ordering within each stratum.
 buckets=collections.defaultdict(list)
 for r in rows:buckets[r['source_tags']].append(r)
 scheduled=[]
 for key,rr in buckets.items():
  rr.sort(key=lambda r:rank(r['mrn']))
  for i,r in enumerate(rr):scheduled.append(((i+.5)/len(rr),rank(key),r))
 return [r for _,__,r in sorted(scheduled,key=lambda x:x[:2])]

def allocate(rows):
 n=len(rows);ordered=source_order(rows)
 if n>=20:
  dev=max(2,round(n*.15));test=max(2,round(n*.15))
 elif n>=5:dev=max(1,round(n*.2));test=0
 else:dev=test=0
 # Spread evaluation slots over the source-interleaved list, then alternate roles.
 neval=dev+test
 slots={int((i+.5)*n/neval) for i in range(neval)} if neval else set()
 roles=[];dc=tc=0
 for i,r in enumerate(ordered):
  role='train_candidate'
  if i in slots:
   if dc<dev and (tc>=test or dc/max(dev,1)<=tc/max(test,1)):role='dev_candidate';dc+=1
   else:role='test_candidate';tc+=1
  r['proposed_role']=role
  r['allocation_reason']='common_70_15_15' if n>=20 else ('small_train_dev_only' if n>=5 else 'rare_train_only')
 assert dc==dev and tc==test

def main():
 OUT.mkdir(exist_ok=True)
 labels=read(ROOT/'expert_holdout_audit/label_audit.tsv');ld={r['mrn']:r for r in labels}
 ir=xlsx(IAN);ian={mrnkey(r['pat_mrn_id']) for r in ir if r.get('pat_mrn_id')}
 ian_diagnoses=collections.defaultdict(set)
 for r in ir:
  if r.get('pat_mrn_id') and r.get('Diagnosis'):ian_diagnoses[mrnkey(r['pat_mrn_id'])].add(r['Diagnosis'])
 images=read(ROOT/'data_release_audit/image_manifest.tsv')
 strict=read(ROOT/'data_release_audit/chop_patient_consistent_candidates.tsv')
 pats={r['mrn']:r for r in read(ROOT/'data_release_audit/patient_manifest.tsv')}
 g=Groups();nodes=[]
 for r in images:
  ns=[]
  if r['source'].startswith('CHOP'):
   ns += ['mrn:'+mrnkey(x) for k in ('mrn','original_mrn','latest_norm_mrn','identity_candidates') for x in (r.get(k) or '').split('|') if x.strip()]
  ns += ['path:'+str(Path(r[k])) for k in ('image_path','cropping_path') if r.get(k)]
  if r.get('crop_sha256'):ns.append('hash:'+r['crop_sha256'])
  for n in ns:g.join(ns[0],n)
  nodes.append(ns)
 for m in set(ld)|ian:g.find('mrn:'+m)
 component_ids=collections.defaultdict(set)
 for node in list(g.parent):
  if node.startswith('mrn:'):component_ids[g.find(node)].add(node[4:])
 ian_roots={g.find('mrn:'+m) for m in ian}
 ian_ids=set().union(*(component_ids[k] for k in ian_roots))
 ian_paths={n[5:] for n in g.parent if n.startswith('path:') and g.find(n) in ian_roots}
 ian_hashes={n[5:] for n in g.parent if n.startswith('hash:') and g.find(n) in ian_roots}
 # Additional reservation protects Ian patients outside the expert-label file too.
 write('ian_protected_mrns.tsv',[{'mrn':m} for m in sorted(ian_ids)],['mrn'])
 write('ian_blocked_paths.tsv',[{'path':p} for p in sorted(ian_paths)],['path'])
 write('ian_blocked_crop_hashes.tsv',[{'sha256':h} for h in sorted(ian_hashes)],['sha256'])
 bypatient=collections.defaultdict(list)
 for r in strict:bypatient[r['mrn']].append(r)
 hard={'missing_mondo','invalid_mondo_cell','ragged_source_row','missing_diagnosis_text','multiple_mondo_requires_adjudication','manual_fix_mondo_disagreement','obsolete_mondo','id_missing_from_local_ontology','diagnosis_assertion_review'}
 population=set(ld)|ian|ian_ids
 plan=[];buckets=collections.defaultdict(list)
 for m in sorted(population):
  label=ld.get(m,{});pr=pats.get(m,{});rr=bypatient[m];issues=set(filter(None,label.get('issues','').split('|')))
  comp=g.find('mrn:'+m);known_ids=component_ids[comp]
  rec=dict(mrn=m,group_id=hashlib.sha256(comp.encode()).hexdigest()[:16],source_label_member=m in ld,ian_record=m in ian,ian_linked=comp in ian_roots,source_tags=label.get('source_tags',''),mondo_ids=label.get('mondo_ids',''),mondo_names=label.get('mondo_names',''),candidate_crops=len(rr),image_sources='|'.join(sorted({r['source'] for r in rr})),historical_splits=pr.get('historical_splits',''),known_linked_mrns=len(known_ids),family_status='not_verified',label_issues=label.get('issues',''),clinical_adjudication='pending',frontal_qc='pending',training_admitted=False,proposed_role='',allocation_reason='')
  rec['ian_diagnosis_raw']='|'.join(sorted(ian_diagnoses.get(m,set())))
  rec['family_check']='pending'
  rec['release_requirements']='identity_and_family_review|frontal_qc|clinical_label_and_source_review|split_lock'
  if comp in ian_roots:rec.update(proposed_role='expert_test_reserve',allocation_reason='ian_record_or_identity_copy_link_not_yet_approved_test')
  elif not rr:rec.update(proposed_role='no_eligible_image',allocation_reason='no_current_conservative_crop')
  elif len(known_ids)>1:rec.update(proposed_role='label_identity_review',allocation_reason='multiple_identity_aliases_require_review')
  elif issues & hard:rec.update(proposed_role='label_identity_review',allocation_reason='|'.join(sorted(issues & hard)))
  else:buckets[rec['mondo_ids']].append(rec)
  plan.append(rec)
 for rr in buckets.values():allocate(rr)
 pd={r['mrn']:r for r in plan}
 trainids={r['mrn'] for r in plan if r['proposed_role']=='train_candidate'}
 traindiseases={d for r in plan if r['mrn'] in trainids for d in r['mondo_ids'].split('|') if d}
 for r in plan:
  ds=set(filter(None,r['mondo_ids'].split('|')))
  r['target_coverage_in_proposed_training']='all' if ds and ds<=traindiseases else ('partial' if ds & traindiseases else ('none' if ds else 'unknown'))
 write('patient_split_proposal.tsv',plan,list(plan[0]))
 # All inventoried photos of a patient/group inherit that role, including non-admitted images.
 rootroles=collections.defaultdict(set)
 for r in plan:rootroles[g.find('mrn:'+r['mrn'])].add(r['proposed_role'])
 photo=[]
 for r,ns in zip(images,nodes):
  if not ns or g.find(ns[0]) not in rootroles:continue
  roles=rootroles[g.find(ns[0])]
  if len(roles)>1:raise AssertionError('Conflicting roles within identity/copy group')
  photo.append({**r,'proposed_role':next(iter(roles)),'training_admitted':False})
 write('image_split_proposal.tsv',photo,list(photo[0]))
 drows=[]
 diseases=sorted({d for r in plan for d in r['mondo_ids'].split('|') if d})
 for d in diseases:
  members=[r for r in plan if d in r['mondo_ids'].split('|') and int(r['candidate_crops'])>0]
  if not members:continue
  cnt=collections.Counter(r['proposed_role'] for r in members)
  drows.append(dict(mondo_id=d,patient_count=len(members),candidate_crops=sum(r['candidate_crops'] for r in members),**{role:cnt[role] for role in ['train_candidate','dev_candidate','test_candidate','expert_test_reserve','label_identity_review']},closed_set_test_supported=bool(cnt['train_candidate'] and cnt['test_candidate'])))
 write('disease_allocation.tsv',drows,list(drows[0]))
 strata=[]
 for r in plan:
  if not r['candidate_crops']:continue
  mids=set(r['mondo_ids'].split('|'));stratum='22q' if mids & {'MONDO:0018923','MONDO:0012020'} else ('other_mondo' if r['mondo_ids'] else 'unresolved')
  strata.append((stratum,r['proposed_role'],r['candidate_crops']))
 strat=[]
 for a,b in sorted({(a,b) for a,b,_ in strata}):
  vals=[n for x,y,n in strata if (x,y)==(a,b)]
  strat.append(dict(stratum=a,role=b,patients=len(vals),candidate_crops=sum(vals)))
 write('stratum_allocation.tsv',strat,['stratum','role','patients','candidate_crops'])
 sc=collections.Counter((r['source_tags'],r['proposed_role']) for r in plan if r['candidate_crops'])
 write('source_allocation.tsv',[dict(source_tags=a,role=b,patients=n) for (a,b),n in sorted(sc.items())],['source_tags','role','patients'])
 review=[r for r in plan if r['candidate_crops'] and (r['label_issues'] or r['ian_linked'])]
 review.sort(key=lambda r:('MONDO:0018923' in r['mondo_ids'] or 'MONDO:0012020' in r['mondo_ids'],r['proposed_role']!='label_identity_review',rank(r['mrn'])))
 write('review_priorities.tsv',review,list(plan[0]))
 checks={'groups_not_split':all(len(x)==1 for x in rootroles.values()),'ian_excluded_from_training':not bool(trainids&ian_ids),'patient_rows_unique':len(pd)==len(plan),'all_roles_assigned':all(r['proposed_role'] for r in plan),'none_admitted_to_training':all(r['training_admitted'] is False for r in plan),'ordinary_test_has_training_targets':all(r['target_coverage_in_proposed_training']=='all' for r in plan if r['proposed_role']=='test_candidate')}
 assert all(checks.values())
 summary={'status':'proposal_only_no_patients_released','seed':SEED,'rules':{'common_min_patients':20,'common_fractions':[.70,.15,.15],'small_min_patients':5,'small_dev_fraction':.2,'rare':'train_only','ian':'reserve_all; unknown labels are not invented'},'patients_with_conservative_images':sum(r['candidate_crops']>0 for r in plan),'expert_source_patients_with_conservative_images':sum(r['source_label_member'] and r['candidate_crops']>0 for r in plan),'all_patient_roles':dict(collections.Counter(r['proposed_role'] for r in plan)),'with_image_roles':dict(collections.Counter(r['proposed_role'] for r in plan if r['candidate_crops'])),'image_counts_by_role':{role:sum(r['candidate_crops'] for r in plan if r['proposed_role']==role) for role in sorted({r['proposed_role'] for r in plan})},'ian_source_patients':len(ian),'ian_protected_ids':len(ian_ids),'ian_with_images':sum(r['mrn'] in ian and r['candidate_crops']>0 for r in plan),'expert_source_ian_overlap_with_images':sum(r['mrn'] in ian and r['source_label_member'] and r['candidate_crops']>0 for r in plan),'proposed_train_disease_ids':len(traindiseases),'ordinary_test_disease_ids':len({r['mondo_ids'] for r in plan if r['proposed_role']=='test_candidate'}),'strata':strat,'checks':checks,'family_links':'not_available_not_verified','actual_training_release':0}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 inputs=[IAN,ROOT/'expert_holdout_audit/label_audit.tsv',ROOT/'data_release_audit/image_manifest.tsv',ROOT/'data_release_audit/chop_patient_consistent_candidates.tsv',ROOT/'data_release_audit/patient_manifest.tsv',Path(__file__)]
 (OUT/'input_versions.json').write_text(json.dumps({str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size} for p in inputs},indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
