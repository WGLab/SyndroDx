"""Full original-image audit. Resumable, non-destructive, never training admission.

Every indexed source path receives a result. Profile/no-landmark cases remain
review candidates. No automatic orientation search or background removal.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,csv,json,hashlib,io,math,argparse,time,warnings
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_metadata_sources import mrnkey
OUT=ROOT/'face_qc/full_original_audit_v1'
CF=Path('/controlled-data/craniofacial_data')
INPUTS=[(CF/'complete_full_craniofacial_metadata_updated_03152026.tsv','craniofacial_full','image_path',['norm_mrn','MRN','chart_mrn']),
 (CF/'craniofacial_metadata_frontal_updated_123025_deduplicated.csv','craniofacial_frontal_supplement','image_path',['norm_mrn','MRN','chart_mrn']),
 (CF/'genetics_photos_metdata_top10dx.csv','genetics_copied_all_views','image_path',['pat_mrn_id']),
 (Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs/CHOP_Images/final_results/arcus_genetics_photos_faith.csv'),'genetics_full_original_index','file_path',['MRN'])]
GUARDS=[ROOT/'expert_holdout_audit/protected_mrns.tsv',ROOT/'split_plan_v0/ian_protected_mrns.tsv']
PATH_GUARDS=[ROOT/'expert_holdout_audit/blocked_paths.tsv',ROOT/'split_plan_v0/ian_blocked_paths.tsv']

def read(p):
 with Path(p).open() as f:yield from csv.DictReader(f,delimiter='\t')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,rr,fields=None):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rr[0]),delimiter='\t');w.writeheader();w.writerows(rr)
def prepare(out,shards):
 if (out/'manifest.json').exists():raise SystemExit('Existing run preserved')
 out.mkdir(parents=True,exist_ok=True)
 records={};source_rows=Counter();modified_index_count=0
 for p,source,key,idkeys in INPUTS:
  for r in read(p):
   source_rows[source]+=1;path=r.get(key,'').strip()
   if not path:continue
   path=str(Path(path) if Path(path).is_absolute() else CF/path)
   v=records.setdefault(path,dict(ids=set(),sources=set(),views=set(),lineage=set()))
   v['sources'].add(source);v['ids'].update(mrnkey(r.get(k)) for k in idkeys if mrnkey(r.get(k)))
   if r.get('EVA02_Organ'):v['views'].add(r['EVA02_Organ'])
   if r.get('modified_file_path'):modified_index_count+=1
 # Preserve conflicts and protected aliases already found in conservative cohort.
 for r in read(ROOT/'data_release_audit/image_manifest.tsv'):
  p=r['image_path']
  if p in records:
   records[p]['ids'].update(mrnkey(v) for k in ['mrn','original_mrn','latest_norm_mrn','identity_candidates'] for v in r.get(k,'').split('|') if mrnkey(v))
 protected={mrnkey(r['mrn']) for p in GUARDS for r in read(p)}
 blocked_paths={r['path'] for p in PATH_GUARDS for r in read(p)}
 # Propagate a protected path or conflicting identity to all linked identities.
 while True:
  new=set().union(*(v['ids'] for p,v in records.items() if p in blocked_paths or v['ids']&protected))
  if new<=protected:break
  protected.update(new)
 conflict_ids=set().union(*(v['ids'] for v in records.values() if len(v['ids'])>1))
 conflict_ids.update(r['mrn'] for r in read(ROOT/'ssl_image_cleaning/identity_review_queue.tsv'))
 tasks=[]
 for p,v in sorted(records.items()):
  tasks.append(dict(sample_id='orig_'+hashlib.sha256(p.encode()).hexdigest()[:24],image_path=p,
    identity_candidates='|'.join(sorted(v['ids'])),sources='|'.join(sorted(v['sources'])),
    metadata_views='|'.join(sorted(v['views'])),reservation_blocked=bool(v['ids']&protected) or p in blocked_paths,
    identity_review=(len(v['ids'])!=1 or bool(v['ids']&conflict_ids)),
    original_provenance='indexed_original_path_not_yet_adjudicated',training_admitted=False))
 assert len({r['sample_id'] for r in tasks})==len(tasks)
 write(out/'input_manifest.tsv',tasks)
 for i in range(shards):write(out/f'input_{i:03d}.tsv',tasks[i::shards])
 files=[p for p,*_ in INPUTS]+GUARDS+PATH_GUARDS+[ROOT/'data_release_audit/image_manifest.tsv',ROOT/'ssl_image_cleaning/identity_review_queue.tsv',ROOT/'DATA_SCOPE.md',Path(__file__),ROOT/'face_qc/models/face_landmarker.task']
 manifest=dict(images=len(tasks),shards=shards,source_rows=dict(source_rows),
  unique_identity_values=len(set().union(*(v['ids'] for v in records.values()))),
  reserved_image_paths=sum(r['reservation_blocked'] for r in tasks),identity_review_paths=sum(r['identity_review'] for r in tasks),
  full_genetics_modified_rows_not_used=modified_index_count,
  full_genetics_path_resolution_root=str(CF),
  scope='All paths in listed CHOP original indexes, including nonface/view/reserved/identity-review records; not all external knowledge resources or physical archive exhaustiveness',
  input_sha256={str(p):digest(p) for p in files},training_admitted=False)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='input_sha256'},indent=2))

def process(row,det):
 import numpy as np,cv2,mediapipe as mp
 from PIL import Image,ImageOps
 from face_qc.pilot import angles
 r=dict(row,status='review',training_admitted=False,background_removed=False,
        orientation_status='EXIF_only_no_rotation_search',quality_adjudicated=False)
 try:
  data=Path(row['image_path']).read_bytes();r['original_sha256']=hashlib.sha256(data).hexdigest()
  with Image.open(io.BytesIO(data)) as im:
   r['exif_orientation']=im.getexif().get(274,1)
   im=ImageOps.exif_transpose(im).convert('RGB');im.load()
  r.update(file_status='decoded',width=im.width,height=im.height)
  thumb=im.copy();thumb.thumbnail((1024,1024));arr=np.ascontiguousarray(thumb)
  gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
  d=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].flatten()
  b=d>np.median(d[1:]);b[0]=False
  r['phash64']=f"{sum(int(x)<<(63-i) for i,x in enumerate(b)):016x}"
  dd=cv2.resize(gray,(9,8));bits=(dd[:,1:]>dd[:,:-1]).flatten()
  r['dhash64']=f"{sum(int(x)<<(63-i) for i,x in enumerate(bits)):016x}"
  result=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=arr));r['faces']=len(result.face_landmarks)
  if not result.face_landmarks:
   r.update(status='view_or_detection_review',reasons='no_landmarks_not_proof_of_nonface',estimated_view='unresolved');return r
  if len(result.face_landmarks)>1:
   r.update(status='target_selection_review',reasons='multiple_faces_no_auto_target',estimated_view='unresolved');return r
  pts=np.array([[p.x*im.width,p.y*im.height] for p in result.face_landmarks[0]])
  eyes=(pts[33]+pts[133])/2,(pts[263]+pts[362])/2
  eye_roll=math.degrees(math.atan2(*(eyes[1]-eyes[0])[::-1]))
  upright=pts[10,1]<pts[152,1] and pts[13,1]>(eyes[0][1]+eyes[1][1])/2 and abs(eye_roll)<=45
  pitch,yaw,roll=angles(result.facial_transformation_matrixes[0])
  r.update(estimated_pitch=round(pitch,2),estimated_yaw=round(yaw,2),eye_roll=round(eye_roll,2))
  if not upright:
   r.update(status='orientation_review',reasons='landmarks_not_upright_no_auto_rotation',estimated_view='unresolved');return r
  # Descriptive uncalibrated pose bins only, never QC acceptance.
  r['estimated_view']='strong_pitch_candidate' if abs(pitch)>30 else 'profile_candidate' if abs(yaw)>=60 else 'oblique_candidate' if abs(yaw)>25 else 'frontal_candidate'
  lo=pts.min(0);hi=pts.max(0);bw,bh=hi-lo
  desired=(math.floor(lo[0]-.35*bw),math.floor(lo[1]-.55*bh),math.ceil(hi[0]+.35*bw),math.ceil(hi[1]+.18*bh))
  box=(max(0,desired[0]),max(0,desired[1]),min(im.width,desired[2]),min(im.height,desired[3]))
  if box[0]>=box[2] or box[1]>=box[3]:raise ValueError('invalid_box')
  r.update(proposed_crop_box=json.dumps(box),margin_clipped=box!=desired,face_width_pixels=round(float(bw),1))
  face=im.crop(box);face.thumbnail((256,256));fg=cv2.cvtColor(np.asarray(face),cv2.COLOR_RGB2GRAY)
  r.update(blur_variance=round(float(cv2.Laplacian(fg,cv2.CV_64F).var()),2),
    dark_fraction=round(float((fg<10).mean()),4),bright_fraction=round(float((fg>245).mean()),4))
  reasons=['uncalibrated_view_and_quality','new_crop_visual_review_pending','original_provenance_pending']
  if box!=desired:reasons.append('margin_clipped')
  if bw<128:reasons.append('small_face')
  if r['blur_variance']<40:reasons.append('possible_blur')
  if r['dark_fraction']>.3 or r['bright_fraction']>.3:reasons.append('possible_exposure')
  if row.get('metadata_views') and not any(t in row['metadata_views'].lower() for t in ['full human face from the front','side-profile portrait']):
   reasons.append('metadata_not_complete_face_requires_cross_check')
  r.update(status='crop_proposal_pending_review',reasons='|'.join(reasons))
 except Exception as e:
  r.update(file_status='missing' if isinstance(e,FileNotFoundError) else 'read_or_processing_error',status='file_or_processing_review',error_type=type(e).__name__,reasons='source_recovery_or_retry_required')
 return r

def run(out,index,limit=None):
 import cv2,mediapipe as mp
 from PIL import Image
 warnings.simplefilter('error',Image.DecompressionBombWarning)
 cv2.setNumThreads(1)
 m=json.loads((out/'manifest.json').read_text())
 assert digest(__file__)==m['input_sha256'][str(Path(__file__))],'Code changed since preparation'
 model=ROOT/'face_qc/models/face_landmarker.task'
 assert digest(model)==m['input_sha256'][str(model)]
 # The pose helper is part of the pinned run, too.
 assert digest(ROOT/'face_qc/pilot.py')==m['helper_sha256']
 dest=out/f'shard_{index:03d}';dest.mkdir(exist_ok=True);progress=dest/'progress.jsonl'
 done={}
 if progress.exists():
  with progress.open('rb+') as f:
   while True:
    pos=f.tell();line=f.readline()
    if not line:break
    try:r=json.loads(line)
    except json.JSONDecodeError:
     if f.read():raise RuntimeError('Corrupt non-tail progress record')
     f.truncate(pos);break
    assert r['sample_id'] not in done
    done[r['sample_id']]=r
 known_path=out/'known_visual_views.json'
 known=json.loads(known_path.read_text()) if known_path.exists() else {}
 if known:assert digest(known_path)==m['known_visual_views_sha256']
 rows=list(read(out/f'input_{index:03d}.tsv'))
 assert set(done)<={r['sample_id'] for r in rows}
 opts=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=3,output_facial_transformation_matrixes=True,min_face_detection_confidence=.5,min_face_presence_confidence=.5)
 processed=0
 with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det:
  for row in rows:
   if row['sample_id'] in done:continue
   t=time.monotonic();r=process(row,det);r['seconds']=round(time.monotonic()-t,3)
   if row['image_path'] in known:
    r['prior_assistant_view']=known[row['image_path']]
    if known[row['image_path']] in {'non_face_or_local_body_document','incomplete_face_original'}:
     r['status']='prior_visual_nonface_or_incomplete_hold'
   with progress.open('a') as f:f.write(json.dumps(r)+'\n');f.flush()
   done[r['sample_id']]=r;processed+=1
   if processed%100==0:print(index,len(done),'/',len(rows),flush=True)
   if limit and processed>=limit:break
 if len(done)==len(rows):
  rr=[done[r['sample_id']] for r in rows];write(dest/'results.tsv',rr,sorted({k for r in rr for k in r}))
  (dest/'summary.json').write_text(json.dumps(dict(images=len(rr),statuses=dict(Counter(r['status'] for r in rr)),training_admitted=False),indent=2)+'\n')
 print(json.dumps(dict(shard=index,completed=len(done),expected=len(rows),complete=len(done)==len(rows))))

def merge(out):
 m=json.loads((out/'manifest.json').read_text());rr=[]
 for i in range(m['shards']):
  results=list(read(out/f'shard_{i:03d}/results.tsv'));expected=list(read(out/f'input_{i:03d}.tsv'))
  assert len(results)==len(expected) and {r['sample_id'] for r in results}=={r['sample_id'] for r in expected}
  rr.extend(results)
 assert len(rr)==m['images']==len({r['sample_id'] for r in rr})
 assert all(r['training_admitted']=='False' for r in rr)
 fields=sorted({k for r in rr for k in r});write(out/'results.tsv',rr,fields)
 groups=defaultdict(list)
 for r in rr:
  if r.get('original_sha256'):groups[r['original_sha256']].append(r)
 duplicates=[]
 for h,g in groups.items():
  if len(g)<2:continue
  ids={v for r in g for v in r['identity_candidates'].split('|') if v}
  duplicates.append(dict(original_sha256=h,paths=len(g),identity_values='|'.join(sorted(ids)),identity_review=len(ids)!=1,
                         any_reserved=any(r['reservation_blocked']=='True' for r in g),sample_ids='|'.join(r['sample_id'] for r in g)))
 write(out/'exact_original_duplicate_groups.tsv',duplicates,['original_sha256','paths','identity_values','identity_review','any_reserved','sample_ids'])
 summary=dict(images=len(rr),file_status=dict(Counter(r.get('file_status','unknown') for r in rr)),
              statuses=dict(Counter(r['status'] for r in rr)),estimated_views=dict(Counter(r.get('estimated_view','unresolved') for r in rr)),
              exact_duplicate_groups=len(duplicates),training_admitted=False,
              limitations=['Automatic measurements are not clinical or visual approval.','Near-duplicate hashes generated, cross-patient similarity adjudication not completed.','Profile detector failure is not nonface exclusion.','Original-path provenance, target identity, family/splits and crop quality still require review.'])
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','merge']);ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--shards',type=int,default=64);ap.add_argument('--index',type=int,default=0);ap.add_argument('--limit',type=int);a=ap.parse_args()
 if a.action=='prepare':
  prepare(a.out,a.shards)
  p=a.out/'manifest.json';m=json.loads(p.read_text());m['helper_sha256']=digest(ROOT/'face_qc/pilot.py');p.write_text(json.dumps(m,indent=2)+'\n')
 elif a.action=='run':run(a.out,a.index,a.limit)
 else:merge(a.out)
