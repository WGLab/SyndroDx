"""CPU-only face orientation/QC pilot. No source edits, masking, or frontalization."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageOps,ImageDraw
import mediapipe as mp
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'face_qc/models/face_landmarker.task'

def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rr):
 with Path(p).open('w') as f:
  fields=sorted({k for r in rr for k in r})
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rr)
def angles(matrix):
 r=np.asarray(matrix)[:3,:3];u,_,v=np.linalg.svd(r);r=u@v
 if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@v
 yaw=math.degrees(math.asin(float(np.clip(-r[2,0],-1,1))))
 pitch=math.degrees(math.atan2(r[2,1],r[2,2]));roll=math.degrees(math.atan2(r[1,0],r[0,0]))
 return pitch,yaw,roll

def select(n):
 rr=[r for r in read(ROOT/'four_subset_inventory/image_subsets.tsv') if r['subset']=='4']
 originals={(r['mrn'],r['crop_sha256']):r for r in read(ROOT/'data_release_audit/chop_patient_consistent_candidates.tsv')}
 by={}
 for r in rr:by.setdefault(r['mrn'],[]).append(r)
 rng=random.Random(20260928);ids=sorted(by);rng.shuffle(ids);out=[]
 for i,m in enumerate(ids[:n],1):
  a=rng.choice(sorted(by[m],key=lambda x:x['crop_sha256']));r=originals[(m,a['crop_sha256'])]
  out.append(dict(sample_id=f'pilot_{i:03d}',sampling='random_patient_then_image',mrn=m,source=r['source'],image_path=r['image_path'],upstream_crop_sha256=r['crop_sha256']))
 demo=ROOT/'data_release_audit/examples/originals/provenance.json'
 if demo.exists():
  for i,r in enumerate(json.loads(demo.read_text()),1):out.append(dict(sample_id=f'demo_{i:02d}',sampling='previously_displayed_not_random',mrn='',source=r['source'],image_path=r['original_path'],upstream_crop_sha256=''))
 return out

def process(rec,det,out):
 started=time.perf_counter();r=dict(rec,decision='review',reasons='',training_admitted=False)
 try:
  with Image.open(rec['image_path']) as im:
   r['exif_orientation']=im.getexif().get(274,1);im=ImageOps.exif_transpose(im).convert('RGB');im.load()
  r['original_size']=f'{im.width}x{im.height}'
  small=im.copy();small.thumbnail((1024,1024))
  candidates=[]
  for deg in (0,90,180,270):
   thumb=small.rotate(deg,expand=True);arr=np.ascontiguousarray(thumb)
   result=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=arr))
   for j,landmarks in enumerate(result.face_landmarks):
    pts=np.array([[p.x*thumb.width,p.y*thumb.height] for p in landmarks])
    eyes=(pts[33]+pts[133])/2,(pts[263]+pts[362])/2
    dx,dy=eyes[1]-eyes[0];eye_roll=math.degrees(math.atan2(dy,dx))
    upright=pts[10,1]<pts[152,1] and (pts[13,1]>(eyes[0][1]+eyes[1][1])/2)
    if not upright or abs(eye_roll)>50:continue
    pitch,yaw,roll=angles(result.facial_transformation_matrixes[j])
    score=-abs(eye_roll)
    candidates.append(dict(rotation=deg,points=pts,width=thumb.width,height=thumb.height,pitch=pitch,yaw=yaw,roll=roll,eye_roll=eye_roll,score=score,faces=len(result.face_landmarks)))
  if not candidates:
   r['reasons']='no_stable_upright_landmarks';preview=small
  else:
   candidates.sort(key=lambda x:x['score'],reverse=True);c=candidates[0]
   r.update({k:round(c[k],3) if isinstance(c[k],float) else c[k] for k in ['rotation','pitch','yaw','roll','eye_roll','faces']})
   r['orientation_candidate_count']=len(candidates)
   reasons=[]
   if c['faces']>1:reasons.append('multiple_faces_subject_ambiguous')
   # Different upright rotation candidates should not be silently chosen.
   alternatives=[x for x in candidates[1:] if x['rotation']!=c['rotation']]
   if alternatives and c['score']-alternatives[0]['score']<15:reasons.append('rotation_ambiguous')
   oriented=im.rotate(c['rotation'],expand=True)
   pts=c['points']/np.array([c['width'],c['height']])*np.array([oriented.width,oriented.height])
   lo=pts.min(0);hi=pts.max(0);bw,bh=hi-lo
   desired=(math.floor(lo[0]-.35*bw),math.floor(lo[1]-.55*bh),math.ceil(hi[0]+.35*bw),math.ceil(hi[1]+.18*bh))
   box=(max(0,desired[0]),max(0,desired[1]),min(oriented.width,desired[2]),min(oriented.height,desired[3]))
   r['crop_box_oriented_original']=json.dumps(box);r['margin_clipped']=box!=desired
   if box!=desired:reasons.append('requested_head_margin_clipped')
   if (lo<1).any() or hi[0]>=oriented.width-1 or hi[1]>=oriented.height-1:reasons.append('face_landmarks_at_image_boundary')
   crop=oriented.crop(box);r['face_width_pixels']=round(float(bw),1)
   face=oriented.crop((max(0,int(lo[0])),max(0,int(lo[1])),min(oriented.width,int(hi[0])+1),min(oriented.height,int(hi[1])+1)))
   face=ImageOps.contain(face,(256,256));gray=cv2.cvtColor(np.asarray(face),cv2.COLOR_RGB2GRAY)
   blur=float(cv2.Laplacian(gray,cv2.CV_64F).var());dark=float((gray<10).mean());bright=float((gray>245).mean())
   r.update(blur_variance=round(blur,2),dark_fraction=round(dark,4),bright_fraction=round(bright,4))
   if bw<128:reasons.append('small_face')
   if blur<40:reasons.append('possible_blur')
   if dark>.3 or bright>.3:reasons.append('possible_exposure_problem')
   if abs(c['eye_roll'])>15:reasons.append('residual_roll_review')
   if abs(c['yaw'])>30 or abs(c['pitch'])>30:
    r['decision']='nonfrontal_candidate' if not any(x in reasons for x in ['multiple_faces_subject_ambiguous','rotation_ambiguous']) else 'review'
    reasons.append('large_estimated_yaw_or_pitch')
   elif abs(c['yaw'])>15 or abs(c['pitch'])>15:reasons.append('borderline_pose');r['decision']='review'
   else:r['decision']='review' if reasons else 'frontal_candidate'
   r['reasons']='|'.join(reasons)
   crop.thumbnail((768,768));target=out/'crops'/f"{rec['sample_id']}.jpg";crop.save(target,quality=95)
   r['crop_preview_path']=str(target.resolve());preview=oriented.copy();preview.thumbnail((768,768))
  target=out/'oriented'/f"{rec['sample_id']}.jpg";preview.save(target,quality=90);r['oriented_preview_path']=str(target.resolve())
 except Exception as exc:
  r['decision']='review';r['reasons']='processing_error_'+type(exc).__name__;r['error_detail']=str(exc)[:300]
 r['elapsed_seconds']=round(time.perf_counter()-started,3)
 return r

def sheet(rr,out):
 for start in range(0,len(rr),30):
  page=Image.new('RGB',(6*220,5*255),'#eeeeee');draw=ImageDraw.Draw(page)
  for j,r in enumerate(rr[start:start+30]):
   x=(j%6)*220;y=(j//6)*255;p=r.get('crop_preview_path') or r.get('oriented_preview_path')
   if p:
    with Image.open(p) as im:
     im.thumbnail((216,205));page.paste(im,(x+(220-im.width)//2,y+45))
   draw.text((x+3,y+2),r['sample_id']+' '+r['decision'],fill='black')
   draw.text((x+3,y+17),f"rot={r.get('rotation','?')} yaw={r.get('yaw','?')} pitch={r.get('pitch','?')}",fill='black')
  page.save(out/f'contact_{start//30+1:02d}.jpg',quality=90)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=300);ap.add_argument('--out',type=Path,default=ROOT/'face_qc/pilot300');args=ap.parse_args()
 if (args.out/'results.tsv').exists():raise SystemExit('Existing pilot retained; choose a new --out.')
 args.out.mkdir(parents=True,exist_ok=True)
 for x in ['crops','oriented']:(args.out/x).mkdir(exist_ok=True)
 cv2.setNumThreads(1)
 selected=select(args.n);write(args.out/'input_manifest.tsv',selected)
 opts=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=3,output_facial_transformation_matrixes=True,min_face_detection_confidence=.5,min_face_presence_confidence=.5)
 t=time.perf_counter();rr=[]
 with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det:
  for i,rec in enumerate(selected,1):
   result=process(rec,det,args.out);rr.append(result)
   with (args.out/'progress.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
   if i%25==0:print(f'Processed {i}/{len(selected)}; {time.perf_counter()-t:.1f}s',flush=True)
 elapsed=time.perf_counter()-t;write(args.out/'results.tsv',rr);sheet(rr,args.out)
 from collections import Counter
 randomrows=[r for r in rr if r['sampling']=='random_patient_then_image']
 summary=dict(random_images=len(randomrows),random_patients=len({r['mrn'] for r in randomrows}),demo_images=len(rr)-len(randomrows),random_decisions=dict(Counter(r['decision'] for r in randomrows)),random_rotations=dict(Counter(str(r.get('rotation','no_detection')) for r in randomrows)),elapsed_processing_seconds=elapsed,seconds_per_image=elapsed/len(rr),device='CPU',cpu_affinity=len(os.sched_getaffinity(0)),mediapipe=mp.__version__,model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),status='uncalibrated_screening_not_training_admission',background_removal=False,small_roll_correction='not_applied_pending_morphology_review',estimated_pose_convention='Rz(roll) Ry(yaw) Rx(pitch), MediaPipe canonical transform; approximate not calibrated anatomical angles')
 (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
