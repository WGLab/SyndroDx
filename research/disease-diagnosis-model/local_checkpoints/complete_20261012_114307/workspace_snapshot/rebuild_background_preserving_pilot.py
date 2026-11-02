"""Rebuild a bounded original-only crop pilot, retaining exact RGB pixels.

No masking, resizing, generative edits, clinical judgments or training admission.
Model-estimated crop boxes require visual review; outputs are full-resolution PNG.
"""
import csv,hashlib,json,io,math,time
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
import numpy as np
from face_qc.pilot import mp,MODEL,cv2
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'background_preserving_pilot_v2'
def sha(data):return hashlib.sha256(data).hexdigest()
def write(p,rr):
 fields=sorted({k for r in rr for k in r})
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rr)
def process(rec,det,out):
 started=time.perf_counter();r=dict(rec,decision='review',reasons='',training_admitted=False,rotation=0)
 try:
  data=Path(rec['image_path']).read_bytes();r['original_file_sha256']=sha(data)
  with Image.open(io.BytesIO(data)) as im:
   r['exif_orientation']=im.getexif().get(274,1)
   im=ImageOps.exif_transpose(im).convert('RGB');im.load()
  r['original_size']=f'{im.width}x{im.height}'
  small=im.copy();small.thumbnail((1024,1024))
  result=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(small)))
  r['faces']=len(result.face_landmarks)
  # All 82 originals have already been visually checked as upright. No automatic
  # 90/180-degree search is allowed to overturn that evidence.
  if r['faces']!=1:
   r['reasons']='no_face' if r['faces']==0 else 'multiple_faces_target_review'
  else:
   pts=np.array([[p.x*im.width,p.y*im.height] for p in result.face_landmarks[0]])
   lo=pts.min(0);hi=pts.max(0);bw,bh=hi-lo
   desired=(math.floor(lo[0]-.35*bw),math.floor(lo[1]-.55*bh),math.ceil(hi[0]+.35*bw),math.ceil(hi[1]+.18*bh))
   box=(max(0,desired[0]),max(0,desired[1]),min(im.width,desired[2]),min(im.height,desired[3]))
   r.update(crop_box_oriented_original=json.dumps(box),margin_clipped=box!=desired,face_width_pixels=round(float(bw),1))
   reasons=[]
   if box!=desired:reasons.append('requested_margin_clipped')
   if bw<128:reasons.append('small_face')
   r['reasons']='|'.join(reasons)
  preview=im.copy();preview.thumbnail((768,768))
  dest=out/'oriented'/f"{rec['sample_id']}.jpg";preview.save(dest,quality=92);r['oriented_preview_path']=str(dest)
 except Exception as e:r.update(reasons='processing_error',error_detail=type(e).__name__)
 r['elapsed_seconds']=round(time.perf_counter()-started,3)
 return r

def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'results.tsv').exists():raise SystemExit('Existing run retained; do not overwrite reviewed crops.')
 for p in ['crops','oriented','fullres_crops','review_sheets']:(OUT/p).mkdir(exist_ok=True)
 inp=ROOT/'ssl_visual_review/priority_original_rebuild_queue.tsv'
 rr=list(csv.DictReader(inp.open(),delimiter='\t'))
 review={r['crop_sha256']:r for r in csv.DictReader((ROOT/'ssl_visual_review/assistant_visual_review.tsv').open(),delimiter='\t')}
 assert len(rr)==82 and all(r['training_admitted']=='False' and r['subset']!='1' for r in rr)
 cv2.setNumThreads(1)
 opts=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=3,output_facial_transformation_matrixes=True,min_face_detection_confidence=.5,min_face_presence_confidence=.5)
 results=[json.loads(line) for line in (OUT/'progress.jsonl').read_text().splitlines()] if (OUT/'progress.jsonl').exists() else []
 done={r['sample_id'] for r in results}
 with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det:
  for row in rr:
   rec=dict(sample_id=review[row['lineage_crop_sha256']]['review_id'],mrn=row['mrn'],source=row['source'],image_path=row['original_image_path'],upstream_crop_sha256=row['lineage_crop_sha256'])
   if rec['sample_id'] in done:continue
   r=process(rec,det,OUT)
   r.update(subset=row['subset'],legacy_crop_used=False,background_removed=False,new_crop_qc='pending',training_admitted=False)
   if r.get('crop_box_oriented_original'):
    try:
     data=Path(r['image_path']).read_bytes();assert r['original_file_sha256']==sha(data), 'Original changed during processing'
     with Image.open(io.BytesIO(data)) as im:
      im=ImageOps.exif_transpose(im).convert('RGB');im.load()
      deg=int(r['rotation']);assert deg in [0,90,180,270]
      oriented=im.rotate(deg,expand=True)
      box=tuple(json.loads(r['crop_box_oriented_original']))
      assert 0<=box[0]<box[2]<=oriented.width and 0<=box[1]<box[3]<=oriented.height
      crop=oriented.crop(box)
      path=OUT/'fullres_crops'/f"{r['sample_id']}.png";crop.save(path,compress_level=1)
      with Image.open(path) as saved:
       assert saved.size==crop.size and saved.tobytes()==crop.tobytes()
      r.update(new_crop_path=str(path),crop_width=crop.width,crop_height=crop.height,
               exact_original_rgb_crop_verified=True,new_crop_sha256=sha(path.read_bytes()),
               pixel_transform='EXIF transpose; optional lossless 90-degree rotation; rectangular crop; RGB conversion; PNG')
    except Exception as e:r.update(rebuild_error=type(e).__name__,exact_original_rgb_crop_verified=False)
   else:r.update(rebuild_error='no_crop_box',exact_original_rgb_crop_verified=False)
   results.append(r)
   with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   if len(results)%20==0:print('Processed',len(results),flush=True)
 write(OUT/'results.tsv',results)
 for start in range(0,len(results),20):
  sheet=Image.new('RGB',(1600,2800),'#ddd');draw=ImageDraw.Draw(sheet)
  for j,r in enumerate(results[start:start+20]):
   x=(j%2)*800;y=(j//2)*280
   draw.text((x+4,y+2),r['sample_id']+' ORIGINAL / NEW CROP',fill='black')
   for k,path in enumerate([r.get('oriented_preview_path'),r.get('new_crop_path')]):
    if path:
     with Image.open(path) as im:
      im=im.convert('RGB');im.thumbnail((390,250));sheet.paste(im,(x+400*k+(400-im.width)//2,y+25+(250-im.height)//2))
  sheet.save(OUT/'review_sheets'/f'contact_{start//20+1:02d}.jpg',quality=92)
 summary=dict(input_images=len(rr),rebuilt=sum(bool(r.get('new_crop_path')) for r in results),
              exact_pixel_verified=sum(r['exact_original_rgb_crop_verified'] for r in results),
              multiple_face_flags=sum(r.get('faces',0)>1 for r in results),
              nonzero_rotation=sum(bool(r.get('rotation')) for r in results),
              input_sha256=sha(inp.read_bytes()),model_sha256=sha(MODEL.read_bytes()),
              code_sha256=sha(Path(__file__).read_bytes()),background_removal=False,training_admitted=False)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
