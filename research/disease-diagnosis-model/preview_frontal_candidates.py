"""Local-only random contact sheets for view QC; no names or diagnoses on sheets."""
import csv,random,json
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parent/'data_release_audit'
if (ROOT/'view_qc_sample.tsv').exists():
 raise SystemExit('Existing QC sample retained: refusing to overwrite annotations.')
with (ROOT/'candidate_inventory.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
rng=random.Random(20260928);selected=[]
for src,n in [('CHOP_craniofacial',32),('CHOP_genetics',16)]:
 rr=[r for r in rows if r['source']==src];selected+=rng.sample(rr,min(n,len(rr)))
canvas=Image.new('RGB',(8*160,6*180),'white');draw=ImageDraw.Draw(canvas);out=[]
for i,r in enumerate(selected):
 x=(i%8)*160;y=(i//8)*180
 rec={'sample_index':i+1,'source':r['source'],'image_path':r['image_path'],'cropping_path':r['cropping_path'],'status':'view_review_pending'}
 try:
  with Image.open(r['cropping_path']) as im:
   im.draft('RGB',(160,160));im=ImageOps.exif_transpose(im);im.thumbnail((156,156));canvas.paste(im.convert('RGB'),(x+(160-im.width)//2,y+20+(156-im.height)//2))
 except Exception as e:rec['status']='file_error_'+type(e).__name__
 draw.text((x+4,y+3),str(i+1)+' '+('CF' if i<32 else 'GEN'),fill='black');out.append(rec)
canvas.save(ROOT/'view_qc_sample.jpg',quality=90)
with (ROOT/'view_qc_sample.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]),delimiter='\t');w.writeheader();w.writerows(out)
print('Saved local contact sheet: 32 CF + 16 genetics candidates; preliminary visual review only.')
