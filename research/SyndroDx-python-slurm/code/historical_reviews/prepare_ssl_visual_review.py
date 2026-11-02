"""Create controlled local original/crop contact sheets for assistant QC review."""
import csv,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'ssl_visual_review';OUT.mkdir(exist_ok=True)
rows=list(csv.DictReader((ROOT/'ssl_admission_inventory/qc_review_sample.tsv').open(),delimiter='\t'))
lineage={r['crop_sha256']:r for r in csv.DictReader((ROOT/'ssl_image_cleaning/image_lineage_ledger.tsv').open(),delimiter='\t')}

def panel(item):
 i,r=item
 tile=Image.new('RGB',(800,280),'#dddddd');d=ImageDraw.Draw(tile)
 d.text((5,3),f'R{i+1:03d} | original LEFT / crop RIGHT',fill='black')
 errors=[]
 for j,key in enumerate(['original_image_path','cropping_path']):
  try:
   with Image.open(r[key]) as im:
    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((390,250))
    tile.paste(im,(j*400+(400-im.width)//2,25+(250-im.height)//2))
  except Exception as e:
   errors.append(key+':'+type(e).__name__);d.text((j*400+5,50),'READ ERROR',fill='red')
 return i,tile,errors
with ThreadPoolExecutor(max_workers=4) as ex:
 panels=list(ex.map(panel,enumerate(rows)))
manifest=[]
for start in range(0,len(rows),20):
 sheet=Image.new('RGB',(1600,2800),'white')
 for i,tile,errors in panels[start:start+20]:
  k=i-start;sheet.paste(tile,((k%2)*800,(k//2)*280))
  r=rows[i];rec=dict(review_id=f'R{i+1:03d}',sheet=f'contact_{start//20+1:02d}.jpg',**r)
  rec['lineage_identity_pending']=lineage[r['crop_sha256']]['patient_identity_review']
  rec['display_errors']='|'.join(errors);manifest.append(rec)
 sheet.save(OUT/f'contact_{start//20+1:02d}.jpg',quality=92)
with (OUT/'review_manifest.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(manifest[0]),delimiter='\t');w.writeheader();w.writerows(manifest)
(OUT/'preparation.json').write_text(json.dumps(dict(images=len(rows),sheets=(len(rows)+19)//20,display_errors=sum(bool(r['display_errors']) for r in manifest),input_sha256=hashlib.sha256((ROOT/'ssl_admission_inventory/qc_review_sample.tsv').read_bytes()).hexdigest()),indent=2))
print('Prepared',len(rows),'pairs; errors:',sum(bool(r['display_errors']) for r in manifest))
