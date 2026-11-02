"""Obtain FairFace from the maintainers' public CC BY 4.0 distribution."""
import datetime,hashlib,json,os,stat,traceback,urllib.request,zipfile
from pathlib import Path
os.umask(0o077)
ROOT=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/assets/fairface')
ROOT.mkdir(parents=True,exist_ok=True)
def save(v):
 v['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p=ROOT/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
try:
 import gdown
 readme=urllib.request.urlopen('https://raw.githubusercontent.com/dchen236/FairFace/master/README.md',timeout=60).read()
 if b'License: CC BY 4.0' not in readme:raise RuntimeError('public_license_not_confirmed')
 (ROOT/'upstream_README.md').write_bytes(readme)
 save({'state':'downloading_public_reference','license':'CC BY 4.0','clinically_verified_healthy':False})
 archive=ROOT/'fairface_padding025.zip'
 if not archive.exists() or not zipfile.is_zipfile(archive):
  if gdown.download(id='1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86',output=str(archive),quiet=True,resume=True) is None:raise RuntimeError('official_reference_download_failed')
 destination=ROOT/'images';destination.mkdir(exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  for entry in z.infolist():
   target=(destination/entry.filename).resolve()
   if not target.is_relative_to(destination.resolve()) or stat.S_ISLNK(entry.external_attr>>16):raise RuntimeError('unsafe_archive_entry')
   if not target.exists() or target.stat().st_size!=entry.file_size:z.extract(entry,destination)
  bad=z.testzip()
  if bad is not None:raise RuntimeError('reference_archive_crc_failed')
 # Public train images only; FairFace validation data remain outside SSL.
 files=sorted(p for p in destination.rglob('*') if p.suffix.lower() in {'.jpg','.png','.jpeg'} and 'train' in p.parts)
 if not files:raise RuntimeError('public_training_reference_missing')
 import csv
 from PIL import Image
 manifest=ROOT/'reference_train.tsv';temporary=manifest.with_suffix('.tmp')
 seen=set()
 with temporary.open('w',newline='') as handle:
  fields=['photo_id','patient_key','crop_path','source','view','crop_sha256','training_admitted','split']
  writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t');writer.writeheader()
  for path in files:
   try:
    with Image.open(path) as im:im.verify()
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in seen:continue
    seen.add(digest)
    writer.writerow(dict(photo_id='fairface:'+digest,patient_key='fairface:'+digest,crop_path=str(path),source='fairface_reference',view='unspecified',crop_sha256=digest,training_admitted='true',split='train'))
   except (OSError,ValueError):continue
 temporary.replace(manifest)
 save({'state':'public_reference_ready','source':'FairFace','license':'CC BY 4.0','provenance':'https://github.com/dchen236/FairFace','manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'clinically_verified_healthy':False,'identity_labels_available':False,'validation_excluded':True,'aligned_upstream':True})
except Exception as e:
 with (ROOT/'download.private.log').open('a') as f:traceback.print_exc(file=f)
 save({'state':'attention_required','error_type':type(e).__name__});raise SystemExit(1)
