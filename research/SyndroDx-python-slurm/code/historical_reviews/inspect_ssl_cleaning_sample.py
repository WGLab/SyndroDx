"""Bounded crop-readability and similarity pilot; never auto-merges photos."""
import csv,hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from PIL import Image, ImageOps
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'ssl_image_cleaning'

def inspect(r):
    x=dict(mrn=r['mrn'],subset=r['subset'],crop_sha256=r['crop_sha256'],cropping_path=r['cropping_path'])
    try:
        data=Path(r['cropping_path']).read_bytes()
        x['actual_sha256']=hashlib.sha256(data).hexdigest()
        x['hash_matches_audit']=x['actual_sha256']==r['crop_sha256']
        import io
        with Image.open(io.BytesIO(data)) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.load()
            x['width'],x['height']=im.size
            small=im.convert('L').resize((9,8),Image.Resampling.LANCZOS)
            pix=list(small.getdata());bits=0
            for y in range(8):
                for z in range(8):bits=(bits<<1)|int(pix[y*9+z]>pix[y*9+z+1])
            x['dhash64']=f'{bits:016x}'
            x['decoded_pixel_sha256']=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            x['read_status']='decoded'
    except Exception as e:
        x['read_status']='error';x['error_type']=type(e).__name__
    x['training_admitted']=False
    return x

def write(path,rr,fields):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rr)
def main():
    with (ROOT/'ssl_admission_inventory/qc_review_sample.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    with ThreadPoolExecutor(max_workers=4) as ex: rr=list(ex.map(inspect,rows))
    OUT.mkdir(exist_ok=True)
    fields=['mrn','subset','crop_sha256','cropping_path','actual_sha256','hash_matches_audit','width','height','dhash64','decoded_pixel_sha256','read_status','error_type','training_admitted']
    write(OUT/'sample_pixel_audit.tsv',rr,fields)
    pairs=[]
    for a,b in combinations([r for r in rr if r['read_status']=='decoded'],2):
        dist=bin(int(a['dhash64'],16)^int(b['dhash64'],16)).count('1')
        exact=a['decoded_pixel_sha256']==b['decoded_pixel_sha256']
        if exact or dist<=4:
            pairs.append(dict(crop_a=a['crop_sha256'],crop_b=b['crop_sha256'],same_patient=a['mrn']==b['mrn'],dhash_distance=dist,exact_decoded_pixels=exact,decision='review_only_no_merge'))
    write(OUT/'sample_similarity_review.tsv',pairs,['crop_a','crop_b','same_patient','dhash_distance','exact_decoded_pixels','decision'])
    s=dict(scope='Only the preselected QC sample; not a full-dataset duplicate audit or face QC',
           images=len(rr),decoded=sum(r['read_status']=='decoded' for r in rr),
           read_errors=sum(r['read_status']=='error' for r in rr),
           changed_file_hashes=sum(r.get('hash_matches_audit') is False for r in rr),
           similarity_pairs_for_review=len(pairs),threshold='64-bit dHash Hamming <=4 or exact decoded RGB pixels',
           automatic_merges=0,frontal_approval=False,
           limits='dHash can miss recrops/rotations and confuse similar faces; no match does not establish independence',
           input_sha256=hashlib.sha256((ROOT/'ssl_admission_inventory/qc_review_sample.tsv').read_bytes()).hexdigest(),
           code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'sample_pixel_summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(s,indent=2))
if __name__=='__main__':main()
