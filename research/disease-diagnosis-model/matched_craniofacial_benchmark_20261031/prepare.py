"""HPC-only patient and visual-overlap audit before any benchmark training."""
import os,sys,csv,json,hashlib,collections,concurrent.futures
from pathlib import Path
import cv2,numpy as np
from common import *
ID=PROJECT/'transface_arcface_identity_cpt_20261026'
SOURCE=PROJECT/'gmdb_arcface_finetune_20261029'
sys.path.insert(0,str(ID))
from packed_cache import PackedRGB

def phash(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);d=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].reshape(-1)
    bits=d[1:]>np.median(d[1:]);return np.uint64(sum(int(v)<<i for i,v in enumerate(bits)))

def hashes(rgb):
    rgb=cv2.resize(rgb,(112,112));return hashlib.sha256(rgb.tobytes()).hexdigest(),[phash(rgb),phash(np.ascontiguousarray(rgb[:,::-1]))]

def main():
    os.umask(0o077);cv2.setNumThreads(1)
    if load(WORK/'audit.json').get('state')=='passed':return
    protocol=load(WORK/'protocol.json');source_receipt=load(SOURCE/'data_receipt.json')
    for name,sha in source_receipt['files'].items():assert digest(SOURCE/name)==sha
    rows=load(SOURCE/'private/records.private.json')['rows'];public=[json.loads(l) for l in (SHARED/'inputs.private.jsonl').open()]
    snapshots={}
    for p in (SHARED/'split_snapshot').glob('*.csv'):
        with p.open() as f:snapshots[p.stem]=list(csv.DictReader(f))
    test={r['subject'] for name,rs in snapshots.items() if 'test' in name for r in rs}
    tr={r['subject'] for r in rows if r['split']=='train'};va={r['subject'] for r in rows if r['split']=='validation'}
    assert not tr&va and not tr&test and not va&test
    images={}
    for r in rows+public:
        key=r['image_id']
        if key in images:assert images[key]['subject']==r['subject'] and images[key]['crop_sha256']==r['crop_sha256']
        images[key]=dict(r)
    ids=sorted(images);source_array=np.load(SOURCE/'private/images.uint8.npy',mmap_mode='r',allow_pickle=False)
    available={r['image_id']:r['array_index'] for r in rows}
    def public_one(key):
        r=images[key]
        if key in available:rgb=np.asarray(source_array[available[key]])
        else:
            if digest(r['crop_path'])!=r['crop_sha256']:raise RuntimeError('public_crop_integrity_changed')
            bgr=cv2.imread(r['crop_path'])
            if bgr is None:raise RuntimeError('local_decode_failed')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        return hashes(rgb)
    save(WORK/'preparation_status.json',dict(state='hashing_public_images',images=len(ids),updated_at_utc=now()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(public_one,ids))
    pixels=[v[0] for v in values];h=np.array([v[1] for v in values],dtype=np.uint64);np.save(WORK/'private/public_phashes.npy',h)
    bypixel=collections.defaultdict(set)
    for key,pixel in zip(ids,pixels):bypixel[pixel].add(images[key]['subject'])
    exact=set().union(*(v for v in bypixel.values() if len(v)>1)) if bypixel else set();near=set()
    for i in range(len(ids)):
        distances=np.minimum(np.bitwise_count(h[i,0]^h[i+1:,0]),np.bitwise_count(h[i,0]^h[i+1:,1]))
        for j in np.flatnonzero(distances<=4)+i+1:
            if images[ids[i]]['subject']!=images[ids[int(j)]]['subject']:near.update([images[ids[i]]['subject'],images[ids[int(j)]]['subject']])
    # Compare approved clinical CPT pixels to public data locally; never export hash rows.
    receipt=load(ID/'cache_receipt.json');packed=PackedRGB(ID/'cache_packed',receipt['index_sha256'])
    spec=load(ID/'protocol.json')['train_manifest'];assert digest(spec['path'])==spec['sha256']
    with Path(spec['path']).open() as f:clinical=list(csv.DictReader(f,delimiter='\t'))
    chashes=sorted({r['crop_sha256'] for r in clinical})
    save(WORK/'preparation_status.json',dict(state='hashing_clinical_CPT_images',images=len(chashes),updated_at_utc=now()))
    packed.image(chashes[0]).close()
    def clinical_one(sha):
        im=packed.image(sha)
        try:return hashes(np.asarray(im))
        finally:im.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:cv=list(pool.map(clinical_one,chashes))
    cpixels={v[0] for v in cv};cph=np.array([v[1] for v in cv],dtype=np.uint64);np.save(WORK/'private/clinical_phashes.npy',cph)
    clinical_overlap=set()
    for i,key in enumerate(ids):
        if pixels[i] in cpixels or np.any(np.minimum(np.bitwise_count(h[i,0]^cph[:,0]),np.bitwise_count(h[i,0]^cph[:,1]))<=4):clinical_overlap.add(images[key]['subject'])
    blocked=exact|near|clinical_overlap
    train_rows=[r for r in rows if r['subject'] not in blocked]
    vocabulary=sorted({r['label'] for r in train_rows if r['split']=='train'});classmap={v:i for i,v in enumerate(vocabulary)}
    for r in train_rows:r['class_index']=classmap.get(r['label'],-1)
    save(WORK/'private/records.private.json',dict(rows=train_rows,labels=vocabulary))
    save(WORK/'private/blocked_patients.json',dict(subjects=sorted(blocked)))
    save(WORK/'private/public_rows.json',dict(rows=public))
    save(WORK/'private/support_ids.json',dict(image_ids=[r['image_id'] for r in train_rows if r['split']=='train']))
    arraypath=WORK/'private/images.uint8.npy'
    if not arraypath.exists():arraypath.symlink_to((SOURCE/'private/images.uint8.npy').resolve())
    pins=dict(records_sha256=digest(WORK/'private/records.private.json'),blocked_sha256=digest(WORK/'private/blocked_patients.json'),public_inputs_sha256=digest(SHARED/'inputs.private.jsonl'),image_array_sha256=source_receipt['files']['private/images.uint8.npy'],splits_sha256={p.stem:digest(p) for p in (SHARED/'split_snapshot').glob('*.csv')})
    save(WORK/'data_pins.json',pins)
    report=dict(state='passed',train_patients=len({r['subject'] for r in train_rows if r['split']=='train'}),validation_patients=len({r['subject'] for r in train_rows if r['split']=='validation' and r['class_index']>=0}),diseases=len(vocabulary),
        exact_duplicate_patient_flags=len(exact),perceptual_duplicate_patient_flags=len(near),CPT_visual_overlap_patient_flags=len(clinical_overlap),whole_patient_quarantined=len(blocked),
        remaining_test_patients=len(test-blocked),train_validation_patient_overlap=0,train_test_patient_overlap=0,validation_test_patient_overlap=0,
        clinical_test_used=False,source_identity_alias_audit_complete=False,near_duplicate_flags_conservatively_excluded=True,
        protocol_sha256=digest(WORK/'protocol.json'),data_pins=pins,updated_at_utc=now())
    if report['train_patients']<100 or report['validation_patients']<30 or report['remaining_test_patients']<100:raise RuntimeError('quarantine_leaves_insufficient_cohort')
    save(WORK/'audit.json',report);print('Local patient and visual-overlap audit complete; aggregate report saved.')

if __name__=='__main__':
    try:main()
    except Exception as e:save(WORK/'audit_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
