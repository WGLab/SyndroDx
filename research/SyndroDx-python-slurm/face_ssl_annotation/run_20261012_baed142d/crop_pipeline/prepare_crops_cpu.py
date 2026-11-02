"""Prepare provisional native-pixel crops from existing local Qwen coordinates."""
from collections import Counter
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time
import traceback

os.umask(0o077)
ROOT=Path(__file__).resolve().parent


def save(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def crop_module(root):
    spec=importlib.util.spec_from_file_location('local_crop_functions',root/'crop_worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare(row,root,crop):
    from PIL import Image,ImageOps
    key=crop.output_key(row['photo_id'])
    item={'photo_id':row['photo_id'],'patient_key':row['patient_key'],'view':row['view'],
          'state':'held','training_admitted':False,'GPU_review_complete':False,'original_untouched':True}
    size=(int(row['width']),int(row['height']))
    try:encoded=json.loads(row.get('face_bbox') or 'null')
    except (ValueError,TypeError):encoded=None
    face_box=crop.normalized_box(encoded,size)
    if face_box is None:
        item['state']='requires_GPU_localization';return item
    target=root/'prepared_native'/key[:2]/(key+'.png')
    if (root/'images_native'/key[:2]/(key+'.png')).exists():
        item['state']='already_materialized_by_GPU';return item
    original=Path(row['original_path']).read_bytes()
    item['source_sha256']=hashlib.sha256(original).hexdigest()
    with Image.open(io.BytesIO(original)) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
    if image.size!=size:
        item['state']='held_source_dimensions_changed';return item
    box=crop.padded_box(face_box,size,row['view'])
    if min(box[2]-box[0],box[3]-box[1])<64:
        item['state']='held_crop_too_small';return item
    item.update(state='prepared_candidate',source_size=list(size),crop_box=list(box),
                output=crop.save_crop(image,box,target),output_path=str(target.relative_to(root)))
    image.close()
    return item


def worker(args):
    root,slot,worker_count=args
    root=Path(root);crop=crop_module(root)
    state=root/'preparation'/f'slot_{slot:02d}.json'
    lock=(root/'preparation'/f'slot_{slot:02d}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db=sqlite3.connect(root/'preparation'/f'slot_{slot:02d}.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS preparations (key TEXT PRIMARY KEY,state TEXT,record TEXT,updated_at REAL)')
    db.commit();counts=Counter(dict(db.execute('SELECT state,count(*) FROM preparations GROUP BY state')))
    recent=time.time()
    save(state,{'state':'preparing','slot':slot,'counts':dict(counts),'updated_at':time.time()})
    try:
        with (root.parent/'accepted_faces.tsv').open(newline='') as handle:
            for row in csv.DictReader(handle,delimiter='\t'):
                key=crop.output_key(row['photo_id'])
                if int(key[:8],16)%worker_count!=slot or db.execute('SELECT 1 FROM preparations WHERE key=?',(key,)).fetchone():
                    continue
                try:item=prepare(row,root,crop)
                except Exception as exc:
                    with (root/'preparation'/f'slot_{slot:02d}.private_errors.log').open('a') as log:
                        traceback.print_exc(file=log)
                    item={'state':'error','reason':type(exc).__name__,'training_admitted':False,
                          'photo_id':row['photo_id'],'patient_key':row['patient_key'],'view':row['view']}
                db.execute('INSERT INTO preparations VALUES (?,?,?,?)',(key,item['state'],json.dumps(item),time.time()))
                db.commit();counts[item['state']]+=1
                if time.time()-recent>=5:
                    save(state,{'state':'preparing','slot':slot,'counts':dict(counts),'updated_at':time.time()});recent=time.time()
        save(state,{'state':'prepared','slot':slot,'counts':dict(counts),'updated_at':time.time(),
                    'training_admitted':False})
        return dict(counts)
    finally:
        db.close();lock.close()


def main(root):
    plan=json.loads((root/'preparation_plan.json').read_text());worker_count=int(plan['worker_count'])
    manifest=json.loads((root/'READY.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('frozen_pipeline_hash_mismatch')
    folder=root/'preparation';folder.mkdir(exist_ok=True)
    with (folder/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(folder/'status.json',{'state':'preparing','updated_at':time.time(),'worker_count':worker_count})
        counts=Counter()
        with multiprocessing.get_context('spawn').Pool(worker_count) as pool:
            for result in pool.imap_unordered(worker,[(str(root),i,worker_count) for i in range(worker_count)]):
                counts.update(result)
        save(folder/'completion.json',{'state':'candidate_preparation_complete','updated_at':time.time(),
             'counts':dict(counts),'training_admitted':False,'GPU_review_complete':False})
        save(folder/'status.json',{'state':'prepared','updated_at':time.time(),'counts':dict(counts)})
        spec=importlib.util.spec_from_file_location('local_stage_mail',root.parent/'notify_pipeline_stages.py')
        mail=importlib.util.module_from_spec(spec);spec.loader.exec_module(mail)
        mail.notify('crop candidate preparation completed','Candidate crop preparation',
                    'Native-pixel candidate crops have been prepared locally from existing Qwen coordinates. GPU crop quality review is still required. Images and patient information remain on HPC; training data have not been released.')
        print('LOCAL_CANDIDATE_CROP_PREPARATION_COMPLETE',flush=True)


if __name__=='__main__':
    try:main(ROOT)
    except Exception as exc:
        folder=ROOT/'preparation';folder.mkdir(exist_ok=True)
        with (folder/'coordinator.private_errors.log').open('a') as log:traceback.print_exc(file=log)
        save(folder/'status.json',{'state':'failed','error_type':type(exc).__name__,'updated_at':time.time()})
        print('CPU_CROP_PREPARATION_FAILED_'+type(exc).__name__,flush=True)
        raise SystemExit(1)
