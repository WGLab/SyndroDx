"""Build derived input cache entirely on HPC; emit aggregate status only."""
import concurrent.futures
import fcntl
import hashlib
import io
import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from control import WORK, load, save, digest, now
from data import verify_manifest, evaluation_cohort

def destination(sha, kind):
    return WORK/'cache'/kind/sha[:2]/(sha+('.png' if kind=='train' else '.npy'))

def materialize(task):
    path, sha, kind = task
    out = destination(sha, kind)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.with_suffix(out.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            record=load(out.with_suffix(out.suffix+'.json'))
            if record.get('source_sha256')==sha and out.exists() and digest(out)==record.get('sha256'):
                return (kind,sha,record['sha256'],out.stat().st_size)
            return write_item(path,sha,kind,out)
    except Exception:
        return None

def write_item(path,sha,kind,out):
    try:
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name+'.tmp.'+str(os.getpid()))
        if kind == 'train':
            with Image.open(io.BytesIO(raw)) as image:
                image = image.convert('RGB')
                image.thumbnail((256,256), Image.Resampling.BILINEAR)
                image.save(tmp, format='PNG', compress_level=1)
        else:
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                return None
            image = cv2.cvtColor(cv2.resize(image,(112,112)),cv2.COLOR_BGR2RGB)
            value = np.ascontiguousarray(image.transpose(2,0,1),dtype=np.float32)
            value /= 255.; value -= .5; value /= .5
            with tmp.open('wb') as stream:
                np.save(stream,value,allow_pickle=False)
        tmp.replace(out)
        output_sha=digest(out)
        save(out.with_suffix(out.suffix+'.json'),dict(source_sha256=sha,sha256=output_sha))
        return (kind,sha,output_sha,out.stat().st_size)
    except Exception:
        return None

def main():
    os.umask(0o077); cv2.setNumThreads(1)
    p=load(WORK/'protocol.json')
    rows=verify_manifest(p['train_manifest'])
    records,_=evaluation_cohort()
    tasks=sorted({(r['crop_path'],r['crop_sha256'],'train') for r in rows} |
                 {(r['crop_path'],r['crop_sha256'],'eval') for r in records})
    # Different paths with identical source bytes share a single derived cache entry.
    tasks=list({(sha,kind):(path,sha,kind) for path,sha,kind in tasks}.values())
    workers=int(os.environ.get('SLURM_CPUS_PER_TASK','8'))
    begun=time.monotonic(); done=0; failed=0; size=0; index={'train':{},'eval':{}}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(materialize,tasks,chunksize=8):
            done+=1
            if result is None:failed+=1
            else:
                kind,sha,output_sha,n=result;index[kind][sha]=output_sha;size+=n
            if done%256==0 or done==len(tasks):
                save(WORK/'cache_status.json',dict(state='building',completed=done,total=len(tasks),
                    failed=failed,elapsed_seconds=time.monotonic()-begun,bytes=size,workers=workers,updated_at_utc=now()))
    if failed:
        save(WORK/'cache_status.json',dict(state='failed',failed=failed,completed=done,total=len(tasks),updated_at_utc=now()))
        raise RuntimeError('local_cache_integrity_or_decode_failure')
    save(WORK/'cache/index.private.json',index)
    save(WORK/'cache_receipt.json',dict(state='complete',train_files=len(index['train']),
        eval_files=len(index['eval']),train_manifest_sha256=p['train_manifest']['sha256'],
        validation_inputs_sha256=p['validation_inputs_sha256'],index_sha256=digest(WORK/'cache/index.private.json'),
        source_sha_verified=True,originals_unchanged=True,bytes=size,elapsed_seconds=time.monotonic()-begun,
        derivation=p['input_cache'],updated_at_utc=now()))

if __name__=='__main__':
    main()
