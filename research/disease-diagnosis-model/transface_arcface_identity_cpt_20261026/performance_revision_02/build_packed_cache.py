"""Losslessly decode approved cache on HPC; emit only aggregate status."""
import collections
import concurrent.futures
import fcntl
import hashlib
import io
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from control import WORK,load,save,digest,now

FOLDER=WORK/'cache_packed'

def decode(item):
    sha,expected=item
    path=WORK/'cache/train'/sha[:2]/(sha+'.png')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise RuntimeError('source_cached_png_checksum_mismatch')
    with Image.open(io.BytesIO(raw)) as source:
        image=source.convert('RGB')
        width,height=image.size
        pixels=image.tobytes()
        image.close()
    if max(width,height)>256 or min(width,height)<1:raise RuntimeError('unexpected_cached_image_dimensions')
    return sha,width,height,pixels

def build():
    os.umask(0o077);FOLDER.mkdir(exist_ok=True)
    with (FOLDER/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cache=load(WORK/'cache_receipt.json');protocol=load(WORK/'protocol.json')
        index_path=WORK/'cache/index.private.json'
        if cache.get('state')!='complete' or digest(index_path)!=cache['index_sha256']:
            raise RuntimeError('verified_source_cache_required')
        if cache['train_manifest_sha256']!=protocol['train_manifest']['sha256']:
            raise RuntimeError('training_cohort_mismatch')
        old=load(FOLDER/'receipt.json')
        if old.get('state')=='complete':
            if old['source_index_sha256']!=cache['index_sha256']:raise RuntimeError('existing_packed_cohort_mismatch')
            return
        source=load(index_path)['train'];tasks=iter(sorted(source.items()))
        workers=min(32,int(os.environ.get('SLURM_CPUS_PER_TASK','16')))
        records={};offset=0;count=0;whole=hashlib.sha256();begun=time.monotonic()
        temporary=FOLDER/'train_rgb.private.bin.tmp'
        save(FOLDER/'status.json',dict(state='building',completed=0,total=len(source),updated_at_utc=now()))
        with temporary.open('wb') as output,concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            waiting=collections.deque()
            for _ in range(workers*2):
                item=next(tasks,None)
                if item is not None:waiting.append(executor.submit(decode,item))
            while waiting:
                sha,width,height,pixels=waiting.popleft().result()
                output.write(pixels);whole.update(pixels)
                records[sha]=dict(offset=offset,bytes=len(pixels),width=width,height=height,
                    pixel_sha256=hashlib.sha256(pixels).hexdigest())
                offset+=len(pixels);count+=1
                item=next(tasks,None)
                if item is not None:waiting.append(executor.submit(decode,item))
                if count%1024==0 or count==len(source):
                    save(FOLDER/'status.json',dict(state='building',completed=count,total=len(source),bytes=offset,
                        elapsed_seconds=time.monotonic()-begun,updated_at_utc=now()))
            output.flush();os.fsync(output.fileno())
        destination=FOLDER/'train_rgb.private.bin';temporary.replace(destination);os.chmod(destination,0o400)
        save(FOLDER/'index.private.json',dict(images=records))
        receipt=dict(state='complete',images=count,bytes=offset,packed_sha256=whole.hexdigest(),
            index_sha256=digest(FOLDER/'index.private.json'),source_index_sha256=cache['index_sha256'],
            train_manifest_sha256=cache['train_manifest_sha256'],
            pixels_identical_to_existing_rgb_cache=True,no_resize_or_augmentation=True,
            elapsed_seconds=time.monotonic()-begun,updated_at_utc=now())
        save(FOLDER/'receipt.json',receipt);save(FOLDER/'status.json',receipt)
        print('verified packed RGB cache complete; no private records exported')

if __name__=='__main__':
    try:build()
    except Exception as error:
        save(FOLDER/'status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise
