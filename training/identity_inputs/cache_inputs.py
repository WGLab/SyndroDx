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
