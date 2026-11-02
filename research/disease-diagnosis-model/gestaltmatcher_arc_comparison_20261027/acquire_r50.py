"""Polite durable retries of the two official public R50 Glint360K links."""
import os,sys,time,fcntl
from common import *
sys.path.insert(0,str(PROJECT/'arcface_glint360k_eval_20261026/dependencies'))
import gdown,onnx

def main():
    os.umask(0o077);lock=(WORK/'assets/download.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    target=WORK/'assets/glint360k_r50.onnx'
    while not target.exists():
        for fileid in ['1MpRhM76OQ6cTzpr2ZSpHp2_CP19Er4PI','164o2Ct42tyJdQjckeMJH2-7KTXolu-EP']:
            tmp=target.with_suffix('.download')
            try:
                gdown.download(id=fileid,output=str(tmp),quiet=True,use_cookies=False)
                model=onnx.load(str(tmp));onnx.checker.check_model(model)
                if model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value!=512 or tmp.stat().st_size<100000000:
                    raise RuntimeError('unexpected_public_model_shape')
                save(WORK/'assets/r50_provenance.json',dict(source='InsightFace official model zoo Google Drive',file_id=fileid,
                    sha256=digest(tmp),bytes=tmp.stat().st_size,onnx_structure_checked=True,updated_at_utc=now()))
                tmp.replace(target);save(WORK/'assets/download_status.json',dict(state='complete',updated_at_utc=now()));return
            except Exception as e:
                save(WORK/'assets/download_status.json',dict(state='waiting_for_public_download_availability',error_type=type(e).__name__,
                    next_retry_seconds=21600,updated_at_utc=now()))
        time.sleep(21600)

if __name__=='__main__':main()
