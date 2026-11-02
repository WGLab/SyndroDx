"""Cache the same verified images once on HPC; no images leave this process."""
import cv2
import numpy as np
from common import *


def main():
    cv2.setNumThreads(1)
    receipt=load(WORK/'data_receipt.json');assert receipt['state']=='passed'
    assert digest(WORK/'private/image_groups.json')==receipt['image_groups_sha256']
    groups=load(WORK/'private/image_groups.json')['patients'];n=receipt['images']
    path=WORK/'private/images_rgb112.npy';tmp=path.with_name(path.name+'.tmp')
    arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(n,112,112,3));offsets={};j=0
    for key in sorted(groups,key=int):
        offsets[key]=[]
        for r in groups[key]:
            if digest(r['crop_path'])!=r['crop_sha256']:raise ValueError('crop_hash_changed')
            im=cv2.imread(r['crop_path'])
            if im is None:raise RuntimeError('decode_failed')
            arr[j]=cv2.cvtColor(cv2.resize(im,(112,112)),cv2.COLOR_BGR2RGB);offsets[key].append(j);j+=1
        if j%100<10:save(WORK/'cache_status.json',dict(state='caching_locally',images_done=j,images_total=n,updated_at_utc=now()))
    assert j==n
    arr.flush();del arr;tmp.replace(path)
    save(WORK/'private/patient_image_offsets.json',offsets)
    save(WORK/'cache_receipt.json',dict(state='passed',images=j,images_sha256=digest(path),offsets_sha256=digest(WORK/'private/patient_image_offsets.json'),
        transform='cv2 resize112 BGR-to-RGB identical to verified native feature extraction',images_exported=False,updated_at_utc=now()))
    print(json.dumps(dict(state='passed',cached_images=j)))


if __name__=='__main__':guarded(main,WORK/'cache_failure.json')
