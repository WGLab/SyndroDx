"""Extract frozen selected vision features locally, without emitting images or rows."""
import sys
from pathlib import Path
NEW = Path(__file__).resolve().parent
sys.path.insert(0, str(NEW.parent / 'matched_craniofacial_benchmark_20261031'))
from common import *
from network import Encoder
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader


def main():
    os.umask(0o077)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    receipt = load(NEW / 'data_receipt.json')
    for name, sha in receipt['files'].items():
        assert digest(NEW / 'private' / name) == sha
    selection = load(NEW / 'visual_selection.json')
    assert digest(selection['path']) == selection['sha256']
    model = Encoder('transface_chop_cpt', 'arcface_full', load(WORK / 'audit.json')['diseases'])
    state = torch.load(selection['path'], map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.cuda().eval().requires_grad_(False)
    rows = load(NEW / 'private/chop_images.json')['images']
    class Inputs(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            r = rows[i]
            if digest(r['crop_path']) != r['crop_sha256']:
                raise RuntimeError('crop_integrity_changed')
            im = cv2.imread(r['crop_path'])
            if im is None: raise RuntimeError('local_image_decode_failed')
            im = cv2.cvtColor(cv2.resize(im, (112,112)), cv2.COLOR_BGR2RGB)
            return torch.from_numpy(im.transpose(2,0,1).copy()).float().div_(127.5).sub_(1)
    values = []
    with torch.inference_mode():
        for x in DataLoader(Inputs(), batch_size=128, num_workers=2, pin_memory=True):
            z = model.embed(x.cuda(non_blocking=True))
            assert torch.isfinite(z).all()
            values.append(z.cpu().numpy())
            save(NEW / 'features/status.json', dict(state='extracting', images_done=sum(len(v) for v in values), images_total=len(rows),updated_at_utc=now()))
    embeddings = np.concatenate(values)
    features = np.zeros((receipt['patient_groups'],512), np.float32)
    counts = np.zeros(len(features), np.int64)
    for r, z in zip(rows, embeddings):
        features[r['patient_index']] += z
        counts[r['patient_index']] += 1
    assert np.all(counts > 0)
    features /= counts[:,None]
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True),1e-12)
    with np.load(NEW / 'private/chop_terms.npz') as d:
        np.savez(NEW / 'private/chop.npz', features=features, ids=d['ids'], states=d['states'])
    save(NEW / 'features/status.json', dict(state='complete', patient_groups=len(features), images=len(rows), path=str(NEW / 'private/chop.npz'),sha256=digest(NEW / 'private/chop.npz'),updated_at_utc=now()))


if __name__ == '__main__':
    try: main()
    except Exception as e:
        save(NEW / 'features/failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        raise
