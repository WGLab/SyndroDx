"""Four fixed checkpoints on the identical, audit-released expert crops."""
import os,sys,json,hashlib,importlib.util
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;STAGE=PROJECT/'transface_arcface_identity_cpt_20261026'
def load(p):return json.loads(Path(p).read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n');os.chmod(p,0o600)
@torch.no_grad()
def main(arm):
    os.umask(0o077);torch.cuda.set_device(0);torch.set_num_threads(4);cv2.setNumThreads(1)
    release=load(ROOT/'release.json')
    if hashlib.sha256((ROOT/'private/released.json').read_bytes()).hexdigest()!=release['release_sha256']:raise RuntimeError('expert_release_changed')
    rows=load(ROOT/'private/released.json')['rows']
    if not rows:raise RuntimeError('no_expert_images_released')
    sys.path[:0]=[str(STAGE),str(STAGE/'dependencies')]
    spec=importlib.util.spec_from_file_location('expert_transface',STAGE/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.author_backbone(optimized=True)
    if arm in ['cpt','cpt_gmdb']:
        chosen=load(PROJECT/'gmdb_supervised_transface_20261026/selected/identity_checkpoint.json')
        if hashlib.sha256(Path(chosen['path']).read_bytes()).hexdigest()!=chosen['sha256']:raise RuntimeError('CPT_checkpoint_changed')
        s=torch.load(chosen['path'],map_location='cpu',mmap=True,weights_only=False);prefix='backbone.encoder.'
        model.load_state_dict({k[len(prefix):]:v for k,v in s['student'].items() if k.startswith(prefix)},strict=True)
    if arm in ['original_gmdb','cpt_gmdb']:
        case='transface_original' if arm=='original_gmdb' else 'transface'
        p=PROJECT/'gestaltmatcher_arc_comparison_20261027/cases'/case/'best.pt'
        s=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
        model.feature.load_state_dict(s['adaptation']['feature'],strict=True)
    model.cuda().eval().requires_grad_(False);pieces=[]
    for start in range(0,len(rows),128):
        images=[]
        for r in rows[start:start+128]:
            p=Path(r['crop_path'])
            if hashlib.sha256(p.read_bytes()).hexdigest()!=r['crop_sha256']:raise RuntimeError('expert_crop_changed')
            im=cv2.imread(str(p))
            if im is None:raise RuntimeError('expert_decode_failure')
            images.append(cv2.cvtColor(cv2.resize(im,(112,112)),cv2.COLOR_BGR2RGB))
        x=torch.from_numpy(np.stack(images)).cuda().permute(0,3,1,2).float().div_(127.5).sub_(1)
        with torch.autocast('cuda',dtype=torch.bfloat16):features=model(x)[0]
        features=F.normalize(features.float(),dim=-1)
        if not torch.isfinite(features).all():raise RuntimeError('nonfinite_expert_features')
        pieces.append(features.cpu().numpy())
        save(ROOT/(arm+'_status.json'),dict(state='extracting',images=start+len(images),total=len(rows)))
    np.save(ROOT/'private'/(arm+'.npy'),np.concatenate(pieces))
    save(ROOT/(arm+'_receipt.json'),dict(state='complete',images=len(rows),release_sha256=release['release_sha256']))
if __name__=='__main__':main(sys.argv[1])
