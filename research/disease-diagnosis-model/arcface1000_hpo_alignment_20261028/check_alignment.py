"""Development-only pairing permutation test, with fixed HPO semantic targets."""
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from model import Alignment,targets

@torch.no_grad()
def main():
    torch.cuda.set_device(0);torch.set_num_threads(4)
    patients=load(WORK/'private/training_index.json')['patients'];ids=[i for i,p in enumerate(patients) if p['split']=='validation']
    f=torch.from_numpy(np.load(WORK/'private/image_features.npy')).cuda()
    with np.load(WORK/'private/text_arrays.npz') as a:arrays={k:torch.from_numpy(a[k]).cuda() for k in a.files}
    model=Alignment().cuda().eval();best=load(WORK/'best_checkpoint.json')
    if digest(best['path'])!=best['sha256']:raise RuntimeError('checkpoint_changed')
    model.load_state_dict(torch.load(best['path'],map_location='cpu',weights_only=False)['model'])
    ims=[];txts=[]
    for start in range(0,len(ids),64):
        ix=ids[start:start+64];x=F.normalize(torch.stack([f[patients[i]['feature_indices']].mean(0) for i in ix]),dim=-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):im,txt,_=model(x,arrays['input_ids'][ix],arrays['attention_mask'][ix])
        ims.append(im.float());txts.append(txt.float())
    scores=(torch.cat(ims)@torch.cat(txts).T).cpu().numpy()
    target,_=targets(arrays['hpo'][ids],arrays['labels'][ids]);positive=(target>0).cpu().numpy()
    observed=float(positive[np.arange(len(ids)),scores.argmax(1)].mean())
    rng=np.random.default_rng(20261028);null=[]
    for _ in range(2000):
        shuffled=scores[rng.permutation(len(ids))]
        null.append(float(positive[np.arange(len(ids)),shuffled.argmax(1)].mean()))
    baseline=load(WORK/'baseline_alignment.json');validation=best['validation']
    pvalue=(1+sum(x>=observed for x in null))/(len(null)+1)
    passed=(pvalue<.01 and validation['loss']<baseline['loss']*.95 and validation['image_embedding_effective_rank']>baseline['image_embedding_effective_rank']*.8)
    save(WORK/'alignment_gate.json',dict(state='passed' if passed else 'needs_inspection',
        semantic_top1=observed,random_pairing_mean=float(np.mean(null)),random_pairing_95_interval=np.quantile(null,[.025,.975]).tolist(),
        pairing_permutation_p=pvalue,repetitions=2000,baseline_validation_loss=baseline['loss'],validation_loss=validation['loss'],
        method='Shuffle image-to-patient associations against fixed recorded HPO targets; exploratory development check',
        confirms_clinical_utility=False,clinical_tests_used=False,checkpoint_sha256=best['sha256'],updated_at_utc=now()))

if __name__=='__main__':main()
