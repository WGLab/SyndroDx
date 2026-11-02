"""GPU numerical/gradient/memory test; never export clinical media or rows."""
import gc
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from data import arrays,Patients,collate
from model import JointModel


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,groups=arrays();p=load(WORK/'protocol.json');reports={}
    # A fixed train-only batch with the most images exercises worst-case pooling.
    train=sorted(map(int,data['train']),key=lambda i:(-len(groups[str(i)]),i))
    micro=p['preferred_micro_patients']
    while micro>=1:
        try:
            for arm in p['arms']:
                save(WORK/'smoke_progress.json',dict(state='checking_arm',arm=arm,micro_patients=micro,updated_at_utc=now()))
                torch.manual_seed(11);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                m=JointModel(arm).cuda();m.eval()
                ds=Patients(train[:micro],groups);ix,x,c=collate([ds[i] for i in range(len(ds))]);x=x.cuda()
                with torch.no_grad():z=m.patient_features(x,c,p['image_chunk'])
                ref=torch.from_numpy(data['features'][ix]).cuda()
                error=float((z-ref).abs().max());cos=float(F.cosine_similarity(z,ref).min())
                save(WORK/'smoke_progress.json',dict(state='initialization_parity',arm=arm,micro_patients=micro,parity_max_abs_error=error,parity_min_cosine=cos,updated_at_utc=now()))
                if error>3e-5 or cos<.99999:raise RuntimeError('native_feature_initialization_parity_failed')
                before=m.vision.body.patch_embed.proj.weight.detach().clone()
                m.train()
                ids=torch.from_numpy(data['ids'][ix]).cuda();states=torch.from_numpy(data['states'][ix]).cuda();y=torch.from_numpy(data['labels'][ix]).cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    zz=m.patient_features(x,c,p['image_chunk']);logits,_=m.fusion(zz,ids,states);loss=F.cross_entropy(logits.float(),y)
                loss.backward()
                trainable=[q for q in m.parameters() if q.requires_grad]
                if any(q.grad is not None and not torch.isfinite(q.grad).all() for q in trainable):raise FloatingPointError('nonfinite_gradient')
                early_grad=m.vision.body.patch_embed.proj.weight.grad
                if arm!='full' and early_grad is not None:raise RuntimeError('unexpected_early_backbone_gradient')
                if arm=='full' and (early_grad is None or not early_grad.abs().sum()):raise RuntimeError('missing_full_backbone_gradient')
                last=m.vision.body.blocks[-1].attn.qkv.weight.grad
                if arm=='frozen' and last is not None:raise RuntimeError('frozen_backbone_received_gradient')
                if arm!='frozen' and (last is None or not last.abs().sum()):raise RuntimeError('missing_unfrozen_gradient')
                if m.fusion.terms.weight.grad is not None:raise RuntimeError('term_embeddings_changed')
                opt=torch.optim.AdamW(trainable,lr=1e-6);opt.step()
                if arm!='full' and not torch.equal(before,m.vision.body.patch_embed.proj.weight):raise RuntimeError('frozen_parameter_changed')
                reports[arm]=dict(parameter_counts=m.parameter_counts(),worst_train_batch_images=len(x),
                    parity_max_abs_error=error,parity_min_cosine=cos,
                    peak_memory_GiB=torch.cuda.max_memory_allocated()/1024**3,finite_loss=float(loss),gradient_checks_passed=True)
                del m,opt,x,z,zz,ref,logits,loss,before,early_grad,last,trainable
                gc.collect();torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            # Retry all arms at the same smaller microbatch; no arm gets a distinct loss batch.
            reports={};micro//=2;gc.collect();torch.cuda.empty_cache()
            if micro<1:raise
    result=dict(state='passed',micro_patients=micro,global_patient_batch=p['global_patient_batch'],image_chunk=p['image_chunk'],arms=reports,
                source_weights_modified=False,training_cases_started=False,updated_at_utc=now())
    save(WORK/'smoke.json',result);print(json.dumps(result))


if __name__=='__main__':guarded(main,WORK/'smoke_failure.json')
