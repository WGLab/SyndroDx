"""Frozen Qwen classifier inference on every image of each expert patient."""
import sys,os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;Q=ROOT.parent/'cranioface_qwen_alignment_20261027';sys.path.insert(0,str(Q))
import numpy as np,torch
from PIL import Image
from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
from runtime import load,save,MODEL,digest,now
from model import Resampler
from features import vision
from classifier_model import classification_head,pooled_hidden
from sft import sequences
@torch.no_grad()
def main():
 os.umask(0o077);torch.set_num_threads(4);torch.cuda.set_device(0)
 arm=sys.argv[1];shard=int(sys.argv[2]);world=int(sys.argv[3]);patients=load(ROOT/'private/patients.json')['patients'];p=load(Q/'classifier/protocol.json')
 ck=load(Q/'classifier'/arm/'best_checkpoint.json')
 if digest(ck['path'])!=ck['sha256']:raise RuntimeError('fixed_classifier_changed')
 state=torch.load(ck['path'],map_location='cpu',weights_only=False)
 llm=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').requires_grad_(False)
 llm=get_peft_model(llm,LoraConfig(r=p['lora_rank'],lora_alpha=p['lora_alpha'],target_modules=p['lora_modules'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'));set_peft_model_state_dict(llm,state['lora']);llm.eval()
 connector=Resampler().cuda().eval();connector.load_state_dict(state['connector']);head=classification_head(classes=p['classes']).cuda().eval();head.load_state_dict(state['head']);del state
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);records=[]
 if arm=='fusion':
  visual_model=vision().cuda().eval();cache={};visual_model.encoder.norm.register_forward_hook(lambda m,i,o:cache.update(patches=o.detach()))
 else:
  with np.load(Q/'private/hpo_constant_prior.npz') as f:pt=torch.from_numpy(f['patches']).cuda().unsqueeze(0);gl=torch.from_numpy(f['global_features']).cuda().unsqueeze(0)
 for index in range(shard,len(patients),world):
  row=patients[index];logits=[]
  if arm=='fusion':
   for start in range(0,len(row['images']),2):
    selected=row['images'][start:start+2]
    if len(selected)==1:selected=selected*2
    images=[]
    for r in selected:
     if digest(r['crop_path'])!=r['crop_sha256']:raise RuntimeError('expert_crop_changed')
     with Image.open(r['crop_path']) as im:images.append(np.asarray(im.convert('RGB').resize((112,112)),dtype=np.uint8))
    x=torch.from_numpy(np.stack(images)).cuda().permute(0,3,1,2).float()/127.5-1
    with torch.autocast('cuda',dtype=torch.bfloat16):
     g=visual_model(x);pt=cache['patches'].reshape(1,-1,768);gl=torch.nn.functional.normalize(g.float(),dim=-1).mean(0,keepdim=True)
     z=connector(pt.float(),gl);logits.append(head(pooled_hidden(llm,z,sequences(tok,[row],targets=False))).float().cpu().numpy()[0])
  else:
   with torch.autocast('cuda',dtype=torch.bfloat16):logits.append(head(pooled_hidden(llm,connector(pt,gl),sequences(tok,[row],targets=False))).float().cpu().numpy()[0])
  s=np.mean(logits,axis=0)
  if not np.isfinite(s).all():raise RuntimeError('nonfinite_expert_classifier_scores')
  records.append(dict(index=index,scores=s.tolist()))
  save(ROOT/'features'/f'qwen_{arm}_{shard}.progress.json',dict(patients=len(records),updated_at_utc=now()))
 save(ROOT/'private'/f'qwen_{arm}_{shard}.json',dict(records=records))
 save(ROOT/'features'/f'qwen_{arm}_{shard}.json',dict(state='complete',patients=len(records),updated_at_utc=now()))
if __name__=='__main__':main()
