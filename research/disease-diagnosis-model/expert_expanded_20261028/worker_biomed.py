"""Frozen BiomedBERT controls on the same expert patients and full image sets."""
import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;B=ROOT.parent/'transface_biomedbert_alignment_20261027';sys.path.insert(0,str(B))
import numpy as np,torch
from PIL import Image
from transformers import AutoTokenizer
from common import load,save,now,digest
from fusion_model import Fusion
from vision import build_vision
@torch.no_grad()
def main():
 os.umask(0o077);torch.set_num_threads(4);torch.cuda.set_device(0)
 arm=sys.argv[1];shard=int(sys.argv[2]);world=int(sys.argv[3]);patients=load(ROOT/'private/patients.json')['patients']
 model=Fusion(216,arm).cuda().eval();s=torch.load(B/'fusion'/arm/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(s['model']);del s
 vision=build_vision().cuda().eval() if arm=='fusion' else None;tok=AutoTokenizer.from_pretrained(B/'assets/biomedbert',local_files_only=True);records=[];terms=load(B/'private/ontology.json')['terms']
 for index in range(shard,len(patients),world):
  row=patients[index];vectors=[]
  if vision is not None:
   for start in range(0,len(row['images']),32):
    ims=[]
    for r in row['images'][start:start+32]:
     if digest(r['crop_path'])!=r['crop_sha256']:raise RuntimeError('expert_crop_changed')
     with Image.open(r['crop_path']) as im:ims.append(np.asarray(im.convert('RGB').resize((112,112)),dtype=np.uint8))
    x=torch.from_numpy(np.stack(ims)).cuda().permute(0,3,1,2).float()/127.5-1
    with torch.autocast('cuda',dtype=torch.bfloat16):vectors.append(torch.nn.functional.normalize(vision(x).float(),dim=-1))
   features=torch.nn.functional.normalize(torch.cat(vectors).mean(0,keepdim=True),dim=-1)
  else:features=torch.zeros(1,512,device='cuda')
  text='Observed findings: '+'; '.join(terms[h]['name'] for h in row['hpo'])+'.'
  encoded=tok(text,return_tensors='pt',truncation=True,max_length=256).to('cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):scores=model(features,encoded['input_ids'],encoded['attention_mask']).float().cpu().numpy()[0]
  if not np.isfinite(scores).all():raise RuntimeError('nonfinite_expert_classifier_scores')
  records.append(dict(index=index,scores=scores.tolist()))
 save(ROOT/'private'/f'biomed_{arm}_{shard}.json',dict(records=records))
 save(ROOT/'features'/f'biomed_{arm}_{shard}.json',dict(state='complete',patients=len(records),updated_at_utc=now()))
if __name__=='__main__':main()
