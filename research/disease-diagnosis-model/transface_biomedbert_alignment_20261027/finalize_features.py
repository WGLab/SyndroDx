"""Assemble each image once, tokenize local HPO names, and seal inputs."""
import numpy as np
from transformers import AutoTokenizer
from common import *

def main(world):
    receipt=load(WORK/'data_receipt.json');patients=load(WORK/'private/patients.json')['patients']
    required={i for p in patients for i in p['image_indices']}; byindex={}
    for rank in range(world):
        status=load(WORK/'features'/f'shard_{rank}.status.json');path=WORK/'features'/f'shard_{rank}.npz'
        if status.get('state')!='complete' or status.get('world')!=world or status.get('patient_file_sha256')!=receipt['patient_file_sha256'] or digest(path)!=status['sha256']:
            raise RuntimeError('incomplete_feature_shard')
        with np.load(path) as data:
            for idx,feat in zip(data['indices'],data['features']):
                if int(idx) in byindex:raise RuntimeError('duplicate_feature_assignment')
                byindex[int(idx)]=feat
    if set(byindex)!=required:raise RuntimeError('incomplete_feature_coverage')
    # Multiple clinical views share a patient-level text. Retain per-image embeddings for training sampling.
    indices=sorted(byindex);offset={v:i for i,v in enumerate(indices)}
    np.save(WORK/'private/image_features.npy',np.stack([byindex[i] for i in indices]))
    ids=sorted({h for p in patients for h in p['hpo']});mapping={h:i for i,h in enumerate(ids)}
    hpo=np.zeros((len(patients),len(ids)),np.float32)
    labels=sorted({p['label'] for p in patients});label_map={x:i for i,x in enumerate(labels)}
    text=AutoTokenizer.from_pretrained(WORK/'assets/biomedbert',local_files_only=True)
    raw=text([p['text'] for p in patients],padding='max_length',truncation=True,max_length=192,return_tensors='np')
    lengths=[len(text(p['text'],truncation=False)['input_ids']) for p in patients]
    for i,p in enumerate(patients):
        hpo[i,[mapping[h] for h in p['hpo']]]=1
        p['feature_indices']=[offset[x] for x in p['image_indices']]
    np.savez(WORK/'private/text_arrays.npz',input_ids=raw['input_ids'],attention_mask=raw['attention_mask'],
        hpo=hpo,labels=np.asarray([label_map[p['label']] for p in patients]))
    save(WORK/'private/training_index.json',dict(patients=patients))
    files=['private/image_features.npy','private/text_arrays.npz','private/training_index.json']
    save(WORK/'feature_receipt.json',dict(state='complete',images=len(indices),patients=len(patients),
        hpo_terms=len(ids),max_text_tokens=192,truncated_patient_texts=sum(n>192 for n in lengths),
        visual_checkpoint_step=2500,visual_backbone_frozen=True,files={p:digest(WORK/p) for p in files},updated_at_utc=now()))

if __name__=='__main__':
    import sys;main(int(sys.argv[1]))
