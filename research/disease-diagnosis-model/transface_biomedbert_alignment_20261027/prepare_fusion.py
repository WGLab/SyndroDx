"""Patient-level full observed HPO, including explicit absences, for diagnosis."""
import numpy as np
from transformers import AutoTokenizer
from common import *

def main():
    if load(WORK/'fusion_data_receipt.json').get('state')=='complete':return
    patients=load(WORK/'private/training_index.json')['patients'];terms=load(WORK/'private/ontology.json')['terms']
    classes=sorted({p['label'] for p in patients if p['split']=='train'});label_map={x:i for i,x in enumerate(classes)}
    features=np.load(WORK/'private/image_features.npy');texts=[];image=[]
    for p in patients:
        text='Observed findings: '+'; '.join(terms[h]['name'] for h in p['all_present_hpo'])+'.'
        if p['absent_hpo']:text+=' Explicitly absent findings: '+'; '.join(terms[h]['name'] for h in p['absent_hpo'])+'.'
        texts.append(text);f=features[p['feature_indices']].mean(0);image.append(f/max(np.linalg.norm(f),1e-12))
    tok=AutoTokenizer.from_pretrained(WORK/'assets/biomedbert',local_files_only=True)
    raw=tok(texts,padding='max_length',truncation=True,max_length=256,return_tensors='np')
    lengths=[len(tok(t,truncation=False)['input_ids']) for t in texts]
    np.savez(WORK/'private/fusion_arrays.npz',features=np.stack(image),input_ids=raw['input_ids'],attention_mask=raw['attention_mask'],
        labels=np.asarray([label_map.get(p['label'],-1) for p in patients]),
        train=np.asarray([i for i,p in enumerate(patients) if p['split']=='train']),
        validation=np.asarray([i for i,p in enumerate(patients) if p['split']=='validation' and p['label'] in label_map]))
    save(WORK/'fusion_data_receipt.json',dict(state='complete',classes=len(classes),train_patients=sum(p['split']=='train' for p in patients),
        validation_patients=sum(p['split']=='validation' and p['label'] in label_map for p in patients),
        unseen_validation_patients_excluded_from_closed_set_probe=sum(p['split']=='validation' and p['label'] not in label_map for p in patients),
        full_observed_patient_HPO=True,explicit_absence_separate=True,missing_is_not_absent=True,
        truncated_texts=sum(x>256 for x in lengths),text_tokens=256,clinical_tests_used=False,
        arrays_sha256=digest(WORK/'private/fusion_arrays.npz'),updated_at_utc=now()))

if __name__=='__main__':main()
