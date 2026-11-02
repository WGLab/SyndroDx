"""Prepare only GMDB train/development images and patient-balanced label budgets."""
import collections
import csv
import hashlib
import json
import os
import numpy as np
import cv2
from common import WORK,STAGE1,SHARED,load,save,digest,now

def label_budget(rows,fraction,seed):
    by_label=collections.defaultdict(set);patient_labels=collections.defaultdict(set)
    for row in rows:
        by_label[row['label']].add(row['subject']);patient_labels[row['subject']].add(row['label'])
    rng=np.random.default_rng(seed);selected=set()
    for label in sorted(by_label):
        patients=sorted(by_label[label]);order=rng.permutation(len(patients))
        n=max(1,int(np.ceil(len(patients)*fraction)))
        selected.update(patients[int(i)] for i in order[:n])
    return [r['array_index'] for r in rows if r['subject'] in selected],len(selected)

def main():
    os.umask(0o077);cv2.setNumThreads(1)
    if load(WORK/'data_receipt.json').get('state')=='complete':return
    reference=load(SHARED/'protocol.json')
    if digest(SHARED/'inputs.private.jsonl')!=reference['inputs_sha256']:raise RuntimeError('frozen_gmdb_inputs_changed')
    with (SHARED/'inputs.private.jsonl').open() as stream:inputs={r['image_id']:r for r in map(json.loads,stream)}
    tables={}
    for split,name in (('train','gmdb_train_images'),('validation','gmdb_val_images')):
        path=SHARED/'split_snapshot'/(name+'.csv')
        if digest(path)!=reference['split_sha256'][name]:raise RuntimeError('provided_split_changed')
        with path.open(newline='') as stream:tables[split]=list(csv.DictReader(stream))
    train_subjects={r['subject'] for r in tables['train']};val_subjects={r['subject'] for r in tables['validation']}
    if train_subjects&val_subjects:raise RuntimeError('train_validation_patient_overlap')
    labels=sorted({r['label'] for r in tables['train']})
    if any(r['label'] not in labels for r in tables['validation']):raise RuntimeError('unknown_validation_class')
    label_index={label:i for i,label in enumerate(labels)}
    rows=[]
    for split,records in tables.items():
        for row in records:
            original=inputs[row['image_id']]
            if original['label']!=row['label'] or original['subject']!=row['subject']:raise RuntimeError('input_split_identity_or_label_mismatch')
            rows.append(dict(original,split=split,class_index=label_index[row['label']],array_index=len(rows)))
    if len({r['image_id'] for r in rows})!=len(rows):raise RuntimeError('duplicate_train_validation_image')
    shape=(len(rows),112,112,3);tmp=WORK/'private/images.uint8.npy.tmp'
    array=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=shape)
    pixel_sets={'train':set(),'validation':set()}
    for i,row in enumerate(rows):
        if digest(row['crop_path'])!=row['crop_sha256']:raise RuntimeError('gmdb_source_image_changed')
        image=cv2.imread(row['crop_path'],cv2.IMREAD_COLOR)
        if image is None:raise RuntimeError('gmdb_source_decode_failure')
        image=cv2.cvtColor(cv2.resize(image,(112,112)),cv2.COLOR_BGR2RGB)
        array[i]=image;pixel_sets[row['split']].add(hashlib.sha256(image.tobytes()).hexdigest())
    array.flush();del array
    if pixel_sets['train']&pixel_sets['validation']:raise RuntimeError('identical_pixels_across_train_validation')
    destination=WORK/'private/images.uint8.npy';tmp.replace(destination);os.chmod(destination,0o400)
    save(WORK/'private/records.private.json',dict(rows=rows,labels=labels))
    plan=load(WORK/'protocol.json')['stage2'];budgets={}
    train=[r for r in rows if r['split']=='train']
    for fraction in plan['fractions']:
        indices,patients=label_budget(train,fraction,plan['seed'])
        token=str(int(round(fraction*100)))
        save(WORK/'private'/f'budget_{token}.private.json',dict(indices=indices))
        counts=np.bincount([rows[i]['class_index'] for i in indices],minlength=len(labels))
        if np.any(counts==0):raise RuntimeError('budget_loses_a_training_disease')
        budgets[token]=dict(requested_patient_fraction=fraction,patients=patients,images=len(indices),diseases=len(labels))
    files=['private/images.uint8.npy','private/records.private.json']+[f'private/budget_{k}.private.json' for k in budgets]
    save(WORK/'data_receipt.json',dict(state='complete',train_images=len(train),train_patients=len(train_subjects),
        validation_images=len(tables['validation']),validation_patients=len(val_subjects),diseases=len(labels),
        patient_disjoint=True,exact_resized_pixel_disjoint=True,budgets=budgets,files={name:digest(WORK/name) for name in files},
        shared_input_sha256=reference['inputs_sha256'],split_sha256={k:reference['split_sha256'][k] for k in ('gmdb_train_images','gmdb_val_images')},
        clinical_test_accessed=False,real_mrn_mapping_accessed=False,updated_at_utc=now()))
    print('GMDB train/development preparation complete; only aggregates are reported')

if __name__=='__main__':main()
