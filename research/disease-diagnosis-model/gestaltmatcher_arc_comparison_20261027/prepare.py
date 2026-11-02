"""One training release for every visual comparator; all test patients quarantined."""
import csv,collections,os,re
from common import *
def main():
    os.umask(0o077)
    if load(WORK/'data_receipt.json').get('state')=='complete':return
    ref=load(PREVIOUS/'data_receipt.json');source=load(PREVIOUS/'private/records.private.json')
    for name in ['private/images.uint8.npy','private/records.private.json']:
        if digest(PREVIOUS/name)!=ref['files'][name]:raise RuntimeError('upstream_GMDB_release_changed')
    protected=set()
    for name in ['gmdb_frequent_test_images','gmdb_rare_test_images']:
        with (SHARED/'split_snapshot'/(name+'.csv')).open() as f:protected.update(r['subject'] for r in csv.DictReader(f))
    metadata=Path('/controlled-data/GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata/image_metadata_v1.1.0.tsv')
    with metadata.open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if re.search(r'chop|philadelphia|bch|boston|childrens[.]harvard|childrenshospital[.]org',' '.join(r.get(k,'') for k in ['username','patient_group','email','author']),re.I):protected.add(r['patient_id'])
    blocked=set()
    for root,prefix in [(PROJECT/'expert_holdout_audit',''),(PROJECT/'split_plan_v0','ian_')]:
        with (root/(prefix+'blocked_crop_hashes.tsv')).open() as f:blocked.update(r['sha256'] for r in csv.DictReader(f,delimiter='\t'))
    protected.update(r['subject'] for r in source['rows'] if r['crop_sha256'] in blocked)
    # Keep the common official class indices; filtering does not rename classes.
    records=[dict(r,split='excluded') if r['subject'] in protected else r for r in source['rows']]
    train=[r['array_index'] for r in records if r['split']=='train'];val=[r for r in records if r['split']=='validation']
    if {records[i]['subject'] for i in train}&{r['subject'] for r in val}:raise RuntimeError('patient_overlap')
    classes=sorted({records[i]['class_index'] for i in train});remap={c:i for i,c in enumerate(classes)}
    removed_classes=len(source['labels'])-len(classes)
    val=[r for r in val if r['class_index'] in remap]
    for r in records:
        old=r['class_index'];r['class_index']=remap.get(old,-1)
        if old not in remap:r['split']='excluded'
    path=WORK/'private/images.uint8.npy'
    if not path.exists():os.link(PREVIOUS/'private/images.uint8.npy',path)
    save(WORK/'private/records.private.json',dict(rows=records,labels=[source['labels'][i] for i in classes]))
    save(WORK/'private/budget_100.private.json',dict(indices=train))
    files=['private/images.uint8.npy','private/records.private.json','private/budget_100.private.json']
    save(WORK/'data_receipt.json',dict(state='complete',patient_disjoint=True,train_images=len(train),train_patients=len({records[i]['subject'] for i in train}),validation_images=len(val),diseases=len(classes),
        test_patients_excluded_from_training=True,known_CHOP_BCH_sources_excluded=True,clinical_test_used=False,
        diseases_removed_by_protection=removed_classes,
        files={x:digest(WORK/x) for x in files},updated_at_utc=now()))
if __name__=='__main__':main()
