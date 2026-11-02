"""Admit training-only identity classes locally; export only aggregate receipts."""
import os
from control import WORK,V1,WEIGHTS,load,save,digest,now
from data import verify_manifest,validate_split_rows,cache_index,evaluation_cohort

def main():
    os.umask(0o077)
    p=load(WORK/'protocol.json');release=load(V1/'dataset_release/release.json')
    if not release.get('patient_split_disjoint') or not release.get('protected_pools_excluded'):
        raise RuntimeError('protected_patient_split_required')
    rows={s:verify_manifest(release['manifests'][s]) for s in ('train','validation','test')}
    report=validate_split_rows(rows['train'],rows['validation'],rows['test'])
    if release['manifests']['train']['sha256']!=p['train_manifest']['sha256']:
        raise RuntimeError('unexpected_training_manifest')
    ids=sorted({r['group_id'] for r in rows['train']})
    if len(ids)!=p['num_classes']:raise RuntimeError('unexpected_identity_class_count')
    path=WORK/p['class_map']
    value=dict(patient_groups_in_class_order=ids,train_manifest_sha256=p['train_manifest']['sha256'])
    if path.exists() and load(path)!=value:raise RuntimeError('existing_identity_map_changed')
    if not path.exists():save(path,value)
    p['class_map_sha256']=digest(path)
    save(WORK/'protocol.json',p)
    cache_index()
    records,_=evaluation_cohort()
    admitted=set(ids);development={r['group_id'] for r in records if r['split']=='validation'}
    if admitted&development:raise RuntimeError('development_patients_in_identity_training')
    if digest(WEIGHTS)!=p['model_sha256']:raise RuntimeError('public_initialization_digest_mismatch')
    save(WORK/'data_preflight.json',dict(state='passed',**report,identity_classes=len(ids),
        class_map_sha256=p['class_map_sha256'],development_overlap=0,test_images_used_for_training=False,
        private_records_exported=False,updated_at_utc=now()))
    print('training-only identity preflight passed; no private records exported')

if __name__=='__main__':main()
