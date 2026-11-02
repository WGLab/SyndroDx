"""Freeze the selected checkpoint and split files without changing training."""
import collections
import csv
import json
import os
import shutil
from pathlib import Path
from settings import *


def table(path):
    with path.open(newline='') as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        if any(not row.get(k) for k in ('image_id', 'subject', 'label')):
            raise RuntimeError('incomplete_official_split_row')
    return result


def main():
    os.umask(0o077)
    if (WORK/'protocol.json').exists():
        raise RuntimeError('evaluation_already_frozen_do_not_select_another_checkpoint')
    ready = []
    for plan in (TRAIN/'attempts').glob('*.json'):
        arm = load(plan)['arm'];folder = TRAIN/arm/'training'
        for path in (folder/'checkpoint_ready').glob('step_*.json'):
            row = load(path)
            ckpt = folder/'ckpt'/str(row['completed_steps']-1)
            if (ckpt/'.metadata').exists():ready.append((row['completed_steps'], folder, row))
    if not ready:raise RuntimeError('no_complete_validated_checkpoint_available')
    step, folder, boundary = max(ready, key=lambda x:x[0])
    source = folder/'ckpt'/str(step-1)
    if digest(source/'.metadata') != boundary['metadata_sha256']:
        raise RuntimeError('checkpoint_metadata_integrity_failed')
    if not load(folder/'validation'/f'step_{step:06d}'/'report.json').get('decision',{}).get('continue_training'):
        raise RuntimeError('selected_checkpoint_not_validated')
    target = WORK/'checkpoint_snapshot/ckpt'/str(step-1)
    target.mkdir(parents=True, exist_ok=False)
    for name, size in boundary['checkpoint_file_sizes'].items():
        p = source/name
        if Path(name).name!=name or not p.is_file() or p.stat().st_size!=size:
            raise RuntimeError('incomplete_checkpoint')
        os.link(p, target/name)
    shutil.copy2(folder/'config.yaml', WORK/'checkpoint_snapshot/config.yaml')
    save(WORK/'checkpoint_snapshot/receipt.json', dict(boundary, pinned_at_utc=now()))
    tables = {name:table(split_path(name)) for name in TABLES}
    destination = WORK/'split_snapshot';destination.mkdir()
    for name in TABLES:shutil.copy2(split_path(name), destination/(name+'.csv'))
    frequent_gallery = tables['gmdb_frequent_gallery_images'];frequent_test=tables['gmdb_frequent_test_images']
    rare_gallery=tables['gmdb_rare_gallery_images'];rare_test=tables['gmdb_rare_test_images']
    fg_ids={r['image_id'] for r in frequent_gallery}
    if fg_ids != {r['image_id'] for r in tables['gmdb_train_images']+tables['gmdb_val_images']}:
        raise RuntimeError('frequent_gallery_not_train_plus_validation')
    if len(fg_ids)!=len(frequent_gallery):raise RuntimeError('duplicate_gallery_image')
    splits=sorted({int(r['split']) for r in rare_gallery})
    if splits!=list(range(10)):raise RuntimeError('ten_official_rare_splits_required')
    for split in splits:
        gallery=[r for r in rare_gallery if int(r['split'])==split]
        queries=[r for r in rare_test if int(r['split'])==split]
        if {r['subject'] for r in gallery}&{r['subject'] for r in queries}:
            raise RuntimeError('rare_gallery_query_patient_overlap')
    all_rows=frequent_gallery+frequent_test+rare_gallery+rare_test
    labels={};subjects={}
    for row in all_rows:
        key=row['image_id']
        if key in labels and (labels[key]!=row['label'] or subjects[key]!=row['subject']):
            raise RuntimeError('inconsistent_split_image_identity_or_label')
        labels[key]=row['label'];subjects[key]=row['subject']
    files=collections.defaultdict(list)
    for path in CROPS.iterdir():
        if path.is_file() and path.suffix.lower() in ('.jpg','.jpeg','.png'):
            files[path.stem.split('_')[0]].append(path)
    if any(len(files[key])!=1 for key in labels):
        raise RuntimeError('missing_or_ambiguous_gmdb_crop_do_not_silently_subset')
    release=load(V1/'dataset_release/release.json')
    spec=release['manifests']['train'];manifest=Path(spec['path'])
    if digest(manifest)!=spec['sha256']:raise RuntimeError('ssl_training_manifest_modified')
    ssl_hashes=set();sources=collections.Counter();gmdb_source_count=0
    with manifest.open(newline='') as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            ssl_hashes.add(row['crop_sha256']);sources[row['source']]+=1
            gmdb_source_count+=int('gmdb' in row['crop_path'].lower())
    if gmdb_source_count:raise RuntimeError('gmdb_source_in_ssl_requires_leakage_audit')
    inputs=[];overlap=0
    for key in sorted(labels,key=int):
        path=files[key][0];sha=digest(path);overlap+=int(sha in ssl_hashes)
        inputs.append(dict(image_id=key,subject=subjects[key],label=labels[key],crop_path=str(path),crop_sha256=sha))
    if overlap:raise RuntimeError('exact_gmdb_image_in_ssl_training_requires_exclusion')
    with (WORK/'inputs.private.jsonl').open('w') as stream:
        for row in inputs:stream.write(json.dumps(row)+'\n')
    shared={r['subject'] for r in frequent_gallery}&{r['subject'] for r in frequent_test}
    exact_overlap=fg_ids&{r['image_id'] for r in frequent_test}
    protocol=dict(state='prepared',prepared_at_utc=now(),checkpoint_step=step,
        checkpoint_source_world=boundary['source_world'],frozen_backbone=True,gmdb_supervised_finetuning=False,
        models=['original_dinov3','craniofacial_cpt'],dataset_version='GMDB v1.1.0',
        split_provenance='provided local distributor-named files; not independently authenticated against a distributor checksum manifest',
        official_protocol_source='https://github.com/igsb/GestaltMatcher-Arc/blob/service/evaluate_ensemble.py',
        split_sha256={name:digest(destination/(name+'.csv')) for name in TABLES},
        inputs_sha256=digest(WORK/'inputs.private.jsonl'),images=len(inputs),
        frequent_gallery_images=len(frequent_gallery),frequent_query_images=len(frequent_test),
        frequent_gallery_query_shared_subjects=len(shared),frequent_gallery_query_shared_image_ids=len(exact_overlap),
        frequent_queries_excluded_from_patient_disjoint_supplement=sum(r['subject'] in shared for r in frequent_test),
        rare_splits=10,metrics='macro per syndrome Top-1/5/10/30; mean across ten rare splits',
        protocols=['provided_split_literal','patient_disjoint_query_supplement'],
        strict_supplement='exclude queries whose subject occurs in its gallery; do not silently alter provided files',
        settings=['frequent_to_frequent','rare_to_rare','frequent_to_combined','rare_to_combined'],
        preprocessing='provided aligned crops; bicubic aspect-preserving resize including upsampling to longest-side 256; mean-color padding; checkpoint RGB normalization',
        tta=False,similarity='cosine; deduplicate disease labels by closest gallery image',
        checkpoint_selection='latest complete validated checkpoint before this one-off evaluation; not selected by GMDB results',
        ssl_source_counts=dict(sources),ssl_exact_crop_hash_overlap=overlap,
        overlap_limitations='exact byte matching only; cross-center identity and near-duplicate photo overlap not established',
        test_use='one-off exploratory diagnostic; future selection uses development data, not this test score',
        official_arc_supervision_difference='Arc models are GMDB-trained; these two encoders are frozen without GMDB supervised adaptation; no matched-training superiority claim',
        patient_data_remains_on_hpc=True,external_apis=False)
    save(WORK/'protocol.json',protocol)
    print(json.dumps({k:protocol[k] for k in ['state','checkpoint_step','images','frequent_gallery_images','frequent_query_images','frequent_gallery_query_shared_subjects','frequent_gallery_query_shared_image_ids','rare_splits','ssl_exact_crop_hash_overlap']}))

if __name__=='__main__':main()
