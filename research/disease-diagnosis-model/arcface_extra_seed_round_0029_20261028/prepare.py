"""Pin common data and protocol. Print only aggregate checks."""
import os, collections
from matched_runtime import *
from data import verify_manifest, validate_split_rows, cache_index, evaluation_cohort

def main():
    os.umask(0o077)
    if (ROOT/'protocol.json').exists():
        if load(ROOT/'audit.json').get('state')!='passed':raise RuntimeError('incomplete_preparation')
        return
    release=load(V1/'dataset_release/release.json')
    assert release['patient_split_disjoint'] and release['protected_pools_excluded']
    rows={s:verify_manifest(release['manifests'][s]) for s in ['train','validation','test']}
    audit=validate_split_rows(rows['train'],rows['validation'],rows['test'])
    old=load(ID/'protocol.json');cache_index()
    assert digest(WEIGHTS)==old['model_sha256']
    assert digest(ID/old['class_map'])==old['class_map_sha256']
    groups=set(r['group_id'] for r in rows['train'])
    internal,_=evaluation_cohort()
    assert not groups&{r['group_id'] for r in internal if r['split']=='validation'}
    receipt=load(PAIRS/'data_receipt.json')
    assert digest(PAIRS/'private/patients.json')==receipt['patient_file_sha256']
    source=load(GMDB/'data_receipt.json')
    assert digest(GMDB/'private/records.private.json')==source['files']['private/records.private.json']
    records=load(GMDB/'private/records.private.json')['rows']
    by_index={r['array_index']:r for r in records}
    assert len(by_index)==len(records)
    train_hashes={r.get(k) for r in rows['train'] for k in ['crop_sha256','pixel_sha256','source_sha256'] if r.get(k)}
    blocked={r['subject'] for r in records if r['crop_sha256'] in train_hashes}
    patients=[p for p in load(PAIRS/'private/patients.json')['patients'] if p['subject'] not in blocked]
    for p in patients:
        assert p['split'] in ['train','validation']
        for i in p['image_indices']:
            r=by_index[i]
            assert r['subject']==p['subject'] and r['split']==p['split'] and r['label']==p['label']
    assert not {p['subject'] for p in patients if p['split']=='train'}&{p['subject'] for p in patients if p['split']=='validation'}
    save(ROOT/'private/gmdb_patients.json',dict(patients=patients))
    protocol=dict(created_at_utc=now(),state='locked',initialization=str(WEIGHTS),initialization_sha256=digest(WEIGHTS),
        arms={a:dict(objective='arcface_identity' if a.startswith('arcface') else 'dino_two_view_ssl',
            backbone_lr=5e-7 if a.endswith('5e7') else 2e-6,head_lr=1e-3 if a.startswith('arcface') else 1e-4) for a in ARMS},
        original_baseline=True,global_patient_draws_per_step=3840,views_per_draw=2,max_steps=4000,
        validation_steps=[0,100,250,500,1000,1500,2000,2500,3000,3500,4000],checkpoint_every=250,
        seed=20261028,head_only_steps=100,backbone_ramp_steps=500,backbone_weight_decay=.01,
        lr_schedule='fixed LR after warmup, no validation-dependent LR changes; equal backbone LR grid',
        teacher_momentum=.996,teacher_temp_start=.04,teacher_temp_end=.07,teacher_temp_warmup=1000,
        student_temp=.1,center_momentum=.9,dino_output_dim=65536,gradient_clip=3.,
        arcface_scale=64.,arcface_margin=.5,num_identity_classes=old['num_classes'],
        common_sampling='same deterministic patient/view/image stream and two augmentations per draw in both objectives',
        matched_budget='same 15,360,000 patient-image draws and 30,720,000 augmented student inputs at step4000; teacher compute reported separately',
        augmentation='112 RGB, crop .85-1 ratio .95-1.05, mild color jitter, no flip, no local crops',
        ssl_description='DINO-style adaptation to TransFace; not an official TransFace SSL recipe or a test of all SSL methods',
        deployment_embedding='512-dimensional normalized original/ArcFace student/DINO EMA teacher; no training heads at evaluation',
        train_manifest=release['manifests']['train'],class_map_sha256=old['class_map_sha256'],
        gmdb_patient_sha256=digest(ROOT/'private/gmdb_patients.json'),
        gmdb_image_array_sha256=source['files']['private/images.uint8.npy'],
        source_pairs_sha256=receipt['patient_file_sha256'],internal_validation_plan_sha256=digest(BASE/'validation/plan.json'),
        selection='GMDB development macro patient retrieval Top1; report Top5, fixed-seed 1/5/10-shot prototypes, internal probes and HPO separately; original remains a candidate',
        HPO='frozen image encoder; per-term probe uses documented positive and explicit negative assertions only; missing is unknown',
        fewshot='support only from GMDB train, queries only GMDB validation; same support draws for every arm; no disease-supervised CPT',
        tests_used=False,expert_used=False,BCH_used=False,
        limitations=['Development cohorts have been repeatedly inspected.','Cross-institution patient identity crosswalk is incomplete; this is not strict unseen-image or unseen-disease external validation.','Two objectives and two backbone learning rates do not exhaust SSL or identity-training recipes.'],
        safety='nonfinite loss/gradients stop the affected arm; severe collapse or repeated statistically supported decline triggers inspection; allocations retained',
        trackers_disabled=True,private_records_exported=False)
    save(ROOT/'protocol.json',protocol)
    save(ROOT/'audit.json',dict(state='passed',**audit,protected_pools_excluded=True,development_patient_overlap=0,
        gmdb_patients={s:sum(p['split']==s for p in patients) for s in ['train','validation']},
        gmdb_exact_overlap_patients_quarantined=len(blocked),cross_source_identity_resolution_complete=False,
        image_array_hash_verification='verified again by GPU evaluation before reading',updated_at_utc=now()))
    for a in ARMS:
        for folder in ['checkpoints','evaluations','logs']:(ROOT/a/folder).mkdir(parents=True,exist_ok=True)
    (ROOT/'logs').mkdir(exist_ok=True)
    print('Matched comparison preparation passed; private records remain on HPC.')

if __name__=='__main__':main()
