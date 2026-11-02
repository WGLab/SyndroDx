"""Reuse approved patient splits; never export patient records or clinical media."""
import collections
import numpy as np
from common import *

BASE = PROJECT / 'hpo_fusion_staged_20261031'

def main():
    os.umask(0o077)
    receipt = load(BASE/'data_receipt.json')
    for name in ['gmdb.npz', 'term_index.json']:
        assert digest(BASE/'private'/name) == receipt['files'][name]
    pins = load(MATCH/'data_pins.json')
    assert digest(SHARED/'inputs.private.jsonl') == pins['public_inputs_sha256']
    assert digest(MATCH/'private/blocked_patients.json') == pins['blocked_sha256']
    assert load(MATCH/'audit.json')['state'] == 'passed'
    old = load(OLD/'data_receipt.json')
    assert digest(OLD/'private/patients.json') == old['patient_file_sha256']
    patients = load(PROJECT/'hpo_fusion_public_20261031/private/patient_index.json')['patients']
    prior = {p['subject']: p for p in load(OLD/'private/patients.json')['patients']}
    mapping = {r['array_index']: r for r in load(PROJECT/'gmdb_supervised_transface_20261026/private/records.private.json')['rows']}
    source = [json.loads(line) for line in (SHARED/'inputs.private.jsonl').open()]
    by_image = {r['image_id']:r for r in source}
    assert len(by_image) == len(source)
    blocked = set(load(MATCH/'private/blocked_patients.json')['subjects'])
    with np.load(BASE/'private/gmdb.npz') as d:
        train, val = d['train'].copy(), d['validation'].copy()
        labels = d['labels'].copy()
        assert len(patients) == len(labels)
    sets = [{patients[i]['subject'] for i in ix} for ix in [train,val]]
    assert not sets[0] & sets[1] and not (sets[0]|sets[1]) & blocked
    assert len(sets[0]) == len(train) and len(sets[1]) == len(val)
    import csv
    held = set()
    for split in ['gmdb_frequent_test_images','gmdb_rare_test_images']:
        with (SHARED/'split_snapshot'/f'{split}.csv').open() as f:
            held.update(r['subject'] for r in csv.DictReader(f))
    assert not (sets[0]|sets[1]) & held
    hashes = [set(),set()]
    image_rows = []
    cap = 4
    truncated = 0
    for role, indices in enumerate([train,val]):
        for i in indices:
            p = patients[i]; oldp = prior[p['subject']]
            assert p['split'] == oldp['split'] and p['label'] == oldp['label']
            rows = [by_image[mapping[k]['image_id']] for k in oldp['image_indices']]
            assert rows and all(r['subject'] == p['subject'] for r in rows)
            rows = sorted(rows,key=lambda r:hashlib.sha256(str(r['image_id']).encode()).hexdigest())
            truncated += len(rows)>cap
            for j,r in enumerate(rows):
                hashes[role].add(r['crop_sha256'])
                image_rows.append(dict(r,patient_index=int(i),token_slot=j if j<cap else -1))
    assert not hashes[0] & hashes[1]
    test_hashes={r['crop_sha256'] for r in source if r['subject'] in held|blocked}
    assert not (hashes[0]|hashes[1]) & test_hashes
    assert len({r['image_id'] for r in image_rows})==len(image_rows)
    save(WORK/'private/images.json',dict(images=image_rows))
    ep=BASE/'features/term_embeddings.npy'
    assert digest(ep)==load(BASE/'term_embedding_receipt.json')['sha256']
    visual=load(BASE/'visual_selection.json')
    assert digest(visual['path'])==visual['sha256']
    hp={}
    for seed in [11,29,47,83]:
        r=load(BASE/'cases'/f'public_pretrained_knowledge__hpo__seed{seed}'/'result.json')
        assert r['state']=='complete' and digest(r['checkpoint_path'])==r['checkpoint_sha256']
        hp[str(seed)] = dict(path=r['checkpoint_path'],sha256=r['checkpoint_sha256'])
    protocol=dict(state='locked_before_training',seeds=[11,29,47,83],
        modes=['conditional_local','unconditioned_local','conditional_global'],
        max_epochs=50,min_epochs=15,patience=10,batch_size=64,learning_rate=1e-4,weight_decay=.01,
        width=256,queries=8,layers=2,heads=8,dropout=.1,max_images=cap,
        max_local_tokens_per_image=144,visual_token_width=768,
        initialization='frozen selected CHOP CPT -> GMDB ArcFace visual encoder; frozen independently trained HPO branch per seed',
        objective='class-weighted disease cross entropy only; train query connector and residual output projection',
        selection='GMDB development disease-macro Top1 then Top5; same patients and 216 candidate diseases in all arms',
        limitation='HPO checkpoints and earlier fusion settings have already been selected on this development set; all new development gains exploratory',
        image_policy='all original eligible images for native patient mean; at most four deterministic hash-selected images for local tokens; same patients in every arm',
        view_policy='GMDB audit does not provide verified view labels; multi-image tokens with positional encoding, no fabricated frontal/profile labels',
        hpo_policy='present and explicit absence only; missing never absent; weak CHOP recorded-state pairs excluded',
        test_policy='No GMDB test, CHOP expert or BCH optimization/selection or inference in this stage; BCH identity manifest unavailable',
        network_during_patient_processing=False,clinical_media_export=False,private_MRN_mapping_read=False,
        immutable_baseline_manifest_sha256=load(PROJECT/'model_releases/CURRENT.json')['sha256'],
        visual=visual,updated_at_utc=now())
    save(WORK/'protocol.json',protocol)
    save(WORK/'data_receipt.json',dict(state='passed',gmdb_path=str(BASE/'private/gmdb.npz'),
        gmdb_sha256=digest(BASE/'private/gmdb.npz'),images_sha256=digest(WORK/'private/images.json'),
        patient_index_sha256=digest(PROJECT/'hpo_fusion_public_20261031/private/patient_index.json'),
        term_embeddings_path=str(ep),term_embeddings_sha256=digest(ep),hpo=hp,counts=receipt['counts'],
        patients_total=len(patients),train_patients=len(train),validation_patients=len(val),
        images=len(image_rows),patients_over_image_cap=truncated,patient_overlap=0,exact_crop_overlap=0,
        inherited_audit_sha256=digest(MATCH/'audit.json'),BCH_cross_source_identity_audit='manifest_unavailable',updated_at_utc=now()))

if __name__=='__main__':
    try: main()
    except Exception as e:
        save(WORK/'preparation_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        raise
