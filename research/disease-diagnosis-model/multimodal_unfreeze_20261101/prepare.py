"""Verify existing patient/image guards; emit aggregate counts only."""
import collections
import csv
import numpy as np
from common import *


def main():
    for name in ['private', 'logs', 'cases']: (WORK / name).mkdir(exist_ok=True)
    registry = load(PROJECT / 'model_releases/CURRENT.json')
    manifest_path = Path(registry['manifest'])
    assert digest(manifest_path) == registry['sha256']
    manifest = load(manifest_path)
    assets = {}
    for name in ['visual_encoder', 'reference_11', 'HPO_term_embeddings', 'HPO_term_index', 'disease_vocabulary', 'code_model.py', 'evidence_input_receipt.json', 'visual_classifier_class_indices']:
        item = manifest['artifacts'][name]
        path = manifest_path.parent / item['path']
        assert digest(path) == item['sha256']
        assets[name] = dict(path=str(path), sha256=item['sha256'])
    old = load(QUERY / 'data_receipt.json')
    assert old['state'] == 'passed'
    image_path = QUERY / 'private/images.json'
    assert digest(image_path) == old['images_sha256']
    gmdb = Path(old['gmdb_path']); assert digest(gmdb) == old['gmdb_sha256']
    patient_path = PROJECT / 'hpo_fusion_public_20261031/private/patient_index.json'
    assert digest(patient_path) == old['patient_index_sha256']
    patients = load(patient_path)['patients']; images = load(image_path)['images']
    pins = load(MATCH / 'data_pins.json')
    assert digest(MATCH / 'private/blocked_patients.json') == pins['blocked_sha256']
    blocked = set(load(MATCH / 'private/blocked_patients.json')['subjects'])
    splitroot = PROJECT / 'gmdb_checkpoint_eval_20261026/split_snapshot'
    held = set()
    for name in ['gmdb_frequent_test_images', 'gmdb_rare_test_images']:
        p = splitroot / (name + '.csv'); assert digest(p) == pins['splits_sha256'][name]
        with p.open() as f: held.update(r['subject'] for r in csv.DictReader(f))
    with np.load(gmdb) as d:
        train, val = d['train'].copy(), d['validation'].copy()
        labels = d['labels'].copy(); assert len(labels) == len(patients)
    vocabulary = load(assets['disease_vocabulary']['path'])['labels']
    groups = [{patients[i]['subject'] for i in ix} for ix in [train, val]]
    assert len(groups[0]) == len(train) and len(groups[1]) == len(val)
    assert not groups[0] & groups[1] and not (groups[0] | groups[1]) & (blocked | held)
    image_by_patient = collections.defaultdict(list)
    hash_sets = [set(), set()]
    active = set(train) | set(val)
    train_set = set(train)
    for r in images:
        i = r['patient_index']; assert i in active
        assert r['subject'] == patients[i]['subject']
        assert vocabulary[labels[i]] == patients[i]['label']
        assert digest(r['crop_path']) == r['crop_sha256']
        image_by_patient[i].append(r)
        hash_sets[0 if i in train_set else 1].add(r['crop_sha256'])
    assert not hash_sets[0] & hash_sets[1]
    assert all(image_by_patient[i] for i in active)
    assert len({r['image_id'] for r in images}) == len(images)
    # The prior verified input list also includes held-out crops; read locally to
    # reconfirm that no crop-identical protected/test patient enters this trial.
    source = PROJECT / 'gmdb_checkpoint_eval_20261026/inputs.private.jsonl'
    assert digest(source) == pins['public_inputs_sha256']
    forbidden_hashes = set()
    with source.open() as f:
        for line in f:
            r = json.loads(line)
            if r['subject'] in held | blocked: forbidden_hashes.add(r['crop_sha256'])
    assert not (hash_sets[0] | hash_sets[1]) & forbidden_hashes
    counts = [len(image_by_patient[i]) for i in active]
    profiles = BASE / 'private/profiles.npz'
    assert digest(profiles) == load(BASE / 'data_receipt.json')['files']['profiles.npz']
    save(WORK / 'private/image_groups.json', {'patients': {str(i): sorted(rs, key=lambda r: str(r['image_id'])) for i, rs in image_by_patient.items()}})
    protocol = dict(state='locked_before_training', seeds=[11,29,47], arms=['frozen','partial','full'],
        initialization='same locked v1 visual and fusion seed11 weights in all nine runs; seeds change optimization randomness only',
        control='frozen arm receives identical extra optimization budget; epoch0 locked v1 retained as reference',
        partial='last 4 of 24 transformer blocks, final norm, SENet, and native512 output projection; earlier layers frozen',
        text='fixed BiomedBERT term embeddings; HPO set encoder, cross-attention, gate and classifier trainable in all arms',
        fusion_learning_rate=1e-5, visual_learning_rate=1e-6, weight_decay=.01,
        global_patient_batch=128, preferred_micro_patients=16, image_chunk=32,
        max_epochs=20, min_epochs=5, patience=5, warmup_epochs=1,
        image_policy='all eligible images per patient; normalize each embedding, mean, renormalize; identical 112 RGB preprocessing in all arms',
        image_augmentation='none in this isolated continuation experiment, matching the prior cached-feature inputs',
        backbone_stochastic_layers='eval mode for all arms; gradients enabled according to arm; frozen BN statistics; activation checkpointing for trainable blocks',
        objective='class-weighted disease CE + 0.5 HPO-only auxiliary CE + 0.1 public profile loss',
        HPO_dropout=.1, face_dropout=.1, HPO_modality_dropout=.1,
        knowledge_microbatch='same micro-patient size for every arm; profile negatives within microbatch',
        selection='GMDB development disease-macro Top1, then Top5; retain epoch0 if continuation does not improve',
        guard='two consecutive development checks with disease-bootstrap upper95 delta below -0.02 for diagnosis Top1 or native retrieval Top1; no loss-only stopping',
        evaluation='full/1/3/5 HPO diagnosis; shuffled-face control; native patient retrieval; 1/5-shot supervised-disease retention probes',
        limitations=['same development set used previously for model selection; exploratory continuation comparison',
            'few-shot on supervised GMDB diseases is not unseen-disease transfer',
            'BCH identity manifest unavailable; no comprehensive BCH overlap certification'],
        test_policy='GMDB test/CHOP expert/BCH excluded from optimization and selection; no new test inference',
        baseline_manifest_sha256=registry['sha256'], requested_resource='up to 9 existing H100 allocations; background seeds checkpoint-yield, never cancel parent allocations',
        network_during_patient_processing=False, updated_at_utc=now())
    if (WORK/'protocol.json').exists(): raise ValueError('refuse_to_overwrite_existing_trial_protocol')
    save(WORK/'protocol.json',protocol)
    save(WORK/'assets.json',assets)
    receipt = dict(state='passed', train_patients=len(train), validation_patients=len(val),
        classes=len(vocabulary), validation_diseases=len(np.unique(labels[val])), images=len(images),
        max_images_per_patient=max(counts), image_count_quantiles=np.quantile(counts,[.5,.9,.99]).tolist(),
        gmdb_path=str(gmdb), gmdb_sha256=digest(gmdb), profiles_path=str(profiles),profiles_sha256=digest(profiles),
        image_groups_sha256=digest(WORK/'private/image_groups.json'), source_image_list_sha256=digest(image_path),
        patient_index_sha256=digest(patient_path), patient_overlap=0, exact_crop_overlap=0,
        known_protected_or_test_overlap=0, all_local_crop_hashes_verified=True,
        private_MRN_mapping_read=False, images_exported=False, raw_notes_read=False,
        model_counts=load(assets['evidence_input_receipt.json']['path'])['model_counts'],updated_at_utc=now())
    save(WORK/'data_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['state','train_patients','validation_patients','classes','images','max_images_per_patient','patient_overlap','exact_crop_overlap']}))


if __name__ == '__main__': guarded(main, WORK/'prepare_failure.json')
