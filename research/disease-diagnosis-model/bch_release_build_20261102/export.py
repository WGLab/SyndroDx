"""Export allowlisted tensor states and public vocabulary; no patient data."""
import csv
import hashlib
import json
import os
import pathlib
import shutil
import sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / 'model_releases/craniofacial_multimodal_v1_20261101'
DST = ROOT / 'model_releases/bch_evaluation_v1_20261102'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    os.umask(0o077)
    torch.set_num_threads(2)
    registry = json.loads((ROOT/'model_releases/CURRENT.json').read_text())
    assert sha(SRC/'manifest.json') == registry['sha256']
    manifest = json.loads((SRC/'manifest.json').read_text())
    def asset(name):
        e = manifest['artifacts'][name]
        p = SRC/e['path']
        assert sha(p) == e['sha256']
        return p
    indices = json.loads(asset('visual_classifier_class_indices').read_text())['indices']
    labels = json.loads(asset('disease_vocabulary').read_text())['labels']
    # Training checkpoint is local/trusted. Only tensor-only model states leave it.
    state = torch.load(asset('visual_encoder'), map_location='cpu', weights_only=False, mmap=True)['model']
    allowed = {k: v.detach().cpu().clone() for k, v in state.items() if k.startswith(('body.', 'feature.'))}
    assert set(state) == set(allowed) | {'head.weight'}
    torch.save(allowed, DST/'weights/visual.pt')
    torch.save(state['head.weight'][indices].float().clone(), DST/'weights/visual_head.pt')
    del allowed, state
    for source, name in [('hpo_11','hpo'), ('reference_11','fusion')]:
        s = torch.load(asset(source), map_location='cpu', weights_only=False)['model']
        s = {k: v.detach().cpu().clone() for k, v in s.items()}
        if s['assertion.weight'].shape[0] == 2:
            w = s['assertion.weight']; s['assertion.weight'] = torch.cat([w, w.mean(0, keepdim=True)], 0)
        assert all(isinstance(v, torch.Tensor) for v in s.values())
        torch.save(s, DST/f'weights/{name}.pt')
        del s
    shutil.copyfile(asset('HPO_term_index'), DST/'assets/term_index.json')
    sys.path.insert(0, str(ROOT/'chop_clinical_readiness_20261101'))
    from audit import parse_mondo, exact_class_mapping
    _, _, exact = parse_mondo(ROOT/'ontology/mondo.obo')
    public = pathlib.Path('/controlled-data/GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata/gmdb_syndromes_v1.1.0.tsv')
    vocab = {}
    with public.open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['syndrome_id'] in labels:
                # Explicit field allowlist. Never export image/subject columns.
                vocab[row['syndrome_id']] = {'gmdb_id':row['syndrome_id'], 'name':row['syndrome_name'],
                    'mondo_id':exact_class_mapping(row['OMIM'], exact)}
    assert set(vocab) == set(labels)
    (DST/'assets/diseases.json').write_text(json.dumps([vocab[c] for c in labels], indent=2)+'\n')
    config = {'release':'bch_evaluation_v1_20261102', 'source_release':manifest['version'],
        'source_manifest_sha256':registry['sha256'], 'seed':11, 'visual_epoch':50,
        'source_visual_checkpoint_sha256':manifest['artifacts']['visual_encoder']['sha256'],
        'source_HPO_checkpoint_sha256':manifest['artifacts']['hpo_11']['sha256'],
        'source_fusion_checkpoint_sha256':manifest['artifacts']['reference_11']['sha256'],
        'visual_probability_temperature':4., 'retrieval_HPO_weight':.5, 'classes':216,
        'public_classes':612, 'max_HPO_terms':128, 'precision':'FP32',
        'patient_image_or_record_files_included':False, 'optimizer_or_training_state_included':False,
        'classification':'cross-attention with gated fusion; modality fallback',
        'retrieval':'independent visual/HPO cosine scores; pairwise visual fallback',
        'aggregation':'normalize(mean(per-image normalized native512 features)); no test-time flips'}
    (DST/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps({'state':'exported_tensor_only_weights','classes':len(vocab),
        'weight_bytes':sum(p.stat().st_size for p in (DST/'weights').glob('*.pt'))}))

if __name__ == '__main__':
    main()
