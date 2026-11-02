"""Create a read-only, checksum-verified HPC model release; exclude patient records."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'protected_visual_fusion_20261101'
RELEASE = ROOT / 'model_releases/craniofacial_multimodal_v1_20261101'
os.umask(0o077)


def load(path): return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')
    path.chmod(0o400)


def main():
    receipt = load(SOURCE / 'input_receipt.json')
    protocol = load(SOURCE / 'protocol.json')
    results = load(SOURCE / 'results.json')
    assert results['state'] == 'complete' and results['visual_branch_bitwise_preserved']
    RELEASE.mkdir(parents=True, exist_ok=True)
    manifest_path = RELEASE / 'manifest.json'
    if manifest_path.exists():
        manifest = load(manifest_path)
        for item in manifest['artifacts'].values():
            assert sha(RELEASE / item['path']) == item['sha256']
        print('Existing release verified; no files changed.')
        return
    if shutil.disk_usage(RELEASE).free < 5 * 1024**3:
        raise RuntimeError('insufficient_space_for_non_destructive_release_copy')
    for folder in ['weights', 'assets', 'code', 'evidence']:
        (RELEASE / folder).mkdir(exist_ok=True)
    artifacts = {}

    def copy(name, source, relative, expected=None):
        source = Path(source)
        expected = expected or sha(source)
        dest = RELEASE / relative
        if not dest.exists():
            tmp = dest.with_name(dest.name + '.tmp')
            h = hashlib.sha256()
            with source.open('rb') as r, tmp.open('wb') as w:
                for chunk in iter(lambda: r.read(8 * 1024 * 1024), b''):
                    h.update(chunk); w.write(chunk)
                w.flush(); os.fsync(w.fileno())
            assert h.hexdigest() == expected
            tmp.chmod(0o400); tmp.replace(dest)
        assert sha(dest) == expected
        artifacts[name] = dict(path=relative, sha256=expected, source=str(source), bytes=dest.stat().st_size)

    visual = protocol['visual_encoder']
    copy('visual_encoder', visual['path'], 'weights/visual_encoder.pt', visual['sha256'])
    for name, pin in receipt['checkpoints'].items():
        copy(name, pin['path'], f'weights/{name}.pt', pin['sha256'])
    copy('HPO_term_embeddings', ROOT/'hpo_fusion_staged_20261031/features/term_embeddings.npy', 'assets/term_embeddings.npy', receipt['term_embeddings_sha256'])
    copy('HPO_term_index', SOURCE/'private/term_index.json', 'assets/term_index.json')
    copy('visual_classifier_class_indices', SOURCE/'private/class_indices.json', 'assets/class_indices.json', receipt['class_indices_sha256'])
    index = load(ROOT/'hpo_fusion_public_20261031/private/patient_index.json')['patients']
    classes = sorted({p['label'] for p in index if p['split'] == 'train'})
    assert len(classes) == receipt['counts']['diseases']
    class_file = RELEASE/'assets/disease_vocabulary.json'
    write(class_file, dict(labels=classes, patient_records_included=False))
    artifacts['disease_vocabulary'] = dict(path='assets/disease_vocabulary.json',sha256=sha(class_file),bytes=class_file.stat().st_size)
    for name in ['model.py','evaluate.py','common.py']:
        copy('code_'+name, SOURCE/name, 'code/'+name)
    copy('visual_loader', ROOT/'matched_craniofacial_benchmark_20261031/network.py','code/visual_network.py')
    copy('visual_author_wrapper', ROOT/'transface_arcface_identity_cpt_20261026/model.py','code/visual_author_wrapper.py')
    for name in ['protocol.json','results.json','branch_policy.json','checks.json','input_receipt.json']:
        copy('evidence_'+name, SOURCE/name, 'evidence/'+name)
    versions = {}
    for package in ['torch','numpy','timm','transformers','onnx2torch','opencv-python']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = 'not_installed_under_this_distribution_name'
    manifest = dict(version='craniofacial_multimodal_v1_20261101',state='locked_development_baseline',
        architecture='independent frozen visual and HPO branches; score-level multimodal retrieval',
        visual_lineage='TransFace -> CHOP identity ArcFace CPT -> GMDB common-LR ArcFace fine-tuning',
        visual_seed=11, visual_epoch=50,visual_dimensions=512,
        HPO_seeds=[11,29,47,83],default_runtime_seed=11,
        default_seed_reason='fixed representative, not selected using CHOP expert; reported results are four-seed means, not default-seed or ensemble claims',
        retrieval=dict(visual_weight=.5,HPO_weight=.5,score='independent cosine similarities',
            native_visual_branch_always_available=True,missing_or_ineligible_HPO='visual-only fallback',
            unresolved_clinician_HPO_state2='outside validated primary use'),
        classification=dict(policy='retain existing cross-attention classifier',checkpoint_keys=[f'reference_{s}' for s in [11,29,47,83]],
            score_level_classification='evaluated alternative; not adopted as default'),
        selection_scope='GMDB development only; not final clinical validation',
        expert_history='Earlier architectures inspected on CHOP expert; this is not a claim of an untouched expert test cohort',
        public_knowledge_variants_and_agent='future versioned modules; not part of this locked release',
        source_checkpoints_unchanged=True,patient_images_MRI_records_or_identifiers_included=False,
        environment=versions,dependency_scope='Selected code and weights snapshotted; full third-party runtime and visual vendor dependencies remain in the HPC environment',
        artifacts=artifacts,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(manifest_path,manifest)
    manifest_hash=sha(manifest_path)
    write(RELEASE/'verification.json',dict(state='passed',artifacts_verified=len(artifacts),manifest_sha256=manifest_hash,
        total_bytes=sum(v['bytes'] for v in artifacts.values()),source_weights_modified=False,
        checkpoint_copies_independent_not_symlinks_or_hardlinks=True,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    # Parent registry points to a version; changes must create a new version directory.
    registry=ROOT/'model_releases/CURRENT.json'
    temporary=registry.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(version=manifest['version'],manifest=str(manifest_path),sha256=manifest_hash,state=manifest['state']),indent=2)+'\n')
    temporary.replace(registry)
    for folder in ['weights','assets','code','evidence']: (RELEASE/folder).chmod(0o500)
    RELEASE.chmod(0o500)
    print(json.dumps(dict(state='locked',version=manifest['version'],artifacts_verified=len(artifacts),total_bytes=sum(v['bytes'] for v in artifacts.values()))))


if __name__ == '__main__':main()
