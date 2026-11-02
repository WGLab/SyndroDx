"""Freeze released splits and bridge authorized inventory labels locally."""
import collections
import csv
import hashlib
import hmac
import json
import re
import sys
from pathlib import Path
from runtime import ROOT, RUN, load, save, digest, now

INVENTORY = Path('/opt/craniofacial/disease-diagnosis-model/complete_data_inventory')
OLD = INVENTORY.parent
METADATA = Path('/controlled-data/chop_data_metadata/metadata_internal.tsv')
PRIVATE_MAPPING = METADATA.parent / 'mrn_mapping_PRIVATE.csv'


def rows(path):
    with path.open(newline='') as source:
        yield from csv.DictReader(source, delimiter='\t')


def mondo(value):
    return {'MONDO:' + n.zfill(7) for n in re.findall(r'MONDO[:_](\d+)', value or '', flags=re.I)}


def norm(value):
    value = (value or '').strip()
    if re.fullmatch(r'\d+\.0', value):
        value = value[:-2]
    return (value.lstrip('0') or '0') if value.isdigit() else value


def private_bridge(identity, eligible, salt):
    """Read only on HPC. Never return identifiers outside this local program."""
    bridge = collections.defaultdict(set)
    if not PRIVATE_MAPPING.is_file():
        return bridge
    with PRIVATE_MAPPING.open(newline='') as source:
        reader = csv.DictReader(source)
        # Exact schema supplied by the user; no header/record inspection output.
        pseudo, actual = 'psedudo_mrn', ['mrn']
        if pseudo not in (reader.fieldnames or []) or 'mrn' not in (reader.fieldnames or []):
            raise RuntimeError('private_mapping_schema_requires_local_review')
        for row in reader:
            value = (row.get(pseudo) or '').strip()
            key = hmac.new(salt, ('patient:' + value).encode(), hashlib.sha256).hexdigest()
            group = identity['patients'].get(key, {}).get('group_id')
            if group not in eligible:
                continue
            for column in actual:
                if norm(row.get(column)):
                    bridge[norm(row[column])].add(group)
    return bridge


def prepare_inputs(world):
    folder = ROOT / 'evaluation'
    folder.mkdir(exist_ok=True)
    release = load(ROOT / 'dataset_release/release.json')
    if not release.get('patient_split_disjoint') or not release.get('protected_pools_excluded'):
        raise RuntimeError('protected_patient_split_required')
    records, group_splits, image_splits = [], {}, {}
    hashes = {}
    for split in ['train', 'validation', 'test']:
        spec = release['manifests'][split]
        path = Path(spec['path'])
        if digest(path) != spec['sha256']:
            raise RuntimeError('evaluation_manifest_changed')
        hashes[split] = spec['sha256']
        for row in rows(path):
            if row['split'] != split or row['training_admitted'] != str(split == 'train').lower():
                raise RuntimeError('evaluation_split_admission_mismatch')
            group = row['group_id']
            if group_splits.setdefault(group, split) != split:
                raise RuntimeError('evaluation_patient_leakage')
            for key in ['crop_sha256', 'pixel_sha256']:
                identity = (key, row[key])
                if image_splits.setdefault(identity, split) != split:
                    raise RuntimeError('evaluation_exact_image_leakage')
            records.append({k: row[k] for k in ['group_id', 'patient_key', 'split', 'view', 'crop_path', 'crop_sha256', 'pixel_sha256']})
    path = folder / 'inputs.private.jsonl'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as out:
        for row in records:
            out.write(json.dumps(row) + '\n')
    temporary.replace(path)
    plan = dict(state='frozen', world_size=world, inputs_sha256=digest(path),
                manifests_sha256=hashes, patient_split_disjoint=True,
                exact_image_split_disjoint=True, rows=len(records), created_at_utc=now())
    save(folder / 'input_plan.json', plan)
    prepare_labels({r['group_id'] for r in records})
    return plan


def prepare_labels(eligible):
    # The user supplied this inventory directory. Follow only its registered
    # image-label sources; never import protected expert labels into training.
    versions = load(INVENTORY / 'input_versions.json')
    paths = [OLD / 'diagnosis_strata/diagnosis_image_manifest.tsv',
             OLD / 'candidate_concept_audit/raw_label_concepts.tsv',
             OLD / 'data_release_audit/image_manifest.tsv']
    if any(not p.is_file() for p in paths) or any(str(p) not in versions for p in paths[:2]):
        save(ROOT / 'evaluation/labels_status.json', dict(state='requires_label_source', updated_at_utc=now()))
        return
    identity = load(ROOT / 'dataset_release/patient_identity_plan.private.json')
    if digest(METADATA) != identity['metadata_sha256']:
        raise RuntimeError('approved_metadata_changed_before_label_join')
    sys.path.insert(0, str(RUN))
    from annotation_pipeline import references, canonical_path
    settings = load(RUN / 'config.json')
    salt = (RUN / 'direct_identity_salt.bin').read_bytes()
    by_path = collections.defaultdict(set)
    for row in rows(METADATA):
        pseudo = (row.get('pseudo_mrn') or '').strip()
        key = hmac.new(salt, ('patient:' + pseudo).encode(), hashlib.sha256).hexdigest()
        group = identity['patients'].get(key, {}).get('group_id')
        if not pseudo or group not in eligible:
            continue
        for value in references(row.get('image_path')):
            by_path[str(canonical_path(value, settings))].add(group)
    # Raw historical identifiers exist only transiently inside this local join.
    # Ambiguous mappings are excluded; only released group hashes are saved.
    bridge = private_bridge(identity, eligible, salt)
    frozen_audit_hash = identity.get('control_sha256', {}).get(str(paths[2]))
    if not frozen_audit_hash or digest(paths[2]) != frozen_audit_hash:
        raise RuntimeError('identity_bridge_audit_changed')
    for row in rows(paths[2]):
        groups = set()
        for field in ['image_path', 'cropping_path']:
            if row.get(field):
                groups.update(by_path.get(str(canonical_path(row[field], settings)), set()))
        if groups:
            for field in ['mrn', 'original_mrn', 'latest_norm_mrn']:
                if norm(row.get(field)):
                    bridge[norm(row[field])].update(groups)
    concepts = {r['raw_label_id']: r for r in rows(paths[1])}
    labels = {name: collections.defaultdict(set) for name in ['disease_or_disorder', 'structural_diagnosis']}
    for row in rows(paths[0]):
        groups = bridge.get(norm(row.get('mrn')), set())
        if len(groups) != 1 or str(row.get('is_diagnosis', '')).lower() not in ['true', '1', 'yes']:
            continue
        group = next(iter(groups))
        if group not in eligible:
            continue
        row_mondo = mondo(row.get('mondo_ids'))
        for raw in re.split(r'[|;,]', row.get('raw_label_ids') or ''):
            raw = raw.strip().strip('[]\'" ')
            concept = concepts.get(raw.strip(), {})
            kind = concept.get('concept_type')
            if kind in labels:
                labels[kind][group].update(row_mondo & mondo(concept.get('mondo_ids')))
    labels = {kind: {g: sorted(values) for g, values in groups.items() if values}
              for kind, groups in labels.items()}
    save(ROOT / 'evaluation/labels.private.json', dict(
        state='exploratory_metadata_labels', labels=labels,
        clinical_adjudication='not established', molecular_confirmation='not established',
        excluded=['protected expert labels', 'ambiguous identity bridges', 'phenotype_only', 'procedures', 'workflow', 'ambiguous abbreviations', 'unresolved concepts'],
        source_sha256={str(p): digest(p) for p in paths}, approved_metadata_sha256=digest(METADATA),
        private_mapping_used=PRIVATE_MAPPING.is_file(), private_mapping_sha256=digest(PRIVATE_MAPPING) if PRIVATE_MAPPING.is_file() else None,
        patient_split_unchanged=True, labels_used_in_ssl=False, created_at_utc=now()))
    save(ROOT / 'evaluation/labels_status.json', dict(
        state='exploratory_metadata_labels_ready' if any(labels.values()) else 'no_eligible_label_matches',
        clinical_gold_standard=False, labels_used_in_ssl=False, updated_at_utc=now()))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, required=True)
    try:
        prepare_inputs(parser.parse_args().world_size)
    except Exception as exc:
        reason = str(exc) if re.fullmatch('[a-z_]+', str(exc)) else 'private_local_error'
        save(ROOT / 'evaluation/preparation_failure.json', dict(state='failed', error_type=type(exc).__name__, reason=reason, updated_at_utc=now()))
        raise SystemExit(1)
