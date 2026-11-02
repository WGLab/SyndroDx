"""Baseline-controlled, patient-isolated evaluation. Private results stay on HPC."""
import collections
import json
import os
from pathlib import Path
import numpy as np
from runtime import ROOT, load, save, digest, now
from evaluation_core import normalized, aggregate_patients, retrieval, macro_score, paired_interval, collapse_statistics, ridge_probe


def features(kind, records, plan):
    directory = ROOT / 'evaluation'
    merged = None
    seen = np.zeros(len(records), dtype=bool)
    for slot in range(plan['world_size']):
        folder = directory / 'features' / f'slot_{slot:02d}'
        receipt = load(folder / (kind + '.json'))
        path = folder / (kind + '.private.npy')
        if receipt.get('state') != 'complete' or receipt.get('inputs_sha256') != plan['inputs_sha256'] or digest(path) != receipt['features_sha256']:
            raise RuntimeError('incomplete_or_changed_evaluation_features')
        indices = np.load(folder / 'indices.private.npy', allow_pickle=False)
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if len(indices) != len(values) or np.any(indices < 0) or np.any(indices >= len(records)) or np.any(seen[indices]) or len(set(indices.tolist())) != len(indices):
            raise RuntimeError('evaluation_feature_partition_conflict')
        if merged is None:
            merged = np.lib.format.open_memmap(directory / (kind + '.merged.private.npy'), mode='w+', dtype=np.float32,
                                               shape=(len(records), values.shape[1]))
        merged[indices] = values
        seen[indices] = True
    if not seen.all():
        raise RuntimeError('evaluation_feature_coverage_incomplete')
    merged.flush()
    return merged


def view_test(records, image_features):
    views = collections.defaultdict(lambda: collections.defaultdict(list))
    hashes = collections.defaultdict(lambda: collections.defaultdict(set))
    for row, feature in zip(records, image_features):
        if row['split'] == 'test':
            views[row['group_id']][row['view']].append(feature)
            hashes[row['group_id']][row['view']].add(row['pixel_sha256'])
    result = {}
    for other in ['profile', 'oblique']:
        ids = sorted(g for g, v in views.items() if v.get('frontal') and v.get(other)
                     and not (hashes[g]['frontal'] & hashes[g][other]))
        if len(ids) < 2:
            result[other] = dict(state='not_evaluable', reason='at_least_two_cross_view_patients_required')
            continue
        frontal = normalized([np.mean(views[g]['frontal'], axis=0) for g in ids])
        query = normalized([np.mean(views[g][other], axis=0) for g in ids])
        correct = np.zeros(len(ids), bool)
        for i in range(0, len(ids), 128):
            correct[i:i + 128] = np.argmax(query[i:i + 128] @ frontal.T, axis=1) == np.arange(i, min(i + 128, len(ids)))
        result[other] = dict(state='complete', patients=len(ids), top1_same_patient=float(correct.mean()),
                             interpretation='cross-view identity consistency only; not evidence of disease diagnosis')
    return result


def labeled_partition(partitions, labels, split, single=False):
    ids, vectors = partitions[split]
    selected = [i for i, group in enumerate(ids) if labels.get(group) and (not single or len(labels[group]) == 1)]
    return ([ids[i] for i in selected], vectors[selected], [set(labels[ids[i]]) for i in selected])


def evaluate():
    folder = ROOT / 'evaluation'
    plan = load(folder / 'input_plan.json')
    protocol = load(folder / 'protocol.json')
    if load(folder / 'labels_status.json').get('state') != 'exploratory_metadata_labels_ready':
        raise RuntimeError('validated_metadata_label_bridge_required')
    if protocol.get('labels_release_sha256') and digest(folder / 'labels.private.json') != protocol['labels_release_sha256']:
        raise RuntimeError('frozen_evaluation_labels_changed')
    if digest(folder / 'inputs.private.jsonl') != plan['inputs_sha256']:
        raise RuntimeError('frozen_evaluation_inputs_changed')
    with (folder / 'inputs.private.jsonl').open() as source_file:
        records = [json.loads(line) for line in source_file]
    by_model, report = {}, dict(state='running', protocol='frozen_encoder_patient_isolated_v1',
        primary_endpoint='disease_or_disorder metadata-label patient-macro top1 retrieval against train-only gallery',
        secondary_endpoints='structural metadata, top5/top10, frozen linear probe, cross-view identity; exploratory without multiplicity correction',
        baseline='original official DINOv3 ViT-7B, identical evaluation preprocessing',
        patient_split='frozen before SSL, validation/test excluded from SSL',
        test_used_for_hyperparameters=False, clinical_adjudication=False,
        external_center_validation=False, annotations_not_used_in_ssl=True,
        claims='Internal exploratory metadata-label evaluation; no clinical diagnostic validation claim.',
        models={})
    for kind in ['original_dinov3', 'craniofacial']:
        image_features = features(kind, records, plan)
        by_model[kind] = {split: aggregate_patients(records, image_features, split) for split in ['train', 'validation', 'test']}
        ids, test_features = by_model[kind]['test']
        report['models'][kind] = dict(
            integrity=dict(finite=bool(np.isfinite(image_features).all()), nonzero=True),
            collapse=collapse_statistics(test_features) if len(ids) else dict(state='not_evaluable'),
            cross_view_identity=view_test(records, image_features))
        del image_features
    source = load(folder / 'labels.private.json')
    report['label_provenance'] = {k: source.get(k) for k in ['state', 'clinical_adjudication', 'molecular_confirmation', 'source_sha256', 'private_mapping_used']}
    report['metadata_endpoints'] = {}
    for category, labels in source.get('labels', {}).items():
        endpoint, patient_results = {'models': {}, 'label_scope': category}, {}
        for kind, partitions in by_model.items():
            gallery_ids, gallery, gallery_y = labeled_partition(partitions, labels, 'train')
            query_ids, query, query_y = labeled_partition(partitions, labels, 'test')
            if not gallery_ids or not query_ids:
                endpoint['models'][kind] = dict(state='not_evaluable', reason='labeled_training_gallery_and_test_required')
                continue
            if set(gallery_ids) & set(query_ids):
                raise RuntimeError('retrieval_patient_leakage')
            hits, covered = retrieval(query, gallery, query_ids, gallery_ids, query_y, gallery_y)
            order = np.random.default_rng(20261015).permutation(len(gallery_y))
            shuffled, _ = retrieval(query, gallery, query_ids, gallery_ids, query_y, [gallery_y[i] for i in order], ks=(1,))
            entry = dict(state='complete', labeled_test_patients=len(query_ids), gallery_patients=len(gallery_ids),
                         test_label_gallery_coverage=float(covered.mean()), retrieval={},
                         single_label_shuffle_sanity_top1_macro=macro_score(shuffled[1], query_y))
            for k, values in hits.items():
                entry['retrieval'][str(k)] = dict(micro_all_labeled=float(values.mean()), macro_all_labeled=macro_score(values, query_y),
                    micro_gallery_covered=float(values[covered].mean()) if covered.any() else None)
            _, train_x, train_y = labeled_partition(partitions, labels, 'train', single=True)
            _, val_x, val_y = labeled_partition(partitions, labels, 'validation', single=True)
            _, test_x, test_y = labeled_partition(partitions, labels, 'test', single=True)
            probe = ridge_probe(train_x, val_x, test_x, [next(iter(x)) for x in train_y],
                                [next(iter(x)) for x in val_y], [next(iter(x)) for x in test_y])
            probe_success = probe.pop('test_success', None)
            entry['linear_probe'] = probe
            endpoint['models'][kind] = entry
            patient_results[kind] = dict(query_ids=query_ids, query_y=query_y, hits=hits,
                                         probe_success=probe_success, probe_labels=test_y)
        if len(patient_results) == 2:
            first, second = patient_results['original_dinov3'], patient_results['craniofacial']
            if first['query_ids'] != second['query_ids']:
                raise RuntimeError('baseline_patient_alignment_mismatch')
            endpoint['paired_comparison'] = {str(k): paired_interval(first['hits'][k], second['hits'][k], first['query_y']) for k in [1, 5, 10]}
            if first['probe_success'] is not None and second['probe_success'] is not None and len(first['probe_success']):
                endpoint['paired_linear_probe'] = paired_interval(first['probe_success'], second['probe_success'], first['probe_labels'])
        report['metadata_endpoints'][category] = endpoint
    original_ids, original = by_model['original_dinov3']['test']
    updated_ids, updated = by_model['craniofacial']['test']
    if original_ids != updated_ids:
        raise RuntimeError('encoder_comparison_alignment_mismatch')
    report['mean_baseline_updated_patient_cosine'] = float(np.mean(np.sum(original * updated, axis=1))) if len(original) else None
    primary = report['metadata_endpoints'].get('disease_or_disorder', {}).get('paired_comparison', {}).get('1')
    if primary:
        lower, upper = primary['paired_patient_bootstrap_95_interval']
        report['primary_result'] = 'improved_on_metadata_endpoint' if lower > 0 else 'worse_on_metadata_endpoint' if upper < 0 else 'improvement_not_established'
    else:
        report['primary_result'] = 'not_evaluable_with_available_labels'
    report['state'] = 'evaluation_complete'
    report['completed_at_utc'] = now()
    save(folder / 'report.private.json', report)
    text = ['# Craniofacial encoder evaluation', '',
            'This is an internal, patient-isolated comparison with the original DINOv3 ViT-7B.',
            'The diagnosis endpoints use existing metadata mappings. They are not clinically adjudicated or molecularly confirmed ground truth.',
            'Structural diagnoses and disease/disorder metadata categories are evaluated separately. Protected expert pools are excluded.',
            'Cross-view patient matching is an identity-consistency check, not a diagnostic endpoint.', '',
            '## Results', '', '```json', json.dumps(report, indent=2), '```', '',
            '## Interpretation', '',
            'A lower training loss is not evidence of clinical usefulness. Inspect the paired confidence intervals and both retrieval and frozen linear-probe results.',
            'An interval crossing zero does not establish an improvement. A positive result here still requires stronger labels and independent external validation.',
            'Do not select a new model or training setting using this test set. Further tuning requires a new, untouched confirmation set.', '']
    (folder / 'REPORT.md').write_text('\n'.join(text))
    save(folder / 'completion.json', dict(state='evaluation_complete', metadata_label_comparison_available=bool(report['metadata_endpoints']),
         clinically_validated=False, report_path=str(folder / 'REPORT.md'), completed_at_utc=now()))


if __name__ == '__main__':
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    try:
        evaluate()
    except Exception as exc:
        save(ROOT / 'evaluation/metrics_failure.json', dict(state='failed', error_type=type(exc).__name__, updated_at_utc=now()))
        raise SystemExit(1)
