"""Compare recorded VLM experiments without treating prior labels as ground truth."""
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / 'face_qc'


def load_progress(directory):
    path = directory / 'progress.jsonl'
    if not path.exists():
        return {}
    records = {}
    lines = path.read_bytes().splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            d = json.loads(line)
        except ValueError:
            if i == len(lines) - 1 and not line.endswith(b'\n'):
                break
            raise
        if d['review_id'] in records:
            raise ValueError('Duplicate review ID')
        records[d['review_id']] = d
    return records


def counts(records):
    return {'processed': len(records),
            'status': dict(collections.Counter(d['status'] for d in records.values())),
            'triage': dict(collections.Counter(d['triage'] for d in records.values())),
            'views': dict(collections.Counter(d.get('prediction', {}).get('view', 'error') for d in records.values()))}


def main():
    a = load_progress(QC / 'vlm_pilot_v1')
    b = load_progress(QC / 'vlm_pilot_v2')
    expanded = load_progress(QC / 'vlm_expanded_v2')
    with (ROOT / 'multiview_review_v1/view_review.tsv').open() as f:
        refs = {r['review_id']: r for r in csv.DictReader(f, delimiter='\t')}
    common = sorted(a.keys() & b.keys())
    changes = []
    for key in common:
        if a[key].get('original_sha256') != b[key].get('original_sha256'):
            raise ValueError('Original image hashes differ between experiments')
        pa, pb = a[key].get('prediction', {}), b[key].get('prediction', {})
        changed = [k for k in ['content_type', 'complete_face_visible', 'view', 'orientation', 'face_count'] if pa.get(k) != pb.get(k)]
        if changed or a[key]['triage'] != b[key]['triage']:
            changes.append({'review_id': key, 'prior_view_label': refs[key]['view_label'],
                            'changed_fields': changed, 'v1': pa, 'v2': pb,
                            'triage_v1': a[key]['triage'], 'triage_v2': b[key]['triage']})
    checks = {}
    for name, records in [('baseline', a), ('revised', b)]:
        certain_faces = {'frontal_candidate', 'oblique_candidate', 'profile_candidate', 'strong_pitch_candidate'}
        checks[name] = {
            'prior_face_sent_to_nonface_hold': [k for k, d in records.items() if refs[k]['view_label'] in certain_faces and d['triage'] == 'nonface_candidate_hold'],
            'prior_incomplete_called_complete': [k for k, d in records.items() if refs[k]['view_label'] == 'incomplete_face_original' and d.get('prediction', {}).get('complete_face_visible') is True],
            'prior_nonface_called_face_candidate': [k for k, d in records.items() if refs[k]['view_label'] == 'non_face_or_local_body_document' and d['triage'] == 'face_candidate_quality_pending']}
    summary = {'baseline': counts(a), 'revised': counts(b), 'expanded': counts(expanded),
               'paired_count': len(common), 'paired_complete': len(common) == len(refs),
               'all_complete': len(common) == len(refs) and len(expanded) == 300,
               'changed_count': len(changes), 'review_flags': checks,
               'limitation': 'Development comparison only. Prior assistant labels are not expert ground truth; expanded images are unlabelled. No validated accuracy or training admission.',
               'training_admitted': 0}
    out = QC / 'vlm_experiment_comparison'
    out.mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'changes.json').write_text(json.dumps(changes, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
