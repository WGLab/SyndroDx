"""Build private, split-preserving review queues; stdout contains aggregates only."""
import collections
import csv
import datetime
import json
import os
from pathlib import Path
import sqlite3

os.umask(0o077)
ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / 'private'


def rows(path):
    with path.open(newline='') as f:
        yield from csv.DictReader(f, delimiter='\t')


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main():
    people = {r['patient_key']: r for r in rows(PRIVATE / 'patient_master.tsv')}
    assert len(people) == 26897
    db = sqlite3.connect(f'file:{PRIVATE / "linked_metadata.sqlite"}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    images = [dict(r) for r in db.execute('SELECT * FROM images')]
    groups = collections.defaultdict(list)
    for p in people.values():
        groups[p['group_id']].append(p)
    reasons = collections.defaultdict(set)
    protected = set()
    for g, members in groups.items():
        if any(p['protected'] == 'True' for p in members):
            protected.add(g)
            reasons[g].add('protected_patient_group')
        if any(p['identity_conflict'] == 'True' for p in members):
            reasons[g].add('identity_conflict')
        if len({p['split'] for p in members}) > 1:
            reasons[g].add('patient_group_split_conflict')
    blocked_paths, blocked_hashes = set(), set()
    for directory, prefix in [('expert_holdout_audit', ''), ('split_plan_v0', 'ian_')]:
        blocked_paths.update(r['path'] for r in rows(ROOT.parent / directory / (prefix + 'blocked_paths.tsv')))
        blocked_hashes.update(r['sha256'] for r in rows(ROOT.parent / directory / (prefix + 'blocked_crop_hashes.tsv')))
    hash_groups = collections.defaultdict(set)
    historical_protected_train = set()
    image_group_mismatches = 0
    for im in images:
        p = people[im['patient_key']]
        g = p['group_id']
        if im['group_id'] != g:
            image_group_mismatches += 1
            reasons[g].add('image_patient_group_mismatch')
        if im['split'] != p['split']:
            reasons[g].add('image_patient_split_mismatch')
        if g in protected and im['split'] == 'train':
            historical_protected_train.add(im['patient_key'])
        if im['crop_path'] in blocked_paths or im['crop_sha256'] in blocked_hashes:
            reasons[g].add('protected_image_reference')
        if im['crop_sha256']:
            hash_groups[im['crop_sha256']].add(g)
        else:
            reasons[g].add('missing_crop_hash')
    duplicate_groups = set()
    for gs in hash_groups.values():
        if len(gs) > 1:
            duplicate_groups.update(gs)
    for g in duplicate_groups:
        reasons[g].add('exact_crop_shared_between_patient_groups')
    out = PRIVATE / 'review_release'
    out.mkdir(exist_ok=True, mode=0o700)
    counts = collections.Counter()
    eligible = set()
    with (out / 'patient_disposition.tsv').open('w', newline='') as f:
        fields = ['patient_key', 'group_id', 'split', 'disposition', 'reasons']
        writer = csv.DictWriter(f, fields, delimiter='\t'); writer.writeheader()
        for key, p in people.items():
            g = p['group_id']
            if g in protected:
                disposition = 'protected'
            elif reasons[g]:
                disposition = 'quarantine'
            elif p['split'] in ['validation', 'test']:
                disposition = p['split']
            elif p['split'] != 'train':
                disposition = 'unassigned'
            else:
                disposition = 'train_existing_split'
                eligible.add(key)
            counts[disposition] += 1
            writer.writerow(dict(patient_key=key, group_id=g, split=p['split'], disposition=disposition,
                                 reasons='|'.join(sorted(reasons[g]))))
    train_images = [im for im in images if im['patient_key'] in eligible and im['split'] == 'train']
    train_keys = {im['patient_key'] for im in train_images}
    with (out / 'existing_train_visual_references.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(images[0]), delimiter='\t'); writer.writeheader(); writer.writerows(train_images)
    queue_counts = collections.Counter()
    queue_patients = collections.defaultdict(set)
    fields = ['patient_key', 'code', 'assertion', 'source', 'source_rows', 'event_date', 'context_present', 'review_priority', 'training_admitted']
    with (out / 'train_face_HPO_review.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, delimiter='\t'); writer.writeheader()
        for r in db.execute('SELECT * FROM hpo ORDER BY patient_key,source,code'):
            item = dict(r)
            if item['patient_key'] not in train_keys:
                continue
            priority = 'clinician_context' if item['source'] == 'Arcus_clinician_HPO' and item['context_present'] else 'clinician_record' if item['source'] == 'Arcus_clinician_HPO' else 'nlp_assertion_recovery'
            item.update(review_priority=priority, training_admitted=False)
            writer.writerow(item)
            queue_counts[priority] += 1
            queue_patients[priority].add(item['patient_key'])
    assert not any(people[k]['group_id'] in protected for k in train_keys)
    assert all(not reasons[people[k]['group_id']] for k in train_keys)
    assert all(people[k]['split'] == 'train' for k in train_keys)
    assert not any(im['crop_path'] in blocked_paths or im['crop_sha256'] in blocked_hashes for im in train_images)
    train_groups = {people[k]['group_id'] for k in train_keys}
    assert all(len(gs) == 1 for gs in hash_groups.values() if gs & train_groups)
    audit = dict(state='review_release_ready_not_new_clinical_supervision',
        patient_disposition=dict(counts), existing_train_visual_patients=len(train_keys), existing_train_visual_images=len(train_images),
        review_records_by_priority=dict(queue_counts), review_patients_by_priority={k: len(v) for k, v in queue_patients.items()},
        unique_train_face_HPO_review_patients=len(set().union(*queue_patients.values())) if queue_patients else 0,
        historical_protected_patients_in_train_manifest=len(historical_protected_train),
        image_group_mismatch_records=image_group_mismatches, groups_with_shared_crop_hashes=len(duplicate_groups),
        excluded_groups_by_reason=dict(collections.Counter(r for rs in reasons.values() for r in rs)),
        protected_patients_with_incorrect_master_split=sum(p['protected'] == 'True' and p['split'] != 'protected' for p in people.values()),
        emitted_train_protected_overlap=0, emitted_train_identity_or_group_conflicts=0,
        new_HPO_supervision_admissions=0, images_read=False, raw_notes_exported=False,
        limitations=['BCH manifest not provided; no independent BCH cross-source overlap certification',
                     'Known patient groups and recorded exact crop hashes checked; unknown relatives and near duplicates remain unaudited',
                     'Clinical assertions still require source, polarity, experiencer and time verification'],
        updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(ROOT / 'review_release_audit.json', audit)
    db.close()
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
