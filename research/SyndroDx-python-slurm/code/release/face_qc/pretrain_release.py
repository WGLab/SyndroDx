"""Versioned, original-only SSL image release; no diagnosis inference or training."""
import argparse
import collections
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'pretrain_faces_v1'
CPU = ROOT / 'face_qc' / 'full_original_audit_v1'
QWEN = ROOT / 'face_qc' / 'qwen_stream_v5'
WORKERS = 11


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def read(path):
    with Path(path).open() as stream:
        yield from csv.DictReader(stream, delimiter='\t')


def write(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / '.prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Frozen release already prepared; resume it, do not overwrite')
    coverage = json.loads((QWEN / 'coverage_checks.json').read_text())
    assert coverage['complete_coverage'] and coverage['duplicate_ids'] == 0
    print('Reading frozen CPU/Qwen inventories', flush=True)
    cpu = {r['sample_id']: r for r in read(CPU / 'results.tsv')}
    protected, blocked_paths, identity_hold, historical_dev, historical_test = set(), set(), set(), set(), set()
    guard_files = []
    for directory, prefix in [('expert_holdout_audit', ''), ('split_plan_v0', 'ian_')]:
        p = ROOT / directory / (prefix + 'protected_mrns.tsv')
        guard_files.append(p)
        protected.update(r['mrn'] for r in read(p))
        p = ROOT / directory / (prefix + 'blocked_paths.tsv')
        guard_files.append(p)
        blocked_paths.update(str(Path(r['path'])) for r in read(p))
    assert protected and blocked_paths
    for r in read(ROOT / 'data_release_audit/patient_manifest.tsv'):
        mrn = r['mrn']
        if r.get('any_known_identity_conflict') == 'True' or r.get('has_other_identity_conflict') == 'True':
            identity_hold.add(mrn)
        splits = set(r.get('historical_splits', '').split('|'))
        if 'test' in splits:
            historical_test.add(mrn)
        elif 'val' in splits or 'dev' in splits:
            historical_dev.add(mrn)
    records, hash_ids, hash_reserved = [], collections.defaultdict(set), set()
    for row in read(QWEN / 'results.tsv'):
        c = cpu.pop(row['sample_id'])
        for key in ('image_path', 'identity_candidates', 'reservation_blocked', 'identity_review'):
            assert row[key] == c[key], (row['sample_id'], key)
        ids = {x for x in row['identity_candidates'].split('|') if x}
        h = row['original_sha256']
        if h:
            hash_ids[h].update(ids)
            if row['reservation_blocked'] == 'True' or ids & protected or row['image_path'] in blocked_paths:
                hash_reserved.add(h)
        if row['identity_review'] == 'True':
            identity_hold.update(ids)
        records.append(dict(row, cpu=c))
    assert not cpu and len(records) == coverage['expected']
    # Conflicting content aliases never establish identity; conservatively hold all linked IDs.
    for h, ids in hash_ids.items():
        if len(ids) != 1:
            identity_hold.update(ids)
        if h in hash_reserved:
            protected.update(ids)
    pending = []
    groups = collections.defaultdict(list)
    counts = collections.Counter()
    for r in records:
        ids = {x for x in r['identity_candidates'].split('|') if x}
        source_ok = bool({'craniofacial_full', 'craniofacial_frontal_supplement'} & set(r['sources'].split('|')))
        reason = ''
        if r['status'] != 'ok':
            reason = 'unavailable_source' if r['status'] in {'upstream_unreadable', 'image_error'} else 'model_output_exception_hold'
        elif ids & protected or r['original_sha256'] in hash_reserved or r['image_path'] in blocked_paths:
            reason = 'expert_or_ian_reserved'
        elif len(ids) != 1 or ids & identity_hold:
            reason = 'identity_conflict_or_unresolved'
        elif ids & historical_test:
            reason = 'historical_test_reserved'
        elif not source_ok or not r['image_path'].startswith('/controlled-data/craniofacial_data/G_drive/'):
            reason = 'source_provenance_hold'
        elif r['triage'] != 'face_candidate_quality_pending':
            reason = ('nonportrait_pool' if r['triage'] == 'nonface_candidate_hold'
                      else 'local_or_procedure_pool' if r['content_type'] in {'partial_face_photo', 'local_body_photo', 'medical_procedure_photo', 'document_or_screen'}
                      else 'unresolved_visual_hold')
        r['policy_disposition'] = reason or 'candidate'
        r['mrn'] = next(iter(ids)) if len(ids) == 1 else ''
        counts[r['policy_disposition']] += 1
        if not reason:
            groups[r['original_sha256']].append(r)
    print('Building disjoint unique-content assignments', len(groups), flush=True)
    # Prefer an existing conservative CPU crop. Full-frame fallback preserves profiles
    # missed by the landmark detector; the new visual QC must check prominence/completeness.
    for h, rows in sorted(groups.items()):
        chosen = min(rows, key=lambda r: (not bool(r['cpu']['proposed_crop_box']), r['sample_id']))
        c = chosen['cpu']
        box = json.loads(c['proposed_crop_box']) if c['proposed_crop_box'] else [0, 0, int(c['width']), int(c['height'])]
        mrn = chosen['mrn']
        split = 'dev' if mrn in historical_dev else ('dev' if int(hashlib.sha256(('ssl-v1-patient-' + mrn).encode()).hexdigest()[:8], 16) % 20 == 0 else 'train')
        pending.append(dict(image_id=h, sample_id=chosen['sample_id'], image_path=chosen['image_path'],
                            original_sha256=h, mrn=mrn, split=split, view=chosen['view'],
                            crop_box=box, crop_method='cpu_landmark_margin' if c['proposed_crop_box'] else 'full_frame_profile_preserving_fallback',
                            original_width=int(c['width']), original_height=int(c['height']),
                            face_width_pixels=c.get('face_width_pixels', ''),
                            blur_variance=c.get('blur_variance', ''),
                            margin_clipped=c.get('margin_clipped', ''),
                            phash64=c.get('phash64', ''), dhash64=c.get('dhash64', ''),
                            sources=chosen['sources'], aliases=[r['sample_id'] for r in rows]))
    db = sqlite3.connect(OUT / 'inventory.sqlite')
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('CREATE TABLE IF NOT EXISTS originals (sample_id TEXT PRIMARY KEY, sha TEXT, mrn TEXT, disposition TEXT, record TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS candidates (image_id TEXT PRIMARY KEY, worker INTEGER, record TEXT)')
    db.execute('DELETE FROM originals')
    db.execute('DELETE FROM candidates')
    db.executemany('INSERT INTO originals VALUES (?,?,?,?,?)',
                   ((r['sample_id'], r['original_sha256'], r['mrn'], r['policy_disposition'], json.dumps(r)) for r in records))
    loads = [0.] * WORKERS
    shards = [[] for _ in range(WORKERS)]
    for r in pending:
        i = min(range(WORKERS), key=lambda i: loads[i])
        r['worker'] = i
        shards[i].append(r)
        loads[i] += 1. / (2. if i < 7 else 1.)
    db.executemany('INSERT INTO candidates VALUES (?,?,?)', ((r['image_id'], r['worker'], json.dumps(r)) for r in pending))
    db.execute('CREATE INDEX IF NOT EXISTS original_sha_idx ON originals(sha)')
    db.commit()
    db.close()
    for i, rows in enumerate(shards):
        directory = OUT / f'worker_{i}'
        directory.mkdir(exist_ok=True)
        (directory / 'input.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    # 500 content-unique examples, with all views and both crop methods represented.
    strata = collections.defaultdict(list)
    for r in pending:
        strata[(r['view'], r['crop_method'], r['margin_clipped'])].append(r)
    sample, sampled_patients, sampled_images = [], set(), set()
    for _, rows in sorted(strata.items()):
        ordered = sorted(rows, key=lambda r: hashlib.sha256(('audit-' + r['image_id']).encode()).hexdigest())
        for r in ordered:
            if r['mrn'] not in sampled_patients:
                sample.append(r); sampled_patients.add(r['mrn']); sampled_images.add(r['image_id'])
            if sum(x['image_id'] in sampled_images for x in rows) >= 20:
                break
    for r in sorted(pending, key=lambda r: hashlib.sha256(('audit-extra-' + r['image_id']).encode()).hexdigest()):
        if len(sample) >= 500:
            break
        if r['mrn'] not in sampled_patients:
            sample.append(r); sampled_patients.add(r['mrn']); sampled_images.add(r['image_id'])
    sample = sample[:500]
    assert len(sample) == 500 and len({r['image_id'] for r in sample}) == 500
    (OUT / 'sample_input.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in sample))
    code = [ROOT / 'face_qc/pretrain_release.py', ROOT / 'face_qc/pretrain_crop.py',
            ROOT / 'face_qc/pretrain_visual_qc.py', ROOT / 'face_qc/pretrain_qc_prompt.txt',
            ROOT / 'face_qc/finalize_pretrain_release.py']
    sources = [CPU / 'results.tsv', QWEN / 'results.tsv', QWEN / 'coverage_checks.json',
               ROOT / 'data_release_audit/patient_manifest.tsv', ROOT / 'DATA_SCOPE.md'] + guard_files
    manifest = dict(version='pretrain_faces_v1', expected_original_paths=len(records),
                    unique_candidates=len(pending), initial_dispositions=dict(counts), workers=WORKERS,
                    views=dict(collections.Counter(r['view'] for r in pending)),
                    crop_methods=dict(collections.Counter(r['crop_method'] for r in pending)),
                    protected_patient_ids=len(protected), identity_hold_patient_ids=len(identity_hold),
                    historical_test_patient_ids=len(historical_test),
                    split_policy='Historical test held; historical val/dev preserved; other patients hash-split 95/5. Expert/Ian always held. No disease-zero-shot or family-independence claim.',
                    source_policy='Original G_drive photographs indexed by full/frontal craniofacial sources; no legacy crops or genetics-only selection.',
                    quality_policy='Single complete upright portrait, usable visual quality, prominent face, no visible text; exact original-preserving crop; uncertain records held.',
                    created_at=time.time(), input_sha256={str(p): digest(p) for p in sources},
                    code_sha256={str(p): digest(p) for p in code},
                    assignment_sha256={str(i): digest(OUT / f'worker_{i}/input.jsonl') for i in range(WORKERS)},
                    sample_sha256=digest(OUT / 'sample_input.jsonl'), state='prepared_not_released', training_admitted=0)
    atomic_json(OUT / 'manifest.json', manifest)
    atomic_json(OUT / 'status.json', {'state': 'prepared', 'unique_candidates': len(pending), 'at': time.time()})
    print(json.dumps({k: v for k, v in manifest.items() if k not in {'input_sha256', 'code_sha256', 'assignment_sha256'}}, indent=2), flush=True)


def verify_code():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for path, sha in manifest['code_sha256'].items():
        assert digest(path) == sha, 'Pinned processing code changed: ' + path
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare'])
    args = parser.parse_args()
    prepare()
