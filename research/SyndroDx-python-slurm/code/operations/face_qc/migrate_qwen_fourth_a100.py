"""Freeze ten disjoint assignments after all seven previous writers stop."""
import collections
import contextlib
import csv
import fcntl
import json
import sqlite3
import subprocess
import shutil
import tempfile
import time
from pathlib import Path

import stream_qwen_review_v5 as worker

OLD = worker.ROOT / 'face_qc/qwen_stream_v4'
OUT = worker.OUT


def main():
    OUT.mkdir(exist_ok=False)
    old_manifest = json.loads((OLD / 'manifest.json').read_text())
    assert worker.file_hash(worker.CPU / 'manifest.json') == old_manifest['cpu_manifest_sha256']
    # Prepare the large, immutable source database while all production writers still run.
    staging = Path(tempfile.mkdtemp(prefix='qwen_assignment_v5_'))
    db_path = staging / 'assignment_audit.sqlite'
    source_db = OLD / 'assignment_audit.sqlite'
    shutil.copyfile(source_db, db_path)
    source_db_hash = worker.file_hash(source_db)
    assert worker.file_hash(db_path) == source_db_hash
    db = sqlite3.connect(db_path)
    db.execute('PRAGMA cache_size=-262144')
    db.execute('PRAGMA journal_mode=OFF')
    db.execute('PRAGMA synchronous=OFF')
    db.execute('PRAGMA temp_store=MEMORY')
    print('Source database staged and verified; requesting durable stops', flush=True)
    steps = []
    for i in range(10):
        runtime = json.loads((OLD / f'worker_{i}/runtime.json').read_text())
        assert runtime['job_id'] in {'25182707', '25185331', '25185332'}
        steps.append(runtime['job_id'] + '.' + runtime['step_id'])
    subprocess.run(['scancel', '--signal=TERM', *steps], check=True)
    deadline = time.monotonic() + 90
    while True:
        states = [json.loads((OLD / f'worker_{i}/status.json').read_text())['state'] for i in range(10)]
        if all(x in {'stopped_after_durable_batch', 'completed'} for x in states):
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Not all old writers stopped; no reassignment performed')
        time.sleep(1)
    print('All ten workers stopped after durable batches', flush=True)
    previous = {}
    owners = {}
    cache = {}
    old_hashes = {}
    with contextlib.ExitStack() as locks:
        for i in range(10):
            dest = OLD / f'worker_{i}'
            lock = locks.enter_context((dest / '.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status = json.loads((dest / 'status.json').read_text())
            assert status['state'] in {'stopped_after_durable_batch', 'completed'}, status
            path = dest / 'progress.jsonl'
            assert not path.stat().st_size or path.read_bytes().endswith(b'\n')
            old_hashes[str(path)] = worker.file_hash(path)
            for line in path.open():
                r = json.loads(line)
                key = r['sample_id']
                assert key not in previous, ('duplicate prior ID', key)
                assert r['training_admitted'] is False
                previous[key] = r
                owners[key] = i
                if r['status'] == 'ok':
                    cache.setdefault(r['original_sha256'], {'prediction': r['prediction'], 'sample_id': key})

        with (worker.CPU / 'input_manifest.tsv').open() as f:
            fields = next(csv.reader(f, delimiter='\t'))
        total = db.execute('SELECT count(*) FROM samples').fetchone()[0]
        assert total == old_manifest['images']
        with db:
            db.execute('UPDATE samples SET done=0,owner=NULL')
            for key, r in previous.items():
                row = db.execute('SELECT cpu FROM samples WHERE id=?', (key,)).fetchone()
                assert row and json.loads(row[0]) == r['cpu_record'], ('upstream mismatch', key)
                db.execute('UPDATE samples SET done=1,owner=? WHERE id=?', (owners[key], key))
        assert db.execute('SELECT count(*) FROM samples WHERE done=1').fetchone()[0] == len(previous)
        # A100 throughput is intentionally weighted below H100; update only before freezing.
        weights = [1.11] * 7 + [0.57] * 4
        load = [0.0] * 11
        pending_groups = db.execute('SELECT digest,count(*) FROM samples WHERE done=0 GROUP BY digest ORDER BY count(*) DESC,digest').fetchall()
        with db:
            for digest, count in pending_groups:
                target = min(range(11), key=lambda i: (load[i] / weights[i], i))
                cost = (0.0 if digest.startswith('unreadable:') or digest in cache else 1.0) + count * 0.01
                load[target] += cost
                db.execute('UPDATE samples SET owner=? WHERE digest=? AND done=0', (target, digest))
        assert db.execute('SELECT count(*) FROM samples WHERE owner IS NULL').fetchone()[0] == 0
        db.execute('CREATE INDEX IF NOT EXISTS sample_owner ON samples(owner)')
        assert db.execute('SELECT count(*) FROM (SELECT digest FROM samples WHERE done=0 GROUP BY digest HAVING count(DISTINCT owner)>1)').fetchone()[0] == 0
        worker.atomic_json(OUT / 'successful_prediction_cache.json', cache)
        assignments = {}
        old_count = new_count = 0
        for i in range(11):
            dest = OUT / f'worker_{i}'
            dest.mkdir()
            states, routes = collections.Counter(), collections.Counter()
            count = completed = 0
            with (dest / 'input.tsv').open('w', newline='') as inp, (dest / 'upstream.jsonl').open('w') as cpu, (dest / 'progress.jsonl').open('w') as progress:
                writer = csv.DictWriter(inp, fieldnames=fields, delimiter='\t')
                writer.writeheader()
                for key, raw_input, raw_cpu, done in db.execute('SELECT id,input,cpu,done FROM samples WHERE owner=? ORDER BY id', (i,)):
                    row = json.loads(raw_input)
                    cr = json.loads(raw_cpu)
                    for field in ['image_path', 'identity_candidates', 'reservation_blocked', 'identity_review', 'sources']:
                        assert str(row[field]) == str(cr[field]), (key, field)
                    writer.writerow(row)
                    cpu.write(raw_cpu.rstrip('\n') + '\n')
                    count += 1
                    if done:
                        r = previous[key]
                        progress.write(json.dumps(r, ensure_ascii=False) + '\n')
                        states[r['status']] += 1
                        routes[r['triage']] += 1
                        completed += 1
            assignments[str(i)] = {'expected': count, 'migrated': completed,
                                   'input_sha256': worker.file_hash(dest / 'input.tsv'),
                                   'upstream_sha256': worker.file_hash(dest / 'upstream.jsonl')}
            old_count += completed
            new_count += count - completed
            print('Prepared worker', i, 'total', count, 'preserved', completed, flush=True)
            worker.atomic_json(dest / 'status.json', {'state': 'not_started', 'worker': i, 'expected': count,
                               'processed': completed, 'status_counts': dict(states), 'triage_counts': dict(routes),
                               'updated_at_unix': time.time(), 'training_admitted': 0})
        assert old_count == len(previous) and old_count + new_count == total
        for path, digest in old_hashes.items():
            assert worker.file_hash(path) == digest, 'Previous writer changed during migration'
        manifest = {**old_manifest, 'workers': 11, 'assignment': 'Frozen disjoint sample IDs; pending exact hashes kept together; 7 H100 + 4 A100',
                    'script_sha256': worker.file_hash(worker.__file__),
                    'worker_inputs': assignments, 'prediction_cache_sha256': worker.file_hash(OUT / 'successful_prediction_cache.json'),
                    'parent_manifest_sha256': worker.file_hash(OLD / 'manifest.json'), 'source_assignment_database_sha256': source_db_hash, 'relative_capacity_weights': weights}
        checks = {'total': total, 'preserved_completed': old_count, 'remaining_assigned': new_count,
                  'completed_remaining_overlap': 0, 'duplicate_assigned_ids': 0,
                  'pending_hashes_shared_across_workers': 0, 'previous_output_hashes': old_hashes,
                  'all_cpu_records_match_original_inputs': True, 'training_admitted': 0}
        worker.atomic_json(OUT / 'migration_checks.json', checks)
        worker.atomic_json(OUT / 'manifest.json', manifest)
        worker.atomic_json(OLD / 'superseded.json', {'successor': str(OUT), 'preserved_completed': old_count,
                           'reason': 'Expand to 7 H100 + 4 A100 with disjoint, content-grouped assignments', 'at_unix': time.time()})
        print(json.dumps({k: v for k, v in checks.items() if k != 'previous_output_hashes'}), flush=True)
        print(json.dumps(assignments), flush=True)
        db.close()
        shutil.copyfile(db_path, OUT / 'assignment_audit.sqlite')
        shutil.rmtree(staging)


if __name__ == '__main__':
    main()
