"""Freeze ten disjoint assignments after all seven previous writers stop."""
import collections
import contextlib
import csv
import fcntl
import json
import sqlite3
import shutil
import tempfile
import time
from pathlib import Path

import stream_qwen_review_v4 as worker

OLD = worker.ROOT / 'face_qc/qwen_stream_v2'
OUT = worker.OUT


def main():
    OUT.mkdir(exist_ok=False)
    old_manifest = json.loads((OLD / 'manifest.json').read_text())
    assert worker.file_hash(worker.CPU / 'manifest.json') == old_manifest['cpu_manifest_sha256']
    previous = {}
    owners = {}
    cache = {}
    old_hashes = {}
    with contextlib.ExitStack() as locks:
        for i in range(7):
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

        staging = Path(tempfile.mkdtemp(prefix='qwen_assignment_'))
        db_path = staging / 'assignment_audit.sqlite'
        db = sqlite3.connect(db_path)
        db.execute('PRAGMA cache_size=-262144')
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('PRAGMA temp_store=MEMORY')
        db.execute('CREATE TABLE samples (id TEXT PRIMARY KEY, input TEXT NOT NULL, cpu TEXT, digest TEXT, done INTEGER DEFAULT 0, owner INTEGER)')
        with (worker.CPU / 'input_manifest.tsv').open() as f:
            reader = csv.DictReader(f, delimiter='\t')
            fields = reader.fieldnames
            with db:
                db.executemany('INSERT INTO samples(id,input) VALUES (?,?)', ((r['sample_id'], json.dumps(r)) for r in reader))
        total = db.execute('SELECT count(*) FROM samples').fetchone()[0]
        print('Loaded original input IDs', total, flush=True)
        assert total == old_manifest['images']
        for shard in range(old_manifest['shards']):
            path = worker.CPU / f'input_{shard:03d}.tsv'
            assert worker.file_hash(path) == old_manifest['input_sha256'][str(path)]
            with db, (worker.CPU / f'shard_{shard:03d}/progress.jsonl').open() as f:
                for line in f:
                    r = json.loads(line)
                    key = r['sample_id']
                    digest = r.get('original_sha256') or 'unreadable:' + key
                    if key in previous:
                        assert previous[key]['cpu_record'] == r, ('prior upstream changed', key)
                    changed = db.execute('UPDATE samples SET cpu=?,digest=?,done=?,owner=? WHERE id=? AND cpu IS NULL',
                                         (line, digest, int(key in previous), owners.get(key), key)).rowcount
                    assert changed == 1, ('duplicate or unexpected upstream ID', key)
            if shard % 8 == 7:
                print('Verified upstream shards', shard + 1, flush=True)
        assert db.execute('SELECT count(*) FROM samples WHERE cpu IS NULL').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM samples WHERE done=1').fetchone()[0] == len(previous)
        db.execute('CREATE INDEX sample_digest ON samples(digest)')
        # A100 throughput is intentionally weighted below H100; update only before freezing.
        weights = [1.18] * 7 + [0.60] * 3
        load = [0.0] * 10
        pending_groups = db.execute('SELECT digest,count(*) FROM samples WHERE done=0 GROUP BY digest ORDER BY count(*) DESC,digest').fetchall()
        with db:
            for digest, count in pending_groups:
                target = min(range(10), key=lambda i: (load[i] / weights[i], i))
                cost = (0.0 if digest.startswith('unreadable:') or digest in cache else 1.0) + count * 0.01
                load[target] += cost
                db.execute('UPDATE samples SET owner=? WHERE digest=? AND done=0', (target, digest))
        assert db.execute('SELECT count(*) FROM samples WHERE owner IS NULL').fetchone()[0] == 0
        db.execute('CREATE INDEX sample_owner ON samples(owner)')
        assert db.execute('SELECT count(*) FROM (SELECT digest FROM samples WHERE done=0 GROUP BY digest HAVING count(DISTINCT owner)>1)').fetchone()[0] == 0
        worker.atomic_json(OUT / 'successful_prediction_cache.json', cache)
        assignments = {}
        old_count = new_count = 0
        for i in range(10):
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
        manifest = {**old_manifest, 'workers': 10, 'assignment': 'Frozen disjoint sample IDs; pending exact hashes kept together; 7 H100 + 3 A100',
                    'script_sha256': worker.file_hash(worker.__file__),
                    'worker_inputs': assignments, 'prediction_cache_sha256': worker.file_hash(OUT / 'successful_prediction_cache.json'),
                    'parent_manifest_sha256': worker.file_hash(OLD / 'manifest.json'), 'relative_capacity_weights': weights}
        checks = {'total': total, 'preserved_completed': old_count, 'remaining_assigned': new_count,
                  'completed_remaining_overlap': 0, 'duplicate_assigned_ids': 0,
                  'pending_hashes_shared_across_workers': 0, 'previous_output_hashes': old_hashes,
                  'all_cpu_records_match_original_inputs': True, 'training_admitted': 0}
        worker.atomic_json(OUT / 'migration_checks.json', checks)
        worker.atomic_json(OUT / 'manifest.json', manifest)
        worker.atomic_json(OLD / 'superseded.json', {'successor': str(OUT), 'preserved_completed': old_count,
                           'reason': 'Expand to 7 H100 + 3 A100 with disjoint, content-grouped assignments', 'at_unix': time.time()})
        print(json.dumps({k: v for k, v in checks.items() if k != 'previous_output_hashes'}), flush=True)
        print(json.dumps(assignments), flush=True)
        db.close()
        shutil.copyfile(db_path, OUT / 'assignment_audit.sqlite')
        shutil.rmtree(staging)


if __name__ == '__main__':
    main()
