"""Read-only live audit: no duplicate IDs or repeated new content inference."""
import collections
import json
from stream_qwen_review_v4 import OUT, atomic_json

old = OUT.parent / 'qwen_stream_v2'
old_ids = set()
old_records = {}
old_success_hashes = set()
for path in sorted(old.glob('worker_*/progress.jsonl')):
    for line in path.open():
        r = json.loads(line)
        assert r['sample_id'] not in old_ids
        old_ids.add(r['sample_id'])
        old_records[r['sample_id']] = r
        if r['status'] == 'ok':
            old_success_hashes.add(r['original_sha256'])
seen, new_hashes = set(), set()
counts = collections.Counter()
states = []
for i in range(10):
    dest = OUT / f'worker_{i}'
    states.append(json.loads((dest / 'status.json').read_text()))
    with (dest / 'progress.jsonl').open('rb') as f:
        for line in f:
            if not line.endswith(b'\n'):
                break
            r = json.loads(line)
            key = r['sample_id']
            assert key not in seen, ('duplicate output ID', key)
            seen.add(key)
            assert r['training_admitted'] is False
            if key in old_ids:
                assert r == old_records[key], ('migrated record altered', key)
                counts['preserved_completed'] += 1
            else:
                counts['new_records'] += 1
                if r.get('raw_response') is not None and not r.get('cache_hit'):
                    digest = r['original_sha256']
                    assert digest not in old_success_hashes, 'Previously successful content inferred again'
                    assert digest not in new_hashes, 'New content inferred multiple times'
                    new_hashes.add(digest)
                    counts['new_unique_content_inferences'] += 1
                if r.get('cache_hit'):
                    counts['new_content_cache_hits'] += 1
                counts['new_status_' + r['status']] += 1
assert old_ids <= seen, 'Lost prior completed results'
result = {'counts': dict(counts), 'unique_output_ids': len(seen), 'duplicate_output_ids': 0,
          'previous_records_preserved_exactly': True, 'repeated_new_content_inferences': 0,
          'previously_successful_content_reinferred': 0, 'training_admitted': 0,
          'worker_states': [{'worker': r['worker'], 'state': r['state'], 'processed': r['processed']} for r in states]}
atomic_json(OUT / 'live_integrity_checks.json', result)
print(json.dumps(result, indent=2))
