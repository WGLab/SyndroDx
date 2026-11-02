"""Refresh aggregate stream progress and validate coverage before final export."""
import collections
import csv
import json
import time
from stream_qwen_review import OUT, CPU, WORKERS, atomic_json


def snapshot():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    workers = []
    statuses, routes = collections.Counter(), collections.Counter()
    for i in range(WORKERS):
        p = OUT / f'worker_{i}' / 'status.json'
        d = json.loads(p.read_text()) if p.exists() else {'worker': i, 'state': 'not_started', 'processed': 0}
        workers.append(d)
        statuses.update(d.get('status_counts', {}))
        routes.update(d.get('triage_counts', {}))
    completed = all(d['state'] == 'completed' for d in workers)
    result = {'expected_original_paths': manifest['images'], 'processed_paths': sum(d['processed'] for d in workers),
              'qwen_labelled_paths': statuses['ok'], 'status_counts': dict(statuses), 'triage_counts': dict(routes),
              'all_workers_completed': completed, 'workers': workers, 'updated_at_unix': time.time(),
              'training_admitted': 0, 'completeness_validated': False}
    atomic_json(OUT / 'summary.json', result)
    return result


def merge():
    with (CPU / 'input_manifest.tsv').open() as f:
        expected = {r['sample_id'] for r in csv.DictReader(f, delimiter='\t')}
    seen = set()
    temp = OUT / 'results.tsv.tmp'
    with temp.open('w', newline='') as output:
        writer = None
        for i in range(WORKERS):
            with (OUT / f'worker_{i}' / 'results.tsv').open() as source:
                reader = csv.DictReader(source, delimiter='\t')
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, delimiter='\t')
                    writer.writeheader()
                if reader.fieldnames != writer.fieldnames:
                    raise RuntimeError('Worker export schemas differ')
                for row in reader:
                    if row['sample_id'] in seen or row['sample_id'] not in expected:
                        raise RuntimeError('Duplicate or unexpected final sample ID')
                    if row['training_admitted'] != 'False':
                        raise RuntimeError('Unexpected training admission')
                    seen.add(row['sample_id'])
                    writer.writerow(row)
    if seen != expected:
        raise RuntimeError('Incomplete final coverage')
    temp.replace(OUT / 'results.tsv')
    atomic_json(OUT / 'coverage_checks.json', {'expected': len(expected), 'observed_unique': len(seen),
                'complete_coverage': True, 'duplicate_ids': 0, 'training_admitted': 0})


if __name__ == '__main__':
    while True:
        result = snapshot()
        print(json.dumps({k: result[k] for k in ['processed_paths', 'qwen_labelled_paths', 'all_workers_completed', 'status_counts']}), flush=True)
        if result['all_workers_completed']:
            merge()
            break
        time.sleep(60)
