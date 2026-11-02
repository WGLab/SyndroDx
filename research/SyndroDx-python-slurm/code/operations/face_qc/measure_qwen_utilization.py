"""Read-only 30-second GPU and useful-output throughput sample."""
import json
from pathlib import Path
import statistics
import subprocess
import time

root = Path(__file__).resolve().parent / 'qwen_stream_v2'


def counts():
    statuses = [json.loads((root / f'worker_{i}/status.json').read_text()) for i in range(7)]
    return {'processed': sum(d['processed'] for d in statuses),
            'labelled': sum(d['status_counts'].get('ok', 0) for d in statuses),
            'workers': statuses}


start = time.monotonic()
before = counts()
samples = []
for _ in range(15):
    csv = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = [[float(v.strip()) for v in line.split(',')] for line in csv.splitlines()]
    samples.append({'time': time.time(), 'gpus': rows})
    time.sleep(2)
after = counts()
elapsed = time.monotonic() - start
gpu_stats = []
for index in range(7):
    rows = [row for s in samples for row in s['gpus'] if int(row[0]) == index]
    gpu_stats.append({'gpu': index, 'mean_utilization_pct': round(statistics.mean(r[1] for r in rows), 1),
                      'min_utilization_pct': min(r[1] for r in rows), 'max_utilization_pct': max(r[1] for r in rows),
                      'max_memory_GiB': round(max(r[2] for r in rows) / 1024, 2)})
result = {'duration_seconds': round(elapsed, 2), 'before': before, 'after': after,
          'new_labelled': after['labelled'] - before['labelled'],
          'labelled_images_per_second': round((after['labelled'] - before['labelled']) / elapsed, 3),
          'gpu_stats': gpu_stats, 'raw_samples': samples}
(root / 'utilization_measurement.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['duration_seconds', 'new_labelled', 'labelled_images_per_second', 'gpu_stats']}))
