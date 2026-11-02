"""NVML telemetry inside authorized GPU cgroups; never reads project data."""
import json
import os
import subprocess
import time
from pathlib import Path
from runtime import ROOT

samples = []
owned = set()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = [value.decode() for value in (proc / 'cmdline').read_bytes().split(b'\0') if value]
        if proc.stat().st_uid == os.getuid() and any(str(ROOT / script) in args for script in ['train_rank.py', 'evaluation_worker.py']):
            owned.add(int(proc.name))
    except (OSError, UnicodeError):
        continue
apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
                      capture_output=True, text=True, check=True, timeout=10)
uuids = {line.split(',')[1].strip() for line in apps.stdout.splitlines()
         if line.split(',')[0].strip().isdigit() and int(line.split(',')[0].strip()) in owned}
for attempt in range(3):
    command = ['nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    for line in result.stdout.splitlines():
        uuid, *values = [v.strip() for v in line.split(',')]
        if uuid in uuids:
            samples.append([float(v) for v in values])
    if attempt < 2:
        time.sleep(2)
if not samples:
    print(json.dumps(dict(state='no_project_GPU_process_observed', sample_seconds=4)))
else:
    print(json.dumps(dict(mean_utilization_percent=sum(v[0] for v in samples) / len(samples),
                     minimum_utilization_percent=min(v[0] for v in samples),
                     maximum_memory_used_GiB=max(v[1] for v in samples) / 1024,
                     minimum_total_memory_GiB=min(v[2] for v in samples) / 1024,
                     observed_project_GPUs=len(uuids), sample_seconds=4)))
