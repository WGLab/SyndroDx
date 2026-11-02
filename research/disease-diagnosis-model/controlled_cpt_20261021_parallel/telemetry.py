"""Numeric resource telemetry only: never records inputs, identifiers or model tensors."""
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def memory_snapshot():
    rows = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, path = line.split(':', 2)
        if 'memory' not in controllers.split(',') and controllers != '':
            continue
        root = Path('/sys/fs/cgroup/memory' if controllers else '/sys/fs/cgroup')
        leaf = root / path.lstrip('/')
        while leaf != root and root in leaf.parents:
            try:
                if controllers:
                    limit = int((leaf/'memory.limit_in_bytes').read_text())
                    usage = int((leaf/'memory.usage_in_bytes').read_text())
                    peak = int((leaf/'memory.max_usage_in_bytes').read_text())
                    stats = dict(x.split() for x in (leaf/'memory.stat').read_text().splitlines())
                    resident = int(stats.get('total_rss', stats.get('rss', 0)))
                    failed = int((leaf/'memory.failcnt').read_text())
                else:
                    raw = (leaf/'memory.max').read_text().strip()
                    limit = int(raw) if raw != 'max' else 2**63
                    usage = int((leaf/'memory.current').read_text())
                    peak_path = leaf/'memory.peak'
                    peak = int(peak_path.read_text()) if peak_path.exists() else usage
                    stats = dict(x.split() for x in (leaf/'memory.stat').read_text().splitlines())
                    resident = int(stats.get('anon', 0))
                    events = dict(x.split() for x in (leaf/'memory.events').read_text().splitlines())
                    failed = int(events.get('oom_kill', 0))
                if 0 < limit < 2**60:
                    rows.append(dict(limit_GiB=limit/2**30, usage_GiB=usage/2**30,
                                     peak_GiB=peak/2**30, resident_GiB=resident/2**30,
                                     resident_fraction=resident/limit, failures=failed))
            except (OSError, ValueError):
                pass
            leaf = leaf.parent
    return dict(cgroups=rows, max_host_resident_fraction=max((r['resident_fraction'] for r in rows), default=0.))


class Monitor:
    def __init__(self, folder, rank):
        self.folder, self.rank = folder, rank
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def sample(self):
        from common import save, now
        import torch
        value = dict(updated_at_utc=now(), rank=self.rank, **memory_snapshot())
        if torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            value.update(gpu_allocated_GiB=torch.cuda.memory_allocated()/2**30,
                         gpu_reserved_GiB=torch.cuda.memory_reserved()/2**30,
                         gpu_peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                         gpu_free_GiB=free/2**30, gpu_total_GiB=total/2**30)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                value['devices'] = [dict(utilization_percent=int(a), memory_used_MiB=int(b))
                    for a, b in (line.split(',') for line in result.stdout.strip().splitlines())]
        except (OSError, ValueError, subprocess.TimeoutExpired):
            value['gpu_query_unavailable'] = True
        save(self.folder/f'rank_{self.rank:02d}.json', value)
        with (self.folder/f'rank_{self.rank:02d}.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.sample()
            except Exception:
                # A sample failure must not crash training; checked separately at admission.
                pass
            self.stop_event.wait(30)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)

