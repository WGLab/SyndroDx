"""Report only the effective local cgroup memory limit, without reading data."""
import json,os
from pathlib import Path
limits=[]
for line in Path('/proc/self/cgroup').read_text().splitlines():
 _,controllers,relative=line.split(':',2)
 prefixes=[Path('/sys/fs/cgroup')] if not controllers else [Path('/sys/fs/cgroup/memory')] if 'memory' in controllers.split(',') else []
 for prefix in prefixes:
  leaf=prefix/relative.lstrip('/')
  while leaf==prefix or leaf.is_relative_to(prefix):
   for name in ['memory.max','memory.limit_in_bytes']:
    try:value=(leaf/name).read_text().strip()
    except OSError:continue
    if value.isdigit() and 0<int(value)<2**60:limits.append(int(value))
   if leaf==prefix:break
   leaf=leaf.parent
print(json.dumps({'effective_cgroup_memory_limit_GiB':round(min(limits)/2**30,2) if limits else None,'Slurm_mem_per_gpu_MB':os.environ.get('SLURM_MEM_PER_GPU'),'Slurm_mem_per_node_MB':os.environ.get('SLURM_MEM_PER_NODE')}))
