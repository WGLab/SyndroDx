"""Aggregate telemetry for the allocated training GPU; never reads training data."""
import json,os,subprocess,time
from pathlib import Path
from runtime import ROOT
owned=set()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
  if proc.stat().st_uid==os.getuid() and str(ROOT/'train_rank.py') in args:owned.add(int(proc.name))
 except (OSError,UnicodeError):continue
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
uuids={line.split(',')[1].strip() for line in apps.splitlines() if line.split(',')[0].strip().isdigit() and int(line.split(',')[0].strip()) in owned}
util={u:[] for u in uuids};memory={u:[] for u in uuids}
for sample in range(12):
 result=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
 for line in result.splitlines():
  uuid,value,used,total=[v.strip() for v in line.split(',')]
  if uuid in util:util[uuid].append(float(value));memory[uuid].append(float(used)/1024)
 if sample<11:time.sleep(3)
print(json.dumps({'training_GPUs':len(uuids),'mean_utilization_percent':round(sum(sum(v) for v in util.values())/sum(map(len,util.values())),1) if uuids else None,'minimum_sample_percent':min(min(v) for v in util.values()) if uuids else None,'mean_memory_used_GiB':round(sum(sum(v) for v in memory.values())/sum(map(len,memory.values())),1) if uuids else None,'sample_seconds':33}))
